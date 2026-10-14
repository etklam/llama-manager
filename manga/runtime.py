"""Runtime capability discovery and evidence, never a backend fallback."""
from dataclasses import dataclass
from pathlib import Path
import re
import json
import platform
import shutil
import subprocess
import time
import httpx


@dataclass(frozen=True)
class Profile:
    model: str
    mmproj: str = ''
    context: int = 8192
    max_tokens: int = 2048
    temperature: float = 0.0

    def validate(self):
        for value in (self.model, self.mmproj):
            if value and not Path(value).is_file():
                raise ValueError('Model/projector file does not exist')
        if not self.model or self.context < 2048 or not 128 <= self.max_tokens < self.context:
            raise ValueError('Invalid model/context/output budget')
        if not 0 <= self.temperature <= 2:
            raise ValueError('Invalid temperature')


def driver_inventory():
    try:
        if platform.system() == 'Windows':
            shell = shutil.which('powershell')
            if shell:
                result = subprocess.run([shell, '-NoProfile', '-Command',
                    'Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion | ConvertTo-Json -Compress'],
                    capture_output=True, encoding='utf-8', errors='replace', timeout=15,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                if result.returncode == 0:
                    return json.loads(result.stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return 'unknown'


def inspect_runtime(executable):
    if not Path(executable).is_file():
        raise ValueError('Select an existing llama-server executable')
    result = {}
    for key, flag in [('version', '--version'), ('help', '--help'), ('device_list', '--list-devices')]:
        proc = subprocess.run([str(executable), flag], capture_output=True,
                              encoding='utf-8', errors='replace', timeout=30,
                              creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if proc.returncode:
            raise RuntimeError(f'Runtime discovery failed: {flag} ({proc.returncode})')
        result[key] = proc.stdout + proc.stderr
    result['drivers'] = driver_inventory()
    result['devices'] = re.findall(r'^\s*(Vulkan\d+):', result['device_list'], re.M)
    return result


def command(executable, discovery, device, profile, port):
    profile.validate()
    if device not in discovery['devices']:
        raise ValueError('Selected Vulkan device was not enumerated by this runtime')
    args = [str(executable), '-m', profile.model, '--host', '127.0.0.1',
            '--port', str(port), '--device', device, '-ngl', '999',
            '-c', str(profile.context), '-b', '256', '--parallel', '1',
            '--flash-attn', 'off', '--cache-type-k', 'f16', '--cache-type-v', 'f16',
            '--jinja']
    if '--fit' in discovery['help']:
        args += ['--fit', 'off']
    if profile.mmproj:
        args += ['--mmproj', profile.mmproj, '--mmproj-offload']
        # Older runtimes do not expose projector selection: evidence must
        # confirm the actual device after a real image request in that case.
        if '--mmproj-device' in discovery['help']:
            args += ['--mmproj-device', device]
    for flag in args[1:]:
        if flag.startswith('-') and not re.search(r'(?<![\w-])' + re.escape(flag) + r'(?![\w-])', discovery['help']):
            raise ValueError(f'Runtime does not advertise required flag: {flag}')
    return args


def evidence(lines, device, vision=False, request_ok=False):
    lines = list(lines)
    text = '\n'.join(lines)
    layers = re.findall(r'offloaded (\d+)/(\d+) layers to GPU', text)
    decoder = 'unknown'
    if any(int(a) == int(b) and int(a) > 0 for a, b in layers) and re.search(
            re.escape(device) + r'\s+compute buffer', text):
        decoder = device
    if layers and all(int(a) == 0 for a, _ in layers):
        decoder = 'CPU-only'
    matches = re.findall(r'CLIP using (\S+) backend', text)
    projector = matches[-1] if matches else 'unknown'
    selected = [line for line in lines if re.search(
        r'offloaded \d+/\d+ layers|CLIP using \S+ backend|(?:Vulkan\d+|CPU)\s+(?:model|compute) buffer', line)]
    vision_compute = bool(re.search(r'(?:reserve_compute_meta|alloc_compute_meta):\s*' +
                                    re.escape(device) + r'\s+compute buffer size\s*=\s*(?!0+(?:\.0+)?\s)([0-9.]+)', text))
    return {'vision_compute_observed': vision_compute, 'requested_device': device, 'decoder_backend': decoder,
            'vision_backend': projector if vision else 'not_applicable',
            'request_ok': request_ok, 'evidence': selected,
            'status': 'verified' if request_ok and decoder == device and
            (not vision or (projector == device and vision_compute)) else 'unverified'}


def wait_ready(server, port, cancelled, timeout=120):
    deadline = time.monotonic() + timeout
    with httpx.Client(trust_env=False, timeout=2) as http:
        while time.monotonic() < deadline:
            if cancelled():
                raise RuntimeError('Cancelled during model loading')
            if not server.running:
                raise RuntimeError('Managed runtime exited during model loading')
            try:
                response = http.get(f'http://127.0.0.1:{port}/health')
                if response.status_code == 200:
                    check_alive = getattr(server, 'assert_alive', None)
                    if check_alive:
                        check_alive()
                    return
                if response.status_code != 503:
                    raise RuntimeError(f'Unexpected health status {response.status_code}')
            except httpx.TransportError:
                pass
            time.sleep(.1)
    raise RuntimeError('Model readiness timed out')
