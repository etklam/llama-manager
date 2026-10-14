#!/usr/bin/env python3
"""Run a real local image request and translation using one Vulkan runtime."""
import argparse
from dataclasses import asdict
from pathlib import Path
import platform
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from manga.images import collect
from manga.runtime import Profile, inspect_runtime
from manga.runner import MangaRunner, RunSnapshot
from manga.store import write_json
from server_controller import ServerController


def main(argv=None):
    parser = argparse.ArgumentParser(description='Real manga Vulkan OCR + translation verification (no cloud/fallback).')
    parser.add_argument('--runtime', required=True, help='llama-server executable')
    parser.add_argument('--device', required=True, help='Enumerated Vulkan device, e.g. Vulkan0')
    parser.add_argument('--ocr-model', help='GLM-OCR Q8_0 GGUF candidate')
    parser.add_argument('--mmproj', help='Matching GLM-OCR projector GGUF')
    parser.add_argument('--translation-model', help='Text GGUF')
    parser.add_argument('--samples', help='User-owned PNG/JPEG/WEBP directory (nonrecursive)')
    parser.add_argument('--report-dir', required=True)
    parser.add_argument('--context', type=int, default=8192)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--locale', default='zh-tw')
    parser.add_argument('--source', default='ja')
    parser.add_argument('--port', type=int, default=8091)
    parser.add_argument('--inspect-only', action='store_true', help='Record runtime/devices; real OCR and quality remain NOT RUN')
    args = parser.parse_args(argv)
    directory = Path(args.report_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    report = {'software': 'Run python -m pytest separately', 'real_vulkan': 'NOT RUN',
              'manga_quality': 'NOT RUN', 'platform': platform.platform(), 'driver': 'unknown',
              'reason': 'Runtime inspection only; no image or translation request executed'}
    try:
        report['runtime'] = inspect_runtime(args.runtime)
        report['driver'] = report['runtime'].get('drivers', 'unknown')
        if args.device not in report['runtime']['devices']:
            raise ValueError('Requested Vulkan device unavailable')
        if not args.inspect_only:
            if not all([args.ocr_model, args.mmproj, args.translation_model, args.samples]):
                raise ValueError('Real verification requires OCR model, mmproj, translation model and samples')
            files = collect([args.samples], directory)
            if not files:
                raise ValueError('No supported sample images')
            # A fresh output directory guarantees this is a real request test,
            # never a cache-only run misreported as hardware verification.
            snapshot = RunSnapshot(tuple(files), str(directory / ('run-' + uuid.uuid4().hex[:12])),
                args.runtime, args.device,
                Profile(args.ocr_model, args.mmproj, args.context, args.max_tokens),
                Profile(args.translation_model, '', args.context, args.max_tokens, .2),
                source=args.source, locale=args.locale, port=args.port)
            runner = MangaRunner(ServerController(Path(args.runtime), lambda line: None), snapshot)
            report['run'] = runner.run()
            stages = {stage['stage']: stage for stage in report['run']['stages']}
            verified = (report['run']['status'] == 'completed' and all(
                key in stages and stages[key]['status'] == 'verified' for key in ('ocr', 'translation')))
            report['real_vulkan'] = 'PASS' if verified else 'UNVERIFIED/FAILED'
            report['manga_quality'] = 'NOT RUN: human transcription/order review required'
            report['samples'] = len(files)
            report['reason'] = 'See per-stage decoder/vision evidence and request results'
    except Exception as error:
        report['error'] = f'{type(error).__name__}: {error}'
    write_json(directory / 'verification.json', report)
    print(f"Report: {directory / 'verification.json'}")
    print(f"Vulkan: {report['real_vulkan']}; quality: {report['manga_quality']}")
    return 0 if args.inspect_only and 'error' not in report or report['real_vulkan'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
