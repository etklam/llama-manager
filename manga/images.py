"""Read-only image ingestion and deterministic source identities."""
import base64
import hashlib
import io
import json
import re
from pathlib import Path
from PIL import Image, ImageOps

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.webp'}
PREPROCESS = 'exif-white-rgb-png-v1-no-resize'
MAX_PIXELS = 24_000_000
MAX_BYTES = 64 * 1024 * 1024


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def natural_key(path):
    return tuple((0, int(x)) if x.isdigit() else (1, x.casefold())
                 for x in re.split(r'(\d+)', str(path)))


def collect(paths, output=None):
    result = []
    excluded = Path(output).resolve() if output else None
    for name in sorted(paths, key=natural_key):
        path = Path(name).resolve()
        items = sorted(path.iterdir(), key=natural_key) if path.is_dir() else [path]
        for item in items:
            if item.is_file() and item.suffix.lower() in EXTENSIONS:
                if excluded and (item == excluded or excluded in item.parents):
                    continue
                result.append(str(item))
    return result


def _image_payload(path):
    path = Path(path)
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Image exceeds 64 MiB; split the page first')
    with Image.open(path) as original:
        if original.format not in ('PNG', 'JPEG', 'WEBP'):
            raise ValueError('Unsupported image format')
        width, height = original.size
        if width * height > MAX_PIXELS or max(width, height) > 16000:
            raise ValueError('Image too large/long; split the page first (no automatic resize)')
        if getattr(original, 'n_frames', 1) != 1:
            raise ValueError('Animated images are not supported')
        original.load()
        oriented = ImageOps.exif_transpose(original)
        rgba = oriented.convert('RGBA')
        white = Image.new('RGBA', rgba.size, 'white')
        white.alpha_composite(rgba)
        result = io.BytesIO()
        white.convert('RGB').save(result, format='PNG')
    return 'data:image/png;base64,' + base64.b64encode(result.getvalue()).decode('ascii')


def image_payload(path):
    try:
        return _image_payload(path)
    except Image.DecompressionBombError as error:
        raise ValueError('Image exceeds safe pixel limit; split the page first') from error
