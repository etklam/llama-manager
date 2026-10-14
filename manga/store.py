"""Validated atomic checkpoints. Source images are never modified."""
from pathlib import Path
import json
import re
from manga.images import digest, file_hash, PREPROCESS
from utils.atomic_io import atomic_write_text


def write_json(path, value):
    if isinstance(value, dict):
        value = {**value, '_checksum': digest(value)}
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def read_json(path):
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8'))
        if isinstance(value, dict):
            checksum = value.pop('_checksum', None)
            if checksum != digest(value):
                raise ValueError('Checkpoint integrity check failed')
        return value
    except (ValueError, OSError) as error:
        raise ValueError(f'Corrupted checkpoint: {Path(path).name}') from error


def page_sources(paths):
    occurrences = {}
    result = []
    for name in paths:
        path = str(Path(name).resolve())
        occurrence = occurrences.get(path, 0)
        occurrences[path] = occurrence + 1
        result.append({'id': digest([path, occurrence])[:24], 'path': path})
    return result


def documents(paths, independent):
    sources = page_sources(paths)
    groups = [[p] for p in sources] if independent else [sources]
    return [(digest(sorted(p['id'] for p in group))[:24], group) for group in groups if group]


class Store:
    def __init__(self, output, document_id, sources, gate):
        self.root = Path(output) / document_id
        self.sources, self.gate = sources, gate
        self.manifest = {'version': 1, 'document_id': document_id, 'status': 'pending',
                         'pages': [p['id'] for p in sources]}
        self.old_manifest = {}
        manifest = self.root / 'manifest.json'
        if manifest.exists():
            old = read_json(manifest)
            self.old_manifest = old
            if not isinstance(old, dict) or old.get('version') != 1 or old.get('document_id') != document_id:
                raise ValueError('Invalid checkpoint manifest identity/version')
        self.pages = []

    def commit(self, path, value):
        if not self.gate.commit(lambda: write_json(path, value)):
            from translation.transport import TransportCancelledError
            raise TransportCancelledError('Cancelled before checkpoint publication')

    def source_revision_key(self):
        return digest([(p['id'], p['fingerprint'], p['revision'], p['status'], p['segments']) for p in self.pages])

    def save_manifest(self, status):
        self.manifest['source_revision_key'] = self.source_revision_key()
        self.manifest['status'] = status
        self.commit(self.root / 'manifest.json', self.manifest)

    def load_page(self, source, ocr_identity):
        path = self.root / 'pages' / (source['id'] + '.json')
        source_hash = file_hash(source['path'])
        identity = digest([source_hash, ocr_identity, PREPROCESS])
        page = None
        if path.exists():
            page = read_json(path)
            if (not isinstance(page, dict) or page.get('id') != source['id']
                    or page.get('path') != source['path'] or not isinstance(page.get('revision'), int)
                    or page.get('revision', -1) < 0 or not isinstance(page.get('warnings'), list)):
                raise ValueError('Invalid page identity')
            if not isinstance(page.get('segments'), list) or not isinstance(page.get('raw'), str):
                raise ValueError('Invalid page schema')
            ids = []
            for segment in page['segments']:
                if (not isinstance(segment, dict) or not isinstance(segment.get('text'), str)
                        or not isinstance(segment.get('id'), str)
                        or not segment['id'].startswith(source['id'] + '-')):
                    raise ValueError('Invalid checkpoint segment')
                ids.append(segment['id'])
            if len(ids) != len(set(ids)) or page.get('status') not in ('pending', 'ok', 'failed', 'needs_review', 'no_text'):
                raise ValueError('Invalid checkpoint status/duplicate IDs')
            if page['status'] == 'no_text' and page['segments']:
                raise ValueError('no_text page cannot contain segments')
            if page['status'] == 'ok' and not page['segments']:
                raise ValueError('Invalid empty successful checkpoint')
            if page.get('fingerprint') != identity:
                page = None
        if page is None:
            page = {**source, 'fingerprint': identity, 'source_hash': source_hash,
                    'status': 'pending', 'raw': '', 'revision': 0, 'segments': [], 'warnings': []}
        self.pages.append(page)
        return page

    def save_page(self, page):
        self.commit(self.root / 'pages' / (page['id'] + '.json'), page)

    def set_ocr(self, page, raw, status, warning=''):
        page['raw'], page['status'] = raw, status
        page['warnings'] = [warning] if warning else []
        page['segments'] = [{'id': f"{page['id']}-{i}", 'text': text, 'bbox': None, 'confidence': None}
                            for i, text in enumerate(re.split(r'\n\s*\n', raw.strip())) if text.strip()]
        self.save_page(page)

    def segments(self):
        return [s for page in self.pages if page['status'] == 'ok' for s in page['segments']]

    def translation_key(self, settings):
        return digest([settings, [(p['id'], p['fingerprint'], p['revision'], p['status'], p['segments'])
                                  for p in self.pages]])

    def invalidate_exports(self, locale=None):
        # Invalidation happens before new work; stale formal exports cannot
        # look like the current document after source/profile changes.
        locale = locale or '*'
        for pattern in (f'translations.{locale}.json', f'translated.{locale}.txt', f'bilingual.{locale}.md'):
            for path in self.root.glob(pattern):
                if '.partial.' not in path.name:
                    def move(path=path):
                        path.replace(path.with_name(path.stem + '.partial' + path.suffix))
                    if not self.gate.commit(move):
                        raise RuntimeError('Cancelled during invalidation')

    def export(self, locale, translations=None, completed=False):
        if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', locale):
            raise ValueError('Invalid locale code')
        self.commit(self.root / 'ocr.json', {'pages': self.pages, 'status': self.manifest['status']})
        source = '\n\n'.join(f"[{p['id']} / {p['status']}]\n" + '\n\n'.join(s['text'] for s in p['segments'])
                             for p in self.pages)
        self.gate.commit(lambda: atomic_write_text(self.root / 'source.txt', source))
        if translations is None:
            return
        suffix = '' if completed else '.partial'
        values = translations.get('segments', {})
        missing = [p['id'] for p in self.pages if p['status'] not in ('ok', 'no_text')]
        header = f"Status: {'completed' if completed else 'PARTIAL'}\nMissing/review pages: {', '.join(missing) or 'none'}\n"
        texts, bilingual = [header], [header]
        for p in self.pages:
            bilingual.append(f"\n## {p['id']} ({p['status']})\n")
            for segment in p['segments']:
                value = values.get(segment['id'], '[MISSING TRANSLATION]')
                texts.append(value)
                # A fence longer than any data fence keeps Markdown data literal.
                longest = max([len(m) for m in re.findall(r'`+', segment['text'] + value)] or [2])
                fence = '`' * max(3, longest + 1)
                bilingual.extend([f"[{segment['id']}]", '原文', fence + 'text', segment['text'],
                                  fence, '譯文', fence + 'text', value, fence, ''])
        payload = {**translations, 'status': 'completed' if completed else 'partial', 'missing_pages': missing}
        self.commit(self.root / f'translations.{locale}{suffix}.json', payload)
        for name, body in [('translated', '\n\n'.join(texts)), ('bilingual', '\n\n'.join(bilingual))]:
            ext = 'md' if name == 'bilingual' else 'txt'
            self.gate.commit(lambda name=name, body=body, ext=ext: atomic_write_text(
                self.root / f'{name}.{locale}{suffix}.{ext}', body))


def correct_page(root, page_id, texts, no_text=False):
    if not re.fullmatch(r'[a-f0-9]{24}', page_id):
        raise ValueError('Invalid page ID')
    root = Path(root)
    path = root / 'pages' / (page_id + '.json')
    page = read_json(path)
    if page.get('id') != page_id:
        raise ValueError('Page identity mismatch')
    texts = [t for t in texts if t.strip()]
    if not texts and not no_text:
        raise ValueError('Empty text requires explicit no_text confirmation')
    # Match each occurrence once, preserving duplicate dialogue identities.
    remaining = list(page['segments'])
    segments = []
    page['revision'] += 1
    for i, text in enumerate(texts):
        old = next((s for s in remaining if s['text'] == text), None)
        if old:
            remaining.remove(old)
            segments.append(old)
        else:
            segments.append({'id': f"{page_id}-r{page['revision']}-{i}", 'text': text,
                             'bbox': None, 'confidence': None})
    page['segments'], page['status'] = segments, 'no_text' if no_text and not texts else 'ok'
    manifest = read_json(root / 'manifest.json')
    manifest['status'] = 'needs_translation'
    # Manifest is invalidated first, then revision checkpoint is authoritative.
    write_json(root / 'manifest.json', manifest)
    from utils.atomic_io import CommitGate
    store = Store(root.parent, root.name, [], CommitGate())
    store.invalidate_exports()
    write_json(path, page)
    return page


def reorder_document(root, page_ids):
    root = Path(root)
    manifest = read_json(root / 'manifest.json')
    if set(manifest['pages']) != set(page_ids) or len(page_ids) != len(set(page_ids)):
        raise ValueError('Page order must retain every page ID exactly once')
    if manifest['pages'] == page_ids:
        return
    manifest['pages'] = list(page_ids)
    manifest['status'] = 'needs_translation'
    write_json(root / 'manifest.json', manifest)
    from utils.atomic_io import CommitGate
    Store(root.parent, root.name, [], CommitGate()).invalidate_exports()
