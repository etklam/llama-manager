"""Batch stage orchestration over the shared server controller and HTTP client."""
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import time
import httpx
from manga.images import file_hash, image_payload, digest
from manga.runtime import Profile, inspect_runtime, command, wait_ready, evidence
from manga.store import Store, documents, read_json, write_json
from manga.translation import DocumentTranslator, ContentError, ocr_body
from translation.openai_client import OpenAIClient
from translation.transport import TransportCancelledError, FatalProviderError, ProviderUnavailableError, ContextLengthError
from utils.atomic_io import CommitGate

OCR_PROMPT = 'Text Recognition:'


@dataclass(frozen=True)
class RunSnapshot:
    files: tuple[str, ...]
    output: str
    executable: str
    device: str
    ocr: Profile
    translation: Profile
    mode: str = 'ocr_translate'
    source: str = 'ja'
    locale: str = 'zh-tw'
    direction: str = 'rtl'
    independent: bool = False
    review: bool = False
    two_step: bool = False
    port: int = 8091
    run_id: int = 0


class MangaRunner:
    def __init__(self, server, snapshot, notify=lambda event: None):
        self.server, self.snapshot, self.notify = server, snapshot, notify
        self.gate = CommitGate()
        self.report = {'status': 'running', 'runtime': {}, 'stages': [], 'restore_error': None,
                       'gpu_driver': 'unknown', 'quality': 'NOT RUN', 'failures': 0}
        self.client = self.http = None

    def cancel(self):
        self.gate.cancel()

    def check(self):
        if self.gate.cancelled:
            raise TransportCancelledError('Cancelled')

    def emit(self, stage, **values):
        self.notify({'run_id': self.snapshot.run_id, 'stage': stage, **values})

    def close_client(self):
        if self.client:
            self.client.close()
            self.client = None
        if self.http:
            self.http.close()
            self.http = None

    def dispatch_cancelled(self):
        if self.gate.cancelled:
            return True
        check_alive = getattr(self.server, 'assert_alive', None)
        if check_alive:
            check_alive()
        return False

    def load(self, profile, name, discovery):
        self.check()
        self.close_client()
        self.server.stop()
        self.check()
        s = self.snapshot
        self.emit('loading', model=name)
        self.server.start_command(command(s.executable, discovery, s.device, profile, s.port), s.port, isolated=True)
        wait_ready(self.server, s.port, lambda: self.gate.cancelled)
        self.check()
        self.stage_logs = list(self.server.logs)
        observed = evidence(self.stage_logs, s.device, bool(profile.mmproj))
        if observed['decoder_backend'] == 'CPU-only' or observed['vision_backend'] not in ('unknown', 'not_applicable', s.device):
            raise FatalProviderError('Incompatible backend detected; refusing fallback')
        self.http = httpx.Client(trust_env=False, timeout=httpx.Timeout(120, connect=5))
        self.client = OpenAIClient(f'http://127.0.0.1:{s.port}/v1', profile.model,
                                   api_key='local-only', timeout=120, http_client=self.http,
                                   cancel_check=self.dispatch_cancelled, redact_errors=True)

    def reject_fallback(self, vision):
        item = evidence(getattr(self, 'stage_logs', []) + list(self.server.logs), self.snapshot.device, vision)
        if item['decoder_backend'] == 'CPU-only' or item['vision_backend'] not in ('unknown', 'not_applicable', self.snapshot.device):
            raise FatalProviderError('Incompatible backend detected; refusing fallback')

    def record_stage(self, name, profile, request_ok, started):
        item = evidence(getattr(self, 'stage_logs', []) + list(self.server.logs), self.snapshot.device, bool(profile.mmproj), request_ok)
        item.update({'stage': name, 'elapsed_seconds': time.monotonic() - started,
                     'model_sha256': self.model_hashes[profile.model],
                     'projector_sha256': self.model_hashes[profile.mmproj] if profile.mmproj else None})
        self.report['stages'].append(item)
        self.emit('diagnostic', diagnostic=item)
        if item['decoder_backend'] == 'CPU-only' or item['vision_backend'] not in ('unknown', 'not_applicable', self.snapshot.device):
            raise FatalProviderError('Incompatible backend detected; results are not Vulkan verified')

    def run(self):
        s = self.snapshot
        started = time.monotonic()
        stores = []
        try:
            self.check()
            if s.mode not in ('ocr', 'ocr_translate', 'retranslate') or not s.files:
                raise ValueError('Select images and a valid mode')
            if not re.fullmatch(r'[A-Za-z0-9_-]{1,40}', s.locale):
                raise ValueError('Use a locale code such as zh-tw')
            if not s.ocr.mmproj:
                raise ValueError('Select matching OCR mmproj')
            s.ocr.validate()
            if s.mode != 'ocr':
                s.translation.validate()
            discovery = inspect_runtime(s.executable)
            self.report['runtime'] = {k: v for k, v in discovery.items() if k != 'help'}
            self.report['gpu_driver'] = discovery.get('drivers', 'unknown')
            # Validate all commands before touching a running model.
            command(s.executable, discovery, s.device, s.ocr, s.port)
            if s.mode != 'ocr':
                command(s.executable, discovery, s.device, s.translation, s.port)
            self.model_hashes = {path: file_hash(path) for path in
                                 (s.ocr.model, s.ocr.mmproj, *([s.translation.model] if s.mode != 'ocr' else []))}
            identity = [self.model_hashes[s.ocr.model], self.model_hashes[s.ocr.mmproj], OCR_PROMPT,
                        asdict(s.ocr), discovery['version']]
            with self.server.lease('manga'):
                for document_id, sources in documents(s.files, s.independent):
                    store = Store(s.output, document_id, sources, self.gate)
                    for source in sources:
                        store.load_page(source, identity)
                    stores.append(store)
                    store.manifest.update({'source_language': s.source, 'reading_direction': s.direction,
                                           'ocr_identity': digest(identity)})
                    changed = store.old_manifest.get('source_revision_key') != store.source_revision_key()
                    store.save_manifest('running')
                    store.invalidate_exports(None if changed else s.locale)
                self.gate.commit(lambda: write_json(Path(s.output) / 'manga-job.json', asdict(s)))
                previous = self.server.last_command if self.server.running else None
                touched = False
                try:
                    pending = [(store, p) for store in stores for p in store.pages
                               if p['status'] not in ('ok', 'no_text')]
                    if s.mode == 'retranslate' and pending:
                        self.emit('review_required', count=len(pending))
                    if s.mode != 'retranslate' and pending:
                        touched = True
                        self.load(s.ocr, 'ocr', discovery)
                        stage_start, successes = time.monotonic(), 0
                        try:
                            for index, (store, page) in enumerate(pending):
                                self.check()
                                self.emit('ocr', current=index + 1, total=len(pending), page_id=page['id'])
                                try:
                                    data = image_payload(page['path'])
                                    if file_hash(page['path']) != page['source_hash']:
                                        raise ValueError('Source changed during run; restart to revalidate')
                                except (OSError, ValueError) as error:
                                    store.set_ocr(page, '', 'failed', str(error))
                                    self.report['failures'] += 1
                                    continue
                                raw = ''
                                try:
                                    for attempt in range(2):
                                        self.check()
                                        result = self.client.complete_with_metadata(
                                            [{'role': 'user', 'content': [
                                                {'type': 'image_url', 'image_url': {'url': data}},
                                                {'type': 'text', 'text': OCR_PROMPT}]}],
                                            s.ocr.model, s.ocr.max_tokens, s.ocr.temperature)
                                        self.check()
                                        raw = result.content
                                        if result.finish_reason == 'stop' and not raw.strip():
                                            raise ContentError('Empty response requires manual no_text confirmation')
                                        try:
                                            ocr_body(result)
                                            break
                                        except ContentError:
                                            if attempt:
                                                raise
                                    self.reject_fallback(bool(s.ocr.mmproj))
                                    store.set_ocr(page, raw, 'ok')
                                    successes += 1
                                except (FatalProviderError, ProviderUnavailableError, TransportCancelledError):
                                    raise
                                except (ContentError, ContextLengthError, RuntimeError) as error:
                                    store.set_ocr(page, raw, 'needs_review', str(error))
                                    self.report['failures'] += 1
                                store.export(s.locale)
                        finally:
                            self.record_stage('ocr', s.ocr, successes > 0, stage_start)
                    self.check()
                    if s.mode == 'ocr' or (s.review and s.mode != 'retranslate'):
                        for store in stores:
                            ready = all(p['status'] in ('ok', 'no_text') for p in store.pages)
                            store.save_manifest('ocr_completed' if ready else 'partial')
                            store.export(s.locale)
                        self.report['status'] = 'awaiting_review' if s.review else 'ocr_finished'
                        return self.report
                    candidates = [store for store in stores if store.segments()]
                    if candidates:
                        touched = True
                        self.load(s.translation, 'translation', discovery)
                        stage_start, translated = time.monotonic(), 0
                        try:
                            for store in stores:
                                self.check()
                                translator = DocumentTranslator(self.client, s.translation, s.source, s.locale,
                                                                s.two_step, lambda: self.gate.cancelled)
                                segments = store.segments()
                                batches = translator.batches(segments)
                                key = store.translation_key([asdict(s.translation), self.model_hashes[s.translation.model],
                                    s.source, s.locale, s.direction, s.two_step, discovery['version'], 'document-v1'])
                                cache_path = store.root / f'checkpoint.{s.locale}.json'
                                cache = {'fingerprint': key, 'parts': {}}
                                expected = {p['id'] for batch in batches for p in batch}
                                if cache_path.exists():
                                    old = read_json(cache_path)
                                    if (not isinstance(old, dict) or not isinstance(old.get('parts'), dict)
                                            or any(not isinstance(v, str) or not v.strip() for v in old['parts'].values())):
                                        raise ValueError('Corrupted translation checkpoint')
                                    if old.get('fingerprint') == key:
                                        if not set(old['parts']).issubset(expected):
                                            raise ValueError('Checkpoint contains unknown IDs')
                                        cache = old
                                for index, batch in enumerate(batches):
                                    self.check()
                                    if all(p['id'] in cache['parts'] for p in batch):
                                        continue
                                    self.emit('translation', current=index + 1, total=len(batches),
                                              document_id=store.root.name)
                                    nearby = [{'text': p['text']} for p in (batches[index - 1][-1:] if index else [])]
                                    try:
                                        values = translator.translate_batch(batch, nearby)
                                        self.check()
                                        self.reject_fallback(False)
                                        cache['parts'].update(values)
                                        store.commit(cache_path, cache)
                                        partial = {}
                                        for segment in segments:
                                            parts = [p for b in batches for p in b if p['parent'] == segment['id']]
                                            if parts and all(p['id'] in cache['parts'] for p in parts):
                                                partial[segment['id']] = ''.join(cache['parts'][p['id']] for p in parts)
                                        store.export(s.locale, {'fingerprint': key, 'segments': partial}, False)
                                        translated += 1
                                    except ContentError:
                                        self.report['failures'] += 1
                                        # Continue independent batches; their IDs remain isolated.
                                merged = {}
                                for segment in segments:
                                    parts = [p for b in batches for p in b if p['parent'] == segment['id']]
                                    if parts and all(p['id'] in cache['parts'] for p in parts):
                                        merged[segment['id']] = ''.join(cache['parts'][p['id']] for p in parts)
                                complete = (len(merged) == len(segments) and
                                            all(p['status'] in ('ok', 'no_text') for p in store.pages))
                                store.export(s.locale, {'fingerprint': key, 'segments': merged}, complete)
                                store.save_manifest('completed' if complete else 'partial')
                        finally:
                            self.record_stage('translation', s.translation, translated > 0, stage_start)
                    else:
                        for store in stores:
                            complete = all(p['status'] == 'no_text' for p in store.pages)
                            store.export(s.locale, {'segments': {}}, complete)
                            store.save_manifest('completed' if complete else 'partial')
                    self.report['status'] = 'completed' if all(
                        store.manifest['status'] == 'completed' for store in stores) else 'partial'
                finally:
                    self.close_client()
                    if touched:
                        try:
                            self.server.stop()
                            if previous:
                                self.server.start_command(*previous)
                                wait_ready(self.server, previous[1], lambda: False)
                        except Exception as error:
                            self.report['restore_error'] = f'{type(error).__name__}: {error}'
                            self.emit('restore_failed', error=self.report['restore_error'])
        except TransportCancelledError:
            self.report['status'] = 'cancelled'
        except Exception as error:
            self.report['status'] = 'cancelled' if self.gate.cancelled else 'failed'
            self.report['error'] = f'{type(error).__name__}: {error}'
            self.report['failures'] += 1
        finally:
            self.close_client()
            self.report['counts'] = {
                'success': sum(p['status'] in ('ok', 'no_text') for store in stores for p in store.pages),
                'failed': sum(p['status'] == 'failed' for store in stores for p in store.pages),
                'needs_review': sum(p['status'] == 'needs_review' for store in stores for p in store.pages),
                'pending': sum(p['status'] == 'pending' for store in stores for p in store.pages),
            }
            self.report['elapsed_seconds'] = time.monotonic() - started
            if stores:
                try:
                    # Diagnostics may record cancellation; they are never a
                    # successful OCR/translation publication.
                    write_json(Path(s.output) / 'runtime-report.json', self.report)
                except OSError as error:
                    self.report['diagnostic_write_error'] = type(error).__name__
            self.emit('done', report=self.report)
        return self.report
