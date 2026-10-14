"""Exercise actual orchestration/checkpoints with controlled inference responses."""
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import httpx
import pytest
from PIL import Image
from manga.runner import MangaRunner, RunSnapshot
from manga.runtime import Profile
from manga.store import read_json, documents, correct_page
from translation.llm_client import CompletionResult
from translation.transport import ProviderUnavailableError
from translation.openai_client import OpenAIClient
from server_controller import ServerController, ServerBusyError


class Runtime:
    def __init__(self):
        self.running = False
        self.logs = []
        self.last_command = None
        self.events = []
        self.busy = False

    @contextmanager
    def lease(self, name):
        if self.busy:
            raise ServerBusyError('busy')
        self.busy = True
        try:
            yield self
        finally:
            self.busy = False

    def stop(self):
        self.events.append('stop')
        self.running = False

    def start_command(self, cmd, port, isolated=False):
        self.events.append(cmd[0])
        self.running = True
        self.last_command = (cmd, port, isolated)
        self.logs = ['load_tensors: offloaded 25/25 layers to GPU',
                     'llama_context: Vulkan0 compute buffer size = 12 MiB',
                     'clip_model_loader: CLIP using Vulkan0 backend']


@pytest.fixture
def job(tmp_path, monkeypatch):
    paths = []
    for i in range(3):
        p = tmp_path / f'漫畫{i}.png'
        Image.new('RGB', (20, 20), 'white').save(p)
        paths.append(str(p))
    models = []
    for name in ('ocr', 'mmproj', 'translation'):
        p = tmp_path / (name + '.gguf')
        p.write_bytes(name.encode())
        models.append(str(p))
    s = RunSnapshot(tuple(paths), str(tmp_path / 'output'), 'server', 'Vulkan0',
                    Profile(models[0], models[1]), Profile(models[2]))
    monkeypatch.setattr('manga.runner.inspect_runtime', lambda exe: {'version': 'test', 'devices': ['Vulkan0']})
    monkeypatch.setattr('manga.runner.command', lambda exe, info, device, profile, port: [Path(profile.model).stem])
    monkeypatch.setattr('manga.runner.wait_ready', lambda *args: None)
    server = Runtime()
    calls = []

    class Client:
        def __init__(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def complete_with_metadata(self, messages, model, tokens, temperature):
            calls.append(('ocr', messages))
            return CompletionResult('同じ台詞\n\n同じ台詞', 'stop')
        def complete_analysis(self, messages, model, tokens, temperature):
            data = json.loads(messages[1]['content'])
            calls.append(('translation', data))
            return CompletionResult(json.dumps([{'id': p['id'], 'translation': '同一句對白'}
                                                for p in data['requested']]), 'stop')
    monkeypatch.setattr('manga.runner.OpenAIClient', Client)
    return s, server, calls, Client


def test_batch_ocr_then_translation_and_resume(job):
    snapshot, server, calls, _ = job
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'completed', report
    names = [c[0] for c in calls]
    assert names[:3] == ['ocr'] * 3
    assert all(n == 'translation' for n in names[3:])
    assert server.events == ['stop', 'ocr', 'stop', 'translation', 'stop']
    roots = list(Path(snapshot.output).glob('*/manifest.json'))
    assert len(roots) == 1
    output = read_json(roots[0].parent / 'translations.zh-tw.json')
    assert len(output['segments']) == 6
    assert len(set(output['segments'].values())) == 1
    calls.clear()
    resumed = MangaRunner(server, snapshot).run()
    assert resumed['status'] == 'completed'
    assert calls == []


def test_ocr_only_review_then_retranslate_no_ocr_model_load(job):
    snapshot, server, calls, _ = job
    assert MangaRunner(server, replace(snapshot, mode='ocr')).run()['status'] == 'ocr_finished'
    assert not any(c[0] == 'translation' for c in calls)
    server.events.clear()
    calls.clear()
    assert MangaRunner(server, replace(snapshot, mode='retranslate')).run()['status'] == 'completed'
    assert 'ocr' not in server.events
    assert all(c[0] == 'translation' for c in calls)


def test_cancel_late_ocr_never_commits_or_switches_stage(job, monkeypatch):
    snapshot, server, calls, Client = job
    runner = MangaRunner(server, snapshot)
    def late(*args):
        runner.cancel()
        return CompletionResult('late text', 'stop')
    monkeypatch.setattr(Client, 'complete_with_metadata', late)
    report = runner.run()
    assert report['status'] == 'cancelled'
    assert 'translation' not in server.events
    assert not list(Path(snapshot.output).glob('*/pages/*.json'))
    assert not list(Path(snapshot.output).glob('*/translations.zh-tw.json'))


def test_systemic_failure_stops_pages_restores_previous(job, monkeypatch):
    snapshot, server, calls, Client = job
    server.running = True
    server.last_command = (['original'], 8080, False)
    failed = Mock(side_effect=ProviderUnavailableError('systemic'))
    monkeypatch.setattr(Client, 'complete_with_metadata', failed)
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'failed'
    assert failed.call_count == 1
    assert server.events[-1] == 'original'
    assert 'translation' not in server.events


def test_bad_image_continues_and_empty_needs_review(job, monkeypatch):
    snapshot, server, calls, Client = job
    Path(snapshot.files[0]).write_bytes(b'bad image')
    monkeypatch.setattr(Client, 'complete_with_metadata', lambda *a: CompletionResult('', 'stop'))
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'partial'
    assert report['counts'] == {'success': 0, 'failed': 1, 'needs_review': 2, 'pending': 0}
    assert not list(Path(snapshot.output).glob('*/translations.zh-tw.json'))
    assert list(Path(snapshot.output).glob('*/translations.zh-tw.partial.json'))


def test_length_has_bounded_content_retry_no_completed_page(job, monkeypatch):
    snapshot, server, calls, Client = job
    truncated = Mock(return_value=CompletionResult('partial', 'length'))
    monkeypatch.setattr(Client, 'complete_with_metadata', truncated)
    report = MangaRunner(server, replace(snapshot, mode='ocr')).run()
    assert truncated.call_count == 6
    assert report['counts']['needs_review'] == 3
    for file in Path(snapshot.output).glob('*/pages/*.json'):
        assert read_json(file)['raw'] == 'partial'


def test_changed_locale_only_retranslates_and_correction_invalidates(job):
    snapshot, server, calls, _ = job
    assert MangaRunner(server, snapshot).run()['status'] == 'completed'
    calls.clear()
    assert MangaRunner(server, replace(snapshot, locale='en')).run()['status'] == 'completed'
    assert not any(c[0] == 'ocr' for c in calls)
    doc, pages = documents(snapshot.files, False)[0]
    root = Path(snapshot.output) / doc
    correct_page(root, pages[0]['id'], ['修正後', '同じ台詞'])
    assert not (root / 'translations.zh-tw.json').exists()
    calls.clear()
    assert MangaRunner(server, replace(snapshot, mode='retranslate')).run()['status'] == 'completed'
    assert not any(c[0] == 'ocr' for c in calls)
    assert '修正後' in json.dumps(calls, ensure_ascii=False)


def test_independent_documents_never_share_context(job):
    snapshot, server, calls, _ = job
    assert MangaRunner(server, replace(snapshot, independent=True)).run()['status'] == 'completed'
    assert len(list(Path(snapshot.output).glob('*/manifest.json'))) == 3
    for kind, data in calls:
        if kind == 'translation':
            ids = {p['id'].split('-')[0] for p in data['requested']}
            assert len(ids) == 1


def test_busy_does_not_modify_checkpoint(job):
    snapshot, server, calls, _ = job
    server.busy = True
    assert MangaRunner(server, snapshot).run()['status'] == 'failed'
    assert calls == []
    assert not Path(snapshot.output).exists()


def test_cpu_vision_is_rejected_before_image_dispatch(job):
    snapshot, server, calls, _ = job
    original = server.start_command
    def cpu(*args, **kwargs):
        original(*args, **kwargs)
        server.logs = ['CLIP using CPU backend']
    server.start_command = cpu
    assert MangaRunner(server, snapshot).run()['status'] == 'failed'
    assert calls == []


def test_real_adapter_preserves_multimodal_payload_metadata_and_no_sdk_retries():
    seen = []
    def handler(request):
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={'id': 'x', 'object': 'chat.completion', 'created': 0,
            'model': 'ocr', 'choices': [{'index': 0, 'finish_reason': 'length',
                                      'message': {'role': 'assistant', 'content': '正文', 'reasoning_content': 'private'}}]})
    with httpx.Client(transport=httpx.MockTransport(handler), trust_env=False) as http:
        client = OpenAIClient('http://127.0.0.1:8091/v1', 'ocr', api_key='local', http_client=http)
        messages = [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}},
                                               {'type': 'text', 'text': 'Text Recognition:'}]}]
        result = client.complete_with_metadata(messages, 'ocr', 100, 0)
        assert result == CompletionResult('正文', 'length')
        assert seen[0]['messages'] == messages
        assert client._client.max_retries == 0
        client.close()
        assert not http.is_closed


def test_external_listener_is_not_killed_or_reused(tmp_path, monkeypatch):
    import socket
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        controller = ServerController(Path('server'), lambda line: None)
        popen = Mock()
        monkeypatch.setattr('server_controller.subprocess.Popen', popen)
        with pytest.raises(OSError):
            controller.start_command(['server'], listener.getsockname()[1])
        popen.assert_not_called()
        controller.stop()
        assert listener.fileno() != -1

def test_crash_resume_keeps_finished_page_and_retries_remaining(job, monkeypatch):
    snapshot, server, calls, Client = job
    original = Client.complete_with_metadata
    count = 0
    def crash(self, *args):
        nonlocal count
        count += 1
        if count == 2:
            raise ProviderUnavailableError('crash')
        return original(self, *args)
    monkeypatch.setattr(Client, 'complete_with_metadata', crash)
    first = MangaRunner(server, snapshot).run()
    assert first['status'] == 'failed'
    assert len(list(Path(snapshot.output).glob('*/pages/*.json'))) == 1
    monkeypatch.setattr(Client, 'complete_with_metadata', original)
    calls.clear()
    second = MangaRunner(server, snapshot).run()
    assert second['status'] == 'completed'
    assert sum(c[0] == 'ocr' for c in calls) == 2


def test_cancel_late_translation_keeps_ocr_but_no_translation_checkpoint(job, monkeypatch):
    snapshot, server, calls, Client = job
    runner = MangaRunner(server, snapshot)
    original = Client.complete_analysis
    def late(self, *args):
        result = original(self, *args)
        runner.cancel()
        return result
    monkeypatch.setattr(Client, 'complete_analysis', late)
    report = runner.run()
    assert report['status'] == 'cancelled'
    assert len(list(Path(snapshot.output).glob('*/pages/*.json'))) == 3
    assert not list(Path(snapshot.output).glob('*/checkpoint.zh-tw.json'))
    assert not list(Path(snapshot.output).glob('*/translations.zh-tw.json'))


def test_pause_after_ocr_never_loads_translation(job):
    snapshot, server, calls, _ = job
    report = MangaRunner(server, replace(snapshot, review=True)).run()
    assert report['status'] == 'awaiting_review'
    assert 'translation' not in server.events


def test_restore_failure_report_does_not_hide_work_failure(job, monkeypatch):
    snapshot, server, calls, Client = job
    server.running = True
    server.last_command = (['original'], 8080, False)
    start = server.start_command
    def restore_fail(cmd, *args, **kwargs):
        if cmd == ['original']:
            raise RuntimeError('restore failure')
        start(cmd, *args, **kwargs)
    server.start_command = restore_fail
    monkeypatch.setattr(Client, 'complete_with_metadata', Mock(side_effect=ProviderUnavailableError('initial failure')))
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'failed'
    assert 'initial failure' in report['error']
    assert 'restore failure' in report['restore_error']


def test_confirmed_no_text_can_finish_without_translation_model_load(job, monkeypatch):
    snapshot, server, calls, Client = job
    monkeypatch.setattr(Client, 'complete_with_metadata', lambda *a: CompletionResult('', 'stop'))
    MangaRunner(server, replace(snapshot, mode='ocr')).run()
    document, pages = documents(snapshot.files, False)[0]
    root = Path(snapshot.output) / document
    for page in pages:
        correct_page(root, page['id'], [], no_text=True)
    server.events.clear()
    report = MangaRunner(server, replace(snapshot, mode='retranslate')).run()
    assert report['status'] == 'completed'
    assert server.events == []


def test_real_adapter_cancel_between_transport_attempts_sends_no_second_request():
    from translation.transport import TransportCancelledError
    cancelled = [False]
    requests = []
    def handler(request):
        requests.append(request)
        cancelled[0] = True
        return httpx.Response(500, json={'error': {'message': 'temporary'}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OpenAIClient('http://127.0.0.1:8091/v1', 'ocr', api_key='local',
                              http_client=http, cancel_check=lambda: cancelled[0])
        with pytest.raises(TransportCancelledError):
            client.complete_with_metadata([{'role': 'user', 'content': []}], 'ocr', 100, 0)
        assert len(requests) == 1


def test_failed_translation_ids_never_published_and_retry_is_bounded(job, monkeypatch):
    snapshot, server, calls, Client = job
    invalid = Mock(return_value=CompletionResult('[{"id":"unknown","translation":"bad"}]', 'stop'))
    monkeypatch.setattr(Client, 'complete_analysis', invalid)
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'partial'
    assert invalid.call_count >= 2 and invalid.call_count % 2 == 0
    assert not list(Path(snapshot.output).glob('*/translations.zh-tw.json'))
    for file in Path(snapshot.output).glob('*/translations.zh-tw.partial.json'):
        assert read_json(file)['segments'] == {}

def test_manual_start_and_stop_are_blocked_by_other_thread_lease(tmp_path):
    import threading
    controller = ServerController(Path('server'), lambda line: None)
    held, release = threading.Event(), threading.Event()
    def holder():
        with controller.lease('manga'):
            held.set()
            release.wait(3)
    thread = threading.Thread(target=holder)
    thread.start()
    try:
        assert held.wait(1)
        with pytest.raises(ServerBusyError):
            controller.stop()
        with pytest.raises(ServerBusyError):
            controller.start_command(['server'], 8091)
    finally:
        release.set()
        thread.join(2)


def test_other_locale_formal_export_survives_new_target(job):
    snapshot, server, calls, _ = job
    MangaRunner(server, snapshot).run()
    original = next(Path(snapshot.output).glob('*/translations.zh-tw.json'))
    before = original.read_bytes()
    report = MangaRunner(server, replace(snapshot, locale='en')).run()
    assert report['status'] == 'completed'
    assert original.read_bytes() == before


def test_source_change_reocr_only_affected_page(job):
    snapshot, server, calls, _ = job
    MangaRunner(server, snapshot).run()
    Image.new('RGB', (21, 21), 'black').save(snapshot.files[1])
    calls.clear()
    report = MangaRunner(server, snapshot).run()
    assert report['status'] == 'completed'
    assert sum(c[0] == 'ocr' for c in calls) == 1


def test_paragraph_two_step_requires_draft_but_keeps_same_text_translation():
    from manga.translation import parse_translation, ContentError
    with pytest.raises(ContentError):
        parse_translation('[{"id":"x","translation":"123"}]', {'x'}, two_step=True)
    assert parse_translation('[{"id":"x","literal":"123","translation":"123"}]',
                             {'x'}, two_step=True) == {'x': '123'}


def test_ocr_explanation_is_pending_review_not_body(job, monkeypatch):
    snapshot, server, calls, Client = job
    monkeypatch.setattr(Client, 'complete_with_metadata', lambda *a: CompletionResult('Here is my analysis of the image.', 'stop'))
    report = MangaRunner(server, replace(snapshot, mode='ocr')).run()
    assert report['counts']['needs_review'] == 3
    assert report['counts']['success'] == 0

def test_page_reorder_invalidates_formal_export_without_changing_raw(job):
    from manga.store import reorder_document
    snapshot, server, calls, _ = job
    MangaRunner(server, snapshot).run()
    document, pages = documents(snapshot.files, False)[0]
    root = Path(snapshot.output) / document
    before = {p.name: p.read_bytes() for p in (root / 'pages').glob('*.json')}
    reorder_document(root, [p['id'] for p in reversed(pages)])
    assert read_json(root / 'manifest.json')['status'] == 'needs_translation'
    assert not (root / 'translations.zh-tw.json').exists()
    assert before == {p.name: p.read_bytes() for p in (root / 'pages').glob('*.json')}


def test_sensitive_provider_errors_do_not_log_or_return_image_or_text(caplog):
    from translation.transport import FatalProviderError
    secret = 'data:image/png;base64,SECRET_COMIC_TEXT'
    def handler(request):
        return httpx.Response(400, json={'error': {'message': secret}})
    with httpx.Client(transport=httpx.MockTransport(handler)) as http:
        client = OpenAIClient('http://127.0.0.1:8091/v1', 'ocr', api_key='local',
                              http_client=http, redact_errors=True)
        with pytest.raises(FatalProviderError) as error:
            client.complete_with_metadata([], 'ocr', 100, 0)
        assert secret not in str(error.value)
        assert secret not in caplog.text
