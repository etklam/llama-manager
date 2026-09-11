CONTEXT_MODE_LABELS = {
    'none': '標準翻譯',
    'story': '全文理解翻譯',
}


def context_mode_from_label(value):
    if value in CONTEXT_MODE_LABELS:
        return value
    for mode, label in CONTEXT_MODE_LABELS.items():
        if value == label:
            return mode
    return 'none'


def api_url_for_port(port):
    """Base URL of the local llama-server's OpenAI-compatible API.

    Shared so the preflight probe and the translator cannot end up pointing at
    different endpoints.
    """
    return f'http://localhost:{port}/v1'


def build_translation_config(config_manager, port, model):
    return {
        'api_url': api_url_for_port(port),
        'model': model,
        'max_tokens': config_manager.get('ui.max_tokens', 16384),
        'temperature': config_manager.get('ui.temperature', 0.2),
        'batch_size': config_manager.get('ui.batch_size', 15),
        'max_workers': config_manager.get('ui.max_workers', 3),
        'single_step': config_manager.get('ui.single_step', True),
        'context_mode': context_mode_from_label(
            config_manager.get('ui.context_mode', 'none')
        ),
        # Filled from Translation Preflight for each run.  max_tokens is an
        # output cap and must never stand in for the server's input capacity.
        'context_size': None,
    }
