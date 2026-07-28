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
    }
