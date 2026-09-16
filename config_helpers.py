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


def build_translation_config(config_manager, target):
    """Translate-run settings, with the LLM endpoint supplied by `target`.

    Consumers no longer decide where the LLM lives: llm_target.resolve_llm_target
    produced the target (local llama-server or a remote profile), and this
    builder spreads it into the config dict the translator reads. Everything
    else is UI-level translation tuning, identical for both backends.
    """
    return {
        # Backend selection, concentrated here and in llm_target.
        'api_url': target.api_url,
        'model': target.model,
        'api_key': target.api_key,
        'proxy': target.proxy,
        'target': target,
        'max_tokens': config_manager.get('ui.max_tokens', 16384),
        'temperature': config_manager.get('ui.temperature', 0.2),
        'batch_size': config_manager.get('ui.batch_size', 15),
        'max_workers': target.max_workers,
        'single_step': config_manager.get('ui.single_step', True),
        'context_mode': context_mode_from_label(
            config_manager.get('ui.context_mode', 'none')
        ),
        # Filled from Translation Preflight for each run.  max_tokens is an
        # output cap and must never stand in for the server's input capacity.
        'context_size': None,
    }
