"""
Business logic for subtitle translation.
Separated from GUI for testability.
"""

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional
from utils.srt_parser import parse_srt_from_file, generate_srt_from_list
from translation.local_llm_translator import LocalLLMTranslator


SUPPORTED_EXTENSIONS = {'.srt', '.txt'}

# Hardcoded common languages
TARGET_LANGUAGES = {
    'zh-cn': 'Simplified Chinese',
    'zh-tw': 'Traditional Chinese',
    'en': 'English',
    'ja': 'Japanese',
    'ko': 'Korean',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'th': 'Thai',
    'vi': 'Vietnamese',
    'it': 'Italian',
    'nl': 'Dutch',
}

SOURCE_LANGUAGES = {
    'auto': 'Auto Detect',
    **TARGET_LANGUAGES,
}


class SubtitleTranslationLogic:
    """Business logic for subtitle translation operations."""

    def __init__(self, config: dict):
        self._config = config
        self._progress_callback: Optional[Callable] = None
        self._log_callback: Optional[Callable] = None
        self._translator: Optional[LocalLLMTranslator] = None

    # --- Language ---

    def get_source_languages(self) -> Dict[str, str]:
        return dict(SOURCE_LANGUAGES)

    def get_target_languages(self) -> Dict[str, str]:
        return dict(TARGET_LANGUAGES)

    # --- File handling ---

    def is_supported_file(self, filepath: str) -> bool:
        ext = Path(filepath).suffix.lower()
        return ext in SUPPORTED_EXTENSIONS

    def get_supported_files(self, paths: List[str]) -> List[str]:
        """Filter and expand paths to supported files."""
        files = []
        for p in paths:
            path = Path(p)
            if path.is_dir():
                for f in path.iterdir():
                    if f.is_file() and self.is_supported_file(str(f)):
                        files.append(str(f))
            elif path.is_file() and self.is_supported_file(str(path)):
                files.append(str(path))
        return files

    # --- Output path ---

    def get_output_path(self, input_path: str, target_lang: str,
                        replace_original: bool = False) -> str:
        if replace_original:
            return input_path
        p = Path(input_path)
        stem = p.stem
        lang_name = TARGET_LANGUAGES.get(target_lang, target_lang)
        return str(p.parent / f"{stem}_{lang_name}{p.suffix}")

    # --- Config ---

    def update_config(self, config: dict):
        self._config.update(config)

    def get_config(self) -> dict:
        return dict(self._config)

    def get_batch_size(self) -> int:
        return self._config.get('batch_size', 20)

    def get_model(self) -> str:
        return self._config.get('model', '')

    def get_api_url(self) -> str:
        return self._config.get('api_url', 'http://localhost:8080/v1')

    # --- Progress callbacks ---

    def set_progress_callback(self, callback: Optional[Callable]):
        self._progress_callback = callback

    def set_log_callback(self, callback: Optional[Callable]):
        self._log_callback = callback

    def _notify_progress(self, file_name: str, current: int, total: int,
                         status: str):
        if self._progress_callback:
            self._progress_callback(file_name, current, total, status)

    def _notify_log(self, level: str, message: str):
        if self._log_callback:
            self._log_callback(level, message)

    # --- Translation ---

    def _get_translator(self) -> LocalLLMTranslator:
        if self._translator is None:
            self._translator = LocalLLMTranslator(self._config)
        return self._translator

    def translate_file(self, input_path: str, target_lang: str,
                       output_path: Optional[str] = None,
                       replace_original: bool = False):
        """Translate a single SRT or TXT file."""
        filepath = Path(input_path)
        ext = filepath.suffix.lower()

        if output_path is None:
            output_path = self.get_output_path(input_path, target_lang,
                                               replace_original)

        self._notify_log("INFO", f"Translating: {filepath.name}")
        self._notify_progress(filepath.name, 0, 1, "starting")

        if ext == '.srt':
            subtitles = parse_srt_from_file(str(filepath))
            total_lines = len(subtitles)
            self._notify_log("INFO", f"Parsed {total_lines} subtitle lines, "
                              f"translating line by line...")
            translator = self._get_translator()

            def file_progress(current, total, status):
                self._notify_progress(filepath.name, current, total, status)

            translated = translator.translate_srt(
                subtitles, target_lang,
                progress_callback=file_progress,
                log_callback=self._notify_log
            )
            srt_content = generate_srt_from_list(translated)
            Path(output_path).write_text(srt_content, encoding='utf-8')
        elif ext == '.txt':
            text = filepath.read_text(encoding='utf-8')
            translator = self._get_translator()
            self._notify_log("INFO", f"Text file, {len(text)} chars")
            translated = translator.translate(text, target_lang)
            Path(output_path).write_text(translated, encoding='utf-8')

        self._notify_progress(filepath.name, 1, 1, "completed")
        self._notify_log("SUCCESS", f"Saved: {output_path}")
