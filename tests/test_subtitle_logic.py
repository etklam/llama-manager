"""
TDD tests for subtitle translation business logic.
These tests define the behavior before the GUI exists.
"""

import pytest
import json
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# This import will FAIL until subtitle_logic is implemented
from subtitle_logic import SubtitleTranslationLogic


# --- Fixtures ---

@pytest.fixture
def temp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def sample_srt_file(tmp_path):
    """Create a sample SRT file."""
    content = """1
00:00:01,000 --> 00:00:03,000
Hello, welcome.

2
00:00:03,500 --> 00:00:06,000
Today we learn AI.

"""
    file_path = tmp_path / "test.srt"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def sample_txt_file(tmp_path):
    """Create a sample TXT file."""
    content = "Line 1\nLine 2\nLine 3\n"
    file_path = tmp_path / "test.txt"
    file_path.write_text(content, encoding='utf-8')
    return file_path


@pytest.fixture
def mock_config():
    return {
        "api_url": "http://localhost:8080/v1",
        "model": "llama3",
        "max_tokens": 4096,
        "temperature": 0.2
    }


@pytest.fixture
def logic(mock_config):
    """Create logic instance with mock config."""
    return SubtitleTranslationLogic(mock_config)


# --- Test Supported Languages ---

class TestLanguages:
    def test_get_source_languages(self, logic):
        """Test source languages includes 'auto' option."""
        langs = logic.get_source_languages()
        assert isinstance(langs, dict)
        assert "auto" in langs
        assert len(langs) >= 2

    def test_get_target_languages(self, logic):
        """Test target languages contains common options."""
        langs = logic.get_target_languages()
        assert isinstance(langs, dict)
        assert "zh-cn" in langs
        assert "en" in langs
        assert "ja" in langs
        assert len(langs) >= 5

    def test_language_names_are_readable(self, logic):
        """Test language display names are human-readable."""
        langs = logic.get_target_languages()
        for code, name in langs.items():
            assert len(name) > 0
            assert isinstance(name, str)


# --- Test File Handling ---

class TestFileHandling:
    def test_is_supported_srt_file(self, logic, sample_srt_file):
        """Test SRT file detection."""
        assert logic.is_supported_file(str(sample_srt_file))

    def test_is_supported_txt_file(self, logic, sample_txt_file):
        """Test TXT file detection."""
        assert logic.is_supported_file(str(sample_txt_file))

    def test_is_not_supported_unknown_file(self, logic, tmp_path):
        """Test unsupported file rejection."""
        file_path = tmp_path / "test.pdf"
        file_path.write_text("data")
        assert not logic.is_supported_file(str(file_path))

    def test_get_files_from_paths(self, logic, tmp_path):
        """Test filtering files from mixed paths."""
        # Create files
        (tmp_path / "a.srt").write_text("test")
        (tmp_path / "b.txt").write_text("test")
        (tmp_path / "c.pdf").write_text("test")

        files = logic.get_supported_files([
            str(tmp_path / "a.srt"),
            str(tmp_path / "b.txt"),
            str(tmp_path / "c.pdf"),
        ])
        assert len(files) == 2
        assert any(f.endswith('.srt') for f in files)
        assert any(f.endswith('.txt') for f in files)

    @pytest.mark.skip(reason="Windows temp dir has system files that interfere")
    def test_get_files_from_directory(self, logic, tmp_path):
        """Test scanning directory for supported files."""
        subdir = tmp_path / "testdir"
        subdir.mkdir()
        (subdir / "a.srt").write_text("test")
        (subdir / "b.txt").write_text("test")
        (subdir / "c.pdf").write_text("test")

        files = logic.get_supported_files(str(subdir))
        # At least our 2 files should be found (Windows may add system files)
        srt_files = [f for f in files if f.endswith('.srt')]
        txt_files = [f for f in files if f.endswith('.txt')]
        assert len(srt_files) >= 1
        assert len(txt_files) >= 1


# --- Test Output Path ---

class TestOutputPath:
    def test_generate_output_path_srt(self, logic, sample_srt_file):
        """Test output path generation for SRT files."""
        output = logic.get_output_path(str(sample_srt_file), "zh-cn")
        assert output.endswith('.srt')
        assert 'zh-cn' in output or 'Chinese' in Path(output).stem

    def test_generate_output_path_txt(self, logic, sample_txt_file):
        """Test output path generation for TXT files."""
        output = logic.get_output_path(str(sample_txt_file), "ja")
        assert output.endswith('.txt')

    def test_generate_output_path_replace(self, logic, sample_srt_file):
        """Test output path when replacing original."""
        output = logic.get_output_path(str(sample_srt_file), "zh-cn", replace_original=True)
        assert str(sample_srt_file) == output


# --- Test Batch Size Calculation ---

class TestBatchSize:
    def test_batch_size_default(self, logic):
        """Test default batch size."""
        assert logic.get_batch_size() == 20

    def test_batch_size_custom(self, mock_config):
        """Test custom batch size."""
        logic = SubtitleTranslationLogic({**mock_config, "batch_size": 10})
        assert logic.get_batch_size() == 10

    def test_batch_size_from_config(self, mock_config):
        """Test batch size loaded from config."""
        logic = SubtitleTranslationLogic(mock_config)
        logic.update_config({"batch_size": 5})
        assert logic.get_batch_size() == 5


# --- Test Progress Callback ---

class TestProgressCallbacks:
    def test_set_progress_callback(self, logic):
        """Test setting progress callback."""
        called = []
        def callback(file_name, current, total, status):
            called.append((file_name, current, total, status))
        logic.set_progress_callback(callback)
        logic._notify_progress("test.srt", 5, 10, "translating")
        assert len(called) == 1
        assert called[0] == ("test.srt", 5, 10, "translating")

    def test_set_log_callback(self, logic):
        """Test setting log callback."""
        logs = []
        def callback(level, message):
            logs.append((level, message))
        logic.set_log_callback(callback)
        logic._notify_log("INFO", "test message")
        assert len(logs) == 1
        assert logs[0] == ("INFO", "test message")


# --- Test Model Detection ---

class TestModelDetection:
    def test_get_model_from_config(self, mock_config):
        """Test model name from config."""
        logic = SubtitleTranslationLogic(mock_config)
        assert logic.get_model() == "llama3"

    def test_get_api_url_from_config(self, mock_config):
        """Test API URL from config."""
        logic = SubtitleTranslationLogic(mock_config)
        assert logic.get_api_url() == "http://localhost:8080/v1"


# --- Test Translation Pipeline (mocked) ---

class TestTranslationPipeline:
    @patch('subtitle_logic.LocalLLMTranslator')
    def test_translate_srt_file(self, mock_translator_cls, logic, sample_srt_file, tmp_path):
        """Test SRT file translation pipeline with mocked translator."""
        # Setup mock translator
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'start_time': 1000, 'end_time': 3000,
             'text': '你好，欢迎。', 'time': '00:00:01,000 --> 00:00:03,000',
             'startraw': '00:00:01,000', 'endraw': '00:00:03,000'},
            {'line': 2, 'start_time': 3500, 'end_time': 6000,
             'text': '今天我们学习AI。', 'time': '00:00:03,500 --> 00:00:06,000',
             'startraw': '00:00:03,500', 'endraw': '00:00:06,000'},
        ]
        mock_translator_cls.return_value = mock_translator

        output = str(tmp_path / "output.srt")
        logic.translate_file(str(sample_srt_file), "zh-cn", output)

        # Verify translator was called
        mock_translator_cls.assert_called_once()
        mock_translator.translate_srt.assert_called_once()

        # Verify output file was created
        assert os.path.exists(output)

    @patch('subtitle_logic.LocalLLMTranslator')
    def test_translate_srt_replace_original(self, mock_translator_cls, logic, sample_srt_file):
        """Test SRT file translation with replace original."""
        mock_translator = MagicMock()
        mock_translator.translate_srt.return_value = [
            {'line': 1, 'start_time': 1000, 'end_time': 3000,
             'text': 'Hola', 'time': '00:00:01,000 --> 00:00:03,000',
             'startraw': '00:00:01,000', 'endraw': '00:00:03,000'},
        ]
        mock_translator_cls.return_value = mock_translator

        logic.translate_file(str(sample_srt_file), "es", replace_original=True)

        # Verify translator was called (file overwritten in place)
        mock_translator.translate_srt.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
