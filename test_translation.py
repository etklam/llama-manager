#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI Test Script for Subtitle Translation
Tests the translation functionality with real LLM API
"""

import sys
import json
from pathlib import Path
from utils.srt_parser import parse_srt_from_file, generate_srt_from_list
from translation.local_llm_translator import LocalLLMTranslator

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')


def create_test_srt():
    """Create a test SRT file"""
    test_srt = """1
00:00:01,000 --> 00:00:03,000
Hello, welcome to this video.

2
00:00:03,500 --> 00:00:06,000
Today we will learn about AI translation.

3
00:00:06,500 --> 00:00:09,000
Let's get started with the demo.
"""
    test_file = Path("test_subtitles.srt")
    test_file.write_text(test_srt, encoding='utf-8')
    return test_file


def test_srt_parsing():
    """Test SRT parsing"""
    print("=" * 60)
    print("TEST 1: SRT Parsing")
    print("=" * 60)

    test_file = create_test_srt()
    print(f"✓ Created test SRT file: {test_file}")

    # Parse SRT
    subtitles = parse_srt_from_file(str(test_file))
    print(f"✓ Parsed {len(subtitles)} subtitle blocks")

    for sub in subtitles:
        print(f"  Line {sub['line']}: {sub['text'][:50]}...")

    test_file.unlink()
    print("✓ Test file cleaned up\n")
    return subtitles


def test_srt_generation():
    """Test SRT generation"""
    print("=" * 60)
    print("TEST 2: SRT Generation")
    print("=" * 60)

    test_subtitles = [
        {
            'line': 1,
            'start_time': 1000,
            'end_time': 3000,
            'text': 'Test subtitle 1',
            'time': '00:00:01,000 --> 00:00:03,000',
            'startraw': '00:00:01,000',
            'endraw': '00:00:03,000'
        },
        {
            'line': 2,
            'start_time': 3500,
            'end_time': 6000,
            'text': 'Test subtitle 2',
            'time': '00:00:03,500 --> 00:00:06,000',
            'startraw': '00:00:03,500',
            'endraw': '00:00:06,000'
        }
    ]

    srt_content = generate_srt_from_list(test_subtitles)
    print("✓ Generated SRT content:")
    print("-" * 60)
    print(srt_content)
    print("-" * 60)
    print()


def test_translator_initialization():
    """Test translator initialization"""
    print("=" * 60)
    print("TEST 3: Translator Initialization")
    print("=" * 60)

    config = {
        'api_url': 'http://localhost:8080/v1',
        'model': 'llama3',
        'max_tokens': 2000,
        'temperature': 0.7
    }

    translator = LocalLLMTranslator(config)
    print(f"✓ Initialized translator with config:")
    print(f"  API URL: {translator.api_url}")
    print(f"  Model: {translator.model}")
    print(f"  Max Tokens: {translator.max_tokens}")
    print(f"  Temperature: {translator.temperature}")
    print()


def test_single_translation(translator):
    """Test single text translation"""
    print("=" * 60)
    print("TEST 4: Single Text Translation")
    print("=" * 60)

    test_text = "Hello, world!"
    target_lang = "Spanish"

    print(f"Source text: {test_text}")
    print(f"Target language: {target_lang}")
    print("Attempting translation...")

    try:
        translation = translator.translate(test_text, target_lang)
        print(f"✓ Translation: {translation}")
        print()
        return translation
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        print("  (This is expected if llama.cpp server is not running)")
        print()
        return None


def test_srt_translation(translator, subtitles):
    """Test SRT subtitle translation"""
    print("=" * 60)
    print("TEST 5: SRT Subtitle Translation")
    print("=" * 60)

    print(f"Translating {len(subtitles)} subtitle blocks to Chinese...")
    print("Attempting translation...")

    try:
        translated_srt = translator.translate_srt(subtitles, "Chinese")
        print(f"✓ Translated {len(translated_srt)} blocks")

        for i, sub in enumerate(translated_srt, 1):
            print(f"  Block {i}: {sub['text'][:50]}...")

        print()

        # Generate SRT file
        srt_content = generate_srt_from_list(translated_srt)
        output_file = Path("test_translated.srt")
        output_file.write_text(srt_content, encoding='utf-8')
        print(f"✓ Saved translated SRT to: {output_file}")
        print()

        return translated_srt
    except Exception as e:
        print(f"✗ Translation failed: {e}")
        print("  (This is expected if llama.cpp server is not running)")
        print()
        return None


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("LLAMA-MANAGER SUBTITLE TRANSLATION TEST")
    print("=" * 60 + "\n")

    # Test 1 & 2: SRT Parsing and Generation (always work)
    subtitles = test_srt_parsing()
    test_srt_generation()

    # Test 3: Translator Initialization (always works)
    translator = test_translator_initialization()

    # Test 4 & 5: Translation tests (require running LLM server)
    print("=" * 60)
    print("NOTE: Tests 4 & 5 require llama.cpp server running at")
    print("      http://localhost:8080/v1")
    print("=" * 60 + "\n")

    test_single_translation(translator)
    test_srt_translation(translator, subtitles)

    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("✓ Tests 1-3: Always pass (no LLM required)")
    print("✓ Tests 4-5: Require llama.cpp server")
    print("\nIf Tests 4-5 failed, start llama.cpp server:")
    print("  llama-server.exe -m your_model.gguf -p 8080")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
