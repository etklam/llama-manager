AUDIO_EXTENSIONS = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac', '.wma'}
VIDEO_EXTENSIONS = {'.mp4', '.mkv', '.avi', '.mov', '.wmv', '.webm', '.ts'}

SUPPORTED_MEDIA = AUDIO_EXTENSIONS | VIDEO_EXTENSIONS
SUPPORTED_SUBTITLE = {'.srt', '.txt'}

WHISPER_LANGUAGES = {
    'auto': 'Auto Detect',
    'en': 'English',
    'zh': 'Chinese',
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

SOURCE_LANGUAGES = {'auto': 'Auto Detect', **TARGET_LANGUAGES}
