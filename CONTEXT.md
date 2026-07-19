# llama.cpp Manager

llama.cpp Manager coordinates local model serving, speech recognition, and subtitle translation workflows.

## Speech Recognition

**Transcription**:
The operation that turns one source audio or video file into one complete SRT subtitle file.
_Avoid_: Whisper run, speech-recognition job

**Chunked Transcription**:
A Transcription performed in time-based sections when long media could cause whisper.cpp context collapse; the sections still produce one final SRT.
_Avoid_: Multiple transcriptions, split output

**Chunking Policy**:
The rule that allows long media to use Chunked Transcription while short media remains a single transcription.
_Avoid_: Chunk flag, chunk seconds

**Committed SRT**:
The complete SRT that becomes visible at the destination only after a Transcription succeeds. Cancellation or failure leaves any previously committed SRT unchanged.
_Avoid_: Partial SRT, temporary SRT

**Transcription Outcome**:
The terminal state of a Transcription: Completed with a Committed SRT, Cancelled by the user, or Failed during processing.
_Avoid_: Empty result, callback status
