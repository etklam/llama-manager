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

**Chunk Plan**:
The output of the Chunking Policy for one piece of media: whether to chunk, and the chunk size and count the engine executes.
_Avoid_: Chunk flag, chunk seconds

**Committed SRT**:
The complete SRT that becomes visible at the destination only after a Transcription succeeds. Cancellation or failure leaves any previously committed SRT unchanged.
_Avoid_: Partial SRT, temporary SRT

**Cue**:
One subtitle entry (start and end time, text, line number). The single element type that crosses between speech recognition and subtitle translation: produced by SRT parsing, consumed by SRT generation.
_Avoid_: SRT dict, subtitle line, entry dict

**Transcription Outcome**:
The terminal state of a Transcription: Completed with a Committed SRT, Cancelled by the user, or Failed during processing.
_Avoid_: Empty result, callback status

**Transcription Run**:
One invocation of the shared run loop over one or more Transcriptions: request building, event routing to the shell, and outcome handling (Committed SRT, Cancelled, Failed).
_Avoid_: Batch loop, run loop

## Subtitle Translation

**Translation Preflight**:
The probe→clamp→plan decision made once before a translation run: whether the server is reachable, how many workers fit, and one note explaining either outcome.
_Avoid_: Probe, preflight check, server health check
