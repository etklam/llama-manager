# Product

## Register

product

## Users

One operator, at their own Windows desktop, with a Vulkan GPU they chose on purpose.

They are technically fluent — they built llama.cpp from source, they know what a
KV cache quant is, they pick GGUF quantizations by name. They are not the
audience for hand-holding, and an interface that hides the numbers from them is
an interface they will work around.

Their context is **unattended duration**. The jobs this tool starts are not
interactive: transcribing two hours of Japanese audio is 20+ minutes of GPU
time, and translating the 1,500 cues that come out of it is another 20. They
start a run and leave. They come back and look at one thing first: *is it still
alive, and how much is left?* The second thing they look at, only when something
went wrong, is the log.

The job to be done: **turn Japanese media into Traditional Chinese subtitles and
translated manga, on their own hardware, without a cloud bill and without
babysitting.**

## Product Purpose

llama.cpp Manager is the control surface for a local AI media pipeline. It owns
four things the operator would otherwise do by hand across three terminals:

1. **Serving** — launch and configure `llama-server` against a chosen GGUF, with
   presets for the three real workloads (translation, chat, long context).
2. **Transcription** — drive `whisper-cli` over a batch of media into committed SRT.
3. **Translation** — translate SRT/TXT through the local server or a remote
   OpenAI-compatible profile, with story-context understanding and recovery for
   missed lines.
4. **Manga** — OCR page images and translate them as documents.

Success is **a finished batch the operator didn't have to watch**, and when a
batch does fail, a log that says why without being reconstructed from four
places.

The underlying engine is already careful about this: runs are frozen into
immutable snapshots, output is published through a commit gate so a cancel can
never truncate a file, retries have exactly one owner. The interface has not
been holding up its end of that bargain.

## Brand Personality

**Instrument, not app.** Three words: *legible, unsurprising, candid.*

- **Legible** — the numbers are the content. Rates, slot counts, ETAs, token
  budgets, VRAM. Show them in a font where digits line up, at a contrast that
  survives a glance from across the room.
- **Unsurprising** — this is a tool in the user's hands for hours. Every control
  behaves the way the same control behaved on the last surface. Earned
  familiarity beats novelty; the interface should disappear into the task.
- **Candid** — when a run fails, say what failed and what was kept. When a
  setting is clamped, say what clamped it. Never report success for a file that
  wasn't written. The engine already works this way; the UI must match it.

## Anti-references

- **Consumer AI chat apps.** No gradient hero, no oversized friendly type, no
  rounded-pill everything, no emoji carrying meaning that text should carry.
  (The current UI has emoji in section titles — `📦 模型選擇`, `🧹 釋放內存` —
  decorative, and in the server panel they're load-bearing labels for controls
  that deserve real words.)
- **Bootstrap-blue admin dashboard.** The saturated `#2563eb`-on-near-white
  default. It's the current theme, and it reads as a template rather than an
  instrument.
- **The "AI made this" dark theme**: pure `#000` background, neon accent,
  glowing borders, monospace everywhere including prose.
- **Hidden state.** The thing this interface is being rebuilt to stop doing:
  starting a 40-minute run and then showing its status only on the one tab the
  user navigated away from.

## Design Principles

1. **Run state is global, not local.** A run belongs to the application, not to
   the tab that started it. If something is executing, that fact and its
   progress are visible from every surface, always.

2. **One log, filtered — never five.** The engine already tags every line with a
   channel. The interface should have exactly one place to read them, with the
   channel as a filter, not four per-tab widgets plus a floating debug window
   that each show a slice.

3. **One way to start each job.** Today a translation can be started from three
   different surfaces with three slightly different option sets. Each workflow
   gets exactly one authoritative control surface; other surfaces link to it
   rather than reimplementing it.

4. **Progressive disclosure by frequency, not by "advanced".** Group controls by
   how often this operator touches them — model and port every session, KV cache
   quant almost never — rather than dumping fourteen fields into one panel and
   hiding a second set behind a toggle that reflows the page.

5. **Legibility is a requirement, not a preference.** Every text/background pair
   meets WCAG AA (4.5:1 body, 3:1 large). Numeric readouts are tabular and
   monospaced. This is checkable, so it gets checked in code, not by eye.

## Accessibility & Inclusion

*Inferred — not yet confirmed with the user.*

- **Target: WCAG 2.1 AA for contrast**, verified programmatically against the
  token palette rather than asserted. Both themes must pass.
- **Keyboard**: every surface reachable without the mouse (`Ctrl+1..5` for
  surfaces, `Ctrl+L` for the log). The app uses native ttk widgets specifically
  so focus order and keyboard behavior stay native — that stays true.
- **Color is never the only signal.** Server state, run state, and log levels
  each carry a glyph and a word alongside the color, so red/green confusion
  (~8% of men with Northern European ancestry) never costs information.
- **Reduced motion**: Tk has no `prefers-reduced-motion`, so a
  `ui.reduce_motion` config flag replaces the animated run spinner with a static
  indicator.
- **CJK first.** The interface language is Traditional Chinese. Font selection
  must resolve a real CJK family per platform rather than hardcoding one
  Windows font, and layouts must tolerate the wider glyphs and the longer
  English strings that sit beside them.

---

*Note: this file was written from the codebase and README rather than from a
product interview. The Users, Brand Personality, Anti-references, and
Accessibility sections are informed inference and are the ones most worth
correcting.*
