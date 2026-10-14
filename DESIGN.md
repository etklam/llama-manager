# Design

The visual system for llama.cpp Manager's desktop shell. Strategy lives in
[PRODUCT.md](PRODUCT.md); this file is how it looks and where each value lives
in code.

Everything here is generated from `ui_tokens.py` and applied by `ui_theme.py`.
No widget in the app carries a literal color — if you need one, add a role to
the palette.

## Visual theme

**An instrument panel, not an app.** Three flat surface levels separated by
real 1px borders, one restrained accent, and semantic color reserved for state.
The numbers are the content: rates, slot counts, ETAs, ports and VRAM are set
in a monospace face so a changing digit never reflows its row.

Two palettes ship complete and switch at runtime (header button, `Ctrl+Shift+T`,
persisted to `ui.theme`). **Dark is the default**, because of how the tool is
actually used: multi-hour unattended runs checked by glance, in a room where
the monitor is the brightest thing. Light is for daylight work, not a fallback.

Three deliberate constraints come from ttk itself, and the design leans into
them rather than fighting them:

| ttk cannot | So depth/feedback comes from |
|---|---|
| Rounded corners | Square edges throughout. No radius token exists. |
| Shadows | Three surface lightness levels + 1px borders |
| Transitions | One periodic-callback spinner, and nothing else |

## Color

Authored in **OKLCH** (`ui_tokens.oklch_hex`), emitted as sRGB hex because Tk
accepts nothing else. Out-of-gamut colors have their chroma reduced to the
gamut boundary rather than a channel clipped, so hue and lightness — the
structural attributes — survive.

**Strategy: Restrained.** Tinted neutrals plus one accent. Accent appears only
on the primary action, the current nav item, selection, focus, and the progress
fill. Never as decoration.

**The accent hue is 265° (indigo) specifically to stay out of the status band.**
This app's most important signal is state — ok (150°), warn (70–80°), danger
(25°). An accent inside that band would make "this tab is selected" and "this
file failed" read as the same class of event. Neutrals carry a trace of the
same 265° (chroma 0.004–0.018) so the greys read as one family without looking
tinted.

### Dark (default)

| Role | Hex | Used for |
|---|---|---|
| `canvas` | `#101318` | App background, nav rail, header, log drawer |
| `surface` | `#1b1e24` | Section panels — the content plane |
| `raised` | `#272a31` | Inputs, secondary buttons |
| `overlay` | `#30343c` | Hover fills |
| `border` | `#70757e` | Hairlines, panel edges |
| `border_strong` | `#7b808b` | Button edges, emphasis |
| `focus` | `#739bf5` | Keyboard focus ring |
| `ink` | `#edeef1` | Body text (not `#fff`: it blooms on dark over long sessions) |
| `ink_muted` | `#b2b6bd` | Field labels, secondary text |
| `ink_faint` | `#8b8f97` | Hints, timestamps |
| `accent` | `#648eed` | Primary action, selection, progress |
| `ok` | `#61cb7c` | Success, server running |
| `warn` | `#f3b539` | Stopped, clamped, unverified |
| `danger` | `#fc6661` | Failure |
| `info` | `#6ec3eb` | Neutral notice |

### Light

| Role | Hex | | Role | Hex |
|---|---|---|---|---|
| `canvas` | `#f3f5f9` | | `accent` | `#2b53be` |
| `surface` | `#ffffff` | | `ok` | `#00672c` |
| `raised` | `#f5f7fb` | | `warn` | `#825100` |
| `border` | `#898c92` | | `danger` | `#b00a1d` |
| `ink` | `#1d2028` | | `info` | `#006485` |
| `ink_muted` | `#525760` | | `focus` | `#3760c9` |

Not a mechanical inversion: chroma rises as lightness falls, so the semantic
colors stay equally identifiable against white.

### Contrast is checked, not asserted

`ui_tokens.audit_palette()` runs 35 text, selection and control-boundary pairs
against WCAG 2.1 for both palettes. **Both pass with zero failures** (all audited
text ≥4.5:1, control boundaries ≥3:1). Small hints, navigation shortcuts and log
timestamps use the same text threshold; button borders are checked against
their normal and hover fills. Verify after any palette edit:

```bash
python tools/preview_ui.py --audit    # exits non-zero on any failure
```

Lowest passing margins, worth knowing before nudging a lightness value:
`accent on accent_soft` in dark mode and `border on canvas` in light mode.

### Color is never the only signal

Every state carries a glyph and a word alongside its color (`ui_tokens.GLYPH`):
`○` idle, `●` busy, `✓` ok, `!` warn, `✕` error. Red/green confusion must never
cost information.

## Typography

Fixed point sizes, not fluid — one person views this at one DPI, and a heading
that shrinks inside a panel looks worse, not better. Ratio ≈1.15, tight on
purpose: this UI has many type roles and a wide ratio becomes noise.

| Token | Size | Weight | Used for |
|---|---|---|---|
| `display` | 17 | bold | Surface title, once per surface |
| `title` | 14 | bold | Dialog titles |
| `heading` | 11 | bold | Section headings |
| `body` | 10 | regular | Default for everything |
| `label` | 9 | regular | Field labels, nav, secondary |
| `micro` | 8 | regular | Hints, timestamps |

**Families are resolved against the system's real font list**, not hardcoded.
The previous theme pinned `Microsoft JhengHei UI`, which silently substituted
on any machine that wasn't the author's. Every UI candidate carries Han glyphs,
because the interface is Traditional Chinese and a Latin-only family makes Tk
substitute per glyph — which is how one label ends up in two fonts.

- **UI:** Microsoft JhengHei UI → Microsoft YaHei UI → PingFang TC → Noto Sans
  CJK TC → Source Han Sans TC → Segoe UI
- **Data:** Cascadia Mono → Consolas → SF Mono → Menlo → DejaVu Sans Mono

Mono is for data only — rates, counts, ports, paths, log lines — never prose.

## Space

4px-based, named steps only (`ui_tokens.SPACE`). The old layout used literal
`5`s and `12`s interchangeably, which is why nothing lined up between tabs.

`hair 2 · tight 4 · snug 6 · base 8 · cozy 10 · roomy 12 · loose 16 · section 20 · page 24`

Fixed metrics that must agree across surfaces (`ui_tokens.METRICS`): nav rail
164px, header 56px, log drawer 190px.

## Layout

```
┌──────────────────────────────────────────────────────────┐
│ llama.cpp Manager    ⠿ 翻譯 ep01.srt    14.2 行/秒  ◐ 日誌 │  header, always visible
│ ● :8080 qwen3-30b   ▓▓▓▓▓▓▓▓░░░░ 62%                     │
├───────────┬──────────────────────────────────────────────┤
│ 工作台  ⌃1 │                                              │
│ 伺服器  ⌃2 │              active surface                  │
│ 字幕    ⌃3●│                                              │
│ 轉錄    ⌃4 │                                              │
│ 漫畫    ⌃5 │                                              │
├───────────┴──────────────────────────────────────────────┤
│ ▸ 執行日誌  Ctrl+L            ! 2 筆警告／錯誤             │  drawer, collapsed
└──────────────────────────────────────────────────────────┘
```

Replaces a five-tab notebook. Three reasons, each a real defect it fixes:

1. **Run state was local.** A translation started on the Subtitle tab showed
   its progress only there. `StatusHeader` subscribes to `run_bus`, so every
   surface reports to one always-visible place.
2. **The log was in five places** — four per-tab widgets plus a floating debug
   window. `LogDrawer` is one view with the channel as a filter, detachable to
   its own window for a second monitor.
3. **Tabs cannot show activity behind them.** Each nav item carries a live dot.

Surfaces are plain `ttk.Frame`s registered with the shell, so they stay
independently constructible. A surface may implement `on_surface_shown()` to
refresh on entry — this replaces the old `<<NotebookTabChanged>>` index routing.

**Keyboard:** `Ctrl+1`–`Ctrl+5` surfaces · `Ctrl+L` log · `Ctrl+Shift+T` theme.
Shortcuts are printed in the nav rail, because an undiscoverable shortcut is
one nobody uses.

## Components

All in `ui_kit.py`. Compose into a component's `.body`, never into the
component itself.

| Component | Purpose |
|---|---|
| `Section` | Titled panel: heading, optional description, trailing control, body. Replaces `LabelFrame` |
| `Disclosure` | Section that starts collapsed. For settings touched monthly |
| `FieldGrid` | Label-and-control rows on one aligned grid, 1–2 columns, optional per-field hint |
| `StatusLine` | Glyph + text + semantic color. The app's one status primitive |
| `ActionBar` | Run controls: `primary()` / `danger()` right, `secondary()` / `ghost()` left |
| `SegmentedControl` | 2–4 exclusive options, visible without a click. Value and label are separate |
| `Readout` | Labelled numeric value in mono, for live metrics |
| `ValidatedSpin` | Numeric field that marks itself and names itself on bad input |
| `EmptyState` | Teaches the surface: title, explanation, resolving action |
| `ScrollArea` | Scrollable region with all three platform wheel bindings |
| `Spinner` | The one animated element |
| `chip()` / `hairline()` / `labelled_scale()` | Status pill, 1px rule, scale + mono readout |

### Button hierarchy

One primary per surface, for the action that surface exists to perform.

`Accent.TButton` filled accent (primary) · `TButton` bordered (secondary) ·
`Danger.TButton` bordered, danger ink (stop/cancel/delete) · `Ghost.TButton`
borderless (browse, scan, tertiary).

Every interactive style defines default, hover, pressed, disabled and focus.

## Motion

Tk has no transitions, and that limit is useful: it rules out decorative
animation and leaves only the case the product register allows — conveying that
work is in progress.

- **The run spinner**, 10 braille frames at 120ms, in the header while a run is
  active. That is the entire motion vocabulary.
- **Reduced motion:** Tk has no `prefers-reduced-motion`, so `ui.reduce_motion`
  in config.json replaces the animation with a static `●`.
- Progress bars recolor to `ok` / `warn` / `danger` on completion, so the bar
  the operator is already watching reports the outcome.

## Working on this

```bash
python tools/preview_ui.py            # component gallery, all states, both themes
python tools/preview_ui.py --theme light
python tools/preview_ui.py --audit    # contrast report, non-zero on failure
python tools/preview_ui.py --app      # the real application
```

The gallery stubs `tkinterdnd2` and `Pillow`, so the design can be reviewed on
a machine that isn't the Windows box this app ships to. `--app` needs the real
dependencies.

**Rules of thumb**

- Need a color? Add a role to `Palette` in both palettes and re-run `--audit`.
- Need a size? Use a `SPACE` step. If none fits, the layout is probably wrong.
- Building a form? `FieldGrid`. Grouping it? `Section`. Rarely used? `Disclosure`.
- Reporting state? `StatusLine` or `run_bus` — never `foreground="red"`.
- Storing a choice? Stable value in the variable, label only in the UI.
