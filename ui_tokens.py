"""Design tokens for the desktop shell: color, type, spacing, motion.

Why a token module instead of hex literals scattered through the tabs: the old
theme hardcoded nine colors in ``ui_theme.apply_theme`` and every tab then
invented its own on top of them (``foreground="blue"``, ``"gray"``, ``"orange"``
appear across five files). There was no way to add a dark theme, and no way to
know whether any pair of colors was actually readable.

Colors are authored in **OKLCH** — lightness, chroma, hue — because that is the
space where "same lightness, different hue" is true perceptually, which is what
makes a semantic ramp (ok / warn / danger at matched weight) look like one
family instead of four unrelated colors. Tk only accepts ``#rrggbb``, so the
conversion to sRGB happens here, once, at import.

Two things this module guarantees that a hex palette cannot:

* **Gamut safety.** A requested OKLCH triple can sit outside sRGB. Naive
  conversion clips a channel and silently shifts the hue (a clipped blue turns
  violet). :func:`oklch_hex` reduces chroma until the color actually fits,
  preserving hue and lightness — the two attributes the palette's structure
  depends on.
* **Checkable contrast.** :func:`contrast_ratio` is real WCAG 2.1 math, and
  :func:`audit_palette` runs every text-on-surface pair in both themes. The
  accessibility claim in PRODUCT.md is therefore testable rather than asserted;
  ``tools/preview_ui.py`` prints the audit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# OKLCH -> sRGB
# ---------------------------------------------------------------------------

# Ottosson's OKLab matrices. Kept as literals rather than imported from a color
# library so the UI layer keeps its only-stdlib dependency profile.
_LMS_FROM_OKLAB = (
    (1.0, +0.3963377774, +0.2158037573),
    (1.0, -0.1055613458, -0.0638541728),
    (1.0, -0.0894841775, -1.2914855480),
)
_LINEAR_RGB_FROM_LMS = (
    (+4.0767416621, -3.3077115913, +0.2309699292),
    (-1.2684380046, +2.6097574011, -0.3413193965),
    (-0.0041960863, -0.7034186147, +1.7076147010),
)


def _linear_to_srgb(c: float) -> float:
    """Linear-light channel to gamma-encoded sRGB (IEC 61966-2-1)."""
    if c <= 0.0031308:
        return 12.92 * c
    return 1.055 * (c ** (1 / 2.4)) - 0.055


def _srgb_to_linear(c: float) -> float:
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def oklch_to_linear_rgb(lightness: float, chroma: float,
                        hue_deg: float) -> Tuple[float, float, float]:
    """OKLCH to linear-light sRGB, unclamped so gamut can be tested."""
    hue = math.radians(hue_deg)
    lab = (lightness, chroma * math.cos(hue), chroma * math.sin(hue))
    lms = tuple(
        sum(coefficient * component for coefficient, component in zip(row, lab))
        for row in _LMS_FROM_OKLAB
    )
    lms_cubed = tuple(value ** 3 for value in lms)
    return tuple(
        sum(coefficient * component
            for coefficient, component in zip(row, lms_cubed))
        for row in _LINEAR_RGB_FROM_LMS
    )


def _in_gamut(rgb: Tuple[float, float, float], tolerance: float = 1e-4) -> bool:
    return all(-tolerance <= channel <= 1 + tolerance for channel in rgb)


def oklch_hex(lightness: float, chroma: float, hue_deg: float) -> str:
    """OKLCH to ``#rrggbb``, reducing chroma rather than clipping channels.

    Clipping is what makes naive OKLCH palettes drift: an out-of-gamut blue has
    its blue channel pinned at 1.0 and comes back violet, so a ramp authored at
    one hue renders at several. Binary-searching chroma down to the gamut
    boundary keeps hue and lightness — the structural attributes — intact and
    gives up only saturation, which is the attribute with slack.
    """
    if _in_gamut(oklch_to_linear_rgb(lightness, chroma, hue_deg)):
        fitted = chroma
    else:
        low, high = 0.0, chroma
        for _ in range(24):  # ~1e-7 of the chroma range; far below 8-bit output
            mid = (low + high) / 2
            if _in_gamut(oklch_to_linear_rgb(lightness, mid, hue_deg)):
                low = mid
            else:
                high = mid
        fitted = low
    channels = oklch_to_linear_rgb(lightness, fitted, hue_deg)
    return "#" + "".join(
        f"{round(255 * min(1.0, max(0.0, _linear_to_srgb(channel)))):02x}"
        for channel in channels
    )


# ---------------------------------------------------------------------------
# WCAG contrast
# ---------------------------------------------------------------------------

def relative_luminance(hex_color: str) -> float:
    """WCAG 2.1 relative luminance of an ``#rrggbb`` string."""
    value = hex_color.lstrip("#")
    channels = [int(value[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    r, g, b = (_srgb_to_linear(channel) for channel in channels)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(foreground: str, background: str) -> float:
    """WCAG 2.1 contrast ratio, 1.0 (identical) to 21.0 (black on white)."""
    a = relative_luminance(foreground)
    b = relative_luminance(background)
    lighter, darker = max(a, b), min(a, b)
    return (lighter + 0.05) / (darker + 0.05)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

# The accent hue is deliberately outside the 20-150 band that carries status.
#
# This app's most important information is state: a run is ok (green), degraded
# (amber), or failed (red). An accent sharing that band competes with the one
# signal the operator scans for, so "selected tab" and "this file failed" would
# read as the same kind of event. 265 is indigo — adjacent to nothing semantic.
_ACCENT_HUE = 265.0

# Neutrals carry a trace of the accent hue (C 0.004-0.018) rather than being
# pure gray. Enough to read as one deliberate family under a monitor's light,
# far too little to look tinted. Not warmed by default: a warm near-white body
# is the saturated default this project is explicitly avoiding.
_NEUTRAL_HUE = _ACCENT_HUE


@dataclass(frozen=True)
class Palette:
    """One complete theme. Field names are roles, never color names.

    ``surface`` is where panels live, ``canvas`` is the app behind them, and
    ``raised`` is for inputs and the nav's selected item. Three levels is
    enough for a desktop shell and keeps every border a real 1px line, since
    ttk has no shadows to lean on.
    """

    name: str
    dark: bool

    canvas: str
    surface: str
    raised: str
    overlay: str

    border: str
    border_strong: str
    focus: str

    ink: str
    ink_muted: str
    ink_faint: str
    ink_on_accent: str

    accent: str
    accent_hover: str
    accent_pressed: str
    accent_soft: str

    ok: str
    warn: str
    danger: str
    info: str

    ok_soft: str
    warn_soft: str
    danger_soft: str

    selection_bg: str
    selection_ink: str
    track: str


def _dark_palette() -> Palette:
    """Default theme: the monitor is the room's main light source.

    Lightness steps are spaced so adjacent surfaces differ by >=0.04 L, which
    survives a cheap panel's gamma. Body ink sits at 0.95 rather than pure
    white: #fff on a dark panel blooms and costs legibility over a long session.
    """
    hue, accent = _NEUTRAL_HUE, _ACCENT_HUE
    return Palette(
        name="dark",
        dark=True,
        canvas=oklch_hex(0.185, 0.012, hue),
        surface=oklch_hex(0.235, 0.013, hue),
        raised=oklch_hex(0.285, 0.014, hue),
        overlay=oklch_hex(0.325, 0.015, hue),
        border=oklch_hex(0.560, 0.015, hue),
        border_strong=oklch_hex(0.600, 0.018, hue),
        focus=oklch_hex(0.700, 0.140, accent),
        ink=oklch_hex(0.950, 0.004, hue),
        ink_muted=oklch_hex(0.775, 0.011, hue),
        ink_faint=oklch_hex(0.650, 0.013, hue),
        ink_on_accent=oklch_hex(0.165, 0.020, accent),
        accent=oklch_hex(0.660, 0.150, accent),
        accent_hover=oklch_hex(0.720, 0.150, accent),
        accent_pressed=oklch_hex(0.600, 0.145, accent),
        accent_soft=oklch_hex(0.285, 0.060, accent),
        ok=oklch_hex(0.760, 0.150, 150.0),
        warn=oklch_hex(0.810, 0.150, 80.0),
        danger=oklch_hex(0.700, 0.185, 25.0),
        info=oklch_hex(0.780, 0.100, 230.0),
        ok_soft=oklch_hex(0.330, 0.070, 150.0),
        warn_soft=oklch_hex(0.340, 0.070, 80.0),
        danger_soft=oklch_hex(0.305, 0.090, 25.0),
        selection_bg=oklch_hex(0.400, 0.110, accent),
        selection_ink=oklch_hex(0.980, 0.005, hue),
        track=oklch_hex(0.310, 0.014, hue),
    )


def _light_palette() -> Palette:
    """Daylight theme: same roles, same hues, inverted lightness structure.

    Not a mechanical inversion — chroma has to rise as lightness falls to keep
    the semantic colors equally identifiable, so the accent and status hues are
    authored darker *and* more saturated than their dark-theme counterparts.
    """
    hue, accent = _NEUTRAL_HUE, _ACCENT_HUE
    return Palette(
        name="light",
        dark=False,
        canvas=oklch_hex(0.970, 0.005, hue),
        surface=oklch_hex(0.999, 0.000, hue),
        raised=oklch_hex(0.975, 0.006, hue),
        overlay=oklch_hex(0.945, 0.008, hue),
        border=oklch_hex(0.640, 0.010, hue),
        border_strong=oklch_hex(0.590, 0.014, hue),
        focus=oklch_hex(0.520, 0.170, accent),
        ink=oklch_hex(0.245, 0.016, hue),
        ink_muted=oklch_hex(0.455, 0.016, hue),
        ink_faint=oklch_hex(0.535, 0.016, hue),
        ink_on_accent=oklch_hex(0.995, 0.002, accent),
        accent=oklch_hex(0.480, 0.175, accent),
        accent_hover=oklch_hex(0.420, 0.170, accent),
        accent_pressed=oklch_hex(0.370, 0.155, accent),
        accent_soft=oklch_hex(0.930, 0.035, accent),
        ok=oklch_hex(0.450, 0.130, 150.0),
        warn=oklch_hex(0.480, 0.125, 70.0),
        danger=oklch_hex(0.480, 0.190, 25.0),
        info=oklch_hex(0.470, 0.110, 230.0),
        ok_soft=oklch_hex(0.940, 0.040, 150.0),
        warn_soft=oklch_hex(0.945, 0.045, 80.0),
        danger_soft=oklch_hex(0.945, 0.040, 25.0),
        selection_bg=oklch_hex(0.480, 0.175, accent),
        selection_ink=oklch_hex(0.995, 0.002, accent),
        track=oklch_hex(0.910, 0.008, hue),
    )


PALETTES: Dict[str, Palette] = {
    "dark": _dark_palette(),
    "light": _light_palette(),
}

# Dark is the default because of how this tool is actually used: long
# unattended runs, checked by glance, in a room where the monitor is the
# brightest thing. A near-white full-window background is the wrong instrument
# for that. ``light`` ships complete for daylight work and is switchable at
# runtime; neither is a fallback for the other.
DEFAULT_THEME = "dark"


def palette_for(name: str) -> Palette:
    """Look up a palette, falling back to the default rather than raising.

    Theme name arrives from config.json, which the user edits by hand; a typo
    there should start the app in the default theme, not crash it on launch.
    """
    return PALETTES.get(name, PALETTES[DEFAULT_THEME])


# ---------------------------------------------------------------------------
# Type
# ---------------------------------------------------------------------------

# Candidate families in preference order, resolved against the system's real
# font list at theme time (see ui_theme.resolve_fonts). The old theme hardcoded
# "Microsoft JhengHei UI", which silently fell back to a default on any machine
# that wasn't the author's — including every non-Windows one.
#
# Every UI candidate carries Han glyphs: the interface is Traditional Chinese,
# and a family without CJK coverage makes Tk substitute per-glyph, which is how
# you get a label where the Latin and Han halves are visibly different fonts.
UI_FONT_CANDIDATES = (
    "Microsoft JhengHei UI",   # Windows, zh-TW UI face
    "Microsoft YaHei UI",      # Windows, zh-CN fallback
    "PingFang TC",             # macOS, zh-TW
    "Noto Sans CJK TC",        # Linux
    "Source Han Sans TC",
    "Segoe UI",                # Latin-only last resorts
    "Helvetica Neue",
)

# Mono is for data, not prose: rates, token counts, ports, VRAM, log lines.
# Tabular digits are the whole point — a changing ETA must not reflow its row.
MONO_FONT_CANDIDATES = (
    "Cascadia Mono",
    "Consolas",
    "SF Mono",
    "Menlo",
    "DejaVu Sans Mono",
    "Courier New",
)

# Fixed point sizes, not fluid. A desktop app is viewed at one DPI by one
# person; clamping type to the window width makes a sidebar label shrink for no
# reason. Ratio is ~1.15 between steps, tight on purpose: this UI has many type
# roles and a wide ratio turns that into noise.
TYPE_SCALE: Dict[str, int] = {
    "display": 17,   # app name in the header, once
    "title": 14,     # surface title, once per surface
    "heading": 11,   # section label
    "body": 10,      # default for everything
    "label": 9,      # field labels, nav items, secondary
    "micro": 8,      # log timestamps, unit suffixes
}

# ---------------------------------------------------------------------------
# Space
# ---------------------------------------------------------------------------

# 4px-based, with a 6 and a 10 kept because ttk padding often needs the
# half-step to align a bordered panel against an unbordered one. Named steps
# only: the old layout code used literal 5s and 12s interchangeably, which is
# why nothing lined up between tabs.
SPACE: Dict[str, int] = {
    "none": 0,
    "hair": 2,
    "tight": 4,
    "snug": 6,
    "base": 8,
    "cozy": 10,
    "roomy": 12,
    "loose": 16,
    "section": 20,
    "page": 24,
}

# Widths that need to agree across surfaces.
METRICS: Dict[str, int] = {
    "nav_width": 164,       # fits the longest nav label at `label` size
    "header_height": 56,
    "drawer_height": 190,
    "field_label": 108,     # right-aligned label column in forms
    "control_min": 220,
}

# ---------------------------------------------------------------------------
# Motion
# ---------------------------------------------------------------------------

# Tk has no transitions, so "motion" here is only what a periodic callback can
# express. That limit is a feature: it rules out decorative animation and
# leaves exactly the one case the product register allows, which is conveying
# that work is in progress.
SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
SPINNER_INTERVAL_MS = 120
STATIC_BUSY_GLYPH = "●"  # substituted when ui.reduce_motion is on

# Status glyphs pair with every status color. Color is never the only carrier
# of state: red/green confusion is common enough that a palette which encodes
# "failed" solely as hue loses information for ~8% of men.
GLYPH = {
    "idle": "○",
    "busy": "●",
    "ok": "✓",
    "warn": "!",
    "error": "✕",
}

LOG_LEVEL_ROLE = {
    "INFO": "ink_muted",
    "SUCCESS": "ok",
    "WARNING": "warn",
    "ERROR": "danger",
    "DEBUG": "ink_faint",
}


# ---------------------------------------------------------------------------
# Self-audit
# ---------------------------------------------------------------------------

# Every pair the UI actually renders, as (foreground_role, background_role,
# minimum_ratio). All text roles used here are below the large-text sizes,
# including bold labels, so they require 4.5. Control boundaries require 3.0.
_CONTRAST_CONTRACT: Tuple[Tuple[str, str, float], ...] = (
    ("ink", "canvas", 4.5),
    ("ink", "surface", 4.5),
    ("ink", "raised", 4.5),
    ("ink_muted", "canvas", 4.5),
    ("ink_muted", "surface", 4.5),
    ("ink_muted", "raised", 4.5),
    ("ink_faint", "surface", 4.5),
    ("ink_faint", "canvas", 4.5),
    ("ink_on_accent", "accent", 4.5),
    ("ink_on_accent", "accent_hover", 4.5),
    ("ink_on_accent", "accent_pressed", 4.5),
    ("accent", "surface", 4.5),
    ("accent", "canvas", 4.5),
    ("accent", "accent_soft", 4.5),
    ("ok", "surface", 4.5),
    ("warn", "surface", 4.5),
    ("danger", "surface", 4.5),
    ("info", "surface", 4.5),
    ("ok", "canvas", 4.5),
    ("warn", "canvas", 4.5),
    ("danger", "canvas", 4.5),
    ("danger", "raised", 4.5),
    ("danger", "danger_soft", 4.5),
    ("ok", "ok_soft", 4.5),
    ("warn", "warn_soft", 4.5),
    ("selection_ink", "selection_bg", 4.5),
    ("border", "canvas", 3.0),
    ("border", "surface", 3.0),
    ("border", "raised", 3.0),
    ("border_strong", "canvas", 3.0),
    ("border_strong", "surface", 3.0),
    ("border_strong", "raised", 3.0),
    ("border_strong", "overlay", 3.0),
    ("focus", "surface", 3.0),
    ("focus", "raised", 3.0),
)


def audit_palette(palette: Palette) -> List[Tuple[str, str, float, float, bool]]:
    """Check one palette against the contrast contract.

    Returns ``(foreground, background, ratio, required, passed)`` per pair.
    Intended for ``tools/preview_ui.py`` and any future CI step — the palette is
    generated code, so its readability is a property that can regress silently
    when a lightness value is nudged.
    """
    results = []
    for fg_role, bg_role, required in _CONTRAST_CONTRACT:
        ratio = contrast_ratio(getattr(palette, fg_role), getattr(palette, bg_role))
        results.append(
            (fg_role, bg_role, round(ratio, 2), required, ratio >= required)
        )
    return results


def audit_failures() -> List[str]:
    """Human-readable failures across every shipped palette (empty when clean)."""
    problems = []
    for palette in PALETTES.values():
        for fg, bg, ratio, required, passed in audit_palette(palette):
            if not passed:
                problems.append(
                    f"{palette.name}: {fg} on {bg} is {ratio}:1, needs {required}:1"
                )
    return problems
