#!/usr/bin/env python3
"""Render the redesigned shell for visual inspection, and audit the palette.

Why this exists: a theme cannot be verified by reading it. Contrast ratios are
checkable arithmetic (``--audit`` does that and exits non-zero on a failure),
but spacing, alignment, type hierarchy and widget states have to be looked at.
This builds the real shell with the real components — no mock widgets — so what
you see is what the app renders.

    python tools/preview_ui.py                # build the gallery window
    python tools/preview_ui.py --audit        # contrast report only, no window
    python tools/preview_ui.py --theme light  # start in the light palette
    python tools/preview_ui.py --app          # drive the real LlamaManager

``--app`` needs the project's real dependencies (tkinterdnd2, Pillow, psutil).
The gallery mode does not: it stubs them, so the design can be inspected on a
machine that is not the Windows box this app ships to.
"""
from __future__ import annotations

import argparse
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def install_stubs() -> None:
    """Stand in for the runtime-only dependencies.

    ``tkinterdnd2`` and ``Pillow`` exist to accept dropped files and decode
    manga pages. Neither is needed to look at a layout, and requiring them
    would mean the design could only be reviewed on the target machine.
    """
    if "tkinterdnd2" not in sys.modules:
        dnd = types.ModuleType("tkinterdnd2")
        dnd.DND_FILES = "DND_Files"

        class _TkinterDnD:
            @staticmethod
            def Tk():  # pragma: no cover - only used by the real app path
                import tkinter
                return tkinter.Tk()

        dnd.TkinterDnD = _TkinterDnD
        sys.modules["tkinterdnd2"] = dnd

    # Listbox has no drop_target_register without the real extension; a no-op
    # keeps FileListbox constructible.
    import tkinter
    if not hasattr(tkinter.Listbox, "drop_target_register"):
        tkinter.Listbox.drop_target_register = lambda self, *a, **k: None
        tkinter.Listbox.dnd_bind = lambda self, *a, **k: None

    if "PIL" not in sys.modules:
        pil = types.ModuleType("PIL")
        image_mod = types.ModuleType("PIL.Image")
        ops_mod = types.ModuleType("PIL.ImageOps")
        tkmod = types.ModuleType("PIL.ImageTk")
        image_mod.open = lambda *a, **k: (_ for _ in ()).throw(
            RuntimeError("Pillow not installed in preview mode"))
        ops_mod.exif_transpose = lambda image: image
        tkmod.PhotoImage = lambda *a, **k: None
        pil.Image, pil.ImageOps, pil.ImageTk = image_mod, ops_mod, tkmod
        sys.modules["PIL"] = pil
        sys.modules["PIL.Image"] = image_mod
        sys.modules["PIL.ImageOps"] = ops_mod
        sys.modules["PIL.ImageTk"] = tkmod


def audit() -> int:
    """Print the contrast report for every palette. Non-zero on any failure."""
    from ui_tokens import PALETTES, audit_palette

    failures = 0
    for name, palette in PALETTES.items():
        print(f"\n{name.upper()}  ({'dark' if palette.dark else 'light'})")
        print(f"  {'pair':38s} {'ratio':>7s} {'needs':>7s}")
        for fg, bg, ratio, required, passed in audit_palette(palette):
            flag = "ok " if passed else "FAIL"
            print(f"  {flag} {fg + ' on ' + bg:34s} {ratio:7.2f} {required:7.1f}")
            failures += 0 if passed else 1
    print(f"\n{failures} failing pair(s)")
    return 1 if failures else 0


def build_gallery(theme_name: str) -> None:
    """Build the shell with one surface per component family."""
    import tkinter as tk
    from tkinter import ttk

    from app_shell import AppShell
    from ui_helpers import (
        CHANNEL_MANGA, CHANNEL_PIPELINE, CHANNEL_SERVER, CHANNEL_TRANSLATE,
        CHANNEL_WHISPER, log_bus, run_bus,
    )
    from ui_kit import (
        ActionBar, Disclosure, EmptyState, FieldGrid, Readout, ScrollArea,
        Section, SegmentedControl, StatusLine, ValidatedSpin, chip,
        labelled_scale,
    )
    from ui_theme import apply_theme
    from ui_tokens import SPACE

    root = tk.Tk()
    root.title("llama.cpp Manager — UI gallery")
    root.geometry("1280x880")
    apply_theme(root, theme_name)

    shell = AppShell(root, None)
    shell.grid(row=0, column=0, sticky="nsew")
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    # ---- Surface 1: every control, in the states they actually appear in.
    controls = ttk.Frame(shell.content, style="Canvas.TFrame")
    area = ScrollArea(controls)
    area.pack(fill="both", expand=True)
    page = area.body
    page.columnconfigure(0, weight=1)

    head = ttk.Frame(page, style="Canvas.TFrame")
    head.grid(row=0, column=0, sticky="ew", padx=SPACE["page"],
              pady=(SPACE["section"], SPACE["loose"]))
    ttk.Label(head, text="元件樣式", style="Display.TLabel").pack(anchor="w")
    ttk.Label(head, text="每個控制項的預設、停用與語意狀態。",
              style="CanvasMuted.TLabel").pack(anchor="w")

    forms = Section(page, "表單控制項", "標籤欄位對齊，說明文字在欄位下方。")
    forms.grid(row=1, column=0, sticky="ew", padx=SPACE["page"],
               pady=(0, SPACE["loose"]))
    grid = FieldGrid(forms.body, columns=2, label_width=10)
    grid.grid(row=0, column=0, sticky="ew")
    ttk.Entry(grid.add("文字欄位")).pack(fill="x")
    ttk.Entry(grid.add("路徑", column=1), style="Mono.TEntry").pack(fill="x")
    ttk.Combobox(grid.add("下拉選單"), values=["a", "b"],
                 state="readonly").pack(fill="x")
    ttk.Spinbox(grid.add("數字", column=1, hint="帶說明的欄位。"),
                from_=1, to=9).pack(anchor="w")
    disabled = ttk.Entry(grid.add("停用欄位"))
    disabled.pack(fill="x")
    disabled.state(("disabled",))
    invalid = ttk.Entry(grid.add("錯誤欄位", column=1), style="Invalid.TEntry")
    invalid.insert(0, "abc")
    invalid.pack(fill="x")
    holder, _ = labelled_scale(grid.add("滑桿"), tk.IntVar(value=3), 1, 8)
    holder.pack(anchor="w")
    seg_var = tk.StringVar(value="b")
    SegmentedControl(grid.add("分段選擇", column=1), seg_var,
                     [("a", "只做 OCR"), ("b", "OCR 後翻譯")]).pack(anchor="w")
    ttk.Checkbutton(forms.body, text="核取方塊").grid(row=1, column=0, sticky="w")

    states = Section(page, "狀態與數據")
    states.grid(row=2, column=0, sticky="ew", padx=SPACE["page"],
                pady=(0, SPACE["loose"]))
    for index, (state, text) in enumerate(
            (("idle", "閒置"), ("busy", "執行中"), ("ok", "完成"),
             ("warn", "已停止"), ("error", "失敗：找不到模型"))):
        StatusLine(states.body, text=text, state=state).grid(
            row=index, column=0, sticky="w")
    chips = ttk.Frame(states.body, style="Surface.TFrame")
    chips.grid(row=5, column=0, sticky="w", pady=(SPACE["base"], 0))
    for kind, text in (("ok", "已驗證"), ("warn", "未驗證"),
                       ("danger", "3 個失敗"), ("accent", "story 模式")):
        chip(chips, text, kind).pack(side="left", padx=(0, SPACE["snug"]))
    meters = ttk.Frame(states.body, style="Surface.TFrame")
    meters.grid(row=6, column=0, sticky="w", pady=(SPACE["loose"], 0))
    for index, (label, value) in enumerate((("系統 RAM", "23.1 / 64 GB"),
                                            ("GPU", "已載入"),
                                            ("伺服器 CPU", "47%"),
                                            ("速率", "14.2 行/秒"))):
        Readout(meters, label, value).grid(row=0, column=index, sticky="w",
                                           padx=(0, SPACE["section"]))
    for index, style in enumerate(("Accent", "Ok", "Warn", "Danger")):
        bar = ttk.Progressbar(states.body, maximum=100,
                              style=f"{style}.Horizontal.TProgressbar")
        bar.configure(value=[62, 100, 40, 80][index])
        bar.grid(row=7 + index, column=0, sticky="ew",
                 pady=(SPACE["snug"] if index == 0 else SPACE["hair"], 0))

    buttons = Section(page, "動作")
    buttons.grid(row=3, column=0, sticky="ew", padx=SPACE["page"],
                 pady=(0, SPACE["loose"]))
    bar = ActionBar(buttons.body)
    bar.grid(row=0, column=0, sticky="ew")
    bar.ghost("次要動作", lambda: None)
    bar.secondary("一般按鈕", lambda: None)
    bar.primary("主要動作", lambda: None)
    bar.danger("停止", lambda: None)
    disabled_row = ttk.Frame(buttons.body, style="Surface.TFrame")
    disabled_row.grid(row=1, column=0, sticky="w", pady=(SPACE["base"], 0))
    for text, style in (("停用主要", "Accent.TButton"), ("停用一般", "TButton"),
                        ("停用危險", "Danger.TButton")):
        button = ttk.Button(disabled_row, text=text, style=style)
        button.state(("disabled",))
        button.pack(side="left", padx=(0, SPACE["snug"]))

    disclosure = Disclosure(page, "折疊區段（預設收起）")
    disclosure.grid(row=4, column=0, sticky="ew", padx=SPACE["page"],
                    pady=(0, SPACE["loose"]))
    ttk.Label(disclosure.body, text="很少動到的設定放這裡。",
              style="Muted.TLabel").grid(row=0, column=0, sticky="w")
    spin = ValidatedSpin(disclosure.body, tk.StringVar(value="8192"),
                         "context", 512, 1_048_576)
    spin.grid(row=1, column=0, sticky="w", pady=(SPACE["snug"], 0))

    empty = Section(page, "空狀態")
    empty.grid(row=5, column=0, sticky="ew", padx=SPACE["page"],
               pady=(0, SPACE["page"]))
    EmptyState(empty.body, "尚未設定模型目錄",
               "指向含有 .gguf 檔的資料夾，之後就會自動掃描。",
               ("選擇資料夾…", lambda: None)).grid(row=0, column=0)

    shell.register("controls", controls, "元件樣式",
                   channels=(CHANNEL_TRANSLATE,))

    # ---- Surface 2: type scale, at render size.
    type_surface = ttk.Frame(shell.content, style="Canvas.TFrame")
    type_section = Section(type_surface, "字級", "固定級距，比例約 1.15。")
    type_section.pack(fill="x", padx=SPACE["page"], pady=SPACE["page"])
    for style, label in (("Display.TLabel", "Display 17 — 介面標題"),
                         ("Title.TLabel", "Title 14 — 區段標題"),
                         ("Heading.TLabel", "Heading 11 — 群組標題"),
                         ("Body.TLabel", "Body 10 — 一般內文"),
                         ("Muted.TLabel", "Label 9 — 欄位標籤與次要說明"),
                         ("Micro.TLabel", "Micro 8 — 提示與時間戳"),
                         ("Mono.TLabel", "Mono 9 — 1234567890 數據對齊"),
                         ("MonoMicro.TLabel", "MonoMicro 8 — 09:41:22 日誌")):
        ttk.Label(type_section.body, text=label, style=style).pack(
            anchor="w", pady=(0, SPACE["tight"]))
    shell.register("type", type_surface, "字級")

    # ---- Surface 3: the empty/first-run case.
    first_run = ttk.Frame(shell.content, style="Canvas.TFrame")
    EmptyState(first_run, "還沒有可用的模型",
               "把 llama.cpp 的建置路徑指好，應用就會掃描 .gguf 並列出來。",
               ("開啟伺服器設定", lambda: shell.select("controls"))).place(
        relx=0.5, rely=0.42, anchor="center")
    shell.register("firstrun", first_run, "首次啟動")

    shell.select("controls")

    # Populate the chrome so the header and drawer are not shown empty.
    shell.header.set_server(True, 8080, "qwen3-30b-a3b-instruct-q5_k_m")
    shell.drawer.show(True)
    for level, message, channel in (
            ("INFO", "llama-server 啟動於 http://localhost:8080", CHANNEL_SERVER),
            ("INFO", "已解析 1482 行字幕，開始逐行翻譯…", CHANNEL_TRANSLATE),
            ("SUCCESS", "已寫出 ep01.zh-tw.srt", CHANNEL_TRANSLATE),
            ("WARNING", "伺服器只有 1 個 slot，並發數已由 3 降為 1", CHANNEL_TRANSLATE),
            ("ERROR", "第 842 行重試後仍失敗，整檔未寫出", CHANNEL_TRANSLATE),
            ("INFO", "辨識 12/48", CHANNEL_MANGA),
            ("INFO", "whisper: 轉錄 00:42:10 / 01:58:33", CHANNEL_WHISPER),
            ("INFO", "管線：ep02.mkv 轉錄完成，接續翻譯", CHANNEL_PIPELINE)):
        log_bus.emit(level, message, channel)
    run_bus.progress(CHANNEL_TRANSLATE, "翻譯 ep01.srt (3/12)", 62.0,
                     "14.2 行/秒 · 剩餘 02:13")

    root.mainloop()


def run_real_app(theme_name: str) -> None:
    from ui_tokens import DEFAULT_THEME  # noqa: F401  (import check)
    import llama_manager

    llama_manager.main()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", action="store_true",
                        help="print the contrast report and exit")
    parser.add_argument("--theme", default="dark", choices=("dark", "light"))
    parser.add_argument("--app", action="store_true",
                        help="launch the real application instead of the gallery")
    args = parser.parse_args()

    if args.audit:
        return audit()
    if args.app:
        run_real_app(args.theme)
        return 0
    install_stubs()
    build_gallery(args.theme)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
