"""FileListbox — shared file-list widget used by subtitle_tab, whisper_tab,
and pipeline_card. Owns a FileListboxModel (pure-Python state) + a tk.Listbox
+ Browse/Clear buttons + DnD registration.

Three callers differ only in valid extensions. Subclasses ttk.Frame so it
composes inside any container; caller wraps in a LabelFrame if it wants a title.
"""
import tkinter as tk
from tkinter import ttk, filedialog
from pathlib import Path
from typing import Callable, Optional, Sequence

from tkinterdnd2 import DND_FILES

from ui_helpers import parse_dropped_paths


class FileListboxModel:
    """Pure-Python state for a file listbox.

    Holds a list of files; enforces: supported-extension filter + dedup.
    Calls on_change (if not None) on real mutations only (add, clear),
    NOT on no-ops (duplicate add, unsupported add, clear-when-empty).
    """

    def __init__(self, valid_extensions: set, on_change: Optional[Callable[[], None]] = None):
        self._valid = {e.lower() for e in valid_extensions}
        self._files: list = []
        self._on_change = on_change

    @property
    def files(self) -> list:
        return list(self._files)

    def is_supported(self, filepath: str) -> bool:
        return Path(filepath).suffix.lower() in self._valid

    def add(self, filepath: str) -> bool:
        if filepath in self._files or not self.is_supported(filepath):
            return False
        self._files.append(filepath)
        self._notify()
        return True

    def add_many(self, paths: Sequence[str]) -> int:
        added = 0
        for p in paths:
            if self.add(p):
                added += 1
        return added

    def add_dropped(self, raw_data: str) -> int:
        # ponytail: parse_dropped_paths already lives in ui_helpers — reuse it.
        paths = (p.strip() for p in parse_dropped_paths(raw_data))
        return self.add_many([p for p in paths if p])

    def clear(self) -> None:
        if not self._files:
            return
        self._files.clear()
        self._notify()

    def _notify(self) -> None:
        if self._on_change is not None:
            self._on_change()


class FileListbox(ttk.Frame):
    """ttk.Frame holding: button row (Browse/Clear) + listbox (with DnD) + scrollbar.

    Caller wraps in a LabelFrame if a titled section is wanted.
    Exposes `.files` (read-only list), `.listbox`, and `.button_row`
    (the ttk.Frame holding Browse/Clear — caller can pack extra widgets there).
    """

    def __init__(self, parent, valid_extensions: set,
                 filetypes_label: str = "Files",
                 filetypes_exts: Optional[Sequence[str]] = None,
                 browse_text: str = "Choose Files",
                 clear_text: str = "Clear",
                 on_change: Optional[Callable[[], None]] = None):
        super().__init__(parent)
        self.columnconfigure(0, weight=1)

        self.model = FileListboxModel(valid_extensions, on_change=self._on_model_changed)
        self._user_on_change = on_change

        self.button_row = ttk.Frame(self)
        self.button_row.grid(row=0, column=0, sticky=tk.W)
        ttk.Button(self.button_row, text=browse_text,
                   command=self._browse).pack(side=tk.LEFT, padx=2)
        ttk.Button(self.button_row, text=clear_text,
                   command=self.clear).pack(side=tk.LEFT, padx=2)

        list_frame = ttk.Frame(self)
        list_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(5, 0))
        list_frame.columnconfigure(0, weight=1)

        self._listbox = tk.Listbox(list_frame, height=4, selectmode=tk.EXTENDED)
        self._listbox.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self._listbox.drop_target_register(DND_FILES)
        self._listbox.dnd_bind('<<Drop>>', self._on_drop)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                  command=self._listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox.config(yscrollcommand=scrollbar.set)

        if filetypes_exts is None:
            filetypes_exts = sorted(valid_extensions)
        # ponytail: filedialog wants a single "*.ext1;*.ext2" glob string.
        self._filetypes = [
            (filetypes_label, ";".join(f"*{e}" for e in filetypes_exts)),
            ("All files", "*.*"),
        ]

    @property
    def files(self) -> list:
        return self.model.files

    @property
    def listbox(self) -> tk.Listbox:
        return self._listbox

    def add(self, filepath: str) -> bool:
        return self.model.add(filepath)

    def clear(self) -> None:
        self.model.clear()
        self._listbox.delete(0, tk.END)  # always wipe display (matches old behavior)

    def _browse(self) -> None:
        files = filedialog.askopenfilenames(filetypes=self._filetypes)
        self.model.add_many(files)

    def _on_drop(self, event) -> None:
        self.model.add_dropped(event.data)

    def _on_model_changed(self) -> None:
        # ponytail: model is single source of truth, rebuild listbox each change.
        self._listbox.delete(0, tk.END)
        for f in self.model.files:
            self._listbox.insert(tk.END, Path(f).name)
        if self._user_on_change is not None:
            self._user_on_change()
