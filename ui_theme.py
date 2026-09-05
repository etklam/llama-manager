"""Shared desktop styling; native widgets retain keyboard and focus behavior."""
from tkinter import ttk


def apply_theme(root):
    root.configure(background="#f4f6fa")
    root.option_add("*Font", "{Microsoft JhengHei UI} 10")
    root.option_add("*Text.background", "#ffffff")
    root.option_add("*Text.foreground", "#243247")
    root.option_add("*Text.selectBackground", "#dbeafe")
    root.option_add("*Listbox.background", "#ffffff")
    root.option_add("*Listbox.selectBackground", "#2563eb")
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure(".", background="#f4f6fa", foreground="#243247")
    style.configure("TNotebook", borderwidth=0, tabmargins=(12, 8, 12, 0))
    style.configure("TNotebook.Tab", padding=(18, 10))
    style.map("TNotebook.Tab", background=[("selected", "#ffffff")],
              foreground=[("selected", "#1d4ed8")])
    style.configure("TLabelframe", bordercolor="#dbe2ec", relief="solid")
    style.configure("TLabelframe.Label", font=("Microsoft JhengHei UI", 10, "bold"))
    style.configure("TButton", padding=(10, 6))
    style.configure("TEntry", padding=5)
    style.configure("TCombobox", padding=5)
    style.configure("Accent.TButton", background="#2563eb", foreground="white")
    style.map("Accent.TButton", background=[("disabled", "#dbe2ec"),
              ("pressed", "#1e40af"), ("active", "#1d4ed8")],
              foreground=[("disabled", "#64748b")])
    style.configure("Title.TLabel", font=("Microsoft JhengHei UI", 20, "bold"))
    style.configure("Muted.TLabel", foreground="#64748b")
    return style
