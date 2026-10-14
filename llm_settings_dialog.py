"""LLM 連線設定 — editor for remote OpenAI-compatible endpoint profiles.

Persisted fields go to config.json via llm_target.upsert_profile (stable ids,
no credentials). The session API key field is the one thing that does not
persist: it feeds llm_target.SESSION_KEYS for the current process only, so a
key can be used without ever reaching disk.
"""
import tkinter as tk
from tkinter import ttk, messagebox

from ui_kit import ActionBar, FieldGrid, Section
from ui_theme import theme
from ui_tokens import SPACE
from llm_target import (
    SESSION_KEYS, list_profiles, new_profile_id, upsert_profile, delete_profile,
)


class LLMSettingsDialog(tk.Toplevel):
    """Modal-ish editor: profile list on the left, fields on the right."""

    def __init__(self, parent, config_manager, on_change=None):
        super().__init__(parent)
        self.title("LLM 連線設定")
        self.transient(parent.winfo_toplevel())
        self.grab_set()
        theme().register_classic(self, "toplevel")

        self._config = config_manager
        self._on_change = on_change or (lambda: None)
        self._profiles = {p.get('id', ''): p for p in list_profiles(config_manager)}
        self._selected_id = ''

        self._create_ui()
        self._reload_list(select_first=True)
        self.protocol("WM_DELETE_WINDOW", self._close)

    # ------------------------------------------------------------------ UI
    def _create_ui(self):
        """Profile list on the left, fields on the right."""
        self.configure(padx=SPACE["roomy"], pady=SPACE["roomy"])
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        left = Section(self, "設定檔", flush=True, padding=0)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, SPACE["loose"]))
        left.body.rowconfigure(0, weight=1)

        list_host = ttk.Frame(left.body, style="Inset.TFrame")
        list_host.grid(row=0, column=0, sticky="nsew")
        self._profile_list = tk.Listbox(list_host, width=20, height=9,
                                        selectmode=tk.SINGLE,
                                        exportselection=False)
        theme().register_classic(self._profile_list, "listbox")
        self._profile_list.pack(fill="both", expand=True)
        self._profile_list.bind('<<ListboxSelect>>', self._on_select)

        btns = ttk.Frame(left.body, style="Surface.TFrame")
        btns.grid(row=1, column=0, sticky="ew", pady=(SPACE["base"], 0))
        ttk.Button(btns, text="新增", style="Ghost.TButton",
                   command=self._new_profile).pack(side=tk.LEFT,
                                                   padx=(0, SPACE["tight"]))
        ttk.Button(btns, text="刪除", style="Ghost.TButton",
                   command=self._delete_profile).pack(side=tk.LEFT)

        form = Section(self, "設定檔內容")
        form.grid(row=0, column=1, sticky="nsew")

        self._vars = {
            'name': tk.StringVar(),
            'base_url': tk.StringVar(),
            'model': tk.StringVar(),
            'api_key_env': tk.StringVar(),
            'proxy': tk.StringVar(),
            'max_workers': tk.StringVar(value='3'),
        }
        rows = (
            ('name', '名稱', ''),
            ('base_url', 'Base URL', 'OpenAI 相容端點，貼上 /chat/completions 也可以。'),
            ('model', '模型', '例如 provider/model-name。'),
            ('api_key_env', 'API Key 環境變數', '只存變數名稱，金鑰本身不會寫入設定檔。'),
            ('proxy', 'Proxy', '可留空。'),
            ('max_workers', '並發數', ''),
        )
        grid = FieldGrid(form.body, label_width=16)
        grid.grid(row=0, column=0, sticky="ew")
        for key, label, hint in rows:
            mono = key in ('base_url', 'model', 'api_key_env', 'proxy')
            ttk.Entry(grid.add(label, hint=hint),
                      textvariable=self._vars[key],
                      style="Mono.TEntry" if mono else "TEntry").pack(fill="x")

        # Session key: typed here, kept in memory only.
        key_cell = grid.add("Session API Key")
        self._session_key_var = tk.StringVar()
        self._session_key_entry = ttk.Entry(
            key_cell, textvariable=self._session_key_var, show='*')
        self._session_key_entry.pack(side=tk.LEFT, fill="x", expand=True)
        self._session_key_status = ttk.Label(key_cell, text="",
                                             style="Muted.TLabel")
        self._session_key_status.pack(side=tk.LEFT, padx=(SPACE["snug"], 0))

        ttk.Label(form.body, style="Micro.TLabel", wraplength=420,
                  justify="left",
                  text="此欄優先於環境變數；不會存檔，關閉程式即失效。"
                       "留空並按儲存可清除本次工作階段的金鑰。").grid(
            row=1, column=0, sticky="w", pady=(SPACE["snug"], 0))

        actions = ActionBar(self)
        actions.grid(row=1, column=0, columnspan=2, sticky="ew",
                     pady=(SPACE["loose"], 0))
        actions.secondary("關閉", self._close)
        actions.primary("儲存", self._save)

    # -------------------------------------------------------------- list
    def _reload_list(self, select_first=False):
        self._profile_list.delete(0, tk.END)
        ids = list(self._profiles)
        for pid in ids:
            self._profile_list.insert(tk.END, self._profiles[pid].get('name', pid))
        if select_first and ids:
            self._profile_list.selection_set(0)
            self._load_profile(ids[0])
        elif self._selected_id in ids:
            index = ids.index(self._selected_id)
            self._profile_list.selection_set(index)
        else:
            self._clear_form()

    def _on_select(self, event=None):
        selection = self._profile_list.curselection()
        if not selection:
            return
        pid = list(self._profiles)[selection[0]]
        self._load_profile(pid)

    def _load_profile(self, pid):
        # An unsaved edit is discarded on selection change; the form is small
        # enough that retyping beats silently keeping stale fields.
        self._selected_id = pid
        profile = self._profiles.get(pid, {})
        for key, var in self._vars.items():
            var.set(str(profile.get(key, '') or ''))
        self._session_key_var.set('')
        self._session_key_status.config(
            text="已設定 (僅本次執行)" if SESSION_KEYS.is_set(pid) else "")

    def _clear_form(self):
        self._selected_id = ''
        for var in self._vars.values():
            var.set('')
        self._vars['max_workers'].set('3')
        self._session_key_var.set('')
        self._session_key_status.config(text='')

    def _new_profile(self):
        pid = new_profile_id()
        self._profiles[pid] = {'id': pid, 'name': f"Profile {len(self._profiles) + 1}"}
        self._selected_id = pid
        self._reload_list()
        index = list(self._profiles).index(pid)
        self._profile_list.selection_set(index)
        self._load_profile(pid)

    def _delete_profile(self):
        if not self._selected_id:
            return
        name = self._profiles[self._selected_id].get('name', self._selected_id)
        if not messagebox.askyesno("刪除設定檔", f"刪除「{name}」？", parent=self):
            return
        delete_profile(self._config, self._selected_id)
        del self._profiles[self._selected_id]
        self._selected_id = ''
        self._reload_list(select_first=True)
        self._on_change()

    # -------------------------------------------------------------- save
    def _save(self):
        name = self._vars['name'].get().strip()
        base_url = self._vars['base_url'].get().strip()
        model = self._vars['model'].get().strip()
        if not (name and base_url and model):
            messagebox.showwarning(
                "缺少欄位", "名稱、Base URL 與模型皆為必填。", parent=self)
            return

        profile = {'id': self._selected_id or new_profile_id()}
        for key, var in self._vars.items():
            profile[key] = var.get()
        pid = upsert_profile(self._config, profile)

        session_key = self._session_key_var.get()
        SESSION_KEYS.set(pid, session_key)
        # An explicit clear (empty field on a profile that had a session key)
        # falls back to the environment variable — which is the point of the
        # field being empty rather than absent.
        self._session_key_status.config(
            text="已設定 (僅本次執行)" if SESSION_KEYS.is_set(pid) else "")

        self._selected_id = pid
        self._profiles[pid] = next(
            (p for p in list_profiles(self._config) if p.get('id') == pid),
            profile)
        self._reload_list()
        self._on_change()

    def _close(self):
        self._on_change()
        self.grab_release()
        self.destroy()
