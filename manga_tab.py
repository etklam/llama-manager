"""漫畫 OCR／翻譯 surface. Workers communicate only through an event queue.

Brought onto the app's shared conventions in this pass. What changed and why:

* **Logs reach the log drawer.** This surface kept every diagnostic in its own
  status line and reported failures through ``messagebox``, so a manga run left
  no trace in the application log at all. It now emits on ``CHANNEL_MANGA``
  like every other subsystem, and progress goes to ``run_bus`` so the header
  shows it from any surface.
* **State values are stable, not display text.** ``group`` used to hold the
  literal string ``'同一文件'`` and was compared against it in three places, so
  editing a label would silently change which documents got grouped. Values are
  now ``document`` / ``page`` with the label kept separate.
* **Numeric fields validate themselves.** Eight free-text entries were read
  straight into ``int()`` / ``float()`` inside the start handler; one typo threw
  a ``ValueError`` into a dialog that did not say which field was wrong.
  :class:`ui_kit.ValidatedSpin` marks the offending field and names it.
"""
from pathlib import Path
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
from manga.images import collect
from manga.runtime import Profile, inspect_runtime
from manga.runner import MangaRunner, RunSnapshot
from manga.store import documents, read_json, correct_page, reorder_document, write_json
from ui_helpers import CHANNEL_MANGA, LogMixin, run_bus
from ui_kit import (
    ActionBar, Disclosure, FieldGrid, Section, SegmentedControl, StatusLine,
    ValidatedSpin,
)
from ui_theme import theme
from ui_tokens import SPACE

# (value, label) pairs. The value is what the runner sees and what lands in
# manga-job.json; the label is only ever shown.
GROUP_OPTIONS = (("document", "同一文件"), ("page", "每張獨立"))
MODE_OPTIONS = (("ocr", "只做 OCR"), ("ocr_translate", "OCR 後翻譯"),
                ("retranslate", "重用 OCR 翻譯"))
DIRECTION_OPTIONS = (("rtl", "右→左"), ("ltr", "左→右"), ("vertical", "直書"))

# Free-text fields that hold paths or language codes.
TEXT_FIELDS = (
    ("executable", "llama-server", ""),
    ("ocr_model", "OCR GGUF", ""),
    ("mmproj", "對應 mmproj", ""),
    ("translation_model", "翻譯 GGUF", ""),
    ("output", "輸出目錄", "output"),
)

# (key, label, default, is_integer, minimum, maximum)
NUMERIC_FIELDS = (
    ("ocr_context", "OCR context", "8192", True, 512, 1_048_576),
    ("ocr_tokens", "OCR tokens", "2048", True, 16, 1_048_576),
    ("ocr_temperature", "OCR 溫度", "0", False, 0.0, 2.0),
    ("translation_context", "翻譯 context", "8192", True, 512, 1_048_576),
    ("translation_tokens", "翻譯 tokens", "2048", True, 16, 1_048_576),
    ("translation_temperature", "翻譯溫度", "0.2", False, 0.0, 2.0),
)

MAX_PREVIEW_PIXELS = 24_000_000


class MangaTab(LogMixin, ttk.Frame):
    log_channel = CHANNEL_MANGA

    def __init__(self, parent, server, config):
        super().__init__(parent, style="Canvas.TFrame")
        self.server, self.config = server, config
        self.files, self.events = [], queue.Queue()
        self.runner, self.run_id, self.active = None, 0, False
        self.preview_image = None
        self.page_location = None
        self.vars = {}
        self._numeric = {}

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._create_heading()
        self._create_body()

        self._poll_id = self.after(80, self.poll)
        self.bind('<Destroy>', self.destroyed, add='+')

    # ------------------------------------------------------------------ UI
    def _create_heading(self):
        heading = ttk.Frame(self, style="Canvas.TFrame")
        heading.grid(row=0, column=0, sticky="ew", padx=SPACE["page"],
                     pady=(SPACE["section"], SPACE["loose"]))
        ttk.Label(heading, text="漫畫 OCR／翻譯",
                  style="Display.TLabel").pack(anchor="w")
        ttk.Label(heading,
                  text="用本地 Vulkan llama-server 辨識頁面文字再翻譯；"
                       "GLM-OCR 為候選模型，成品仍需人工驗收。",
                  style="CanvasMuted.TLabel").pack(anchor="w",
                                                   pady=(SPACE["hair"], 0))

    def _create_body(self):
        split = ttk.Panedwindow(self, orient="horizontal")
        split.grid(row=1, column=0, sticky="nsew",
                   padx=SPACE["page"], pady=(0, SPACE["base"]))

        settings_host = ttk.Frame(split, style="Canvas.TFrame")
        review_host = ttk.Frame(split, style="Canvas.TFrame")
        split.add(settings_host, weight=2)
        split.add(review_host, weight=3)

        self._create_settings(settings_host)
        self._create_review(review_host)
        self._create_controls()

    def _create_settings(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        runtime = Section(parent, "執行環境與模型")
        runtime.grid(row=0, column=0, sticky="ew", pady=(0, SPACE["loose"]))

        grid = FieldGrid(runtime.body, label_width=12)
        grid.grid(row=0, column=0, sticky="ew")

        device_cell = grid.add("Vulkan 裝置")
        self.vars["device"] = tk.StringVar(
            value=self.config.get("manga.device", ""))
        self.device_combo = ttk.Combobox(device_cell,
                                         textvariable=self.vars["device"],
                                         state="readonly")
        self.device_combo.pack(side="left", fill="x", expand=True)
        ttk.Button(device_cell, text="枚舉／診斷", style="Ghost.TButton",
                   command=self.discover).pack(side="left",
                                               padx=(SPACE["snug"], 0))

        for key, label, default in TEXT_FIELDS:
            cell = grid.add(label)
            self.vars[key] = tk.StringVar(
                value=self.config.get("manga." + key, default))
            ttk.Entry(cell, textvariable=self.vars[key],
                      style="Mono.TEntry").pack(side="left", fill="x",
                                                expand=True)
            ttk.Button(cell, text="瀏覽…", style="Ghost.TButton",
                       command=lambda k=key: self.browse(k)).pack(
                side="left", padx=(SPACE["snug"], 0))

        params = Disclosure(parent, "OCR 與翻譯參數")
        params.grid(row=1, column=0, sticky="ew", pady=(0, SPACE["loose"]))
        param_grid = FieldGrid(params.body, columns=2, label_width=14)
        param_grid.grid(row=0, column=0, sticky="ew")

        for index, (key, label, default, integer, low, high) in enumerate(
                NUMERIC_FIELDS):
            self.vars[key] = tk.StringVar(
                value=str(self.config.get("manga." + key, default)))
            field = ValidatedSpin(
                param_grid.add(label, column=index % 2), self.vars[key],
                label, minimum=low, maximum=high, integer=integer)
            field.pack(anchor="w")
            self._numeric[key] = field

        for key, label, default in (("source", "來源語言", "ja"),
                                    ("locale", "目標語言", "zh-tw")):
            self.vars[key] = tk.StringVar(
                value=self.config.get("manga." + key, default))
            ttk.Entry(param_grid.add(label), textvariable=self.vars[key],
                      width=10).pack(anchor="w")

        modes = Section(parent, "處理方式")
        modes.grid(row=2, column=0, sticky="new")
        mode_grid = FieldGrid(modes.body, label_width=12)
        mode_grid.grid(row=0, column=0, sticky="ew")

        self.mode = tk.StringVar(value="ocr_translate")
        SegmentedControl(mode_grid.add("模式"), self.mode,
                         MODE_OPTIONS).pack(anchor="w")

        self.group = tk.StringVar(value="document")
        SegmentedControl(mode_grid.add("分組", hint="「同一文件」會把整批頁面視為一本書。"),
                         self.group, GROUP_OPTIONS).pack(anchor="w")

        self.direction = tk.StringVar(value="rtl")
        SegmentedControl(mode_grid.add("閱讀方向"), self.direction,
                         DIRECTION_OPTIONS).pack(anchor="w")

        self.review = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes.body, text="先校對再翻譯",
                        variable=self.review).grid(row=1, column=0, sticky="w",
                                                   pady=(SPACE["snug"], 0))
        self.two_step = tk.BooleanVar(value=False)
        ttk.Checkbutton(modes.body, text="兩步翻譯（直譯後意譯）",
                        variable=self.two_step).grid(row=2, column=0,
                                                     sticky="w")

    def _create_review(self, parent):
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        section = Section(parent, "頁面與校正",
                          "空行分段；可剪貼調整段落次序，重複對白會保留。")
        section.grid(row=0, column=0, sticky="nsew")
        section.body.columnconfigure(1, weight=1)
        section.body.rowconfigure(1, weight=1)

        pages_host = ttk.Frame(section.body, style="Surface.TFrame")
        pages_host.grid(row=0, column=0, rowspan=2, sticky="nsew",
                        padx=(0, SPACE["loose"]))
        pages_host.columnconfigure(0, weight=1)
        pages_host.rowconfigure(1, weight=1)

        buttons = ttk.Frame(pages_host, style="Surface.TFrame")
        buttons.grid(row=0, column=0, sticky="ew", pady=(0, SPACE["tight"]))
        for label, action in (("加入圖片", self.add_files),
                              ("資料夾", self.add_folder),
                              ("↑", lambda: self.move(-1)),
                              ("↓", lambda: self.move(1)),
                              ("移除", self.remove)):
            ttk.Button(buttons, text=label, style="Ghost.TButton",
                       command=action).pack(side="left",
                                            padx=(0, SPACE["hair"]))

        list_host = ttk.Frame(pages_host, style="Inset.TFrame")
        list_host.grid(row=1, column=0, sticky="nsew")
        list_host.columnconfigure(0, weight=1)
        list_host.rowconfigure(0, weight=1)
        self.pages = tk.Listbox(list_host, exportselection=False, width=26)
        theme().register_classic(self.pages, "listbox")
        self.pages.grid(row=0, column=0, sticky="nsew")
        pages_scroll = ttk.Scrollbar(list_host, orient="vertical",
                                     command=self.pages.yview)
        pages_scroll.grid(row=0, column=1, sticky="ns")
        self.pages.configure(yscrollcommand=pages_scroll.set)
        self.pages.bind('<<ListboxSelect>>', self.select_page)

        preview_host = ttk.Frame(section.body, style="Inset.TFrame")
        preview_host.grid(row=0, column=1, sticky="nsew",
                          pady=(0, SPACE["base"]))
        self.preview = ttk.Label(preview_host,
                                 text="選一頁預覽（預覽縮放不影響 OCR 輸入）",
                                 style="Faint.TLabel", anchor="center")
        self.preview.pack(fill="both", expand=True, padx=SPACE["base"],
                          pady=SPACE["loose"])

        editor_host = ttk.Frame(section.body, style="Surface.TFrame")
        editor_host.grid(row=1, column=1, sticky="nsew")
        editor_host.columnconfigure(0, weight=1)
        editor_host.rowconfigure(0, weight=1)
        self.editor = tk.Text(editor_host, height=8, width=45, wrap="word")
        theme().register_classic(self.editor, "text")
        self.editor.grid(row=0, column=0, sticky="nsew")
        editor_scroll = ttk.Scrollbar(editor_host, orient="vertical",
                                      command=self.editor.yview)
        editor_scroll.grid(row=0, column=1, sticky="ns")
        self.editor.configure(yscrollcommand=editor_scroll.set)

        edit_bar = ActionBar(editor_host)
        edit_bar.grid(row=1, column=0, columnspan=2, sticky="ew",
                      pady=(SPACE["base"], 0))
        edit_bar.ghost("儲存校正／順序", self.save_correction)
        edit_bar.ghost("確認無文字", lambda: self.save_correction(True))
        edit_bar.ghost("查看 Raw OCR", self.show_raw)

    def _create_controls(self):
        controls = ttk.Frame(self, style="Canvas.TFrame")
        controls.grid(row=2, column=0, sticky="ew", padx=SPACE["page"],
                      pady=(0, SPACE["page"]))
        controls.columnconfigure(0, weight=1)

        self.status = tk.StringVar(
            value="尚未執行；Vulkan 未驗證；GLM-OCR 為候選模型，漫畫品質需人工驗收")
        self._status_line = StatusLine(controls, state="idle", on_canvas=True)
        self._status_line.grid(row=0, column=0, sticky="ew")
        # The StringVar stays the surface's status API (load_job, poll and the
        # worker handlers all write it); this trace is what paints it.
        self.status.trace_add("write", lambda *_: self._render_status())

        self.progress = ttk.Progressbar(
            controls, mode="determinate",
            style="Accent.Horizontal.TProgressbar")
        self.progress.grid(row=1, column=0, sticky="ew",
                           pady=(SPACE["snug"], SPACE["base"]))

        bar = ActionBar(controls, on_canvas=True)
        bar.grid(row=2, column=0, sticky="ew")
        bar.ghost("載入工作記錄", self.load_job)
        bar.ghost("開啟輸出", self.open_output)
        self._start_button = bar.primary("執行", self.start)
        self._resume_button = bar.secondary("續跑／重試失敗", self.start)
        self._cancel_button = bar.danger("取消", self.cancel)
        self._cancel_button.state(("disabled",))
        self._render_status()

    # -------------------------------------------------------------- status
    def _render_status(self, state=None):
        """Paint the status line, and mirror it to the log and the header."""
        message = self.status.get()
        if state is None:
            state = "busy" if self.active else "idle"
        self._status_line.set(state, message)

    def _set_status(self, message, state=None, level="INFO"):
        """Set status, log it, and publish run state in one call."""
        self.status.set(message)
        self._render_status(state)
        self._log(level, message)
        if state in ("ok", "warn", "error"):
            run_bus.end(CHANNEL_MANGA, "漫畫 OCR／翻譯", state, message)
        elif self.active:
            run_bus.progress(CHANNEL_MANGA, "漫畫 OCR／翻譯", None, message)

    def _set_running(self, running):
        self.active = running
        self._start_button.state(("disabled",) if running else ("!disabled",))
        self._resume_button.state(("disabled",) if running else ("!disabled",))
        self._cancel_button.state(("!disabled",) if running else ("disabled",))

    def destroyed(self, event):
        if event.widget is self:
            if self.runner:
                self.runner.cancel()
            if self._poll_id:
                self.after_cancel(self._poll_id)
                self._poll_id = None

    def browse(self, key):
        value = (filedialog.askdirectory() if key == 'output'
                 else filedialog.askopenfilename())
        if value:
            self.vars[key].set(value)

    def discover(self):
        executable = self.vars['executable'].get()
        self._set_status('讀取 runtime version／help／devices…')
        run_id = self.run_id

        def worker():
            try:
                self.events.put({'run_id': run_id, 'stage': 'devices',
                                 'info': inspect_runtime(executable)})
            except Exception as error:
                self.events.put({'run_id': run_id, 'stage': 'error',
                                 'error': str(error)})

        threading.Thread(target=worker, daemon=True).start()

    def add_files(self):
        if not self.active:
            self.files.extend(collect(
                filedialog.askopenfilenames(
                    filetypes=[('Images', '*.png *.jpg *.jpeg *.webp')]),
                self.vars['output'].get()))
            self.refresh_pages()

    def add_folder(self):
        if not self.active:
            folder = filedialog.askdirectory()
            if folder:
                self.files.extend(collect([folder], self.vars['output'].get()))
                self.refresh_pages()

    def refresh_pages(self):
        self.pages.delete(0, 'end')
        for index, path in enumerate(self.files):
            self.pages.insert('end', f'{index + 1}. {Path(path).name}')

    @property
    def _independent(self):
        """True when each image is its own document."""
        return self.group.get() == 'page'

    def move(self, delta):
        selection = self.pages.curselection()
        if self.active or not selection:
            return
        index = selection[0]
        other = index + delta
        if 0 <= other < len(self.files):
            self.files[index], self.files[other] = (self.files[other],
                                                    self.files[index])
            self.refresh_pages()
            self.pages.selection_set(other)
            try:
                output = Path(self.vars['output'].get()).resolve()
                for doc, pages in documents(self.files, self._independent):
                    if (output / doc / 'manifest.json').exists():
                        reorder_document(output / doc, [p['id'] for p in pages])
                job_path = output / 'manga-job.json'
                if job_path.exists():
                    job = read_json(job_path)
                    if sorted(job['files']) == sorted(self.files):
                        job['files'] = list(self.files)
                        write_json(job_path, job)
            except Exception as error:
                self._set_status('頁序 checkpoint 更新失敗：' + str(error),
                                 state="warn", level="WARNING")
            self.select_page()

    def remove(self):
        if not self.active:
            for index in reversed(self.pages.curselection()):
                del self.files[index]
            self.refresh_pages()

    def select_page(self, event=None):
        selection = self.pages.curselection()
        if not selection:
            return
        index = selection[0]
        path = self.files[index]
        try:
            with Image.open(path) as original:
                if original.width * original.height > MAX_PREVIEW_PIXELS:
                    raise ValueError('圖片過大，請先拆頁')
                image = ImageOps.exif_transpose(original)
                image.thumbnail((500, 280))
                self.preview_image = ImageTk.PhotoImage(image.convert('RGB'))
            self.preview.configure(image=self.preview_image, text='')
        except Exception:
            self.preview.configure(image='', text='無法預覽圖片／需拆頁')
        groups = documents(self.files, self._independent)
        flattened = [(doc, page) for doc, pages in groups for page in pages]
        doc, page = flattened[index]
        self.page_location = (
            Path(self.vars['output'].get()).resolve() / doc, page['id'])
        checkpoint = self.page_location[0] / 'pages' / (page['id'] + '.json')
        self.editor.delete('1.0', 'end')
        if checkpoint.exists():
            try:
                value = read_json(checkpoint)
                self.editor.insert(
                    '1.0', '\n\n'.join(s['text'] for s in value['segments']))
                self.status.set(
                    f"頁面 {value['status']}；revision {value['revision']}；"
                    + '; '.join(value['warnings']))
                self._render_status()
            except Exception as error:
                self._set_status(str(error), state="warn", level="WARNING")

    def show_raw(self):
        if not self.page_location:
            return
        root, page_id = self.page_location
        try:
            value = read_json(root / 'pages' / (page_id + '.json'))
            window = tk.Toplevel(self)
            window.title('Raw OCR（唯讀）')
            window.geometry('620x480')
            theme().register_classic(window, "toplevel")
            text = tk.Text(window, wrap='word')
            theme().register_classic(text, "text")
            text.pack(fill='both', expand=True, padx=SPACE["roomy"],
                      pady=SPACE["roomy"])
            text.insert('1.0', value['raw'])
            text.configure(state='disabled')
        except Exception as error:
            messagebox.showerror('Raw OCR', str(error))

    def save_correction(self, no_text=False):
        if self.active or not self.page_location:
            return
        try:
            text = '' if no_text else self.editor.get('1.0', 'end-1c')
            correct_page(*self.page_location, re.split(r'\n\s*\n', text),
                         no_text)
            self._set_status(
                '校正已儲存；raw OCR 保留；該文件譯文已失效，請選「重用 OCR 翻譯」。',
                state="ok", level="SUCCESS")
        except Exception as error:
            self._set_status('校正失敗：' + str(error), state="error",
                             level="ERROR")
            messagebox.showerror('校正失敗', str(error))

    # --------------------------------------------------------------- run
    def _profiles(self):
        """Validated OCR and translation profiles, or ``None`` with a report.

        Returns ``(ocr, translation, errors)``. Collecting every bad field
        before reporting beats failing on the first one: a mistyped context
        size and a mistyped temperature are one correction trip, not two.
        """
        errors = [field.error() for key, field in self._numeric.items()
                  if field.value() is None]
        errors = [message for message in errors if message]
        if errors:
            return None, None, errors
        values = {key: field.value() for key, field in self._numeric.items()}
        ocr = Profile(self.vars['ocr_model'].get(), self.vars['mmproj'].get(),
                      values['ocr_context'], values['ocr_tokens'],
                      values['ocr_temperature'])
        translation = Profile(self.vars['translation_model'].get(), '',
                              values['translation_context'],
                              values['translation_tokens'],
                              values['translation_temperature'])
        return ocr, translation, []

    def start(self):
        if self.active or getattr(self, '_closing', False):
            return
        ocr, translation, errors = self._profiles()
        if errors:
            self._set_status('；'.join(errors), state="error", level="ERROR")
            messagebox.showerror('參數有誤', '\n'.join(errors))
            return
        try:
            values = {key: variable.get() for key, variable in self.vars.items()}
            self.run_id += 1
            snapshot = RunSnapshot(
                tuple(self.files), str(Path(values['output']).resolve()),
                values['executable'], values['device'], ocr, translation,
                mode=self.mode.get(), source=values['source'],
                locale=values['locale'], direction=self.direction.get(),
                independent=self._independent, review=self.review.get(),
                two_step=self.two_step.get(), run_id=self.run_id)
            for key, value in values.items():
                self.config.set('manga.' + key, value)
            self.runner = MangaRunner(self.server, snapshot, self.events.put)
            self._set_running(True)
            run_bus.begin(CHANNEL_MANGA, "漫畫 OCR／翻譯", "驗證輸入與 checkpoint…")
            self._set_status('驗證輸入與 checkpoint…')
            threading.Thread(target=self.runner.run, daemon=True).start()
        except Exception as error:
            self._set_running(False)
            self._set_status('無法開始：' + str(error), state="error",
                             level="ERROR")
            messagebox.showerror('無法開始', str(error))

    def cancel(self):
        if self.active and self.runner:
            self.runner.cancel()
            self._set_status(
                '停止中：不再派發新頁，等待目前 HTTP 返回（最多 120 秒），再還原 server',
                level="WARNING")

    def load_job(self):
        if self.active:
            return
        filename = filedialog.askopenfilename(
            filetypes=[('Manga job', 'manga-job.json')])
        if not filename:
            return
        try:
            job = read_json(filename)
            if (not isinstance(job.get('files'), list)
                    or not all(isinstance(p, str) for p in job['files'])):
                raise ValueError('Invalid job image list')
            for key in ('executable', 'device', 'output', 'source', 'locale'):
                self.vars[key].set(job[key])
            for prefix, name in (('ocr', 'ocr'), ('translation', 'translation')):
                profile = job[name]
                self.vars[prefix + '_model'].set(profile['model'])
                self.vars[prefix + '_context'].set(profile['context'])
                self.vars[prefix + '_tokens'].set(profile['max_tokens'])
                self.vars[prefix + '_temperature'].set(profile['temperature'])
            self.vars['mmproj'].set(job['ocr']['mmproj'])
            self.files = job['files']
            self.mode.set(job['mode'])
            self.group.set('page' if job['independent'] else 'document')
            self.direction.set(job['direction'])
            self.review.set(job['review'])
            self.two_step.set(job['two_step'])
            self.refresh_pages()
            self._set_status(
                '工作記錄已載入；按「續跑」會重新驗證圖片與 checkpoints',
                state="ok")
        except Exception as error:
            self._set_status('無法載入工作記錄：' + str(error), state="error",
                             level="ERROR")
            messagebox.showerror('無法載入', str(error))

    def open_output(self):
        path = Path(self.vars['output'].get()).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    # -------------------------------------------------------------- events
    def poll(self):
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event['run_id'] != self.run_id:
                continue
            self._handle_event(event)
        self._poll_id = self.after(80, self.poll)

    def _handle_event(self, event):
        stage = event['stage']
        if stage == 'devices':
            info = event['info']
            self.device_combo.configure(values=info['devices'])
            if info['devices'] and self.vars['device'].get() not in info['devices']:
                self.vars['device'].set(info['devices'][0])
            self._set_status(info['version'].strip() + ' | '
                             + info['device_list'].strip() + ' | 未驗證')
        elif stage == 'done':
            self._set_running(False)
            report = event['report']
            verified = report['stages'] and all(
                x['status'] == 'verified' for x in report['stages'])
            counts = report.get('counts', {})
            self.progress.configure(value=self.progress.cget('maximum'))
            failures = report['failures']
            message = (
                f"{report['status']}；OCR 成功 {counts.get('success', 0)}／"
                f"失敗 {counts.get('failed', 0)}／"
                f"待確認 {counts.get('needs_review', 0)}；錯誤 {failures}；Vulkan "
                + ('已驗證（本輪有證據的階段）' if verified else '未驗證')
                + ('；' + report['error'] if 'error' in report else '')
                + ('；還原失敗：' + report['restore_error']
                   if report['restore_error'] else ''))
            outcome = "error" if (failures or report.get('error')) else "ok"
            self.progress.configure(
                style="Danger.Horizontal.TProgressbar" if outcome == "error"
                else "Ok.Horizontal.TProgressbar")
            self._set_status(message, state=outcome,
                             level="ERROR" if outcome == "error" else "SUCCESS")
        elif stage in ('ocr', 'translation'):
            self.progress.configure(maximum=event['total'],
                                    value=event['current'] - 1)
            label = '辨識' if stage == 'ocr' else '翻譯'
            message = f"{label} {event['current']}/{event['total']}"
            self.status.set(message)
            self._render_status()
            percent = (event['current'] - 1) / max(1, event['total']) * 100
            run_bus.progress(CHANNEL_MANGA, f"漫畫{label}", percent,
                             f"{event['current']}/{event['total']}")
        elif stage == 'diagnostic':
            self._set_status('Vulkan ' + event['diagnostic']['status'])
        else:
            self._set_status(
                stage + ': ' + str(event.get('error', event.get('model', ''))),
                state="warn", level="WARNING")
