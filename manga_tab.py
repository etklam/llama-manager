"""Native Tk shell. Workers communicate only through an event queue."""
from dataclasses import asdict
from pathlib import Path
import json
import os
import queue
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageOps, ImageTk
from manga.images import collect, natural_key
from manga.runtime import Profile, inspect_runtime
from manga.runner import MangaRunner, RunSnapshot
from manga.store import documents, read_json, correct_page, reorder_document, write_json


class MangaTab(ttk.Frame):
    def __init__(self, parent, server, config):
        super().__init__(parent, padding=8)
        self.server, self.config = server, config
        self.files, self.events = [], queue.Queue()
        self.runner, self.run_id, self.active = None, 0, False
        self.preview_image = None
        self.page_location = None
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)
        settings = ttk.LabelFrame(self, text='本地 Vulkan profiles（漫畫專用）', padding=5)
        settings.grid(row=0, column=0, sticky='ew')
        settings.columnconfigure(1, weight=1)
        self.vars = {}
        fields = [('executable', 'llama-server', ''), ('device', 'Vulkan device', ''),
                  ('ocr_model', 'OCR GGUF (GLM-OCR Q8_0)', ''), ('mmproj', 'Matching mmproj', ''),
                  ('translation_model', 'Translation GGUF', ''), ('output', '輸出目錄', 'output')]
        for row, (key, label, default) in enumerate(fields):
            self.vars[key] = tk.StringVar(value=config.get('manga.' + key, default))
            ttk.Label(settings, text=label).grid(row=row, column=0, sticky='w')
            if key == 'device':
                self.device_combo = ttk.Combobox(settings, textvariable=self.vars[key], state='readonly')
                self.device_combo.grid(row=row, column=1, sticky='ew')
                ttk.Button(settings, text='枚舉／診斷', command=self.discover).grid(row=row, column=2)
            else:
                ttk.Entry(settings, textvariable=self.vars[key]).grid(row=row, column=1, sticky='ew')
                ttk.Button(settings, text='瀏覽', command=lambda key=key: self.browse(key)).grid(row=row, column=2)
        options = ttk.Frame(self)
        options.grid(row=1, column=0, sticky='ew')
        settings_fields = [('ocr_context', 'OCR context', '8192'), ('ocr_tokens', 'OCR tokens', '2048'),
                           ('translation_context', '翻譯 context', '8192'), ('translation_tokens', '翻譯 tokens', '2048'),
                           ('ocr_temperature', 'OCR 溫度', '0'), ('translation_temperature', '翻譯溫度', '0.2'),
                           ('source', '來源語言', 'ja'), ('locale', '目標語言', 'zh-tw')]
        for index, (key, label, default) in enumerate(settings_fields):
            row, col = divmod(index, 4)
            self.vars[key] = tk.StringVar(value=config.get('manga.' + key, default))
            ttk.Label(options, text=label).grid(row=row, column=col * 2, padx=3)
            ttk.Entry(options, textvariable=self.vars[key], width=9).grid(row=row, column=col * 2 + 1)
        self.mode = tk.StringVar(value='ocr_translate')
        self.group = tk.StringVar(value='同一文件')
        self.direction = tk.StringVar(value='rtl')
        self.review = tk.BooleanVar(value=False)
        self.two_step = tk.BooleanVar(value=False)
        modes = ttk.Frame(self)
        modes.grid(row=2, column=0, sticky='ew', pady=5)
        for label, mode in [('只做 OCR', 'ocr'), ('OCR 後翻譯', 'ocr_translate'), ('重用 OCR 翻譯', 'retranslate')]:
            ttk.Radiobutton(modes, text=label, variable=self.mode, value=mode).pack(side='left')
        ttk.Combobox(modes, textvariable=self.group, values=['同一文件', '每張獨立'], state='readonly', width=10).pack(side='left')
        ttk.Combobox(modes, textvariable=self.direction, values=['rtl', 'ltr', 'vertical'], state='readonly', width=8).pack(side='left')
        ttk.Checkbutton(modes, text='先校對', variable=self.review).pack(side='left')
        ttk.Checkbutton(modes, text='兩步翻譯', variable=self.two_step).pack(side='left')
        body = ttk.Panedwindow(self, orient='horizontal')
        body.grid(row=3, column=0, sticky='nsew')
        left, right = ttk.Frame(body), ttk.Frame(body)
        body.add(left, weight=1)
        body.add(right, weight=2)
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        buttons = ttk.Frame(left)
        buttons.grid(row=0, column=0, sticky='ew')
        for label, action in [('加入圖片', self.add_files), ('資料夾', self.add_folder),
                              ('↑', lambda: self.move(-1)), ('↓', lambda: self.move(1)), ('移除', self.remove)]:
            ttk.Button(buttons, text=label, command=action).pack(side='left')
        self.pages = tk.Listbox(left, exportselection=False)
        self.pages.grid(row=1, column=0, sticky='nsew')
        self.pages.bind('<<ListboxSelect>>', self.select_page)
        self.preview = ttk.Label(right, text='選頁預覽（預覽縮放不影響 OCR 輸入）', anchor='center')
        self.preview.pack(fill='both', expand=True)
        ttk.Label(right, text='校正文字：空行分段；可剪貼調整段落次序，重複對白會保留').pack()
        self.editor = tk.Text(right, height=8, width=45, wrap='word')
        self.editor.pack(fill='both', expand=True)
        edit_buttons = ttk.Frame(right)
        edit_buttons.pack(fill='x')
        ttk.Button(edit_buttons, text='儲存校正／順序', command=self.save_correction).pack(side='left')
        ttk.Button(edit_buttons, text='確認無文字', command=lambda: self.save_correction(True)).pack(side='left')
        ttk.Button(edit_buttons, text='查看 Raw OCR', command=self.show_raw).pack(side='left')
        controls = ttk.Frame(self)
        controls.grid(row=4, column=0, sticky='ew', pady=5)
        for label, action in [('執行', self.start), ('續跑／重試失敗', self.start),
                              ('取消', self.cancel), ('載入工作記錄', self.load_job), ('開啟輸出', self.open_output)]:
            ttk.Button(controls, text=label, command=action).pack(side='left')
        self.status = tk.StringVar(value='尚未執行；Vulkan unverified；GLM-OCR 為候選模型，漫畫品質需人工驗收')
        ttk.Label(self, textvariable=self.status, wraplength=1000).grid(row=5, column=0, sticky='ew')
        self.progress = ttk.Progressbar(self, mode='determinate')
        self.progress.grid(row=6, column=0, sticky='ew')
        self._poll_id = self.after(80, self.poll)
        self.bind('<Destroy>', self.destroyed, add='+')

    def destroyed(self, event):
        if event.widget is self:
            if self.runner:
                self.runner.cancel()
            if self._poll_id:
                self.after_cancel(self._poll_id)
                self._poll_id = None

    def browse(self, key):
        value = filedialog.askdirectory() if key == 'output' else filedialog.askopenfilename()
        if value:
            self.vars[key].set(value)

    def discover(self):
        executable = self.vars['executable'].get()
        self.status.set('讀取 runtime version／help／devices…')
        run_id = self.run_id
        def worker():
            try:
                self.events.put({'run_id': run_id, 'stage': 'devices', 'info': inspect_runtime(executable)})
            except Exception as error:
                self.events.put({'run_id': run_id, 'stage': 'error', 'error': str(error)})
        threading.Thread(target=worker, daemon=True).start()

    def add_files(self):
        if not self.active:
            self.files.extend(collect(filedialog.askopenfilenames(filetypes=[('Images', '*.png *.jpg *.jpeg *.webp')]), self.vars['output'].get()))
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

    def move(self, delta):
        selection = self.pages.curselection()
        if self.active or not selection:
            return
        index = selection[0]
        other = index + delta
        if 0 <= other < len(self.files):
            self.files[index], self.files[other] = self.files[other], self.files[index]
            self.refresh_pages()
            self.pages.selection_set(other)
            try:
                output = Path(self.vars['output'].get()).resolve()
                for doc, pages in documents(self.files, self.group.get() == '每張獨立'):
                    if (output / doc / 'manifest.json').exists():
                        reorder_document(output / doc, [p['id'] for p in pages])
                job_path = output / 'manga-job.json'
                if job_path.exists():
                    job = read_json(job_path)
                    if sorted(job['files']) == sorted(self.files):
                        job['files'] = list(self.files)
                        write_json(job_path, job)
            except Exception as error:
                self.status.set('頁序 checkpoint 更新失敗：' + str(error))
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
                if original.width * original.height > 24_000_000:
                    raise ValueError('圖片過大，請先拆頁')
                image = ImageOps.exif_transpose(original)
                image.thumbnail((500, 280))
                self.preview_image = ImageTk.PhotoImage(image.convert('RGB'))
            self.preview.configure(image=self.preview_image, text='')
        except Exception:
            self.preview.configure(image='', text='無法預覽圖片／需拆頁')
        groups = documents(self.files, self.group.get() == '每張獨立')
        flattened = [(doc, page) for doc, pages in groups for page in pages]
        doc, page = flattened[index]
        self.page_location = (Path(self.vars['output'].get()).resolve() / doc, page['id'])
        checkpoint = self.page_location[0] / 'pages' / (page['id'] + '.json')
        self.editor.delete('1.0', 'end')
        if checkpoint.exists():
            try:
                value = read_json(checkpoint)
                self.editor.insert('1.0', '\n\n'.join(s['text'] for s in value['segments']))
                self.status.set(f"頁面 {value['status']}；revision {value['revision']}；" + '; '.join(value['warnings']))
            except Exception as error:
                self.status.set(str(error))

    def show_raw(self):
        if self.page_location:
            root, page_id = self.page_location
            try:
                value = read_json(root / 'pages' / (page_id + '.json'))
                window = tk.Toplevel(self)
                window.title('Raw OCR（唯讀）')
                text = tk.Text(window, wrap='word')
                text.pack(fill='both', expand=True)
                text.insert('1.0', value['raw'])
                text.configure(state='disabled')
            except Exception as error:
                messagebox.showerror('Raw OCR', str(error))

    def save_correction(self, no_text=False):
        if self.active or not self.page_location:
            return
        try:
            import re
            text = '' if no_text else self.editor.get('1.0', 'end-1c')
            correct_page(*self.page_location, re.split(r'\n\s*\n', text), no_text)
            self.status.set('校正已儲存；raw OCR 保留；該文件譯文已失效。選重用 OCR 翻譯。')
        except Exception as error:
            messagebox.showerror('校正失敗', str(error))

    def start(self):
        if self.active or getattr(self, '_closing', False):
            return
        try:
            values = {key: variable.get() for key, variable in self.vars.items()}
            self.run_id += 1
            snapshot = RunSnapshot(tuple(self.files), str(Path(values['output']).resolve()),
                values['executable'], values['device'],
                Profile(values['ocr_model'], values['mmproj'], int(values['ocr_context']),
                        int(values['ocr_tokens']), float(values['ocr_temperature'])),
                Profile(values['translation_model'], '', int(values['translation_context']),
                        int(values['translation_tokens']), float(values['translation_temperature'])),
                mode=self.mode.get(), source=values['source'], locale=values['locale'],
                direction=self.direction.get(), independent=self.group.get() == '每張獨立',
                review=self.review.get(), two_step=self.two_step.get(), run_id=self.run_id)
            for key, value in values.items():
                self.config.set('manga.' + key, value)
            self.runner = MangaRunner(self.server, snapshot, self.events.put)
            self.active = True
            self.status.set('驗證輸入與 checkpoint…')
            threading.Thread(target=self.runner.run, daemon=True).start()
        except Exception as error:
            messagebox.showerror('無法開始', str(error))

    def cancel(self):
        if self.active and self.runner:
            self.runner.cancel()
            self.status.set('Stopping：停止新派發，等待目前 HTTP 返回（最多 120 秒），再還原 server')

    def load_job(self):
        if self.active:
            return
        filename = filedialog.askopenfilename(filetypes=[('Manga job', 'manga-job.json')])
        if not filename:
            return
        try:
            job = read_json(filename)
            if not isinstance(job.get('files'), list) or not all(isinstance(p, str) for p in job['files']):
                raise ValueError('Invalid job image list')
            for key in ('executable', 'device', 'output', 'source', 'locale'):
                self.vars[key].set(job[key])
            for prefix, name in [('ocr', 'ocr'), ('translation', 'translation')]:
                profile = job[name]
                self.vars[prefix + '_model'].set(profile['model'])
                self.vars[prefix + '_context'].set(profile['context'])
                self.vars[prefix + '_tokens'].set(profile['max_tokens'])
                self.vars[prefix + '_temperature'].set(profile['temperature'])
            self.vars['mmproj'].set(job['ocr']['mmproj'])
            self.files = job['files']
            self.mode.set(job['mode'])
            self.group.set('每張獨立' if job['independent'] else '同一文件')
            self.direction.set(job['direction'])
            self.review.set(job['review'])
            self.two_step.set(job['two_step'])
            self.refresh_pages()
            self.status.set('工作記錄已載入；按續跑會重新驗證圖片與 checkpoints')
        except Exception as error:
            messagebox.showerror('無法載入', str(error))

    def open_output(self):
        path = Path(self.vars['output'].get()).resolve()
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)

    def poll(self):
        for _ in range(100):
            try:
                event = self.events.get_nowait()
            except queue.Empty:
                break
            if event['run_id'] != self.run_id:
                continue
            stage = event['stage']
            if stage == 'devices':
                info = event['info']
                self.device_combo.configure(values=info['devices'])
                if info['devices'] and self.vars['device'].get() not in info['devices']:
                    self.vars['device'].set(info['devices'][0])
                self.status.set(info['version'].strip() + ' | ' + info['device_list'].strip() + ' | unverified')
            elif stage == 'done':
                self.active = False
                report = event['report']
                verified = report['stages'] and all(x['status'] == 'verified' for x in report['stages'])
                counts = report.get('counts', {})
                self.progress.configure(value=self.progress.cget('maximum'))
                self.status.set(f"{report['status']}；OCR 成功 {counts.get('success', 0)}／失敗 {counts.get('failed', 0)}／待確認 {counts.get('needs_review', 0)}；錯誤 {report['failures']}；Vulkan " +
                    ('verified (本輪有證據的階段)' if verified else 'unverified') +
                    ('；' + report['error'] if 'error' in report else '') +
                    ('；還原失敗：' + report['restore_error'] if report['restore_error'] else ''))
            elif stage in ('ocr', 'translation'):
                self.progress.configure(maximum=event['total'], value=event['current'] - 1)
                self.status.set(f"{stage} {event['current']}/{event['total']}")
            elif stage == 'diagnostic':
                self.status.set('Vulkan ' + event['diagnostic']['status'])
            else:
                self.status.set(stage + ': ' + str(event.get('error', event.get('model', ''))))
        self._poll_id = self.after(80, self.poll)
