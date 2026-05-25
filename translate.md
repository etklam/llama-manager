# Batch Subtitle Translation 移植指南

> 將 pyvideotrans 的批量字幕翻譯功能獨立遷移到另一個 Project。

---

## 目錄

1. [完整依賴樹](#1-完整依賴樹)
2. [最小文件清單](#2-最小文件清單)
3. [分層解耦方案](#3-分層解耦方案)
4. [【第 1 層】SRT 解析／組裝（純 stdlib，直接搬）](#4-第-1-層srt-解析組裝純-stdlib直接搬)
5. [【第 2 層】BaseTrans 去基類化（核心重構）](#5-第-2-層basetrans-去基類化核心重構)
6. [【第 3 層】各翻譯引擎（選搬）](#6-第-3-層各翻譯引擎選搬)
7. [【第 4 層】翻譯入口 + TranslateSrt（組合層）](#7-第-4-層翻譯入口--translatesrt組合層)
8. [【第 5 層】CLI 入口](#8-第-5-層cli-入口)
9. [外部套件依賴](#9-外部套件依賴)
10. [目標 Project 結構](#10-目標-project-結構)
11. [術語表、Prompt 文件](#11-術語表prompt-文件)

---

## 1. 完整依賴樹

下面是 TranslateSrt 完整 import 鏈，從上到下依次是「誰依賴誰」：

```
cli.py (入口)
  └─ TranslateSrt  (videotrans/task/_translate_srt.py)
       ├─ tools.get_subtitle_from_srt()     ← 解析 SRT 文件
       ├─ translator.run()                  ← 中央調度
       └─ BaseTask._check_target_sub()      ← 校驗行數對齊
            └─ BaseTask  (videotrans/task/_base.py)
                 └─ BaseCon  (videotrans/configure/_base.py)
                      ├─ _signal() → tools.set_process()
                      ├─ _set_proxy() → tools.set_proxy()
                      ├─ proxy_str (代理字串)
                      ├─ no_proxy (不需要代理的 host)
                      └─ _exit() → app_cfg.exit_soft / stoped_uuid_set
  └─ BaseTrans  (videotrans/translator/_base.py)
       ├─ BaseCon (同上)
       ├─ _signal() → tools.set_process()
       ├─ _get_cache / _set_cache          ← 翻譯緩存
       ├─ _get_key                          ← MD5 緩存鍵
       ├─ tools.get_md5()
       ├─ tools.cleartext()
       ├─ tools.get_subtitle_from_srt()    ← 解析 SRT 字串
       └─ 各引擎 _item_task()  ← 子類實作
            ├─ Google    (_google.py)    → requests + tenacity
            ├─ Microsoft (_microsoft.py) → requests + tenacity
            ├─ ChatGPT   (_chatgpt.py)   → openai + httpx + tenacity
            ├─ DeepSeek  (_deepseek.py)  → openai
            ├─ DeepL     (_deepl.py)     → deepl SDK + tenacity
            ├─ Baidu     (_baidu.py)     → requests + tenacity
            ├─ Tencent   (_tencent.py)   → tencentcloud-sdk-python
            ├─ Gemini    (_gemini.py)    → google-genai
            ├─ ... 其他 15 個引擎
            └─ 每個都引用 tools.get_prompt() ← 讀取 prompt/*.txt

  └─ tools.*   (videotrans/util/)
       ├─ help_srt.py  → get_subtitle_from_srt, get_srt_from_list, cleartext, format_time, ms_to_time_string, srt_str_to_listdict, format_srt
       ├─ help_misc.py → set_proxy, set_process, get_md5, vail_file, get_prompt, get_prompt_file
       └─ help_ffmpeg.py → format_video, send_notification
```

---

## 2. 最小文件清單

### 絕對需要（核心邏輯）

這些文件包含翻譯流程的本質邏輯，必須搬遷：

| 源文件 | 搬遷方式 | 說明 |
|--------|----------|------|
| `videotrans/translator/_base.py` | **重構** | BaseTrans，去掉 BaseCon 繼承 |
| `videotrans/translator/__init__.py` | **重構** | `run()` 調度 + LANG_CODE 語言映射 |
| `videotrans/task/_translate_srt.py` | **重構** | TranslateSrt，去掉 BaseTask/BaseCon |
| `videotrans/util/help_srt.py` | **直接搬** | SRT 解析/組裝（無外部依賴） |
| `videotrans/util/help_misc.py` | **拆搬** | 只取 `get_md5`, `cleartext`（已含在 SRT 模組）, `get_prompt` |
| `videotrans/translator/_google.py` | **輕量修改** | 去掉 BaseCon 的 _exit() |
| `videotrans/translator/_chatgpt.py` | **輕量修改** | 去掉 BaseCon |
| `videotrans/translator/_deepseek.py` | **輕量修改** | 去掉 BaseCon |
| `videotrans/translator/_deepl.py` | **輕量修改** | 去掉 BaseCon |
| `videotrans/prompts/` | **直接複製** | Prompt 模板（可選，可自行寫） |

### 可以替換的

這些是 pyvideotrans 特有的基礎設施，目標 Project 一定有等價物：

| 源文件 | 替換方案 |
|--------|---------|
| `videotrans/configure/config.py` | 你自己的 config / logger 系統 |
| `videotrans/configure/_base.py` (BaseCon) | 拆掉，功能分散到各層 |
| `videotrans/task/_base.py` (BaseTask) | 拆掉，只保留 `_check_target_sub` |
| `tools.set_proxy()` | 用你自己的 proxy 配置 |
| `tools.set_process()` | 換成你自己的 logger / callback |
| `tools.send_notification()` | 去掉或用自己的通知 |
| `tools.show_error()` | GUI only，CLI 模式不需要 |
| `tools.show_glossary_editor()` | GUI only，術語表改為純文件讀取 |
| `tools.vail_file()` | 用 `Path(file).is_file()` 一行取代 |
| `tools.format_video()` | 只在 CLI 入口用，可內聯簡化 |

### 不需要的（純 GUI ／其他功能）

這些完全不需搬：
- `videotrans/winform/fn_fanyisrt.py`（GUI 窗口）
- `videotrans/task/child_win_sign.py`（GUI 進度線程）
- `videotrans/configure/_except.py`（500 行異常處理，太重，可自寫簡版）
- `videotrans/process/`（子進程管理）
- `videotrans/recognition/`, `videotrans/tts/`（非翻譯功能）
- 其他非翻譯引擎

---

## 3. 分層解耦方案

核心問題：BaseTrans / TranslateSrt 都繼承了 BaseCon，而 BaseCon 依賴 `app_cfg`（dataclass + Queue + set）、`tools.set_process()` 等繁重基礎設施。

解法：**把依賴「外化」為構造參數和 callback**。

```
改造前：                        改造後：

BaseCon                         (不存在了)
  ├─ _signal()        ──→       __init__ 接收 progress_callback(text, type)
  ├─ _exit()          ──→       __init__ 接收 cancel_checker() -> bool
  ├─ proxy_str        ──→       __init__ 接收 proxy: str | None
  └─ _set_proxy()     ──→       由調用方在外部設置環境變量
```

---

## 4. 【第 1 層】SRT 解析／組裝（純 stdlib，直接搬）

### 4.1 需要的函數

從 `videotrans/util/help_srt.py` 提取以下，無需修改（全部只依賴 stdlib）：

| 函數 | 行號 | 用途 |
|------|------|------|
| `ms_to_time_string()` | 85-95 | ms → `HH:MM:SS,mmm` |
| `format_time()` | 98-123 | 規範化時間格式 |
| `srt_str_to_listdict()` | 126-195 | SRT 字串 → `[{line, start_time, end_time, startraw, endraw, time, text}]` |
| `format_srt()` | 200-206 | 調用 `srt_str_to_listdict`，失敗時 fallback |
| `get_subtitle_from_srt()` | 210-247 | 文件路徑 → dict 列表 |
| `get_srt_from_list()` | 252-281 | dict 列表 → SRT 字串 |
| `cleartext()` | 78-82 | 清理翻譯結果（移除特殊字符） |

### 4.2 直接搬遷方式

把 `help_srt.py` 整個文件複製到新 Project，刪掉第 5 行的 pyvideotrans import：

```python
# 原第 5 行：
from videotrans.configure.config import ROOT_DIR,tr,app_cfg,settings,params,TEMP_DIR,logger,defaulelang,HOME_DIR
# 改為：刪掉這行

# 然後修復內部引用 tr() 的地方：
# 行 122: 把 raise Exception(tr("...")) 改為 raise Exception("No valid timestamp")
# 行 266-269: 把 tr("...") 改為硬編碼字符串
```

實質需要改的只有 3 處 `tr()` 調用（錯誤訊息字串）。

### 4.3 精簡版（如果你不需要 edge case 處理）

也可以只複製核心 3 個函數（不到 120 行）：

```python
# srt_utils.py —— 精簡但足夠的版本
import re
import copy
from datetime import timedelta
from pathlib import Path
from typing import List, Dict

def ms_to_time_string(*, ms=0, sepflag=','):
    td = timedelta(milliseconds=ms)
    hours, remainder = divmod(td.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    milliseconds = td.microseconds // 1000
    return f"{hours:02}:{minutes:02}:{seconds:02}{sepflag}{milliseconds:03}"

def get_subtitle_from_srt(file_path: str, *, is_file=True) -> List[Dict]:
    """Parse SRT file or string into list of {line, start_time, end_time, startraw, endraw, time, text}"""
    if is_file:
        content = Path(file_path).read_text(encoding='utf-8').strip()
    else:
        content = file_path.strip()
    if not content:
        return [{"line": 1, "start_time": 0, "end_time": 2000,
                 "startraw": "00:00:00,000", "endraw": "00:00:02,000",
                 "time": "00:00:00,000 --> 00:00:02,000", "text": ""}]

    result = []
    blocks = re.split(r'\n\s*\n', content)
    for i, block in enumerate(blocks, 1):
        lines = block.strip().split('\n')
        if len(lines) < 2:
            continue
        # Parse timing line
        time_match = re.match(
            r'(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{3})',
            lines[0] if not lines[0].isdigit() else lines[1]
        )
        if not time_match:
            continue
        h1, m1, s1, ms1, h2, m2, s2, ms2 = map(int, time_match.groups())
        start_ms = h1 * 3600000 + m1 * 60000 + s1 * 1000 + ms1
        end_ms = h2 * 3600000 + m2 * 60000 + s2 * 1000 + ms2
        startraw = f"{h1:02}:{m1:02}:{s1:02},{ms1:03}"
        endraw = f"{h2:02}:{m2:02}:{s2:02},{ms2:03}"

        # Find text lines
        text_start = 0 if lines[0].isdigit() else 1
        text_start += 1 if lines[text_start] and re.match(r'\d{1,2}:\d{2}', lines[text_start]) else 0
        text = '\n'.join(lines[text_start:]).strip()

        result.append({
            "line": i,
            "start_time": start_ms,
            "end_time": end_ms,
            "startraw": startraw,
            "endraw": endraw,
            "time": f"{startraw} --> {endraw}",
            "text": text
        })
    return result if result else [{"line": 1, "start_time": 0, "end_time": 2000,
                                    "startraw": "00:00:00,000", "endraw": "00:00:02,000",
                                    "time": "00:00:00,000 --> 00:00:02,000", "text": content}]

def get_srt_from_list(srt_list: List[Dict]) -> str:
    """Convert list of {line, text, time} back to SRT string"""
    result = []
    for it in srt_list:
        if not it.get('text', '').strip():
            continue
        if 'time' in it:
            time_str = it['time']
        elif 'startraw' in it and 'endraw' in it:
            time_str = f"{it['startraw']} --> {it['endraw']}"
        else:
            startraw = ms_to_time_string(ms=it.get('start_time', 0))
            endraw = ms_to_time_string(ms=it.get('end_time', 0))
            time_str = f"{startraw} --> {endraw}"
        result.append(f"{it['line']}\n{time_str}\n{it['text'].strip()}\n")
    return "\n".join(result)

def cleartext(text: str) -> str:
    return text.replace('&#39;', '').replace('&quot;', '').replace('\u200b', ' ').strip()
```

---

## 5. 【第 2 層】BaseTrans 去基類化（核心重構）

這是最大的改動點：把 `BaseTrans(BaseCon)` 改成一個純粹的組合類。

### 5.1 原始 BaseTrans 依賴了哪些 BaseCon 功能

| BaseCon 功能 | BaseTrans 中的使用 | 解耦方式 |
|-------------|-------------------|----------|
| `BaseCon.__post_init__()` 設 proxy | `__post_init__()` 調用 `super().__post_init__()` | 構造參數傳入 proxy |
| `self._signal(text, type)` | `run()`, `_run_text()`, `_run_srt()` | callback: `self._progress(text, type)` |
| `self._exit()` | `_item_task()` 內每個引擎都調用 | callback: `self._should_cancel()` |
| `self.proxy_str` | 引擎內部用於 httpx/requests | 構造參數傳入 |
| 從 `config` 導入 `tr, settings, params, logger, TEMP_DIR, TEMP_ROOT` | 全文件 | 構造參數 + 外部 logger |
| `NO_RETRY_EXCEPT` from `_except.py` | 引擎 retry 裝飾器 | 簡化為 connection-only exceptions |

### 5.2 改造後的 BaseTrans

```python
# translator_base.py —— 獨立版 BaseTrans，無任何 pyvideotrans 依賴
import json
import hashlib
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable, Union

from srt_utils import get_subtitle_from_srt, cleartext  # 你的 SRT 模組

# 翻譯緩存目錄（可由外部設置）
CACHE_DIR = Path("translate_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

class BaseTranslator:
    """
    翻譯基類，不再繼承任何東西。
    子類只需實作 _item_task(data: Union[List[str], str]) -> str。
    """

    def __init__(
        self,
        translate_type: int = 0,
        text_list: List[Dict] = None,
        source_lang: str = "auto",
        target_lang: str = "",
        target_lang_name: str = "",
        model: str = "",
        api_url: str = "",
        proxy: str = None,
        batch_size: int = 5,
        wait_seconds: float = 0.0,
        aisendsrt: bool = False,
        full_context: str = "",
        uuid: str = None,
        progress_callback: Callable = None,
        cancel_checker: Callable = None,
    ):
        self.translate_type = translate_type
        self.text_list = text_list or []
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.target_lang_name = target_lang_name
        self.model = model
        self.api_url = api_url
        self.proxy = proxy
        self.batch_size = batch_size
        self.wait_seconds = wait_seconds
        self.aisendsrt = aisendsrt
        self.full_context = full_context
        self.uuid = uuid
        self._progress = progress_callback or (lambda text, type=None: None)
        self._cancel = cancel_checker or (lambda: False)

    # ---- 子類必須實作 ----
    def _item_task(self, data: Union[List[str], str]) -> str:
        raise NotImplementedError

    # ---- 緩存 ----
    def _cache_key(self, data):
        raw = json.dumps(data, ensure_ascii=False) if not isinstance(data, str) else data
        key = f"{self.translate_type}-{self.api_url}-{self.aisendsrt}-{self.model}-{self.source_lang}-{self.target_lang}-{raw}"
        return hashlib.md5(key.encode()).hexdigest()

    def _get_cache(self, data):
        f = CACHE_DIR / f"{self._cache_key(data)}.txt"
        return f.read_text(encoding='utf-8') if f.exists() else None

    def _set_cache(self, data, result):
        if result.strip():
            (CACHE_DIR / f"{self._cache_key(data)}.txt").write_text(result, encoding='utf-8')

    # ---- 主入口 ----
    def run(self):
        t0 = time.time()
        self._progress("")

        source_rows = self.text_list
        if self.aisendsrt:
            chunks = [source_rows[i:i + self.batch_size] for i in range(0, len(source_rows), self.batch_size)]
            return self._run_srt(chunks)
        else:
            texts = [t['text'] for t in source_rows]
            chunks = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
            return self._run_text(chunks)

    def _run_text(self, chunks: List[List[str]]):
        result_texts = []
        for i, batch in enumerate(chunks):
            if self._cancel():
                return []
            self._progress(f'Translating batch {i+1}', type='logs')

            cached = self._get_cache(batch)
            if cached:
                translated = cached
            else:
                translated = cleartext(self._item_task(batch))
                self._set_cache(batch, translated)

            lines = translated.split('\n')
            for j, line in enumerate(lines):
                if j < len(batch):
                    result_texts.append(line.strip())
                    self._progress(line.strip() + '\n', type='subtitle')
            # 行數不夠時補空
            if len(lines) < len(batch):
                result_texts.extend([''] * (len(batch) - len(lines)))
            time.sleep(self.wait_seconds)

        # 回填
        for i, srt in enumerate(self.text_list):
            srt['text'] = result_texts[i] if i < len(result_texts) else ''
        return self.text_list

    def _run_srt(self, chunks: List[List[Dict]]):
        result = []
        for i, batch in enumerate(chunks):
            if self._cancel():
                return []
            self._progress(f'Translating batch {i+1}', type='logs')

            # 組裝 SRT 字串
            srt_str = "\n\n".join(
                f"{s['line']}\n{s['time']}\n{s['text'].strip()}" for s in batch
            )

            cached = self._get_cache(srt_str)
            if cached:
                translated = cached
            else:
                translated = self._item_task(srt_str)
                if not translated.strip():
                    raise RuntimeError("Translation result is empty")
                self._set_cache(srt_str, translated)

            self._progress(translated, type='subtitle')
            parsed = get_subtitle_from_srt(translated, is_file=False)
            result.extend(parsed)
            time.sleep(self.wait_seconds)

        return result
```

### 5.3 關鍵改動說明

| 原始 | 改造後 |
|------|--------|
| `self._signal(text=..., type=...)` | `self._progress(text, type=type)` |
| `self._exit()` | `self._cancel()` |
| `tools.get_md5(key_str)` | `hashlib.md5(...).hexdigest()` (內聯) |
| `tools.cleartext(result)` | `cleartext(result)` (第 1 層的函數) |
| `tools.get_subtitle_from_srt(result, is_file=False)` | `get_subtitle_from_srt(result, is_file=False)` (第 1 層) |
| `settings['aisendsrt']` | 構造參數 `aisendsrt` |
| `settings['trans_thread']` | 構造參數 `batch_size` |
| `settings['translation_wait']` | 構造參數 `wait_seconds` |
| `app_cfg.exit_soft` | callback: `cancel_checker()` |
| `BaseCon.__post_init__()` 設 proxy | 構造參數 `proxy` |
| `_GLOBAL_CONTEXT` prompt 模板 | 構造參數 `full_context` |
| `tr('...')` 日誌字串 | 硬編碼英文（或傳入翻譯函數） |

---

## 6. 【第 3 層】各翻譯引擎（選搬）

每個引擎只需繼承新的 `BaseTranslator`，實作 `_item_task()`。以下以 4 個常用引擎為例。

### 6.1 Google（免費，無需 API Key）

```python
# translator_google.py
import re
import requests
from typing import Union, List

from translator_base import BaseTranslator

class GoogleTranslator(BaseTranslator):
    def _item_task(self, data: Union[List[str], str]) -> str:
        text = data if isinstance(data, str) else "\n".join(data)
        url = f"https://translate.google.com/m?sl={self.source_lang}&tl={self.target_lang}&hl={self.target_lang}&q={requests.utils.quote(text)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15'
        }
        resp = requests.get(url, headers=headers, verify=False, proxies={'http': self.proxy, 'https': self.proxy} if self.proxy else None, timeout=30)
        match = re.search(r'<div\s+class="?result-container"?[^>]*>([^<]+?)<', resp.text)
        if match:
            return match.group(1)
        raise RuntimeError(f"Google Translate parse error: {resp.status_code}")
```

### 6.2 ChatGPT / OpenAI 兼容

```python
# translator_openai.py
import re
from typing import Union, List
from openai import OpenAI

from translator_base import BaseTranslator

class OpenAITranslator(BaseTranslator):
    """適用於 ChatGPT / DeepSeek / 任何 OpenAI 兼容 API"""

    def __init__(self, api_key: str, system_prompt: str = "", user_prompt_template: str = "", **kwargs):
        super().__init__(**kwargs)
        self.api_key = api_key
        self.system_prompt = system_prompt
        self.user_prompt_template = user_prompt_template  # 包含 {batch_input} 和 {context_block}

    def _item_task(self, data: Union[List[str], str]) -> str:
        text = data if isinstance(data, str) else "\n".join(data)
        user_content = self.user_prompt_template.replace('{batch_input}', text).replace('{context_block}', self.full_context)

        client = OpenAI(api_key=self.api_key, base_url=self.api_url)
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            timeout=300,
        )
        result = response.choices[0].message.content
        # 提取 <TRANSLATE_TEXT>...</TRANSLATE_TEXT> 標籤
        match = re.search(r'<TRANSLATE_TEXT>(.*?)</TRANSLATE_TEXT>', result, re.S)
        return match.group(1) if match else result
```

### 6.3 DeepL

```python
# translator_deepl.py
from typing import Union, List
import deepl

from translator_base import BaseTranslator

class DeepLTranslator(BaseTranslator):
    def __init__(self, auth_key: str, **kwargs):
        super().__init__(**kwargs)
        self.translator = deepl.Translator(auth_key, server_url=self.api_url)

    def _item_task(self, data: Union[List[str], str]) -> str:
        text = data if isinstance(data, str) else "\n".join(data)
        # 特殊符號全部跳過
        if re.match(r'^[\s~`!@#$%^&*()_+\-=\[\]{}\\|;,./?><:"\'，。、；''""《》？【】｛｝（）—！·￥…ー]+$', text):
            return text
        result = self.translator.translate_text(text, target_lang=self.target_lang)
        return result.text
```

### 6.4 Microsoft（免費，無需 API Key）

```python
# translator_microsoft.py
import re, requests, uuid
from typing import Union, List

from translator_base import BaseTranslator

class MicrosoftTranslator(BaseTranslator):
    def _item_task(self, data: Union[List[str], str]) -> str:
        text = data if isinstance(data, str) else "\n".join(data)
        url = f"https://api-edge.cognitive.microsofttranslator.com/translate?from={self.source_lang}&to={self.target_lang}&api-version=3.0"
        headers = {
            'Ocp-Apim-Subscription-Region': 'global',
            'Content-Type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4()),
        }
        payload = [{'text': t} for t in text.split('\n')]
        resp = requests.post(url, json=payload, headers=headers,
                           proxies={'http': self.proxy, 'https': self.proxy} if self.proxy else None, timeout=30)
        results = resp.json()
        return '\n'.join(r['translations'][0]['text'] for r in results)
```

---

## 7. 【第 4 層】翻譯入口 + TranslateSrt（組合層）

### 7.1 調度器

```python
# translator.py —— 模組級入口
from typing import List, Dict, Callable

from translator_base import BaseTranslator
from translator_google import GoogleTranslator
from translator_openai import OpenAITranslator
from translator_deepl import DeepLTranslator
from translator_microsoft import MicrosoftTranslator

LANG_CODE = {
    "zh-cn": ["zh-cn", "zh", "ZH-HANS", "Simplified Chinese"],
    "zh-tw": ["zh-tw", "cht", "ZH-HANT", "Traditional Chinese"],
    "en": ["en", "en", "EN-US", "English"],
    "ja": ["ja", "jp", "JA", "Japanese"],
    "ko": ["ko", "kor", "KO", "Korean"],
    "fr": ["fr", "fra", "FR", "French"],
    "de": ["de", "de", "DE", "German"],
    "es": ["es", "spa", "ES", "Spanish"],
    "ru": ["ru", "ru", "RU", "Russian"],
    # ... 按需添加
    "auto": ["auto", "auto", "auto", "auto"],
}

# 語言代碼 → 對各渠道的映射索引
# [0]=Google/Microsoft/通用, [1]=Baidu, [2]=DeepL, [3]=AI 自然語言名稱
LANG_IDX = {
    "google": 0, "microsoft": 0, "deepl": 2, "openai": 3
}

def get_lang_code(lang: str, provider: str) -> str:
    """根據提供者返回對應的語言代碼"""
    if lang not in LANG_CODE:
        return lang
    idx = LANG_IDX.get(provider, 0)
    code = LANG_CODE[lang][idx]
    return code if code != "No" else lang

def translate(
    provider: str,           # "google" | "openai" | "deepl" | "microsoft"
    text_list: List[Dict],   # SRT dict list
    source_lang: str,
    target_lang: str,
    api_key: str = "",
    api_url: str = "",
    model: str = "",
    proxy: str = None,
    batch_size: int = None,
    progress_callback: Callable = None,
    cancel_checker: Callable = None,
    aisendsrt: bool = False,
    full_context: str = "",
    system_prompt: str = "",
    user_prompt_template: str = "",
    uuid: str = None,
    **kwargs,
) -> List[Dict]:
    """
    翻譯入口，根據 provider 選擇引擎，返回翻譯後的 text_list。
    """
    source_code = get_lang_code(source_lang, provider)
    target_code = get_lang_code(target_lang, provider)

    base_kwargs = dict(
        text_list=text_list,
        source_lang=source_code,
        target_lang=target_code,
        target_lang_name=get_lang_code(target_lang, "openai") if provider in ("openai",) else target_code,
        api_url=api_url,
        model=model,
        proxy=proxy,
        batch_size=batch_size or (20 if provider in ("openai",) else 5),
        aisendsrt=aisendsrt,
        full_context=full_context,
        uuid=uuid,
        progress_callback=progress_callback,
        cancel_checker=cancel_checker,
    )

    if provider == "google":
        return GoogleTranslator(**base_kwargs).run()
    elif provider == "microsoft":
        return MicrosoftTranslator(**base_kwargs).run()
    elif provider == "openai":
        return OpenAITranslator(
            api_key=api_key,
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            **base_kwargs
        ).run()
    elif provider == "deepl":
        return DeepLTranslator(auth_key=api_key, **base_kwargs).run()
    else:
        raise ValueError(f"Unknown provider: {provider}")
```

### 7.2 TranslateSrt（簡化版）

```python
# translate_srt.py
import copy
import shutil
from pathlib import Path
from typing import List, Dict, Callable

from srt_utils import get_subtitle_from_srt, get_srt_from_list
from translator import translate

class SubtitleTranslator:
    """
    獨立字幕翻譯任務，無 GUI、無 BaseTask 依賴。
    """
    def __init__(
        self,
        srt_path: str,
        target_lang: str,
        source_lang: str = "auto",
        provider: str = "google",
        api_key: str = "",
        api_url: str = "",
        model: str = "",
        proxy: str = None,
        output_dir: str = None,
        output_format: int = 0,  # 0=單語, 1=目標在上雙語, 2=源在上雙語
        progress_callback: Callable = None,
        cancel_checker: Callable = None,
        **kwargs
    ):
        self.srt_path = Path(srt_path)
        self.target_lang = target_lang
        self.source_lang = source_lang
        self.provider = provider
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.proxy = proxy
        self.output_dir = Path(output_dir or self.srt_path.parent)
        self.output_format = output_format
        self._progress = progress_callback or (lambda text, type=None: None)
        self._cancel = cancel_checker or (lambda: False)
        self.kwargs = kwargs

        name_stem = self.srt_path.stem
        self.target_path = self.output_dir / f"{name_stem}.{target_lang}.srt"

    def translate(self) -> Path:
        """執行翻譯，返回輸出文件路徑"""
        self._progress("Reading SRT file...", "logs")
        source_subs = get_subtitle_from_srt(str(self.srt_path))

        self._progress(f"Translating {len(source_subs)} lines via {self.provider}...", "logs")
        translated = translate(
            provider=self.provider,
            text_list=copy.deepcopy(source_subs),
            source_lang=self.source_lang,
            target_lang=self.target_lang,
            api_key=self.api_key,
            api_url=self.api_url,
            model=self.model,
            proxy=self.proxy,
            progress_callback=self._progress,
            cancel_checker=self._cancel,
            **self.kwargs
        )

        if not translated:
            raise RuntimeError("Translation result is empty")

        # 對齊行數
        translated = self._align_lines(source_subs, translated)

        if self.output_format == 0:
            self._save(translated, self.target_path)
        else:
            srt_str = self._make_bilingual(source_subs, translated, self.output_format)
            bilingual_path = self.output_dir / f"{self.srt_path.stem}.{self.target_lang}-{self.output_format}.srt"
            bilingual_path.write_text(srt_str, encoding='utf-8')
            self._progress(srt_str, type='replace')
            self.target_path = bilingual_path

        self._progress(f"Done: {self.target_path}", "logs")
        return self.target_path

    def _align_lines(self, source, target):
        """確保源和目標行數一致"""
        if len(source) == len(target):
            return target
        if len(target) > len(source):
            return target[:len(source)]
        # target 少於 source：補空行
        while len(target) < len(source):
            target.append({"line": len(target)+1, "start_time": 0, "end_time": 0,
                          "startraw": "00:00:00,000", "endraw": "00:00:00,000",
                          "time": "00:00:00,000 --> 00:00:00,000", "text": ""})
        return target

    def _save(self, srt_list, path):
        srt_string = get_srt_from_list(srt_list)
        path.write_text(srt_string, encoding='utf-8')
        self._progress(srt_string, type='replace')

    def _make_bilingual(self, source, target, fmt):
        """fmt=1: 目標在上; fmt=2: 源在上"""
        result = []
        for i, src in enumerate(source):
            src_text = src['text'].strip()
            tgt_text = target[i]['text'].strip() if i < len(target) else ''
            if fmt == 1:
                combined = f"{tgt_text}\n{src_text}" if tgt_text else src_text
            else:
                combined = f"{src_text}\n{tgt_text}" if tgt_text else src_text
            result.append(f"{src['line']}\n{src['time']}\n{combined}\n")
        return "\n".join(result)
```

---

## 8. 【第 5 層】CLI 入口

```python
# cli.py
import argparse
from pathlib import Path
from translate_srt import SubtitleTranslator

def main():
    parser = argparse.ArgumentParser(description="SRT Subtitle Translation")
    parser.add_argument('srt_file', help='Path to .srt file')
    parser.add_argument('target_lang', help='Target language code (e.g., zh-cn, en, ja)')
    parser.add_argument('--source-lang', default='auto', help='Source language (default: auto)')
    parser.add_argument('--provider', default='google', choices=['google', 'microsoft', 'openai', 'deepl'])
    parser.add_argument('--api-key', default='', help='API key (for openai/deepl)')
    parser.add_argument('--api-url', default='', help='API base URL (for openai-compatible)')
    parser.add_argument('--model', default='gpt-4o-mini', help='Model name (for openai)')
    parser.add_argument('--proxy', default=None, help='HTTP proxy')
    parser.add_argument('--output-dir', default=None, help='Output directory')
    parser.add_argument('--format', type=int, default=0,
                        help='0=single lang, 1=target+source bilingual, 2=source+target bilingual')
    parser.add_argument('--aisendsrt', action='store_true', help='Send full SRT format to AI provider')

    args = parser.parse_args()

    def cli_progress(text, type=None):
        print(f"[{type or 'info'}] {text}")

    translator = SubtitleTranslator(
        srt_path=args.srt_file,
        target_lang=args.target_lang,
        source_lang=args.source_lang,
        provider=args.provider,
        api_key=args.api_key,
        api_url=args.api_url,
        model=args.model,
        proxy=args.proxy,
        output_dir=args.output_dir,
        output_format=args.format,
        progress_callback=cli_progress,
        aisendsrt=args.aisendsrt,
    )
    result = translator.translate()
    print(f"Output: {result}")

if __name__ == "__main__":
    main()
```

### 使用示例

```bash
# Google 免費翻譯
python cli.py /path/to/subtitles.srt zh-cn

# ChatGPT
python cli.py /path/to/subtitles.srt zh-cn --provider openai --api-key sk-xxx --model gpt-4o-mini

# DeepL
python cli.py /path/to/subtitles.srt zh-cn --provider deepl --api-key your-auth-key

# 雙語字幕（目標在上）
python cli.py /path/to/subtitles.srt zh-cn --format 1

# AI 引擎以完整 SRT 格式發送
python cli.py /path/to/subtitles.srt zh-cn --provider openai --api-key sk-xxx --aisendsrt
```

---

## 9. 外部套件依賴

### 核心（按引擎選擇）

| 引擎 | pip 套件 |
|------|---------|
| Google | `requests` |
| Microsoft | `requests` |
| OpenAI / DeepSeek | `openai` |
| DeepL | `deepl` |

### 可選重試（建議）

```bash
pip install tenacity
```

如果不裝，Google/Microsoft/DeepL 引擎的 retry 裝飾器需要去掉（或手寫 while True + sleep 循環）。

### 不需要的

- `PySide6` — GUI only，CLI 模式不需要
- `httpx` — 如果不用 ChatGPT override http_client 可省略
- `plyer` — 桌面通知，CLI 不需要
- `tqdm` — CLI 進度條有 print 就夠
- `pydub`, `edge_tts`, `aiohttp` — TTS 相關，不需要
- `torch`, `faster_whisper` — STT 相關，不需要
- `huggingface_hub` — 模型下載，不需要

### pip install 一行

```bash
pip install requests openai deepl tenacity
```

---

## 10. 目標 Project 結構

```
your-project/
├── translator/
│   ├── __init__.py
│   ├── translator_base.py      # BaseTranslator (第 2 層)
│   ├── translator.py           # translate() 調度器 (第 4 層)
│   ├── translate_srt.py        # SubtitleTranslator (第 4 層)
│   ├── engines/
│   │   ├── __init__.py
│   │   ├── google.py           # GoogleTranslator (第 3 層)
│   │   ├── microsoft.py        # MicrosoftTranslator
│   │   ├── openai.py           # OpenAITranslator
│   │   └── deepl.py           # DeepLTranslator
│   ├── srt_utils.py            # SRT 解析/組裝 (第 1 層)
│   └── prompts/                # Prompt 模板 (可選)
│       ├── chatgpt.txt
│       ├── deepseek.txt
│       └── ...
├── cli.py                      # CLI 入口 (第 5 層)
├── requirements.txt
└── translate_cache/            # 翻譯緩存目錄 (自動創建)
```

---

## 11. 術語表、Prompt 文件

### 術語表 (`translator/glossary.txt`)

可選，每行一個詞條，格式：`原文=譯文`。

```
emergency brake=緊急制動
Ballistic Missile Defense=BMD
```

在 `BaseTranslator` 中注入到 `full_context` 或 `user_prompt_template`。

### Prompt 文件

AI 引擎依賴 prompt 模板。原專案使用 `videotrans/prompts/srt/chatgpt.txt` 等文件，透過 `get_prompt()` 讀取。

**可以直接複製原專案 `videotrans/prompts/` 目錄下的 `chatgpt.txt`, `deepseek.txt` 等**，然後在初始化時讀入：

```python
import os
prompt_dir = os.path.join(os.path.dirname(__file__), 'prompts')
with open(os.path.join(prompt_dir, 'chatgpt.txt'), encoding='utf-8') as f:
    chatgpt_prompt = f.read()
```

---

## 附錄 A：完整改動對照

| 原始 pyvideotrans | 移植後 | 改動量 |
|-----------------|--------|--------|
| `BaseTrans(BaseCon)` | `BaseTranslator` (無繼承) | 重寫 `__init__`，去掉 super()，外化所有依賴 |
| `self._signal()` | `self._progress(text, type)` | 全局搜索替換 |
| `self._exit()` | `self._cancel()` | 全局搜索替換 |
| `tools.get_subtitle_from_srt()` | `srt_utils.get_subtitle_from_srt()` | 直接搬 |
| `tools.get_srt_from_list()` | `srt_utils.get_srt_from_list()` | 直接搬 |
| `tools.cleartext()` | `srt_utils.cleartext()` | 直接搬 |
| `tools.get_md5()` | `hashlib.md5(key.encode()).hexdigest()` | 內聯 1 行 |
| `tools.set_proxy()` | 外部設置 `os.environ['HTTPS_PROXY']` | 不再由 BaseTrans 管理 |
| `tools.get_prompt()` | 外部讀取文件，傳入構造參數 | 簡化 |
| `tools.format_video()` | 內聯 pathlib 操作 | 5 行取代 |
| `config.logger` | 你自己的 logger | 外部傳入 |
| `config.tr()` i18n | 硬編碼英文或傳入 lambda | 字串替換 |
| `config.settings['...']` | 構造參數 | 逐個參數化 |
| `TranslateSrt(BaseTask, BaseCon)` | `SubtitleTranslator` (無繼承) | 重寫 |
| `_check_target_sub()` | `_align_lines()` (內聯簡化版) | 保留邏輯 |

## 附錄 B：如需更多引擎

原專案還有 20 個其他引擎（Gemini, Baidu, Tencent, Azure, ZhipuAI, etc.），移植方式一致：

1. 從原文件複製 `_item_task()` 方法
2. 改繼承為 `BaseTranslator`
3. 替換 `self._exit()` → `self._cancel()`
4. 替換 proxy 來源（原用 `self.proxy_str` 來自 BaseCon，改為 `self.proxy`）

以 Baidu 為例，頂多 30 行程式碼即可完成移植。
