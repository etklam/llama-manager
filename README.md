# llama.cpp Manager

桌面工具，用於啟動本地 llama.cpp 模型、透過 whisper.cpp 轉錄音訊／影片，再將字幕翻譯成指定語言。主線支援 Windows；macOS／Linux 先以原始碼執行及實機驗收為目標。

**日文語音轉錄推薦使用 `kotoba-whisper-v2.0`。** 本程式使用 whisper.cpp，請下載下方介紹的 GGML `.bin` 版本。

## 功能與工作流程

| 畫面 | 用途 |
| --- | --- |
| 工作台 | 啟停模型伺服器；執行「轉錄 → 翻譯」管線，分開顯示待處理／已完成 |
| 伺服器設定 | 選擇 GGUF 模型、設定 GPU／上下文／並行槽、啟停 llama-server |
| 字幕翻譯 | 將 `.srt`／`.txt` 翻譯成指定語言，支援批次、單步／兩步及 SRT 全文理解模式 |
| 語音轉錄 | 使用 whisper.cpp 將音訊／影片轉成 SRT，支援長音訊分段 |
| 漫畫 OCR／翻譯（Beta） | 批量圖片 → GLM-OCR 候選模型 → 本地 Vulkan 文字模型；校正、checkpoint、取消／續跑與雙語文字 |

只做語音轉錄不需要啟動 llama-server；字幕翻譯需要 LLM 伺服器；媒體管線需要兩者。

## 介面操作

左側導覽列切換畫面，頂部狀態列**固定顯示**伺服器狀態與目前執行中的工作（檔名、進度、速率、剩餘時間）──
不論你切到哪個畫面都看得到。所有子系統的日誌集中在底部的「執行日誌」抽屜，可依頻道篩選，也能另開視窗放到第二螢幕。

| 快捷鍵 | 動作 |
| --- | --- |
| `Ctrl+1` … `Ctrl+5` | 切換畫面（順序同導覽列） |
| `Ctrl+L` | 展開／收起執行日誌 |
| `Ctrl+Shift+T` | 切換深色／淺色主題 |

主題預設為深色，可用狀態列右上角的按鈕或快捷鍵切換，選擇會存入使用者資料目錄的 `config.json`。
若不想看到執行中的動態指示，可將 `ui.reduce_motion` 設為 `true`。

介面的設計系統（色票、字級、間距、元件）記錄在 [DESIGN.md](DESIGN.md)；
產品定位與設計原則記錄在 [PRODUCT.md](PRODUCT.md)。開發時可用元件畫廊檢視：

```powershell
python tools\preview_ui.py            # 元件畫廊（含所有狀態與兩種主題）
python tools\preview_ui.py --audit    # 對比度檢查，未達 WCAG AA 會回傳非零
```

## 安裝（Windows／PowerShell）

### 1. 安裝 Python 與專案依賴

以下步驟以 **Python 3.12 64-bit** 為基準（開發測試使用的版本）。安裝 Python 時包含 Tcl/Tk 與 Python Launcher。下載／解壓本專案後，在專案根目錄開啟 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

依賴包含 `tkinterdnd2`、`psutil`、`openai`、`httpx`（`requirements.txt` 記錄實際測試版本與相容性依據）。模型與外部執行檔需另行準備。

macOS／Linux 先使用 Python 3.12 的原始碼流程。macOS 需要 Tcl/Tk（Homebrew Python 通常已附）；Ubuntu 需要 `python3-tk`。建立虛擬環境後使用相同的 `pip install -r requirements.txt`，再把 `llama-server`、`whisper-cli`、`ffmpeg` 與 `ffprobe` 的完整路徑填入介面；留空的 FFmpeg 工具會從 PATH 尋找。

`llama-server`（llama.cpp）與 `whisper-cli`（whisper.cpp）都可以使用 Vulkan build，這兩者是各自的推論 backend。macOS 通常由 MoltenVK 將 Vulkan 轉接到 Metal，Linux 可使用原生 Vulkan；目前仍需按工具、作業系統與 GPU 實機驗收。FFmpeg／ffprobe 只負責影音預處理，自己的解碼 hwaccel 不等於 llama.cpp 或 whisper.cpp 已啟用 Vulkan。

### 2. 準備外部工具

| 工具 | 取得方式 | 本程式用途 |
| --- | --- | --- |
| `llama-server.exe` | [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases) | 載入文字 GGUF 模型並提供翻譯 API |
| `whisper-cli.exe` | [whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases) | 載入 Whisper GGML 模型並轉錄 |
| `ffmpeg.exe`、`ffprobe.exe` | [FFmpeg 下載頁](https://ffmpeg.org/download.html)的 Windows builds | 音訊預處理與長音訊時長探測 |

llama.cpp 與 whisper.cpp 請分別選擇配合電腦與 GPU 的執行版本，保留下載套件內的 DLL／共享庫。FFmpeg 的 `bin` 目錄可加入 PATH，重新開啟 PowerShell 後確認：

```powershell
ffmpeg -version
ffprobe -version
```

本程式另有可選的 **DFlash** 加速設定；首次設定先關閉。啟用時需使用支援本程式 `--spec-type draft-dflash` 等參數的 llama-server 與相容 draft 模型，不能假設任何下載版都支援。

### 3. 設定工具與文字模型路徑

程式會在使用者資料目錄尋找工具與模型；未設定時不再使用作者機器的固定路徑。最簡單的方式是在「伺服器設定」選擇模型目錄與 `llama-server`，再在「語音轉錄」選擇 `whisper-cli`、FFmpeg、ffprobe。也可在 `config.json` 的 `paths` 區段填入自己的路徑，例如：

沒有既有設定的首次啟動會自動開啟「設定精靈」；日後可在「伺服器設定」按同名按鈕重新開啟。

```json
{
  "paths": {
    "base_dir": "C:\\AI\\llama.cpp",
    "runtime_dir": "C:\\AI\\llama.cpp\\bin",
    "model_dir": "C:\\AI\\models\\llm",
    "server_exe": "C:\\AI\\llama.cpp\\bin\\llama-server.exe",
    "ffmpeg": "C:\\AI\\ffmpeg\\bin\\ffmpeg.exe",
    "ffprobe": "C:\\AI\\ffmpeg\\bin\\ffprobe.exe"
  }
}
```

把文字模型的 `.gguf` 放入 `model_dir`，或啟動後在「伺服器設定」按「新增」選取模型。LLM 的 GGUF 與 Whisper 的 GGML `.bin` 是兩種不同模型，不能互換。

### 4. 下載日文 Whisper 推薦模型

**推薦：`kotoba-whisper-v2.0`。** 官方提供 [kotoba-whisper-v2.0-ggml](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml)，可直接供 whisper.cpp 使用，無需自行轉換原始權重。

在專案根目錄執行以下下載指令，或從官方模型頁手動下載同名檔案：

```powershell
New-Item -ItemType Directory -Force .\models\whisper
curl.exe -L --fail --output .\models\whisper\ggml-kotoba-whisper-v2.0.bin https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/resolve/main/ggml-kotoba-whisper-v2.0.bin
```

官方亦提供 `ggml-kotoba-whisper-v2.0-q5_0.bin` 量化版本，可按需要選用。兩個檔案都在同一個[官方模型頁](https://huggingface.co/kotoba-tech/kotoba-whisper-v2.0-ggml/tree/main)。請選 GGML `.bin`，不要把原始 Transformers 權重放進模型清單。

### 5. 啟動並設定 Whisper

```powershell
.\.venv\Scripts\python.exe llama_manager.py
```

打開「語音轉錄」→ **Settings**：

1. **whisper-cli**：按 Browse，選擇 `whisper-cli.exe`（macOS/Linux 選無副檔名的可執行檔）。
2. **Model Dir**：選剛才的 `models\whisper` 目錄，再按 **Scan**。掃描目錄內的 `.bin` 檔，請勿放在更深層子目錄。
3. **Model**：選 `kotoba-whisper-v2.0`（清單會移除檔名的 `ggml-` 前綴與 `.bin`）。
4. **Language**：日文音訊選 **ja - Japanese**。
5. **Threads**：預設 8，可按電腦資源調整。
6. 長音訊可勾 **Chunk long audio (30 min, anti-repeat)**。設定會存入 `config.json`，管線共用 Whisper 設定。

「語音轉錄」的 FFmpeg 與 ffprobe 欄位支援完整路徑及檢查；留空代表使用 PATH。這兩個欄位與工作台共用同一個轉錄器，修改後下一次工作即生效。

日後可繼續用上述 Python 指令啟動，不需啟用虛擬環境。`start.bat` 使用 PATH 中的 `python`，不會自動選 `.venv`，而且只檢查 `psutil`；使用該捷徑前，請確保同一個 Python 已安裝完整 `requirements.txt`。

## 使用教學

### 日文影片 → 繁體中文字幕（一鍵管線）

1. 先依安裝步驟設定 `kotoba-whisper-v2.0` 與 `whisper-cli`。
2. 到「伺服器設定」選文字 GGUF 模型，按 **翻譯模式** preset，再啟動伺服器。等待模型載入，檢查日誌是否有錯誤。
3. 到「工作台」的管線卡片，按 **Choose Files** 或拖入日文影片，檔案會出現在 **Pending**。
4. **Whisper Model** 選 `kotoba-whisper-v2.0`，**Language** 選 `ja - Japanese`，**Target** 選 `zh-tw - Traditional Chinese`。
5. 初次使用先不勾 **Replace original**，按 **Start**。
6. 程式自動抽取音訊、產生日文 SRT，再交給 LLM 翻譯。成功檔案移到 **Completed**；失敗檔案留在 Pending，診斷日誌會記錄原因。

第一次啟動時，伺服器程序可能已執行但模型仍在載入。Quick Whisper 遇到 HTTP 503 或暫時連線失敗會顯示 **Waiting for llama-server**，在約 120 秒的等待期限內重試，準備好便自動繼續。Stop 可取消等待（正在進行的短 HTTP 檢查需先返回）。超時或其他無法重試的錯誤會顯示 **Cannot start** 與原因，不會誤報為使用者停止。

例如輸入 `movie.mp4`，會在來源目錄產生：

```text
movie.srt                       # 日文轉錄
movie_Traditional Chinese.srt   # 繁體中文翻譯
```

**Replace original** 在媒體管線中是覆寫產生的 `.srt`，不會覆寫影片。Whisper 轉錄本身會在成功時更新同名 `.srt`。管線的 **Clear** 會清空 Pending 與 Completed 清單；Stop 會取消目前轉錄，翻譯期間則需等待當前翻譯呼叫返回。

### 批量漫畫 OCR → 本地翻譯（Beta）

**不穩定功能：可能執行失敗或產生不準確的結果，請人工核對輸出。**

開「漫畫 OCR／翻譯」，選 llama-server executable，按「枚舉／診斷」選 Vulkan device，再選 OCR GGUF＋matching mmproj 與翻譯 GGUF。加入圖片／資料夾、調整頁序與文件分組，選語言後按「執行」。漫畫路徑與設定全部可由 GUI 設定，不跟隨全域 remote profile。

支援只做 OCR、OCR 後翻譯、重用 OCR 重新翻譯；可選先校對。每頁／每批立即 atomic checkpoint，原圖唯讀。空結果需要人工確認無文字；不完整文件只發布帶缺漏標記的 partial 譯文。重啟後按「載入工作記錄」選輸出根目錄 `manga-job.json`，再續跑。

第一個候選為 `ggml-org/GLM-OCR-GGUF:Q8_0` 與匹配 projector。**目前完整真機 Vulkan OCR＋翻譯與漫畫品質驗收均為 NOT RUN**；runtime 裝置可見、模型載入或 mocks 通過不等於真機通過。沒有消字、嵌字或翻譯後漫畫圖片。

安裝、模型配對、校正、輸出、限制及 10–20 頁驗收步驟見 [漫畫 OCR Vulkan 文件](docs/manga-ocr-vulkan.md)。本機驗證入口：`python tools/verify_manga_vulkan.py --help`。

### 只做語音轉錄

1. 到「語音轉錄」完成 Settings 設定。
2. 在 File Selection 選音訊／影片（例如 mp4、mkv、mp3、wav、flac、m4a）。按 **Add Directory** 可一次加入整個目錄下所有支援的檔案（含子目錄），重複檔案會自動略過。勾 **Skip existing SRT** 後，掃目錄與開始轉錄都會跳過已有同名 `.srt` 的檔案（設定存於 `config.json`）。
3. 日文選 `ja - Japanese`，按 **Start Transcription**。
4. 程式以 FFmpeg 轉成 16kHz 單聲道 WAV，再交給 whisper-cli 產生同名 SRT；完成後也會加入「字幕翻譯」檔案清單。

這個流程不需要 llama-server。短檔案可先單次轉錄確認結果；超過 30 分鐘的檔案勾選分段後，會獨立轉錄各段並合併時間軸。

### 只翻譯已有字幕

1. 在「伺服器設定」啟動文字模型。
2. 到「字幕翻譯」，按 **Choose Files** 或拖入 `.srt`／`.txt`。
3. 選 **Target**（例如繁體中文）；來源文字由模型辨識，Source 選單目前不會作為明確的來源語言指令送入 prompt。
4. **Replace original** 不勾時另存帶目標語言名稱的檔案；勾選則覆寫輸入字幕／文字檔。
5. 按 **Start Translation**。Stop 會在目前的模型 request 返回後阻止下一個分析／翻譯 request，並且不寫入未完成檔案。

翻譯模式可選：

- **標準翻譯**：沿用原本的批次翻譯與全檔相同文字去重。
- **全文理解翻譯**：先按時間順序讀完一份 SRT，整理精簡的話題／故事、人物稱呼、術語、語氣及不確定資訊，再把同一背景與附近原文加入每批翻譯及補譯。這個選項可分別搭配快速模式或兩步模式。

全文理解模式只適用於 SRT，也適用於工作台的「Whisper → SRT → 翻譯」。TXT 會清楚記錄後使用原本的標準翻譯路徑。長 SRT 若無法放進單一 context window，會按連續 cue 分段全部讀取，再逐層合併背景；「每份檔案分析一次流程」因此不代表永遠只發出一個 API request，也不會只抽樣開頭或靜默截尾。

全文分析、合併與 JSON 修復共用同一份完整欄位規格。若模型的 chat template 支援 thinking 控制，這些結構化分析請求會逐次停用 thinking，將輸出額度留給 JSON；不需重啟伺服器，也不會改動一般字幕翻譯的設定。

支援標準 `response_format` 的伺服器會以 strict JSON Schema 約束背景輸出：最多保留 4 位人物、6 個術語、2 個語氣及 2 項不確定資訊，優先選擇重複出現且會影響翻譯的內容。這些上限只壓縮背景，不會少讀、截斷或略過來源字幕；程式仍會追蹤並驗證全文涵蓋範圍，也不會把不合規項目靜默刪除。伺服器拒絕 schema 或輸出仍無法驗證時，分析會明確失敗。

分析與翻譯會分別顯示進度。Stop 會在模型請求之間生效；已送出的同步 HTTP request 仍需等待回覆或 timeout。分析失敗、取消或翻譯不完整時不會寫入輸出，也不會沿用上一個檔案的背景或自動降級成標準模式。

工作台管線也接受 SRT，會直接跳過 Whisper；一般文字 `.txt` 兩個入口都走同一條全文翻譯路徑：整份文件一次請求、先檢查 context 預算、以 finish reason 判斷完整性，不會經過 SRT 解析或逐句補譯。超過 context window 的 TXT 會明確失敗（檔案與既有輸出保持原狀），不會截斷或分段；本程式不提供無上限的長文件翻譯。

### 翻譯設定

開始翻譯時，檔案清單、目標語言、Replace original、模型／端點、翻譯模式與生成設定會在 UI 執行緒凍結成不可變的執行快照（Run Snapshot）：執行中切換分頁或修改設定只會影響下一輪，preflight 探測到的並行數與 context 容量也只套用到當輪。

「伺服器設定」的 **翻譯模式** preset 設為 16,384 context、3 個並行槽、512 batch、FlashAttention 與 q8_0 KV cache，並同步翻譯 Workers。這是程式內建起始設定，可再按模型與硬體調整。

「字幕翻譯」→ **Show Advanced**：

| 選項 | 預設值 | 說明 |
| --- | --- | --- |
| Batch Size | 15 | 每次請求的字幕句數 |
| Temperature | 0.2 | 生成隨機度 |
| Max Tokens | 16384 | 輸出 token 上限；實際請求會按字幕句數縮小預算 |
| Workers | 3 | 批次並行數；啟動翻譯前會依伺服器可用槽調整 |
| 快速模式 | 開啟 | 單步輸出 `translation`；關閉後要求 `step1` 直譯及 `step2` 意譯 |
| 翻譯模式 | 標準翻譯 | `全文理解翻譯` 先分析整份 SRT；與快速／兩步設定互相獨立 |

快速模式日誌顯示 `translation="..."`；兩步模式顯示 `literal="..." -> paraphrase="..."`。快速模式沒有要求直譯欄位，因此不顯示空白的直譯。兩步模式會產生更多輸出，實際品質差異需用自己的字幕比較。System 與 user prompt 的指令均為英文，**不會因此把目標語言改成英文**。

全文理解背景只能協助消除歧義與統一術語；原句及附近原文永遠優先。模型仍可能誤判摘要、人物或關係，功能也不能修正 Whisper 已聽錯的原文。Story 模式會增加分析時間與 request 數，同文在不同場景亦會分開翻譯，因此通常比標準模式耗時。

若要評估實際品質，請固定同一模型、同一 SRT、目標語言、Temperature 與快速／兩步設定，各跑一次標準及全文理解翻譯。人工抽查約 30–50 個省略主語、跨批次指代、重複人名／術語及跨場景同句，分別記錄錯指代、稱呼不一致、誤補劇情、漏句與總耗時。Mock 測試只驗證管線與 prompt contract，不代表翻譯準確度已提升。

### LLM 連線：本地 llama-server 或遠端 API

「字幕翻譯」分頁的 **LLM 連線** 選擇翻譯使用的後端，**字幕翻譯** 與 **Quick Whisper → Translate** 兩處共用同一選擇：

- **本地 llama-server**（預設）：沿用既有流程 — 啟動前探測 `/props`、按伺服器槽數調整 Workers、等待模型載入。
- **遠端 API**：使用已儲存的 OpenAI 相容端點設定檔，不需要本地 llama-server。Workers 使用設定檔的並發數，不探測 `/props`、不等待冷啟動；設定不完整（缺少 Base URL、模型或金鑰）會在啟動前以設定檔名稱提示。

點 **設定...** 管理設定檔（名稱、Base URL、模型、API Key 環境變數、Proxy、並發數）。Base URL 貼上整條 `/v1/chat/completions` 或 `/chat/completions` 結尾會自動歸一化。API Key 不會存入 `config.json`：填 **API Key 環境變數** 名稱供每次啟動讀取，或在設定檔對話框輸入 **Session API Key**（僅本次執行有效，優先於環境變數）。

### 日誌與失敗處理

工作台右上角 **診斷日誌** 可查看所有分頁的詳細訊息。分頁內的日誌只顯示對應工作。

翻譯請求的傳輸層重試只有一個擁有者：SDK 用戶端以 `max_retries=0` 建立，重試完全由內建的傳輸政策（`translation/transport.py`）決定 — 最多 3 次 HTTP 嘗試、30 秒累計預算、指數退避並尊重（有上限的）`Retry-After`。認證／權限／模型不存在／不支援參數等錯誤立即失敗；context 超限視為請求過大，由呼叫端縮小請求而非原樣重送。同一輪執行中伺服器持續失敗時，整輪中止，不會展開成逐句請求。

漏掉的字幕會自動逐句補譯；補譯仍失敗時，該檔案不會儲存為成功結果，也不會覆寫既有輸出。請找出 `Batch ... failed` 或 `L12: ... retry failed` 的具體錯誤。管線會保留失敗檔案供重新執行。

輸出檔案與 `config.json` 都以暫存檔＋原子替換寫入：寫入或替換失敗時舊檔案內容保持原狀。非空但解析不到任何字幕的 SRT 會明確失敗，不會寫出空的「成功」輸出（勾選 Replace original 時尤其重要）。

## 🧪 開發與測試

介面使用共用的主題系統（預設深色），工作台、伺服器設定、字幕翻譯、語音轉錄與漫畫 OCR／翻譯五個分頁保留既有工作流程。伺服器參數分組排列，日誌區隨視窗伸展。

日誌顯示每 80ms 最多處理 200 筆，以一次 Tk 呼叫批次插入。各顯示佇列最多保留 4,000 筆待處理訊息；超量會略過最舊訊息並顯示略過筆數。向上捲動閱讀時會保留閱讀位置。此優化針對介面回應速度與日誌記憶體用量，不改變模型推論參數。

`tests/test_desktop_ui.py` 使用實際 Tk 元件驗證版面與日誌生命週期；環境無法初始化 Tk 時會跳過。

```powershell
# 跑全套測試（pytest + coverage）
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest

# 跳過翻譯子套件的緩慢測試
.\.venv\Scripts\python.exe -m pytest --ignore=tests/translation --no-cov -q
```

歷史漫畫 MVP 測試記錄（2026-09-18，非目前工作樹基線）：

- Baseline：`python -m pytest --basetemp=temp_test_dir_manga_baseline` → **731 passed**。
- 最終全套：`python -m pytest --basetemp=temp_test_dir_manga_delivery -rs` → **784 passed，0 failed，0 skipped**（45.26 秒，coverage 90%）。既有字幕、TXT、全文理解、Whisper、local／remote 測試均保留。
- `python tools/verify_manga_vulkan.py --help`、本機 `--inspect-only` 已執行。完整真機 OCR＋翻譯與漫畫品質仍為 **NOT RUN**，原因及驗收程序見 [完整紀錄](docs/manga-ocr-vulkan.md)。

目前工作樹的測試紀錄（macOS、Python 3.14.7、`.venv` 安裝 `requirements-dev.txt`，排除 10 個需要 Tk 的 GUI 測試模組）：**671 passed、13 failed**。13 個失敗全部源自環境，不是本次改動：12 個來自 `subtitle_tab` 延遲 import 觸發缺少的 `_tkinter`，1 個是 `test_srt_parser.py::test_windows_backslashes` 在 POSIX 上沒有 `C:` drive letter。真 Tk 測試與完整數字仍需 Python 3.12＋Tcl/Tk 的 CI（見 `.github/workflows/test-matrix.yml`）。

主要架構模組：

- `log_bus`（在 `ui_helpers.py`）- 統一的日誌 pub/sub，所有分頁共用
- `FileListbox`（`file_listbox.py`）- 共用的檔案清單 widget（清單 + 瀏覽 + 拖放 + 副檔名過濾）
- `TranscriptionRunner`（`transcription_runner.py`）- Whisper 與管線共用的轉錄生命週期
- `ToolManager`／`SetupWizard`（`tool_manager.py`、`setup_wizard.py`）- 外部工具路徑檢查與首次設定
- `build_translation_config`（`config_helpers.py`）- 翻譯設定的單一來源
- `output_path_for`（`utils/srt_parser.py`）- 翻譯輸出路徑的單一來源

## 📁 配置檔案

### config.json

設定儲存在使用者資料目錄的 `config.json`，不會寫入程式安裝目錄。預設位置是 Windows `%LOCALAPPDATA%/llama-manager`、macOS `~/Library/Application Support/llama-manager`、Linux `$XDG_DATA_HOME/llama-manager`（未設定時 `~/.local/share/llama-manager`）。目前模型清單也存於這個檔案的 `models.models` 與 `whisper_models.models`；根目錄的 `models.json` 不是目前 registry 的讀取來源。從舊版本升級時，程式會首次啟動時匯入程式目錄的舊 `config.json`。

版本庫內的 `config.default.json` 與 `models.default.json` 只有空白預設值；`config.json`／`models.json` 是執行時狀態，不能拿來當發布設定。

主要設定群組：

- `server`：port、GPU 層數、context、並行槽與加速選項。
- `whisper`：CLI 路徑、模型目錄、語言、執行緒數、長音訊分段選項、跳過已有 SRT 選項。
- `paths`：llama-server、模型、FFmpeg 與 ffprobe 的完整路徑；FFmpeg／ffprobe 留空時會嘗試使用系統 PATH。
- `ui`：最後選擇的文字模型、目標語言及翻譯參數。
- `ui.context_mode`：`none`（標準翻譯）或 `story`（全文理解翻譯）；舊設定缺少時使用 `none`。
- `llm`：翻譯後端 — `mode`（`local`／`remote`）、`active_profile_id` 與 `profiles`（遠端 OpenAI 相容端點設定檔）。設定檔只存 `api_key_env` 環境變數名稱，不存金鑰本身；舊設定缺少時使用 `local`。
- `pipeline`：管線語言、目標語言與覆寫選項。

一般透過 GUI 設定即可。手動編輯前先關閉程式；路徑變更會在下一次工作啟動時生效。

## 🔌 API 使用

伺服器啟動後，可以透過以下方式使用：

### Python (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="no-key"
)

response = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "你好！"}]
)

print(response.choices[0].message.content)
```

### cURL

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "你好！"}]
  }'
```

## 🛠️ 故障排除

### 翻譯漏句或出現「retry failed」

實際使用的 prompt 由 `translation/prompt_builder.py` 產生（`translation/prompts/srt_translation.txt` 是未載入的舊參考模板）。System 與 user prompt 的指令統一使用英文；原文與譯文仍依所選語言處理。單句、批次與補譯共用目標語言及輸出要求：每個 ID 回覆一次、只輸出指定 YAML 欄位、保留原意與語氣、不合併或自行補完字幕。單步輸出 `translation`；兩步輸出 `step1` 和 `step2`。原文字串會跳脫引號與換行，Context 只作背景資料。這些規則改善指令一致性，但實際模型的格式遵從與譯文品質仍需以樣本驗證。

模型回覆漏掉字幕 ID 或編號時，程式只補譯缺失的句子。回覆前的開場白與 Markdown 標記不會當成第一句譯文。

如果補譯仍失敗，日誌會列出行號與錯誤原因（例如 `L12: RuntimeError: ...`），該檔案不會寫入或覆蓋輸出；管線中仍留在 Pending，介面顯示失敗檔案數。修復伺服器或調整設定後可重新執行。這避免先前「保留原句但回報完成」或儲存空白字幕的情況。

排查時請保留 `Batch ... failed`、`retry failed` 附近的日誌：連線／timeout、context 超限、回覆格式問題需要不同處理方式，單靠「保留原句」無法判斷上游原因。批次回覆的字幕 ID 會經過驗證：重排的明確 ID 依 ID 對應；重複 ID 保留第一筆；缺少 ID 或混合明確／缺少 ID 的回覆不會用位置猜測，而是交給逐句補譯，避免譯文錯位。

全文分析 JSON 若因 `finish_reason=length` 截斷，程式會縮小輸入並作有上限的分段分析；若 JSON 為空、格式無效或超過欄位上限，每次請求最多作一次格式修復，仍無法驗證便令該檔失敗，不會把半份摘要送入翻譯。容量使用 llama-server `/props` 回報的每個 request／slot `n_ctx`；不會把輸出用的 Max Tokens 當 context，也不會再按 Workers 除一次。舊 build 未回報容量時會在日誌明示使用保守估算；最小合法請求仍放不下時會明確失敗，不截斷原句。

### 伺服器無法啟動

1. **檢查模型路徑**: 確保選擇的模型檔案存在
2. **檢查連接埠佔用**: 確保連接埠未被其他程式佔用
3. **查看日誌**: 查看日誌輸出了解錯誤訊息

### GPU 無法使用

- 檢查 `GPU 層數` 設定是否為 0 (應設定為 99)
- 確認 GPU 驅動與所用執行後端相容（依下載／編譯的版本而定）
- 查看執行檔目錄中的 DLL 檔案是否完整

### 顯示編碼問題

如果中文顯示為亂碼，請確保：
- 系統預設編碼為 UTF-8
- 終端機支援中文字型

### Whisper 只轉出同一句 / 字幕重複

若 Whisper 轉錄只產生一句重複的字幕，通常是 whisper-cli 分段參數過度限制所致：

- `--max-len`：限制每段最大字元數。設成 `200` 會強制合併，造成多句被擠成一段或重複輸出。改為 `0`（關閉限制）交由 whisper 內建邏輯分段即可。
- `--suppress-nst`：抑制非語音 token，在中文或某些語言上會誤刪正常 token，導致「只剩同一句」。建議移除。
- `--entropy-thold` / `--logprob-thold`：閾值過嚴會丟棄正常 token，必要時可適度放寬（例如 `entropy-thold` 提高到 `2.8`）。

#### 長音訊（2 小時以上）跑到一半出現大量重複

這是 whisper.cpp 經典的 **context collapse / 解碼無限迴圈**：處理到某個時間點後解碼器卡在 token 循環，後續全部變成同一段重複。根因是模型把前面已產生的（錯誤）token 當成 prompt 上下文繼續餵給自己，錯誤一路累積最終崩潰。

目前 `whisper_transcription.py` 的 whisper-cli adapter 已加入以下抗重複參數：

- `--max-context 0`：不保留前一個 segment 的文字 context，避免錯誤跨段累積（對長音訊最重要）。
- `--temperature 0.0` + `--no-fallback`：使用固定的貪婪解碼，不進入逐步提高 temperature 的 fallback，避免隨機發散把迴圈帶回來。
- `--max-len 0`：關閉最大段長限制。

若上述仍無法解決，可再嘗試：

- 換**較大**的模型（例如 `large-v3`），小模型在長音訊上特別容易崩潰。
- 把音訊切成 30 分鐘以下的段落分批轉錄，再合併 SRT。

#### 內建分段轉錄（推薦用於長音訊）

語音轉錄分頁與工作台管線卡片提供了 **「Chunk long audio (30 min, anti-repeat)」** 勾選框。勾選後：

1. `whisper_transcription.py` 先用 `ffprobe` 探測時長；30 分鐘以下仍使用單次轉錄，只有超過 30 分鐘才進入 chunked transcription。
2. 長音訊由 ffmpeg 產生 30 分鐘的 16kHz mono WAV chunk，每個 chunk 各自獨立跑一次 whisper-cli，避免跨段 context collapse。
3. module 將 chunk SRT 依時間偏移量合併、重新編號，再以 transactional replace 發佈最終 `.srt`；取消或失敗不會覆蓋既有的完整字幕。

設定會儲存到 `config.json` 的 `whisper.chunk_long_audio`，管線也會讀取同一個設定。Whisper 分頁與管線共用同一個 synchronous transcription lifecycle，Stop 會立即取消目前的外部 process 並清理暫存檔。

## 📊 系統需求

- **作業系統**: Windows 10/11；macOS Apple Silicon、Ubuntu 24.04 先支援原始碼執行與手動工具配置
- **Python**: 建議 Python 3.12 64-bit（目前測試版本）
- **GPU／RAM／VRAM**：需求取決於文字模型、Whisper 模型及所用執行後端；沒有固定必須使用的 GPU 型號。

## 🔧 進階選項

### 打包為 Windows 發布包

如需打包為獨立可執行檔：

```bash
pip install -r requirements-dev.txt
python packaging/build.py --zip
```

產生的 `dist/llama-manager/` 目錄包含可執行檔及 Tcl/Tk、tkinterdnd2 等資源；Windows、macOS、Linux 必須在各自平台分別建置。正式發布前仍需驗證外部 llama／Whisper／FFmpeg 工具及其共享庫。

## 📝 更新日誌

### v1.3.0 (2026-06-25)
- 🏗️ **架構深化**：抽出 LogBus、FileListbox、BatchRunner、output_path_for、統一翻譯設定
- 🧪 新增 62 個測試（全套 375 全綠），涵蓋 log_bus、config_helpers、file_listbox、batch_runner、output_path、translator 初始化
- 📝 README 補上 Whisper / 翻譯 / 管線說明

### v1.2.0 (2026-06)
- ✅ Whisper 語音轉錄分頁（whisper.cpp + ffmpeg 預處理）
- ✅ 一鍵管線分頁（Whisper → 翻譯）
- ✅ 抽出 ServerTab、PipelineCard、PipelineRunner、prompt_builder
- ✅ 翻譯加速：批次並行、單步模式、加大 batch size
- ✅ 持久化上次選擇的模型與語言

### v1.1.0 (2026-05)
- ✅ 字幕翻譯分頁（本地 LLM，兩步直譯→意譯，OpenAI 相容 API）

### v1.0.0 (2026-04-28)
- ✅ 初始版本發布
- ✅ 支援模型管理
- ✅ 支援伺服器控制
- ✅ 支援即時日誌顯示
- ✅ 支援資源監控

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 授權條款

MIT License

## 🙏 銘謝

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 核心推論引擎
- [llama-swap](https://github.com/mostlygeek/llama-swap) - 靈感來源

---

*Created with ❤️ by Claude Code*
