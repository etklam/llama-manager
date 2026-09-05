# llama.cpp Manager 🚀

簡單易用的 GUI 管理器，用於管理 llama.cpp 伺服器、模型、Whisper 語音轉錄與字幕翻譯。

## ✨ 功能特點

- ✅ **圖形化界面** - 簡潔直觀的 GUI（伺服器、Whisper、字幕翻譯、管線四個分頁）
- ✅ **模型管理** - 自動掃描和新增 GGUF 模型
- ✅ **伺服器控制** - 一鍵啟動/停止 llama-server
- ✅ **參數配置** - 視覺化配置伺服器參數
- ✅ **Whisper 語音轉錄** - 透過 whisper.cpp 將音訊/視訊轉成 SRT 字幕（自動 ffmpeg 預處理）
- ✅ **字幕翻譯** - 透過本地 LLM（OpenAI 相容 API）做兩步翻譯（直譯→意譯），支援批次並行
- ✅ **一鍵管線** - 媒體檔案自動「Whisper 轉錄 → 字幕翻譯」一氣呵成
- ✅ **即時日誌** - 統一的 log_bus pub/sub，顯示伺服器、轉錄、翻譯日誌
- ✅ **資源監控** - 監控 CPU/RAM 使用情況
- ✅ **配置儲存** - 自動保存配置、模型清單、上次選擇的模型與語言

## 🚀 快速開始

### 方法 1: 雙擊啟動 (推薦)

直接雙擊 `start.bat` 即可啟動！

### 方法 2: 命令行啟動

```bash
# 安裝依賴
pip install -r requirements.txt

# 執行管理器
python llama_manager.py
```

### 外部依賴

- **llama-server** - llama.cpp 編譯產物（伺服器分頁要用）
- **whisper-cli** - whisper.cpp 編譯產物（Whisper 分頁、管線分頁要用）
- **ffmpeg** - 必須在 PATH 中（Whisper 自動將音訊/視訊轉成 16kHz WAV）

> 三者都不在 repo 內，請自行編譯或下載預編譯版。

## 📖 使用說明

主視窗有四個分頁：**伺服器**、**Whisper**、**翻譯**、**管線**。

### 1. 伺服器分頁 — 模型與 llama-server

#### 模型選擇（📦 模型選擇）

- **🔄 掃描**：自動掃描 `llama-hip` 目錄下的 `.gguf` 檔案
- **📂 新增**：手動挑選模型檔
- 選單下方顯示模型大小與量化格式（Q4_K_M、Q5_K_S…）

#### 伺服器設定（⚙️ 伺服器設定）

- **端口**：API 連接埠（預設 8080）
- **GPU 層數**：卸載到 GPU 的層數（0–99，預設 99）
- **上下文大小**：文字上下文長度（512–16384，預設 4096）
- **批次大小**：批處理大小（預設 512）

#### 啟動 / 停止 / 釋放

1. 選好模型、調好參數
2. 點 **▶️ 啟動伺服器**
3. 下方日誌顯示啟動過程，狀態變成「● 執行中」即可使用 API
4. 點 **⏹️ 停止伺服器** 或關閉視窗可終止
5. **🧹 釋放記憶體**：伺服器停止後清出 VRAM/RAM（切換大模型前用）
6. 右下角即時顯示 GPU / VRAM / RAM 用量
7. Debug Log 勾選框會開啟獨立視窗顯示詳細日誌（所有分頁的 log 都會匯流到這裡）

### 2. Whisper 分頁 — 語音轉錄

> 需要先在設定區填好 `whisper-cli` 路徑與 GGML 模型目錄；`ffmpeg` 必須在 PATH。

1. **Settings**：設定 whisper-cli 路徑、Whisper 模型、模型目錄（點 **Scan** 掃描 `.bin` 模型）
2. **File Selection**：拖放或點 **Browse** 選音訊/視訊檔（mp4、mkv、avi、mp3、wav、flac、m4a…）
3. 選 **Language**（或 Auto Detect）與 **Threads**（執行緒數，預設 8）
4. 點 **Start Transcription**
   - ffmpeg 會先把輸入轉成 16kHz 單聲道 WAV（視訊檔會先抽出音軌）
   - 再交給 whisper-cli 產生 SRT
5. 完成後 **Completed**，SRT 路徑顯示在日誌，可直接拖到翻譯分頁

### 3. 翻譯分頁 — 字幕翻譯

> 需要先在伺服器分頁啟動 llama-server 並選好模型。

1. **File Selection**：拖放或 **Browse** 選 `.srt` / `.txt`
2. **Source → Target**：選來源語言與目標語言（簡中/繁中/英/日/韓/法/德/西/葡/俄/阿拉伯/印地/泰/越/義/荷，共 15+ 語言）
3. **Model**：顯示目前用的模型；選 "(no model loaded)" 會無法翻譯
4. **Replace original**：勾起來則直接覆寫原檔；不勾則輸出 `<原名>_<目標語言全名>.<原副檔名>`（例如 `video_Simplified Chinese.srt`）
5. **Show Advanced**：展開後可調
   - **Batch Size**（預設 15）
   - **Temperature**（預設 0.2）
   - **Max Tokens**（預設 16384）
   - **並行數** / Workers（預設 3，批次並行數；按鈕原文為「并发數」）
   - **快速模式**（按鈕原文：`快速模式 (跳过直译, 仅意译)`）：勾起來只跑意譯一步，較快但品質略降
6. 點 **Start Translation**，日誌顯示每批 / 每行進度，**Stop** 可中途停止

### 4. 管線分頁 — 一鍵 Whisper → 翻譯

> 需要同時設定好 whisper-cli、模型、且伺服器已啟動。

1. 拖放或 **Browse** 選媒體檔（支援音訊與視訊副檔名聯集）
2. 選 **Language**（來源）與 **Target**（目標語言）
3. **Whisper Model**：選 whisper 模型
4. **Replace original**：是否覆寫原檔
5. 點 **▶ Start**：自動依序執行「Whisper 轉錄 → 翻譯」，**⏹ Stop** 可停


## 🧪 開發與測試

介面使用共用的淺色主題，工作台、伺服器設定、字幕翻譯與語音轉錄四個分頁保留既有工作流程。伺服器參數分組排列，日誌區隨視窗伸展。

日誌顯示每 80ms 最多處理 200 筆，以一次 Tk 呼叫批次插入。各顯示佇列最多保留 4,000 筆待處理訊息；超量會略過最舊訊息並顯示略過筆數。向上捲動閱讀時會保留閱讀位置。此優化針對介面回應速度與日誌記憶體用量，不改變模型推論參數。

`tests/test_desktop_ui.py` 使用實際 Tk 元件驗證版面與日誌生命週期；環境無法初始化 Tk 時會跳過。

```bash
# 跑全套測試（pytest + coverage）
python -m pytest

# 跳過翻譯子套件的緩慢測試
python -m pytest --ignore=tests/translation --no-cov -q
```

主要架構模組：

- `log_bus`（在 `ui_helpers.py`）- 統一的日誌 pub/sub，所有分頁共用
- `FileListbox`（`file_listbox.py`）- 共用的檔案清單 widget（清單 + 瀏覽 + 拖放 + 副檔名過濾）
- `BatchRunner`（在 `ui_helpers.py`）- 共用的批次任務生命週期（按鈕切換、停止旗標、進度回報）
- `build_translation_config`（`config_helpers.py`）- 翻譯設定的單一來源
- `output_path_for`（`utils/srt_parser.py`）- 翻譯輸出路徑的單一來源

## 📁 配置檔案

### models.json
儲存模型清單：
```json
{
  "models": [
    {
      "name": "supergemma4-26b-uncensored-fast-v2-Q4_K_M",
      "path": "D:\\AI\\llama\\llama.cpp\\llama-hip\\supergemma4-26b-uncensored-fast-v2-Q4_K_M.gguf",
      "size": "16.80GB",
      "format": "Q4_K_M"
    }
  ]
}
```

### config.json
儲存伺服器配置：
```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0",
    "gpu_layers": 99,
    "context_size": 4096,
    "batch_size": 512,
    "threads": -1
  }
}
```

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

### 伺服器無法啟動

1. **檢查模型路徑**: 確保選擇的模型檔案存在
2. **檢查連接埠佔用**: 確保連接埠未被其他程式佔用
3. **查看日誌**: 查看日誌輸出了解錯誤訊息

### GPU 無法使用

- 檢查 `GPU 層數` 設定是否為 0 (應設定為 99)
- 確認 ROCm HIP 驅動正常安裝
- 查看 llama.cpp 目錄中的 DLL 檔案是否完整

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

Whisper 分頁與管線分頁的 Settings 區提供了 **「Chunk long audio (30 min, anti-repeat)」** 勾選框。勾選後：

1. `whisper_transcription.py` 先用 `ffprobe` 探測時長；30 分鐘以下仍使用單次轉錄，只有超過 30 分鐘才進入 chunked transcription。
2. 長音訊由 ffmpeg 產生 30 分鐘的 16kHz mono WAV chunk，每個 chunk 各自獨立跑一次 whisper-cli，避免跨段 context collapse。
3. module 將 chunk SRT 依時間偏移量合併、重新編號，再以 transactional replace 發佈最終 `.srt`；取消或失敗不會覆蓋既有的完整字幕。

設定會儲存到 `config.json` 的 `whisper.chunk_long_audio`，管線分頁也會讀取同一個設定。Whisper 分頁與管線共用同一個 synchronous transcription lifecycle，Stop 會立即取消目前的外部 process 並清理暫存檔。

## 📊 系統需求

- **作業系統**: Windows 10/11
- **Python**: 3.7 或更高版本
- **GPU**: AMD Radeon RX 7900 XTX (或其他 AMD GPU)
- **RAM**: 建議 32GB 或更多
- **VRAM**: 建議 24GB 或更多

## 🔧 進階選項

### 打包為 .exe

如需打包為獨立可執行檔：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name="llama-manager" llama_manager.py
```

產生的 `llama-manager.exe` 將在 `dist` 目錄中。

### 自動啟動伺服器

在 `config.json` 中新增：
```json
{
  "ui": {
    "auto_start": true,
    "auto_start_model": "your-model-name"
  }
}
```

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
