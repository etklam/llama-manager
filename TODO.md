# 新用戶安裝改善 TODO

## 目標與優先順序

目標平台：Windows 10／11 64-bit。

新用戶流程：下載 ZIP → 解壓 → 雙擊 EXE → 設定精靈準備工具及模型 → 成功輸出第一份字幕。

第一版設定精靈先覆蓋語音轉錄與字幕翻譯，再加入漫畫與 DFlash。

### 執行順序（2026-10-15 修訂）

原定先做第 3 項，但工具下載目錄取決於第 1 項的用戶資料目錄決定，而且目前測試套件在有 Tk 的機上係紅的，CI 一開即失敗。修訂為：

1. 第 0 節（立即修正）—— 細改動，係其餘工作的前提。
2. 第 2 節的 Windows CI（2.4）—— 開發機為 macOS，Tk 測試及打包都只能在 CI 驗證。
3. 第 1 節的 1.1–1.3（程式資源／用戶資料目錄、移除 D:\AI fallback）。
4. 第 3 節（精簡版，見該節範圍說明）。
5. 第 2 節其餘、第 4 節、第 5 節。

以下項目均尚未完成。

## 0. 立即修正（阻塞後續工作）

- [ ] 收窄 `requirements.txt` 的 `openai` 上限：現時 `>=2.0.0,<3.0.0` 會裝到 2.54，新版拒絕空 `api_key`，本地翻譯在 `OpenAIClient` 構造階段即 raise `OpenAIError: Missing credentials`（15 個 transport／txt 測試因此失敗）。
- [ ] `translation/openai_client.py` 對本地端點送出佔位金鑰（`self.api_key or 'local-only'`，與 `manga/runner.py` 一致），令新裝環境不依賴 SDK 版本行為。
- [ ] 修正 `tests/test_desktop_ui.py`：仍 reference UI 重構前已移除的 `_debug_after`、`_debug_queue`、`_debug_text`、`_poll_debug_queue`、`_on_tab_changed`，在有 Tk 的機上必定失敗。
- [ ] 更新 README 的測試紀錄（「784 passed，0 failed」已失效），重跑後記錄新數字。
- [ ] `server.host` 預設改為 `127.0.0.1`：現時 `0.0.0.0` 令首次啟動觸發 Windows 防火牆彈窗，並把模型服務曝露在 LAN；需要 LAN 存取改為明確 opt-in。
- [ ] 修正 `config_manager.py` `paths` 註解與實作不符：註解聲稱未設定時 Server 面板顯示 first-run 指引，但該指引不存在。

## 1. 整理設定及資料目錄

- [ ] 區分程式資源目錄與可寫入的用戶資料目錄，兼容原始碼執行及打包後執行。現時 `ConfigManager` 讀寫 `Path(__file__).parent / "config.json"`，打包後落在 `_internal/`：Program Files 無寫入權限，升級亦會覆蓋。
- [ ] 移除新安裝對作者 `D:\AI\...` 路徑的依賴；未設定時提供可操作的提示。注意 `_resolve_paths` 會把 fallback 寫回 config，令「未設定」狀態無法持續存在。
- [ ] `ServerController.start()` 增加 `llama-server.exe` 存在性前置檢查；現時只檢查模型路徑，執行檔缺失時拋出未經處理的 `FileNotFoundError`。
- [ ] `BaseModelRegistry.scan()` 在掃描目錄不存在時回報成因，而非靜默回傳空清單。
- [ ] 決定設定、日誌及下載工具的預設用戶資料目錄（第 3 節的工具安裝目錄取決於此）。
- [ ] 讓用戶選擇模型儲存位置，支援其他硬碟。
- [ ] 發布包只附乾淨預設值，排除本機 `config.json`／`models.json` 狀態。
- [ ] 改為 track `config.default.json`，把帶 `D:\AI\...` 路徑的 `config.json`／`models.json` 移出版本控制並更新 `.gitignore` 註解；現時 clone 下來即帶著作者的機器狀態。
- [ ] 提供舊設定匯入與遷移，保留既有模型路徑、遠端設定檔及自訂工具。
- [ ] 升級程式時保留用戶設定、模型與下載工具。
- [ ] 驗收：全新電腦首次啟動正常；舊用戶升級後仍能使用原有設定及模型。

## 2. 製作免裝 Python 的發布包

- [ ] **先做**：建立 Windows CI 流程（`windows-latest` 跑 pytest，再跑打包），產出版本化發布檔案。開發機為 macOS，Tk 測試與 Windows 打包只能在此驗證；開啟前需完成第 0 節的測試修正。
- [ ] 使用 PyInstaller `onedir` 建立 Windows 64-bit 發布包，壓縮為 ZIP。README 現時建議 `--onefile --windowed`，需一併更正。
- [ ] 包含 Python、Tcl/Tk、Python 依賴及 tkinterdnd2 原生 DLL。
- [ ] 固定發布依賴版本（含 Python 3.12 64-bit），建立可追蹤的打包設定及腳本。`.gitignore` 現時 ignore `*.spec` 及 `docs/*`，會令 spec 檔與打包文件無法提交，需先調整。
- [ ] 處理未簽名執行檔的 SmartScreen「不明發行者」警告及防毒誤報：最少在新用戶文件寫明繞過步驟，並評估程式碼簽署憑證。
- [ ] 打包後驗證程式資源定位、圖片預覽與拖放功能。
- [ ] 驗收：沒有安裝 Python 的 Windows 電腦可以雙擊開啟程式。

## 3. 統一管理外部工具

第一版範圍：固定版本清單 → 下載 → 校驗 → 解壓 → 檢查 → 切換。中斷續傳及多配套選擇延後，先換到可用狀態。

### 管理介面與接駁

- [ ] 建立共用工具管理模組，集中管理 llama-server、whisper-cli、FFmpeg 與 ffprobe。
- [ ] 定義工具目錄及設定結構，記錄來源、版本、執行檔與使用中的配套。
- [ ] 支援選擇既有執行檔及自訂 build，保留現有使用方式。
- [ ] 建立 GUI 工具管理入口：查看狀態、選擇路徑、檢查、下載及重新安裝。
- [ ] 依工作用途顯示所需工具；只做轉錄可獨立準備轉錄工具。
- [ ] Server、Whisper、工作台管線與漫畫流程統一使用管理器提供的路徑。
- [ ] FFmpeg／ffprobe 使用完整路徑，免用戶修改系統 PATH。`whisper_transcription.py` 現時直接呼叫 `"ffmpeg"`／`"ffprobe"`；改動細、可先獨立完成。

### 版本與下載

- [ ] 選定經相容性驗證的固定版本，建立版本、下載來源及 checksum 清單。llama.cpp 與 whisper.cpp 的 release asset 無官方 checksum，清單須自行計算並維護，每次升版重新驗證。
- [ ] 選定 FFmpeg Windows build 來源（ffmpeg.org 無官方 Windows build，須在 BtbN／gyan.dev 之間取捨），並按授權附上 GPL 文字。
- [ ] 提供經驗證的 Windows CPU／Vulkan 配套選項，清楚顯示適用條件。（第一版可只出一個預設配套。）
- [ ] 下載及解壓保留完整 DLL、必要資料檔及授權文件。
- [ ] 顯示下載進度、大小及目前步驟；支援取消及重試。（中斷後續傳延後。）
- [ ] 校驗下載檔案，完成解壓及檢查後才切換使用中的版本。
- [ ] 取消、下載或安裝失敗時保留既有可用版本。
- [ ] 檢查執行檔、版本、必要參數及 CPU／GPU 後端，提供具體錯誤與修復動作。
- [ ] 檢查 Vulkan 前置條件（顯卡驅動／Vulkan runtime）；缺失時指出成因，而非只回報後端不可用。
- [ ] 在背景執行下載與檢查，維持 GUI 可操作，診斷接到共用日誌。
- [ ] 驗收：無須自行搜尋工具、整理 DLL 或設定 PATH，即可執行所選工作流程。

## 4. 加入首次設定精靈

精靈作為工具管理器的 wizard 模式實作，避免同一套下載／檢查邏輯出兩個介面。

- [ ] 用戶先選用途：只做轉錄、本地字幕翻譯或遠端 API。
- [ ] 選擇模型儲存位置，檢查寫入權限及磁碟空間。
- [ ] 按用途接駁工具管理器，準備所需工具。
- [ ] 提供少量經驗證的模型選項，顯示用途、下載大小及硬體需求。
- [ ] 支援使用已有模型，清楚區分文字 GGUF 與 Whisper GGML `.bin`。
- [ ] 模型下載顯示進度，支援取消、重試及校驗。
- [ ] 遠端 API 流程可設定端點、模型及金鑰，沿用現有金鑰保存規則。
- [ ] 執行短樣本，驗證工具、模型、連線及實際字幕輸出。驗證用模型取 tiny／base 一級，不可用 large-v2，否則精靈變成長時間下載。
- [ ] 缺工具、GPU 不可用或磁碟不足時，顯示原因及可執行的處理方法。
- [ ] 允許略過已完成步驟，並可從程式內重新開啟精靈。
- [ ] 後續加入漫畫 OCR／翻譯模型配對及 DFlash 相容設定。
- [ ] 驗收：完成精靈後，成功產生第一份短樣本字幕。

## 5. 乾淨環境驗收及發布

- [ ] 準備乾淨 Windows VM 或實機：本節所有驗收都無法在 macOS 開發機上進行（該機連 Tk 測試都跑不到）。
- [ ] 在沒有 Python、外部工具及本機設定的乾淨 Windows 環境驗收。
- [ ] 驗證沒有 FFmpeg PATH 時仍可正常處理影音及長音訊。
- [ ] 覆蓋中文／空格路徑、普通用戶權限、磁碟不足及 GPU 不可用情況。
- [ ] 覆蓋下載中斷、取消、檔案校驗失敗及重新啟動後恢復。
- [ ] 覆蓋舊版設定匯入、工具版本切換與升級後資料保留。
- [ ] 更新 README 的過時安裝內容，新用戶文件以幾步操作為主。至少包括：「llama-server 執行路徑須修改 `llama_manager.py`」已不成立（路徑已可在設定中配置）、PyInstaller 指令改為 `onedir`、測試紀錄數字。
- [ ] 將開發者 Python／venv 安裝方式另外整理。
- [ ] ZIP 發布流程穩定後，製作安裝程式、開始選單／桌面捷徑及更新提示。
- [ ] 驗收：新用戶無須開 terminal、修改 source 或手動設定 PATH，便能完成第一次字幕輸出。

## 參考

- [PyInstaller 打包方式](https://pyinstaller.org/en/stable/operating-mode.html)
- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases)
- [FFmpeg 下載](https://ffmpeg.org/download.html)
