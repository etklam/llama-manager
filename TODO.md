# 新用戶安裝與跨平台支援 TODO

## 目標與優先順序

既有發布目標：Windows 10／11 64-bit。建議新增支援：macOS Apple Silicon（arm64）及 Ubuntu 24.04 x86_64；兩者先提供原始碼執行，通過實機驗收後才承諾免 Python 發布包。這是規劃範圍，尚未代表已支援。

新用戶流程：下載 ZIP → 解壓 → 雙擊 EXE → 設定精靈準備工具及模型 → 成功輸出第一份字幕。

**目前主線：先完成字幕轉錄與字幕翻譯。** 包括語音／影片 → SRT、已有 SRT／TXT 翻譯，以及「轉錄 → 翻譯」完整管線；先讓新用戶可靠完成第一次字幕輸出。

主線完成標準：工具及模型可配置、本地／遠端翻譯可用、失敗與取消不破壞既有輸出、設定可保留，並完成乾淨環境端到端驗收及使用文件。

跨平台工作先處理這條主線所需的相容性；macOS／Linux 的擴展發布包不阻塞核心流程完成。漫畫維持 Beta，不擴展後端；Intel NPU、DFlash 與漫畫後續能力排在主線驗收完成後。第一版設定精靈只覆蓋語音轉錄與字幕翻譯。

### 執行順序（2026-10-15 review）

1. 第 0 節：驗證目前工作樹的修正，取得可信的測試基線。
2. 第 1 節的資源／用戶資料目錄與移除 Windows fallback，同時完成第 3 節的 FFmpeg 路徑接駁。
3. 建立 Windows／macOS／Linux 測試矩陣；先驗證原始碼 GUI 與核心流程，再加入各平台打包。
4. 第 3 節先做「選擇既有工具＋檢查＋完整路徑」，再加入固定配套下載；先讓三平台可手動配置。
5. 第 2、4、5 節完成 Windows 新用戶發布；macOS／Linux 通過各自驗收後再加入自動安裝與發布包。
6. 主線驗收完成後，再安排 Intel NPU spike、漫畫後端擴展與 DFlash 跨後端驗證；目前保留路線，不同步開工。

## 0. 立即修正與基線

以下「已修改」只表示 review 時工作樹已有實作，尚未經本次測試驗證，也尚未提交；不可當成發布驗收通過。

- [x] 已修改：`requirements.txt` 收窄為 `openai>=2.8.1,<2.9.0`；發布時仍需固定整套相依版本。
- [x] 已修改：`translation/openai_client.py` 使用 `self.api_key or 'local-only'`。
- [x] 已修改：`tests/test_desktop_ui.py` 改用目前 drawer／tab 介面；仍需在可初始化 Tk 的環境執行。
- [x] 已修改：`config_manager.py` 及 `server_tab.py` 的 host 預設為 `127.0.0.1`。已有 config 的明確值不會因此改變；LAN 存取仍需明確 opt-in。
- [x] 已修改：`config_manager.py` 的 paths 註解移除不存在的 first-run 指引描述。
- [ ] 在 Python 3.12、依賴齊全且有 Tk 的環境重跑全套，記錄 passed／failed／skipped 及平台。
- [x] 更新 README 的測試紀錄；2026-09-18 的 784 passed 已標為歷史結果，並記錄目前可執行的 64 passed 針對性測試。

本次執行：建立 `.venv`（macOS、Python 3.14.7）安裝 `requirements-dev.txt`，排除 10 個需要 Tk 的 GUI 測試模組後 **671 passed、13 failed**。13 個失敗全部源自環境：12 個因 `subtitle_tab` 延遲 import 觸發缺少的 `_tkinter`，1 個是 `test_srt_parser.py::test_windows_backslashes` 假設 POSIX 上有 `C:` drive letter。沒有失敗可歸因於本次改動。真 Tk 測試與完整 passed／failed／skipped 數字仍待 Python 3.12＋Tcl/Tk 的 CI 實跑。

## 1. 整理設定及資料目錄

- [x] 區分程式資源目錄與可寫入的用戶資料目錄，兼容原始碼執行及打包後執行。`app_paths.py` 按平台選資料目錄，`ConfigManager` 支援從舊程式目錄匯入。
- [x] 移除新安裝對作者 `D:\AI\...` 路徑的依賴；空設定使用使用者資料目錄下的工具／模型提示路徑，不寫回假裝已配置。
- [x] `ServerController.start()` 對已配置的 llama-server 路徑做存在性前置檢查；PATH 中的 bare command 保留給進階使用者。
- [x] `BaseModelRegistry.scan()` 在掃描目錄不存在或不是目錄時保留可顯示的原因。
- [x] 決定設定、模型及工具的預設用戶資料目錄（Windows、macOS、Linux）。
- [x] 讓用戶選擇模型儲存位置，支援其他硬碟；Server 分頁可選目錄並立即重新掃描。
- [x] 發布包只附乾淨預設值；已加入 `config.default.json`／`models.default.json`，spec 不打包 runtime 狀態。
- [x] 改為 track `config.default.json`，帶 `D:\AI\...` 路徑的 `config.json`／`models.json` 已 `git rm --cached` 移出版本控制並更新 `.gitignore` 註解；本機檔案保留，首次啟動仍可匯入。
- [x] 提供舊設定匯入與遷移，保留既有模型路徑、遠端設定檔及自訂工具；首次啟動會把舊設定複製到用戶資料目錄。
- [ ] 升級程式時保留用戶設定、模型與下載工具。
- [ ] 驗收：全新電腦首次啟動正常；舊用戶升級後仍能使用原有設定及模型。

## 2. 製作免裝 Python 的發布包

- [x] 建立 Windows CI（pytest 含真 Tk 測試，再跑打包），並接入第 6 節測試矩陣；已加入 Windows／macOS／Ubuntu workflow 骨架，仍需 CI 實跑。
- [x] 使用 PyInstaller `onedir` 建立 Windows 64-bit 發布包，壓縮為 ZIP；已加入跨平台 spec，仍需在各 runner 實跑。
- [ ] 包含 Python、Tcl/Tk、Python 依賴及 tkinterdnd2 原生 DLL。
- [x] 建立 Python 3.12 64-bit 的可追蹤打包設定及開發依賴；已加入 `requirements-dev.txt`、PyInstaller spec，並放行 `packaging/*.spec`。完整 lockfile 仍待發布環境確定後產生。
- [ ] 處理未簽名執行檔的 SmartScreen「不明發行者」警告及防毒誤報：最少在新用戶文件寫明繞過步驟，並評估程式碼簽署憑證。
- [ ] 打包後驗證程式資源定位、圖片預覽與拖放功能。
- [ ] 驗收：沒有安裝 Python 的 Windows 電腦可以雙擊開啟程式。

## 3. 統一管理外部工具

第一版範圍：固定版本清單 → 下載 → 校驗 → 解壓 → 檢查 → 切換。中斷續傳及多配套選擇延後，先換到可用狀態。

### 管理介面與接駁

- [x] 建立共用工具管理模組，集中管理 llama-server、whisper-cli、FFmpeg 與 ffprobe。
- [x] 定義工具目錄及設定結構，記錄執行檔路徑與使用中的配套；版本／來源欄位留待下載器接入。
- [x] 支援選擇既有執行檔及自訂 build，保留現有使用方式。
- [x] 建立 GUI 工具管理入口：Server／Whisper 分頁可查看狀態、選擇路徑及檢查；下載及重新安裝仍待接入。
- [ ] 依工作用途顯示所需工具；只做轉錄可獨立準備轉錄工具。
- [x] Server、Whisper、工作台管線與漫畫流程統一使用管理器提供的路徑。
- [x] FFmpeg／ffprobe 支援 config 的完整路徑；未填時才回退到 PATH。Whisper 分頁與工作台共用同一個 adapter。

### 版本與下載

- [ ] 選定經相容性驗證的固定版本，建立版本、下載來源及 checksum 清單。llama.cpp 與 whisper.cpp 的 release asset 無官方 checksum，清單須自行計算並維護，每次升版重新驗證。
- [ ] 選定 FFmpeg Windows build 來源（ffmpeg.org 無官方 Windows build，須在 BtbN／gyan.dev 之間取捨），並按授權附上 GPL 文字。
- [ ] 提供經驗證的 Windows CPU／Vulkan 配套選項，清楚顯示適用條件。（第一版可只出一個預設配套。）
- [ ] 下載及解壓保留完整 DLL、必要資料檔及授權文件。
- [ ] 顯示下載進度、大小及目前步驟；支援取消及重試。（中斷後續傳延後。）
- [ ] 校驗下載檔案，完成解壓及檢查後才切換使用中的版本。
- [ ] 取消、下載或安裝失敗時保留既有可用版本。
- [ ] 檢查執行檔、版本、必要參數及各自的 CPU／GPU／NPU 後端，提供具體錯誤與修復動作。
- [ ] 分別驗證 `llama-server`（llama.cpp）與 `whisper-cli`（whisper.cpp）的 Vulkan build、裝置選擇及實際 offload；FFmpeg／ffprobe 的影音解碼／hwaccel 檢查另列，不以它代替推論 backend 驗證。
- [ ] 檢查選用 Vulkan build 的前置條件（顯卡驅動／Vulkan runtime）；缺失時指出成因，而非只回報後端不可用。
- [ ] 在背景執行下載與檢查，維持 GUI 可操作，診斷接到共用日誌。
- [ ] 驗收：無須自行搜尋工具、整理 DLL 或設定 PATH，即可執行所選工作流程。

## 4. 加入首次設定精靈

精靈作為工具管理器的 wizard 模式實作，避免同一套下載／檢查邏輯出兩個介面。

- [x] 用戶先選用途：只做轉錄、本地字幕翻譯或遠端 API；「設定精靈」已提供三個選項。
- [x] 選擇模型儲存位置；精靈與 Server／Whisper 分頁都有目錄選擇，寫入權限及磁碟空間檢查待補。
- [x] 按用途接駁工具管理器，準備所需工具；目前準備既有工具，下載器待補。
- [ ] 提供少量經驗證的模型選項，顯示用途、下載大小及硬體需求。
- [ ] 支援使用已有模型，清楚區分文字 GGUF 與 Whisper GGML `.bin`。
- [ ] 模型下載顯示進度，支援取消、重試及校驗。
- [ ] 遠端 API 流程可設定端點、模型及金鑰，沿用現有金鑰保存規則。
- [ ] 執行短樣本，驗證工具、模型、連線及實際字幕輸出。驗證用模型取 tiny／base 一級，不可用 large-v2，否則精靈變成長時間下載。
- [ ] 缺工具、GPU 不可用或磁碟不足時，顯示原因及可執行的處理方法。
- [x] 允許略過已完成步驟，並可從「伺服器設定」重新開啟精靈。
- [ ] 後續加入漫畫 OCR／翻譯模型配對及 DFlash 相容設定。
- [ ] 驗收：完成精靈後，成功產生第一份短樣本字幕。

## 5. 乾淨環境驗收及發布

- [ ] 準備乾淨 Windows VM 或實機，驗收 Windows 發布包；macOS／Linux 分別使用對應環境，不以另一平台的通過結果代替。
- [ ] 在沒有 Python、外部工具及本機設定的乾淨 Windows 環境驗收。
- [ ] 驗證沒有 FFmpeg PATH 時仍可正常處理影音及長音訊。
- [ ] 覆蓋中文／空格路徑、普通用戶權限、磁碟不足及 GPU 不可用情況。
- [ ] 覆蓋下載中斷、取消、檔案校驗失敗及重新啟動後恢復。
- [ ] 覆蓋舊版設定匯入、工具版本切換與升級後資料保留。
- [x] 更新 README 的過時安裝內容；路徑可由 UI 配置、PyInstaller 改用 `packaging/build.py --zip`，並記錄目前測試限制。
- [ ] 將開發者 Python／venv 安裝方式另外整理。
- [ ] ZIP 發布流程穩定後，製作安裝程式、開始選單／桌面捷徑及更新提示。
- [ ] 驗收：新用戶無須開 terminal、修改 source 或手動設定 PATH，便能完成第一次字幕輸出。

## 6. macOS／Linux 可行性與最小支援方案

### 評估結論

**可行，核心字幕／轉錄不需要換 GUI 框架或重寫；正式三平台發布的成本主要在工具配套、原生元件與實機驗收。** 建議保留 Windows，並行開放 macOS arm64 與單一 Linux 發行版的核心流程；暫不承諾所有功能與 GPU 配套一致。

| 範圍 | 判斷 | 依據與限制 |
| --- | --- | --- |
| 遠端 API、SRT／TXT 翻譯 | 高可行性、改動較小 | HTTP、Path、解析與原子寫入已可共用；仍受 GUI 啟動與資料目錄影響 |
| 本地 llama-server＋Whisper／FFmpeg | 高可行性、改動中等 | Python 以 subprocess 接 CLI；llama.cpp 與 whisper.cpp 都有 Vulkan build；macOS 通常經 MoltenVK 落到 Metal，Linux 可走原生 Vulkan，執行檔、權限、共享庫與後端仍須分平台驗證 |
| Tk 桌面介面 | 可保留，需實測 | 字型已有 PingFang／Noto fallback，滾輪已有 Linux／macOS 分支；拖放原生庫、快捷鍵、縮放與中文顯示尚未驗收 |
| 漫畫 OCR／翻譯（Beta，不穩定） | Linux Vulkan 可評估；macOS Vulkan／MoltenVK 可評估 | `manga/runtime.py` 只接受 `Vulkan\d+`，command、evidence、runner 均綁定 Vulkan；仍需驗證 macOS 的 Vulkan build、裝置枚舉與真實多模態推理 |
| DFlash | 分開驗證 | 預設 `Vulkan0` 且需要特殊 runtime／模型／參數；一般 llama.cpp 平台支援不等於 DFlash 可用 |
| 免 Python 發布／工具自動下載 | 成本最高 | 每個 OS／CPU 架構分開建置、配套與驗收；Linux 另有 glibc 與桌面環境差異 |

### 第一階段：原始碼執行、手動配置工具

- [x] `llama_manager.py` 移除 `D:\AI` fallback 與寫回副作用；執行檔依平台使用 `llama-server.exe`／`llama-server`，可由設定提供完整路徑。未設定仍可使用遠端 API。
- [x] 設定／工具資料落在可寫入的用戶目錄：Windows `%LOCALAPPDATA%/llama-manager`、macOS `~/Library/Application Support/llama-manager`、Linux `$XDG_DATA_HOME/llama-manager`（未設則 `~/.local/share/llama-manager`）；保留舊設定匯入。
- [x] 外部工具檢查涵蓋檔案存在與 POSIX 執行權限；架構不符及動態庫缺失仍待實際執行檢查。FFmpeg／ffprobe 使用配置的完整路徑。
- [x] `manga_tab.py:open_output()` 分流 Windows startfile、macOS `open`、Linux `xdg-open`，並處理 opener 缺失／失敗。
- [x] `whisper_tab.py` 執行檔選擇器按平台顯示，不再只優先 `.exe`；Command 快捷鍵與真機 UI 驗收仍待完成。
- [x] `server_tab.py:release_memory()` 的 EmptyStandbyList 僅在 Windows 使用；程序清理只作用於本程式擁有的 server。
- [x] ServerController 在 POSIX 以 process group 啟動及停止；轉錄 runner 已有取消清理。關閉視窗及真工具殘留程序仍待實測。
- [ ] macOS 核心字幕流程分別驗收 llama.cpp 與 whisper.cpp 的 Vulkan build（通常由 MoltenVK 轉接到 Metal）；Linux 先驗收 CPU，再驗收兩個工具各一款指定 GPU／Vulkan 配套。FFmpeg／ffprobe 是獨立的影音預處理工具，非 Windows 目前走軟體解碼可先保留。
- [ ] 漫畫及 DFlash 在不符合能力條件時停用入口並顯示原因；不將未驗證的平台能力顯示為成功。

### 第二階段：測試與發布

- [x] CI 固定 Python 3.12，Windows／macOS arm64／Ubuntu 24.04 x86_64 都跑核心測試；Linux GUI 測試使用 Xvfb 的 workflow 已建立，仍需 CI 實跑確認真 Tk 測試。
- [ ] 每平台跑一次真工具 smoke test：短音訊 → SRT → 翻譯、遠端 API、取消、中文／空格路徑、重新啟動後設定保留。GPU 品質與速度另用實機記錄，mock 不代替驗收。
- [ ] PyInstaller 分 OS／架構建置，驗證 Tcl/Tk、tkinterdnd2 原生庫與 Pillow；macOS app 驗證簽署／notarization／Gatekeeper，Linux 明定最低 glibc 與桌面環境。
- [ ] 工具下載清單按 OS／架構／後端索引，處理 `.dll`／`.dylib`／`.so`、執行權限與授權；沒有已驗證 asset 時提供手動選取，不臨時自動編譯。
- [ ] 初版不擴展至 Intel Mac、Linux arm64、所有發行版或所有 GPU；有需求及驗收資源才加入。

### 主線完成後：漫畫 Beta 的 macOS Vulkan 與後端驗收

- [x] 漫畫導覽入口、頁面標題與使用文件標示 Beta，頁面明示不穩定及需要人工核對。
- [ ] 先做漫畫流程的 macOS Vulkan／MoltenVK spike：llama.cpp 的 OCR／翻譯 runtime 與 whisper.cpp 的主線轉錄各自驗證；本專案仍需確認裝置枚舉、OCR projector、翻譯與 evidence，不能直接宣稱可用。macOS 的 Vulkan 通常透過轉譯層落到 Metal。

- [ ] 將裝置探測、命令與執行證據從 Vulkan 專用改為能力判斷，同步修改 `manga/runner.py` 的驗證契約與 UI 文案；保持「無法驗證不得報 verified」。
- [ ] 在 Apple Silicon 真機驗證 OCR 模型＋projector＋翻譯模型，再完成既有 10–20 頁漫畫品質驗收，通過前維持未支援狀態。

上述工作量是相對評估，未作三平台真機 spike，暫不給可靠工期承諾。

## 7. Intel NPU（主線完成後，實驗性路線，尚未支援）

定位：Windows／Linux 上具相容 Intel NPU 的裝置，先選一款 Intel Core Ultra 實機做 spike，再訂支援世代、OS、驅動與 runtime 版本。macOS 不列入此路線。此項不阻塞三平台核心流程發布。

上游已有 llama.cpp OpenVINO backend 的 Intel CPU／GPU／NPU 路線；OpenVINO GenAI 亦有 Whisper NPU 路線，但不能視為目前 whisper-cli、所有 GGUF、kotoba 模型或漫畫 OCR 均可直接使用。

- [ ] **先驗證字幕翻譯**：固定 llama.cpp＋OpenVINO＋NPU driver 版本，選上游列為已驗證的小型文字模型，以 llama-server 完成 `/health`、翻譯 request 與 SRT 輸出。確認本程式使用的 context、parallel、FlashAttention 與 KV cache 參數相容性，建立 NPU 適用 preset。
- [ ] **再驗證 Whisper 轉錄**：比較現有 whisper.cpp OpenVINO 路線與 OpenVINO GenAI WhisperPipeline 的實際 NPU 支援範圍；先用已驗證的標準 Whisper 模型，再單獨評估 kotoba。若須轉換為 OpenVINO 模型格式或另接 adapter，清楚區分現有 GGML `.bin` 與新格式。
- [ ] 工具管理器辨識 NPU、驅動、OpenVINO runtime 與模型相容性；記錄 OS／架構／NPU 世代／版本。安裝 runtime 不代表已安裝可用驅動。
- [ ] 明確選擇 NPU 並記錄實際執行裝置與 CPU／GPU 混合執行情況；NPU 不可用時提供原因及手動切換選項，不靜默降級後仍顯示 NPU 成功。
- [ ] 用同一輸入比較 CPU／GPU／NPU：首次模型編譯、暖啟動、總耗時、吞吐、記憶體、可取得的功耗資料與字幕品質；通過結果決定是否值得提供正式配套，不預設 NPU 一定更快。
- [ ] 驗收取消、timeout、重啟、編譯快取及磁碟空間，確認沒有殘留程序；通過實機測試後才提供可選 Beta 配套。
- [ ] 漫畫 OCR／projector 與 DFlash 的 NPU 支援另行評估；不得由文字翻譯成功推論它們相容，也不得沿用 Vulkan 專用 evidence 宣稱 NPU 已驗證。

## 參考

- [PyInstaller 打包方式](https://pyinstaller.org/en/stable/operating-mode.html)
- [llama.cpp Releases](https://github.com/ggml-org/llama.cpp/releases)
- [whisper.cpp Releases](https://github.com/ggml-org/whisper.cpp/releases)
- [FFmpeg 下載](https://ffmpeg.org/download.html)
- [llama.cpp 平台與後端建置](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md)
- [whisper.cpp 支援平台與加速](https://github.com/ggml-org/whisper.cpp)
- [tkinterdnd2 平台與打包說明](https://github.com/pmgagne/tkinterdnd2)
- [PyInstaller 各平台建置限制](https://pyinstaller.org/en/stable/usage.html)
- [MoltenVK：在 Metal 上執行 Vulkan](https://github.com/KhronosGroup/MoltenVK)
- [llama.cpp OpenVINO backend／Intel NPU](https://github.com/ggml-org/llama.cpp/blob/master/docs/backend/OPENVINO.md)
- [OpenVINO 模型支援矩陣](https://docs.openvino.ai/2026/documentation/compatibility-and-support/supported-models.html)
