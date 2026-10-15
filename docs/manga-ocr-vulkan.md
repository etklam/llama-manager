# 漫畫 OCR → 本地 Vulkan LLM 翻譯（Beta）

> Beta・不穩定功能：可能執行失敗或產生不準確的結果，請人工核對輸出。

本功能提供原生 Tkinter 分頁，輸入 PNG／JPEG／WEBP，輸出原文、譯文及雙語文字。OCR 與翻譯依序使用同一個 llama-server executable，共用 `ServerController`，不需要兩個模型同時常駐。

**GLM-OCR 是起始候選，不是已認證的漫畫辨識方案。** 本次沒有真實漫畫品質或完整 Vulkan OCR＋翻譯通過宣稱。也不包含 PDF／CBZ／CBR、氣泡偵測、消字、inpainting、嵌字或翻譯後圖片。

本文件的 Vulkan 驗證專指漫畫流程使用的 llama.cpp OCR／翻譯 runtime。主線的 `whisper-cli`（whisper.cpp）也可使用 Vulkan build，但它的轉錄 backend 驗收與 FFmpeg／ffprobe 的影音預處理驗收分開計算。

## 安裝與模型配對

```powershell
python -m pip install -r requirements.txt
python llama_manager.py
```

Python／Tkinter 架構保持不變。新增 Pillow 只負責圖片解碼、EXIF、透明背景與預覽；沒有新增 ONNX Runtime、RapidOCR、DirectML、PyTorch、PaddlePaddle、CUDA、ROCm 或雲端推理依賴。

1. 從 [llama.cpp releases](https://github.com/ggml-org/llama.cpp/releases) 準備支援所需模型的 Vulkan build，保留同套 DLL。binary 能列出 Vulkan device 不等於已通過多模態推理。
2. 從 [ggml-org/GLM-OCR-GGUF 檔案頁](https://huggingface.co/ggml-org/GLM-OCR-GGUF/tree/main) 取得 `GLM-OCR-Q8_0.gguf` 與同一模型版本的 `mmproj-GLM-OCR-Q8_0.gguf`。不要拿其他 vision 模型的 projector 混用。下載不由本工具自動進行，模型不加入 Git。
3. 準備自己選定的文字 GGUF 作翻譯模型。
4. 打開「漫畫 OCR／翻譯」，選 executable，按「枚舉／診斷」，再從 runtime 實際列出的 Vulkan devices 選擇裝置。這裡不以 GPU 廠商名稱分支。
5. 分別選 OCR GGUF、matching mmproj、translation GGUF；設定各自 context、輸出 tokens、溫度。全程不需修改 Python source。

漫畫設定存於 `config.json` 的 `manga` 區段，保留既有設定及未知欄位。漫畫 profile 不跟隨字幕的 local／remote 設定，也不借用 remote API key 或 proxy。

## 最短操作流程

1. 加入圖片或資料夾，確認頁序；資料夾只掃本層，排除輸出目錄，檔名採自然排序。不同來源同名檔、內容相同的不同頁面都保留。
2. 選「同一文件」或「每張獨立」。一次一個章節可選同一文件；不同章節可分次加入與執行。
3. 設來源語言、目標語言代碼（例如 `ja`、`zh-tw`、`en`），選閱讀方向。方向是工作紀錄與校對提示，不會自動解決分鏡次序。
4. 預設「OCR 後翻譯」，按「執行」。不需要逐頁人工批准。
5. 完成後按「開啟輸出」。若有失敗／待確認，先校正或修正素材，再續跑。

三種模式：

| 模式 | 行為 |
| --- | --- |
| 只做 OCR | 逐頁辨識及 checkpoint；不載入 translation 模型 |
| OCR 後翻譯 | 整批 OCR 完成後停止 OCR process，再載入 translation 模型 |
| 重用 OCR 翻譯 | 重用驗證有效的 OCR／校正文字；不載入 OCR 模型；無效或缺漏頁保留 partial |

「先校對」使前兩種流程在 OCR 後暫停。校正完成後，選「重用 OCR 翻譯」執行。「兩步翻譯」會在同一個 ID 回應中要求直譯草稿及自然意譯，最終輸出使用意譯；預設單步。

## 校正、無字頁與限制

點選頁面可看原圖預覽與 OCR 文字。預覽會縮放，實際 OCR 輸入不由本程式縮小。文字編輯器以空行分段；剪貼段落可調整閱讀順序，按「儲存校正／順序」。頁序用左側 ↑／↓ 調整，再執行使輸出依新順序發布。

「查看 Raw OCR」顯示唯讀原始回應。校正保留 raw、更新 revision／segments，並使該文件各語言譯文失效。相同台詞的每次出現都有獨立 ID，不做字幕式全檔去重或重複句折疊。page／segment ID 是程式身份，不是氣泡座標；bbox／confidence 為 null。

空回應會標 `needs_review`，不是自動判成無字頁。確認真的沒有文字後，按「確認無文字」。此操作會清空校正段落，但保留 raw。`finish_reason=length`、推理標記、常見解說／拒答開頭、錯誤 JSON 或無正文都不會成為成功結果。文字內容仍可能包含模型未被程式辨識出的幻覺或解說，必須人工核對。

圖片經 EXIF 方向修正、白底透明合成、轉 RGB PNG，再以正確 `image/png` MIME 傳送。只接受靜態 PNG／JPEG／WEBP。上限為 64 MiB 檔案、2,400 萬像素、任一邊 16,000 像素；超大／超長頁會要求拆頁，不偷偷縮小。runtime 本身仍可能依模型規則做 vision 預處理；其細字品質必須透過真圖驗收。

特別人工檢查直排、振假名、多氣泡、跨格對話、擬聲詞、細字與重複對白。沒有 layout detector，也不會讓 OCR 模型憑空回報座標／信心值。閱讀方向選項不能代替分鏡與段落校對。

## Checkpoint、取消與重啟續跑

每份文件使用來源路徑及出現次數建立隔離 ID；source content SHA-256 與設定身份決定有效性，不只用檔名或 mtime。模型及 projector 也計算內容 SHA-256，大模型首次驗證需要讀取整個檔案。

- 每頁立即寫 `pages/<page-id>.json`；每個翻譯批次立即寫 `checkpoint.<locale>.json`。
- JSON checkpoints 有內容 checksum，讀取時再驗證結構與 ID。損毀會清楚失敗，不當作快取。不要直接編輯帶 checksum 的 JSON；使用 GUI 校正。
- 目標語言／翻譯模型／生成設定變更，只重做翻譯，不重做有效 OCR。
- 來源內容／OCR 模型／mmproj／prompt／預處理身份改變，相關頁 OCR 及文件譯文失效。
- 修改段落／頁序會保守重翻整份文件，避免鄰近上下文依賴錯配。
- 取消先停止新派發與所有重試；在途同步 HTTP 有 120 秒 timeout。狀態顯示 stopping，回應晚到也不能越過 CommitGate 發布。
- 不會在取消後自動轉到 translation 階段。已提交的 checkpoints 保留。
- 重啟後按「載入工作記錄」，選輸出根目錄的 `manga-job.json`，再按「續跑／重試失敗」。頁序、profiles 與模式會還原，重新驗證快取後只做未完成工作。
- 正常工作失敗或停止後，都先清理自己擁有的 process，再按工作前快照還原原 server；還原失敗另列 `restore_error`，不掩蓋原始結果。載入 readiness 上限 120 秒。
- 關閉視窗時若漫畫工作仍在執行，先取消並保持 UI dispatch，待 worker 清理與還原完成才關閉。

單張壞圖或內容辨識失敗可以繼續其他頁；HTTP／裝置／程序系統性故障停止整批派發。損毀 checkpoint 需要先移到備份位置，再重新執行；程式不自行刪除使用者資料。

## 輸出

```text
output/
  manga-job.json
  runtime-report.json
  <document-id>/
    manifest.json
    pages/<page-id>.json
    ocr.json
    source.txt
    checkpoint.<locale>.json
    translations.<locale>.json
    translated.<locale>.txt
    bilingual.<locale>.md
```

有未完成頁、失敗、未確認空結果或漏譯 ID 時，翻譯輸出使用 `translations.<locale>.partial.json`、`translated.<locale>.partial.txt`、`bilingual.<locale>.partial.md`，含 PARTIAL／缺頁／缺譯標記。

所有文字檔案以同目錄暫存檔＋fsync＋atomic replace 發布。只有完整文件才寫正式譯文，最後提交 `manifest.status=completed`；manifest 是整組正式輸出的完成標記。若程序在多檔發布之間崩潰，manifest 仍不會宣告完成，續跑會再次驗證並重建。原圖始終唯讀。先前失效的正式譯文移為 partial，不會繼續冒充目前版本。

OCR raw、人工修訂與翻譯分開存放。輸出路徑只由程式 ID／驗證過的 locale 建立；不執行 OCR 或模型回應內的命令、不開網址、不使用模型提供的檔名。雙語 Markdown 的原文／譯文置於文字 fenced blocks。

## 推理與 server ownership

新增工作沿用 `ServerController` 的 process 啟停、monitor 與 HTTP client 的傳輸政策。漫畫、本地字幕、本地管線取得同一個獨佔 lease，手動 server 啟停也受保護。remote 字幕工作維持原行為，不取得本地模型工作權。

只停止本程式建立且追蹤的 process；不使用 taskkill，不按名稱掃殺程序。port 衝突報錯，不接管外部 server。舊 process 必須真正退出才啟新 profile；log monitor 使用 process-local reference／log buffer 與狀態鎖，舊 monitor 不能清掉新 process。

漫畫 server 只綁 `127.0.0.1`，預設 port 8091。HTTP client `trust_env=False`，不使用環境 proxy；SDK `max_retries=0`，`translation.transport` 是唯一 HTTP retry owner。內容解析最多兩次邏輯嘗試，不吞掉系統性 transport failure。

OCR 預設單 worker／slot、f16 KV、FlashAttention off，不繼承 DFlash、量化 KV 或並行 preset。啟動前檢查 binary `--version`、`--help`、`--list-devices`。只用它實際宣告的 flags；新 runtime 支援 `--mmproj-device` 才傳該選項，舊 runtime 不猜參數。清除漫畫 subprocess 的推理環境覆寫，避免環境偷偷更換 profile。

依 [官方 GLM-OCR prompt](https://huggingface.co/zai-org/GLM-OCR) 與 [官方使用範例](https://huggingface.co/docs/transformers/model_doc/glm_ocr)，message 先放 image，再放 `Text Recognition:`。透過 llama-server OpenAI 相容 chat API、GGUF 內建 chat template 與 `--jinja` 執行。此處只參考訊息格式，不安裝 Transformers 或 PyTorch。實際 template 輸出仍需下面的真機驗證。

文件翻譯使用獨立入口，重用共用 client，沒有偽造 SRT timestamp。按完整指令、同文件鄰近原文、輸入、輸出預留和安全餘量估算 context。以保守 UTF-8 bytes 上界估計文字 token；超長段落按 Unicode 字元切成穩定子 ID，完整拼回，不截尾。每個回應嚴格檢查 missing／duplicate／unknown ID、型別及空值；不以位置猜配對、不因譯文等於原文就拒絕。不支援任意混雜 Markdown 的寬鬆修復；非法 JSON 最多重問一次後標為 partial。

## Vulkan 證據與驗證入口

`runtime-report.json` 分別記錄要求 device、實際 decoder／vision backend、runtime version、GPU／driver（可取得時）、模型／projector SHA-256、真實 request 結果、耗時、失敗與頁面狀態數。日誌不輸出 credentials、base64 或完整漫畫原文。

驗證需要真實 request 成功，並有 decoder offload／compute 與 vision backend／compute 證據。一般 CPU host buffer 不會被當成整個模型 CPU-only；反過來，只有 executable 名稱、CLI 參數、health 200 或成功載入，也不會被判為 verified。已知 CPU／錯誤 vision backend 會拒絕；log 版本無法解析或證據不全標 `unknown`／`unverified`，不偷偷換 backend。

參數格式依 [llama.cpp server 官方文件](https://github.com/ggml-org/llama.cpp/tree/master/tools/server) 與 [mtmd 文件](https://github.com/ggml-org/llama.cpp/tree/master/tools/mtmd) 核對，但所選 binary 的 help 才是執行時依據。verified 只適用該次記錄的組合，不代表所有 GPU／OS 認證，也不代表漫畫文字正確。

```powershell
python tools/verify_manga_vulkan.py --help
python tools/verify_manga_vulkan.py `
  --runtime C:/llama/llama-server.exe --device Vulkan0 `
  --ocr-model C:/models/GLM-OCR-Q8_0.gguf `
  --mmproj C:/models/mmproj-GLM-OCR-Q8_0.gguf `
  --translation-model C:/models/text.gguf `
  --samples C:/samples/manga --report-dir verification-reports/my-run
```

此入口每次建立全新輸出目錄，確保不能靠 cache 冒充真機驗證。同一 runtime 先送真正圖片 OCR，再切換文字模型翻譯；`verification.json` 是 machine-readable 報告。PASS 必須兩個階段都有真實 request 與 Vulkan 證據。失敗／證據不足回傳非零 exit code。

若只想查裝置，加 `--inspect-only`，此時不需要模型／素材，報告必須維持 `real_vulkan=NOT RUN`、`manga_quality=NOT RUN`。

## 10–20 頁漫畫驗收

使用自己創作、合法授權或自己有權使用的 10–20 頁素材，不隨程式散布漫畫。建議準備 16 頁：3 頁日文直排、3 頁多氣泡／跨格、2 頁振假名、2 頁細字、2 頁擬聲詞、1 頁特殊頁序、1 頁重複對白、1 頁無字、1 個壞檔。特徵可重疊，但各項都要有人工標記。

1. 先固定 runtime、device、模型 SHA、profile，跑驗證入口，保留報告。
2. 逐頁對原圖抄錄人工基準，分別記錄漏字、錯字、多出文字／幻覺與閱讀次序錯誤。特別記錄振假名被當正文、擬聲詞遺漏及重複對白漏頁。
3. 檢查空頁為待確認，壞圖為失敗，整本不誤標 completed。GUI 確認 no_text／修好壞檔後續跑。
4. 修改一頁原文／段落順序，再重用 OCR 翻譯；核對 revision、ID 與失效範圍。
5. OCR 中途取消、翻譯中途取消、退出重啟續跑；確認來源 hash 不變、已完成頁不重辨、取消後不切階段。
6. 分別報告 A 軟體測試、B Vulkan 硬體路徑、C 漫畫品質；普通橫排印刷文字或 mocks 不能替代 B／C。

## 本次工程驗證紀錄

- 起始 commit：`061d5c553e285a213a9d293df6a9a08ddf923fea`；新分支 `feat/manga-ocr-vulkan`；起始工作樹乾淨，未切回舊版、未修改真實 config。
- 軟體環境：Windows、Python 3.12.10、Pillow 11.3.0、openai 2.8.1、httpx 0.28.1。
- Baseline：`python -m pytest --basetemp=temp_test_dir_manga_baseline`，731 passed（20.03 秒）。
- 中間整合全套：`python -m pytest --basetemp=temp_test_dir_manga_full -rs`，775 passed（37.11 秒），0 failed／0 skipped。
- 最終全套：`python -m pytest --basetemp=temp_test_dir_manga_delivery -rs`，**784 passed（45.26 秒），0 failed／0 skipped，coverage 90%**。
- 聚焦整合：`python -m pytest tests/test_manga_runner.py tests/test_manga_ui.py tests/translation/test_openai_client.py --no-cov --basetemp=temp_test_dir_manga_last_focused -q -rs`，55 passed（29.50 秒）。
- 過程中 `temp_test_dir_manga_final` 一輪為 779 passed／1 failed：既有 `test_applies_cm_overrides` 在 Windows atomic replace 遇 WinError 5。未修改測試或原子寫入斷言；原測試以新暫存目錄單獨重跑 1 passed（0.09 秒），上列最終全套也通過。無法由一次錯誤確定外部檔案鎖來源。
- 新增測試包含真實 Tk 控件、批次階段順序、三種模式、圖片處理、嚴格 ID、重複對白、取消／晚到回應、crash/resume、來源變更、校正／頁序失效、跨語言隔離、partial、checksum、atomic failure、lease、外部 port 保護、舊 monitor race 及錯誤內容遮蔽。
- 真實 runtime 探測：`version: 1 (ba360ef)`、MSVC 19.51.36248.0；枚舉到 Vulkan0（Radeon RX 7900 XTX；系統回報 driver `32.0.31041.1004`）。只 tested version/help/device discovery，沒有聲稱此 build 已通過 GLM-OCR。
- `tools/verify_manga_vulkan.py --help` 與 `--inspect-only` 已實際執行；本機報告在忽略追蹤的 `verification-reports/runtime-inspection/verification.json`。
- **B 真機 Vulkan OCR＋翻譯：NOT RUN**。未找到已確認匹配的 GLM-OCR decoder/projector 與使用者漫畫驗收素材。
- **C 漫畫辨識品質：NOT RUN**。未測漏字率、錯字率、幻覺率或閱讀次序準確度；不編造吞吐量、VRAM 或品質數字。
