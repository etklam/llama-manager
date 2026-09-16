# 翻譯可靠性／效能強化工程報告（Audit Hardening Phase）

日期：2026-09-16
環境：Windows 11 Pro、Python 3.12.10、openai 2.8.1、httpx 0.28.1、pytest 9.0.2（tenacity 已從依賴移除，環境中殘留的 9.1.4 不再被任何模組引用）

## 版本與檔案

- 起始 SHA：`4375a13`（HEAD；靜態稽核基準為其前兩個 commit 的 `85e16bd`，其間的 Remote LLM 連線與 `config.json` `llm` 區段變更皆保留並通過測試）
- 結束狀態：工作樹（未要求 commit；`git stash create` 樹快照 `2799109`，未追蹤新檔另列於下）
- 變更：19 個既有檔案（+1041 / −278）＋ 8 個新檔

新檔：
- `translation/transport.py` — 傳輸重試政策（唯一重試擁有者）
- `utils/atomic_io.py` — 原子寫入 + CommitGate
- `tests/translation/test_transport_retry.py`（17 測試）
- `tests/translation/test_run_guards.py`（8）
- `tests/translation/test_subtitle_id_integrity.py`（12）
- `tests/translation/test_txt_full_text.py`（9）
- `tests/test_run_snapshot.py`（6）
- `tests/utils/test_atomic_io.py`（10）

## 各項發現：重現、修復、回歸測試

### 1. 重試放大

**重現（修復前）**：`OpenAIClient` 以 SDK 預設 `max_retries=2` 建立用戶端，外面再包 tenacity `stop_after_attempt(3)` — 兩個重試擁有者相乘，一個邏輯請求最多 9 次 HTTP 嘗試。錯誤未分型：401/404/422 同樣重試 3×3 次；`Retry-After` 被忽略；llama-server 的 500「prompt is too long」被當成暫時性 5xx 原樣重送。重試耗盡後，`translate_srt` 的重建迴圈把失敗批次丟進 `_recover_missing`，再對每句各付一輪完整重試（N×3 次更多請求）。

**修復**：
- SDK 用戶端以 `max_retries=0` 建立（`translation/openai_client.py`），重試唯一擁有者為 `translation/transport.py` 的 `request_with_retry`。
- 型別化分類（`classify`）：`AuthenticationError`/`PermissionDeniedError`/`NotFoundError`/`UnprocessableEntityError`/`BadRequestError` → `FatalProviderError`，立即失敗；`APIConnectionError`/`APITimeoutError`/`RateLimitError`(429)/5xx → 可重試；錯誤文字含 context 標記（含 llama-server 的「prompt is too long」）或 `LengthFinishReasonError` → `ContextLengthError`，不重試、由呼叫端縮小請求。未知例外型別一律視為 fatal，不盲重試。
- 上限：最多 `MAX_ATTEMPTS=3` 次 HTTP 嘗試、累計預算 `RETRY_BUDGET_SECONDS=30s`；退避指數 1s→2s（上限 10s）；`Retry-After` 有數值才採用、上限 30s 且受預算約束（會吃光剩餘預算的等待直接判定預算耗盡）。
- 每次嘗試的診斷只含狀態碼／錯誤型別／時間，不含請求內容、金鑰或回應物件。
- `translate_srt` 執行迴圈：批次失敗若為 `FatalProviderError`/`ProviderUnavailableError`，記為 fatal、停止派發佇列中的批次、整輪中止（不再進入逐句補譯）；`_recover_missing` 遇同類錯誤立即 re-raise。context 超限維持既有的縮小請求策略（去掉 nearby → 對半切 → 壓縮背景），不是相同重試。

**回歸測試**：`tests/translation/test_transport_retry.py` 以實際安裝的 SDK＋`httpx.MockTransport`（非 mock `chat.completions.create`）＋假時鐘／假 sleep 驗證：500→200 恰 2 次嘗試、持續 500 恰 3 次（無 SDK 放大）、401/404/422 恰 1 次、prompt-too-long 500 恰 1 次且拋 `ContextLengthError`、Retry-After 2s 覆蓋退避、超額 Retry-After 1 次即預算中止、假時鐘下的預算上限、SDK 用戶端 `max_retries==0`。`tests/translation/test_run_guards.py` 驗證 fatal／transport 耗盡不展開成逐句請求（批次 1 次＋補譯第一句即中止）。

### 2. 取消邊界

**重現（修復前）**：`_call_api` 送出前不檢查取消 token；傳輸重試間不檢查；context 超限的遞迴切分再派發前不檢查；`translate()`（TXT 用）完全沒有取消檢查。

**修復**：`translate_srt`／`translate_full_text` 以 `_cancellation_scope` 把該輪 token 綁到 translator 與 HTTP adapter（`client.cancel_check`）：
- 每次邏輯請求派發前（`_check_run_cancelled`）、每次傳輸重試與退避前後（transport 層）、每個 context 切分遞迴進入點（`_translate_batch_lines` 開頭）、補譯每句前、Story 分析請求之間（既有）皆檢查。
- 執行迴圈維持 at-most-workers 派發；取消或 fatal 已知時 `submit_next` 停止派發佇列工作。
- 同步請求仍在途時不承諾瞬時中斷；返回後不再送出新請求，且取消的結果不發佈（見第 4 項 CommitGate）。

**回歸測試**：`test_run_guards.py`：取消於 context 切分之間（第一個請求失敗後 token 翻轉 → 不送出切分的第二個請求）、Story 分析完成與翻譯批次之間取消、派發前取消零請求；`test_transport_retry.py`：退避中被取消（假 sleep 內翻轉 token → `TransportCancelledError`，僅 1 次 HTTP）；`test_txt_full_text.py`：stop 落在翻譯返回與寫入之間 → 不寫出、不記 Success。

### 3. 凍結執行狀態（RunSnapshot）

**重現（修復前）**：`_on_tab_changed` 以數字索引路由，進入「字幕翻譯」分頁即呼叫 `refresh_model` → 無條件重建 `self._translator`，進行中的多檔批次在檔案之間被換掉模型／端點；`_run_translation` 在 worker 執行緒重讀 `_get_target_code()` 與 `_replace_var.get()`（跨執行緒讀 Tk 變數）；舊輪的晚期 callback（如 preflight 失敗的 abort）可能翻動新一輪的按鈕狀態。PipelineCard 的 `_run_pipeline` 也在 worker 執行緒讀 Tk 變數。

**修復**：
- `subtitle_tab.RunSnapshot`（frozen dataclass）：檔案、目標語言、replace 旗標、模型／端點、context 模式、請求 workers、生成設定，全部在 `_start_translation` 的 UI 執行緒讀取後凍結；worker 只讀快照。
- Preflight 與執行同一目標（`config['target']`）；探測得的 workers／context 只寫入該輪的 config **副本**，快照原值不動（重跑仍請求原設定）。
- `refresh_model` 在 `self._translating` 時不重建 translator（只更新顯示）；執行中的執行緒使用 run 區域變數的 translator。
- `_on_tab_changed` 改為 widget 身分比對（`nametowidget(select())` 與 `is`），不再用索引。
- `_on_translation_done`／`_on_translation_aborted` 帶 run_id，與 `_run_seq` 不符即忽略。
- PipelineCard `_start_pipeline` 在 UI 執行緒讀完所有 Tk 變數再開 worker；PipelineRunner 既有的 preflight 後快照保留不變。
- 執行結束時關閉 run 擁有的 HTTP client（subtitle tab 在 `_preflight_and_translate` 的 finally；pipeline 在 `run()` 的 finally）；注入的共用 client 永不被關閉（`OpenAIClient.close()` 對注入的 http_client 完全跳過 SDK close）。

**回歸測試**：`tests/test_run_snapshot.py`：兩檔案輪次在檔案之間切換分頁＋改 Tk 變數，兩檔仍用原 translator／目標語言／replace；`refresh_model` 不重建執行中 translator；晚期 done／abort callback 不影響新輪；preflight 容量只套用到副本；Pipeline 兩檔之間改設定仍只建一次 translator、設定不變。`test_desktop_ui.py` 新增身分路由測試。另以原始碼檢查確保 `_run_translation` 內不出現 Tk 變數讀取。

### 4. 輸出與設定檔提交保護

**重現（修復前）**：`subtitle_tab._translate_file` 與 `pipeline_runner._translate_file` 以 `Path.write_text` 直接覆寫輸出（Replace original 時中途失敗即毀損輸入檔）；`ConfigManager._save_to_file` 以 `open('w')` 直接寫；非空但無效的 SRT 在字幕分頁解析為 0 句 → `translate_srt([])` → 寫出空檔並記錄 Success。

**修復**：
- `utils/atomic_io.atomic_write_text`：同目錄暫存檔、完整寫入＋flush＋fsync、關閉 handle 後 `os.replace`（Windows 上的替換在 handle 關閉後為原子操作）、失敗清理暫存檔、舊檔案位元組在任何失敗點都保持原狀。
- `CommitGate`：以 whisper `CancellationToken._commit` 為參考的 race-aware 邊界 — cancel 與 commit 在同一鎖下序列化，取消贏得競爭時不發佈。Subtitle 與 Pipeline 兩個入口的輸出寫入都經過它。
- `ConfigManager` 使用同一原子寫入；未知設定鍵永遠保留；save 失敗時檔案保持舊位元組、記憶體狀態的行為已在 docstring 明確記錄（記憶體已更新、檔案未動，直到下次成功儲存）。
- 非空 SRT 解析 0 句 → 明確 RuntimeError（兩入口）；空檔在 pipeline 維持明確 skip。

**回歸測試**：`tests/utils/test_atomic_io.py`：fsync 失敗／replace 失敗保留舊位元組且無暫存殘留、暫存檔在目標同目錄、CommitGate 的取消／提交／競爭序列化（執行緒級）；`tests/test_config_manager.py` 新增：save 失敗保留檔案、未知鍵存活、save 失敗的記憶體狀態；`test_txt_full_text.py`：非空 0 句 SRT 在 Replace original 下原檔不動；`test_pipeline_runner.py`：0 句 SRT 記為失敗檔、無輸出。Whisper 的 candidate/commit 未改動，`tests/test_whisper_transcription.py` 全數通過。

### 5. 字幕內容與 ID 完整性

**重現（修復前）**：`parse_srt_from_string` 見到純數字行即終止當前 cue 的文字收集 — 對白「123」會截斷多行對白且數字被丟棄。`_parse_yaml_result` 對缺 ID 的項目以結果位置發明 ID（`len(results)+1`），混合明確／缺 ID 或短缺的回覆會把譯文錯位到錯誤的 cue。

**修復**：
- 數字行僅在**下一個非空行是時間戳**時才視為區塊序號（前瞻）；否則是對白內容。多行對白與破折號行原樣保留。
- `_parse_yaml_result` 標記 `explicit_id`；新 `_map_batch_ids` 驗證：明確 ID 需在 1..N 內、重複保留第一筆（記警告）、範圍外丟棄（記警告）；混合明確／缺 ID 只信任明確映射；全無 ID 只在筆數與批次完全相符時才按位置映射，否則全部交由逐句補譯安全回收。註解／拒絕文字（無欄位項目）本就不會成為譯文；截斷的最後欄位（`translation:` 空值或欄位名殘缺）視為缺句進補譯。

**回歸測試**：`tests/translation/test_subtitle_id_integrity.py`（A/B/C 全套）：重排明確 ID、重複 ID、範圍外 ID、ID-less B/C（不按位置猜）、筆數相符的 ID-less（按位置）、混合回覆、截斷最終項、拒絕文字；SRT 解析：數字對白行、序號 123＋時間戳、多行破折號對白、結尾數字行。

### 6. TXT 路由與完整性

**重現（修復前）**：Pipeline 把 `.txt` 送進 `parse_srt_from_file` → 無時間戳 → 0 句 → 永遠失敗；字幕分頁用 `translate()` 單發請求，無預算檢查，length 截斷的回覆會被 `_parse_yaml_result` 撈出第一項當成整份譯文存檔記 Success。

**修復**：新增 `LocalLLMTranslator.translate_full_text` — 兩個入口（字幕分頁、Pipeline）共用：整份文件一次請求；輸出預算按文件大小估算（下限 1024、上限使用者設定）；`request_fits` 先檢查（未知 context 用保守 4096 估算），放不下 → `TxtTooLargeError` 明確失敗（不引入無上限長文件實作，既有輸出不變）；`finish_reason == 'length'` → `TranslationIncompleteError`，不發佈部分譯文；成功路徑沿用 YAML／標籤解析。與短字幕補譯完全分離。Story 模式對 TXT 明確記錄後走標準全文路徑。

**回歸測試**：`tests/translation/test_txt_full_text.py`（譯者層＋字幕分頁入口：完成文件、length 截斷不寫檔、超額拒絕零請求、空文字、派發前取消、路由、0 句 SRT、stop 前取消不發佈）；`tests/test_pipeline_runner.py`（Pipeline 入口：TXT 不經 SRT 解析、逐項斷言 `translate_full_text` 呼叫與輸出內容、超額 TXT 失敗且無輸出）。

### 7. 配套變更

- HTTP client：`OpenAIClient` 建構時即決定性初始化（非惰性）；`close()` 冪等、注入的 http_client 不關閉；subtitle tab 與 pipeline 各自在輪次結束關閉 run 擁有的 client（輪內連線重用不變）。
- 指標：`TransportStats`（logical／HTTP 請求數、重試數），`translate_srt` 結束日誌附上；Story 分析耗時、總耗時與 lines/s 維持既有。
- 日誌：移除完整回應物件／完整回應文字的 debug 記錄（改為字數）；傳輸診斷不含金鑰／請求內容；`LLMTarget.__repr__` 原有金鑰遮罩保留。
- `requirements.txt`：記錄實測版本與理由（openai>=2,<3、httpx>=0.28,<1；tenacity 移除），無憑空猜測的 pin。
- 文件：README（重試／取消／TXT／原子寫入／RunSnapshot 說明）、CONTEXT.md（Transport Policy、Transport Attempt、Run Snapshot、Commit Gate 詞彙）、本報告。

## 測試指令與實際結果

全部離線執行（mocked HTTP transport、假時鐘、暫存目錄；不觸及真實 config／模型路徑 — repo 的 `config.json` 未被測試寫入）：

| 指令 | 結果 |
| --- | --- |
| `python -m pytest`（全套，含 coverage） | **731 passed, 0 failed, 0 skipped**（12.4s；覆蓋率 90%） |
| 基準（變更前）`python -m pytest --no-cov -q` | 663 passed, 0 failed, 0 skipped（25.4s） |
| `python -m pytest tests/translation/test_transport_retry.py tests/translation/test_run_guards.py tests/translation/test_subtitle_id_integrity.py tests/translation/test_txt_full_text.py tests/test_run_snapshot.py tests/utils/test_atomic_io.py --no-cov -q` | 62 passed |
| 本地＋Story Context 回歸：`python -m pytest tests/translation/test_story_context_translation.py tests/translation/test_local_llm_translator.py tests/translation/test_translation_recovery.py tests/test_whisper_transcription.py --no-cov -q` | 111 passed |
| `python -m pytest tests/ --ignore=tests/translation --no-cov -q` | 432 passed |

未執行的檢查：無（無 skipped；Tk 可用故 UI 測試皆執行）。測試期間一次執行曾把 mocked 輸出寫到 `D:\test\`（舊測試以 `/test/...` 假路徑＋patch `write_text`，被原子寫入繞過）— 已修正測試改用 `tmp_path` 並刪除該次污染。

## 已驗證的傳輸嘗試上限

- 持續 500：恰 3 次 HTTP（SDK 重試為 0，無 9 次放大）— `test_persistent_500_stops_at_max_attempts_without_sdk_amplification`
- 401／404／422／prompt-too-long：各恰 1 次
- 預算耗盡先於次數上限：2 次
- 退避中取消：1 次，之後零派發
- 逐句補譯遇 fatal／耗盡：立即中止，不逐句放大

## 取消／輸出完整性檢查彙總

派發前、重試退避前後、context 切分遞迴、補譯每句、Story 分析之間皆檢查 token；stop 落於最後回應與寫入之間時 CommitGate 抑制發佈；取消輪次不寫出任何檔案、不記 Success；非空 0 句 SRT 在 Replace original 下原檔保持原狀。

## 剩餘限制（無未經測量的加速宣稱）

- 同步 HTTP 請求在途時無法瞬時中斷；取消在其返回或 timeout 後生效（文件與行為一致）。
- 長文件 TXT 為明確拒絕而非實作；超過 context 的文件需使用者自行切分。
- `_map_batch_ids` 對「全無 ID 且筆數相符」的回覆仍按位置映射 — 這是該形狀下唯一可用資訊，錯位風險以筆數相符收窄，無法完全消除。
- 拒絕文字／文前註解的判斷依賴結構（無欄位即非譯文），不做語意層面的拒答偵測。
- Story 模式的分段／合併請求數仍取決於文件與 context 大小（既有上限內），本階段未改變其請求形狀。
- 遠端 API 設定檔 UI 為既有功能（保留並測試）；OCR 不存在亦未新增。
- 效能：本階段消除的是失敗路徑的請求放大（worst case 每邏輯請求 9→3 次、逐句放大歸零）；正常路徑未做未經測量的速度宣稱。
