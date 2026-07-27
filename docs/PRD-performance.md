# PRD：llama-manager 性能提升

- 狀態：P0 / P1 已實作，P2 待排期
- 版本：v0.2
- 日期：2026-07-27
- 範圍：翻譯吞吐、VRAM 使用、輸出清理、Whisper 轉錄、UI/UX
- 目標讀者：本 repo 維護者（Windows + AMD ROCm HIP 環境）

---

## 0. 實作摘要（v0.2）

已完成：
- P0-1：llama-server 多 slot（`--parallel`）、continuous batching、FlashAttention。
- P0-2：KV cache K/V 量化設定，預設 `q8_0`；context 預設降至 16384。
- P1-1：行級 / batch 級進度、lines/s、ETA。
- P1-2：併發模式保留即時逐行日誌，經 `log_bus` marshal 回 Tk 主執行緒。
- P1-3：`max_tokens` 按批次大小動態計算，使用者設定改作上限 cap。
- P1-4：清理 malformed YAML 回應，避免 `- translation:` 等標記污染字幕。
- P1-5：壓縮句內大量重複短單位；相同字幕句子只翻譯一次再回填。

本機 llama.cpp ROCm build 已確認支援：
- `--parallel N`
- `--cont-batching`
- `--flash-attn on|off|auto`
- `--cache-type-k/v` 的 `q8_0`、`q4_0` 等型別

回歸測試：`python -m pytest -q`，**384 passed**。

---

## 1. 背景與問題陳述

llama-manager 目前可用，但有一個結構性瓶頸讓「並行翻譯」形同虛設，另外幾處在 VRAM 與體感上有明顯改善空間。以下皆基於實際程式碼查證，非泛泛而論。

### 1.1 核心問題：併發送請求，但 server 只有 1 個 slot（最高優先）

- `subtitle_tab.py` 提供「并发数 / Workers」滑桿（預設 3，`subtitle_tab.py:142`），`local_llm_translator.py:492` 用 `ThreadPoolExecutor(max_workers=self.max_workers)` 同時送出多個批次請求。
- 但 `server_controller.py:52-60` 組 llama-server 指令時**完全沒有 `--parallel` / `-np` 參數**，llama-server 預設只有 1 個 slot。
- 結果：3 條 worker thread 併發送出的請求，在 server 端會**排隊逐一處理**，並行度實際上是 1。UI 讓使用者以為調高 workers 會更快，實際幾乎沒差。

這是目前投入最小、回報最大的一項。

### 1.2 VRAM：context 預設過大 + KV cache 未量化

- `config.json` 目前 `server.context_size = 131072`。字幕/文字翻譯每則請求的實際 token 遠小於此。
- 131072 的 KV cache 以 f16 儲存時非常吃 VRAM，且啟動未帶 `--flash-attn` 與 KV 量化。
- 這直接壓縮了「能開多大模型 / 能開幾個 slot」的空間。

### 1.3 Whisper 可能未用 GPU

- `whisper_transcription.py:298-312` 的 whisper-cli 指令沒有帶 GPU 相關 flag。若你的 whisper.cpp build 支援 HIP/Vulkan，轉錄可大幅加速；需先確認 build 後端。

### 1.4 UI/UX：進度與日誌回饋不足

- 進度條按「檔案數」算百分比（`subtitle_tab.py:334`），翻譯單一大檔時進度條長時間不動。
- translator 內部已算出行級進度與 lines/s（`local_llm_translator.py:553`），但沒送到 UI。
- 併發模式把 `log_callback` 設為 `None`（`local_llm_translator.py:497`，為執行緒安全），導致開併發後看不到逐行翻譯過程，只剩批次完成訊息。

---

## 2. 目標與非目標

### 2.1 目標
- G1：讓「并发数」真正生效，翻譯吞吐相對現況提升 ≥ 2x（3 slot 情境）。
- G2：在相同模型下降低 VRAM 佔用，或在相同 VRAM 下可開更大模型 / 更多 slot。
- G3：翻譯與轉錄過程進度平滑、可見 ETA 與速率。
- G4：常見用途（翻譯 vs 長對話）可一鍵切換參數 preset。

### 2.2 非目標
- 不更換推論引擎（維持 llama.cpp）。
- 不重寫翻譯 prompt 策略（兩步/單步邏輯維持）。
- 不做多 GPU 分散式。

### 2.3 成功指標
| 指標 | 現況（基準） | 目標 |
|---|---|---|
| 翻譯吞吐（lines/s，3 worker） | 待量測 | ≥ 2x 基準 |
| 131072 ctx 下 KV VRAM | f16 基準 | q8_0 約減半 |
| 單一大檔進度條更新頻率 | 每檔一次 | 每行/每批 |
| 用途切換操作步數 | 手動改多個欄位 | 1 鍵 |

> 基準值需在實作前先跑一次量測（見 §6 驗收）。

---

## 3. 需求細節

### P0-1　server 開併發槽 + FlashAttention（對應 G1）

**改動位置**：`server_controller.py` `start()`；`server_tab.py` 傳參；`config.json` schema。

**新增 server 參數**
- `--parallel N`（`-np N`）：slot 數量。與翻譯分頁 workers 對齊。
- `--cont-batching`：連續批次（新版預設開，明確帶上較保險）。
- `--flash-attn`（新版可用 `--flash-attn on`）：省 attention VRAM 並加速。

**KV cache 與 slot 的取捨**
- `-np N` 會把 KV cache 均分為 N 份，每 slot 得 `context_size / N`。
- 例：ctx=16384、`-np 3`，每 slot ≈ 5461 token，對字幕翻譯充足。
- 因此「併發」與「大 context」在翻譯用途下不衝突，但要避免同時把 ctx 拉到 131072 又開多 slot（VRAM 會爆）。

**`ServerController.start()` 簽名擴充（建議）**
```python
def start(self, model_path, port, host, gpu_layers, context_size, batch_size,
          parallel=1, flash_attn=True, cont_batching=True,
          cache_type_k=None, cache_type_v=None):
    cmd = [
        str(self._server_exe),
        "-m", model_path,
        "--port", str(port),
        "--host", host,
        "-ngl", str(gpu_layers),
        "-c", str(context_size),
        "-b", str(batch_size),
    ]
    if parallel and parallel > 1:
        cmd += ["--parallel", str(parallel)]
    if cont_batching:
        cmd += ["--cont-batching"]
    if flash_attn:
        cmd += ["--flash-attn", "on"]   # 舊版為無值 flag：僅 "--flash-attn"
    if cache_type_k:
        cmd += ["--cache-type-k", cache_type_k]
    if cache_type_v:
        cmd += ["--cache-type-v", cache_type_v]
    ...
```

**驗收**：啟動後 server log 應顯示 `n_parallel = N` / 對應 slot 數；翻譯開 3 worker 時吞吐相對 1 worker 明顯上升。

> ⚠️ flag 相容性：`--flash-attn` 在不同 llama.cpp 版本有「無值旗標」與 `on/off/auto` 兩種形式；`--cache-type-k/v` 也依 build 而定。實作前先跑 `llama-server --help` 對照，並在 §5 setup 記錄實測結果。

---

### P0-2　KV cache 量化（對應 G2）

**改動位置**：同 P0-1 的 `--cache-type-k/v`。

- 加 `--cache-type-k q8_0 --cache-type-v q8_0`：KV cache 由 f16 降 q8_0，約省一半 KV VRAM，翻譯品質幾乎無損。
- 需搭配 `--flash-attn`（部分 build 的量化 KV 需 FA 支援）。
- UI 提供下拉：`f16`（預設穩定）/ `q8_0`（省 VRAM，建議）/ `q4_0`（最省，品質略降）。

**驗收**：同模型同 ctx，開 q8_0 後 VRAM 佔用下降；翻譯結果抽樣比對無明顯劣化。

---

### P1-1　行級進度 + ETA + 速率（對應 G3）

**改動位置**：`subtitle_tab.py` 進度回報。

- 目前 `_run_translation` 只在每檔結束時更新百分比（`subtitle_tab.py:334`）。
- translator 已透過 `progress_callback(current, total, status)` 回報行級進度（`translate_srt` → `file_progress`）。
- 改法：進度條改綁行級 `current/total`，並顯示 `已翻 X/Y 行 · N.N lines/s · ETA mm:ss`。速率 translator 內部已算（`local_llm_translator.py:553`），可一併往上送或在 UI 端估算。

**驗收**：翻譯單一大 SRT 時進度條平滑推進，顯示 ETA。

---

### P1-2　併發模式的即時日誌（對應 G3）

**改動位置**：`local_llm_translator.py:490-508`。

- 併發分支目前 `log_callback=None`（避免跨執行緒直接操作 Tk）。
- 改法：改走 `log_bus.emit(level, msg)`（`ui_helpers.py` 的 pub/sub 本就設計為跨執行緒安全的中介），由訂閱端在 Tk 主執行緒 marshal（`llama_manager.py:280` 已用 `root.after(0, ...)` 這樣做）。
- 效果：開併發後仍可看到逐行 / 逐批翻譯進度，Debug Log 視窗同步。

**驗收**：workers>1 時，日誌區可見即時翻譯行，且無 Tk 跨執行緒錯誤。

---

### P1-3　max_tokens 隨 batch 動態調整（對應 G2）

**改動位置**：`subtitle_tab.py:298-302` 組 config 處，或 translator 內。

- 現況：batch=15、max_tokens=16384 固定。15 行字幕的輸出遠用不到 16K，過大的 max_tokens 會（a）預留過多 KV、（b）在模型跑飛時放大 hang 風險。
- 改法：`max_tokens = clamp(batch_size * per_line_budget, min=512, max=使用者上限)`，`per_line_budget` 取保守值（如 120-200 token/行）。
- 保留使用者手動覆寫（進階選項）。

**驗收**：翻譯批次的請求 KV 佔用下降、逾時 / 截斷（LengthFinishReasonError → 逐行 fallback）次數下降。

---

### P1-4　LLM 回應標記清理

**改動位置**：`translation/local_llm_translator.py` `_parse_yaml_result()`。

部分模型會漏掉 YAML `id`，或將下一個 item 的 `- translation:` / `step1:` / `step2:` 標記吞入上一個欄位。實作內容：
- 使用 YAML list marker `- ` 切分 item，不再依賴 `- id:`。
- 欄位擷取在下一個欄位或 item 邊界停止。
- `_clean_field_value()` 清除洩漏標記、巢狀引號及跨行污染。

**驗收**：輸出字幕不包含 `- translation:` 等格式標記；漏 `id` 的 batch 仍能逐項解析。

---

### P1-5　重複內容壓縮與字幕去重

**改動位置**：`utils/srt_parser.py` `collapse_repeats()`；`translation/local_llm_translator.py` `translate_srt()`。

1. **句內重複壓縮**
   - 將至少 5 次、長度 1-4 字元的重複單位壓成 `<unit>...`。
   - 支援有分隔符及連續形式，例如 `あ、あ、あ、...`、`ああああ...`。
   - 普通短強調（如 `はは`、`あああ`）保持不變。
2. **跨字幕去重**
   - 以空白正規化後的 source text 建立唯一句子清單。
   - 每個唯一句子只送 LLM 一次，再將翻譯回填至所有原 timestamp / line number。
   - 不修改 caller 傳入的 subtitle dictionaries。

**預期收益**：若 1000 行字幕只有 200 個唯一句子，batch size 15 時，LLM batch calls 可由約 67 次降至約 14 次；完全相同的 100 行只需翻譯 1 次。

**驗收**：重複句共用同一翻譯，原時間軸與順序完整保留；log 顯示 `Deduplicated N repeated line(s): X -> Y LLM inputs`。

---

### P2-1　用途 preset（對應 G4）

**改動位置**：`server_tab.py` 加 preset 按鈕群。

- 「翻譯模式」：context 8192-16384、`-np 3`、KV q8_0、flash-attn on。
- 「長對話模式」：context 65536+、`-np 1`、KV f16。
- 一鍵套用到現有欄位，使用者仍可微調後再啟動。

**驗收**：切 preset 後對應欄位即時更新；啟動指令符合 preset。

---

### P2-2　切換模型自動釋放（對應 G4）

**改動位置**：`server_tab.py` `_do_start_server`。

- 現況需手動：停 server → 釋放記憶體 → 啟新模型。
- 改法：啟動前若偵測 `self._server.running`，先 `stop()` + `gc.collect()`（server_tab 已 import gc）再啟動，並在 log 說明。
- 屬破壞性較低操作，但仍建議在 log 明確標示「已停止舊 server 並釋放」。

**驗收**：切換模型只需選模型 + 按啟動一步。

---

### P2-3　Whisper GPU 加速（對應 G1，需前置確認）

**改動位置**：`whisper_transcription.py:298` `WhisperCliAdapter.transcribe`。

- 先確認 whisper.cpp build 後端：跑 `whisper-cli --help`，看是否有 GPU 相關 flag（`-ng` 為「停用 GPU」；HIP/Vulkan build 通常預設用 GPU，Vulkan 可用 `--device`）。
- 若 build 為 CPU-only，此項需重新編譯 whisper.cpp（HIP 或 Vulkan）才有效，屬較大前置工作。
- 不要盲加 flag；以 build 實測為準。

**驗收**：轉錄同一檔案，GPU 後端相對 CPU 明顯加速；`nvidia-smi` / AMD 對應工具可見 GPU 使用率。

---

## 4. 影響檔案總覽

| 檔案 | 改動 | 對應需求 |
|---|---|---|
| `server_controller.py` | `start()` 加併發/FA/KV 參數 | P0-1, P0-2 |
| `server_tab.py` | 新設定欄位、preset、自動釋放 | P0-1, P0-2, P2-1, P2-2 |
| `config.json` / `config_manager.py` | 新增 server 參數 schema 與預設 | P0-1, P0-2 |
| `subtitle_tab.py` | 行級進度、速率與 ETA | P1-1 |
| `translation/local_llm_translator.py` | 併發即時 log、動態 max_tokens、回應清理、字幕去重 | P1-2～P1-5 |
| `utils/srt_parser.py` | 句內大量重複內容壓縮 | P1-5 |
| `whisper_transcription.py` | GPU flag（視 build） | P2-3 |

---

## 5. 詳細 Setup 與環境需求

### 5.1 前置：確認你的 llama.cpp / whisper.cpp 支援哪些 flag

在啟動前先對照本機 build 的說明（flag 依版本 / 後端而異）：

```bash
# llama-server 支援的參數（重點看 parallel / flash-attn / cache-type）
D:/AI/llama/llama.cpp/llama-hip/llama-server.exe --help | grep -iE "parallel|flash|cache-type|cont-batch"

# whisper-cli 支援的參數（重點看 GPU / device / threads）
D:/AI/llama/llama-manager/whisper.cpp/build/bin/Release/whisper-cli.exe --help
```

本機 ROCm build 實測結果：
- `--flash-attn`：需 `on/off/auto`（目前使用 `--flash-attn on`）。
- `--cache-type-k/v`：支援 `q8_0`、`q4_0`，亦支援 f32/f16/bf16/q4_1/iq4_nl/q5_0/q5_1。
- `--parallel N` 與 `--cont-batching`：支援。
- whisper GPU：本輪不處理；現有速度足夠。

### 5.2 建議的 config.json（新增欄位）

在 `server` 區塊新增（保留舊欄位）：

```json
{
  "server": {
    "port": 8080,
    "host": "0.0.0.0",
    "gpu_layers": 99,
    "context_size": 16384,
    "batch_size": 512,
    "threads": -1,
    "parallel": 3,
    "flash_attn": true,
    "cont_batching": true,
    "cache_type_k": "q8_0",
    "cache_type_v": "q8_0"
  }
}
```

> 注意：把 `context_size` 從 131072 調回 16384 是翻譯用途的建議值；若你有長對話需求，用 §3 P2-1 的 preset 切換，而非長期把 ctx 開到最大。

### 5.3 翻譯分頁建議設定

- Workers（并发数）：與 server `parallel` 對齊（例如同為 3）。
- 快速模式（單步）：維持勾選（已是預設，速度優先）。
- Batch Size：10-20；配合 P1-3 讓 max_tokens 自動縮小。

### 5.4 手動驗證啟動指令（不改碼先驗證效果）

在改碼前，可先手動用目標參數起 server，確認吞吐與 VRAM 改善，再回頭實作 UI：

```bash
D:/AI/llama/llama.cpp/llama-hip/llama-server.exe \
  -m "D:/AI/llama/llama.cpp/llama-hip/gemma-4-E2B-it-uncensored-Q8_0.gguf" \
  --port 8080 --host 0.0.0.0 \
  -ngl 99 -c 16384 -b 512 \
  --parallel 3 --cont-batching --flash-attn on \
  --cache-type-k q8_0 --cache-type-v q8_0
```

啟動後在 server log 確認：
- `n_parallel`（或 slot 數）= 3
- KV cache 大小 / VRAM 佔用下降（對照未加 KV 量化時）

---

## 6. 驗收計劃

### 6.1 先建立基準（實作前必做）
1. 用現況 config（131072 ctx、無 parallel）翻譯一份固定的 SRT（建議 300-500 行），記錄總時間與 lines/s（log 已輸出，`local_llm_translator.py:553`）。
2. 記錄啟動後 VRAM 佔用。

### 6.2 分階段驗收
| 階段 | 驗收動作 | 通過標準 |
|---|---|---|
| P0-1 | 同 SRT 開 3 worker + `-np 3` | lines/s ≥ 2x 基準 |
| P0-2 | 開 KV q8_0 | VRAM 下降；翻譯抽樣品質無明顯劣化 |
| P1-1 | 翻譯大檔 | 進度條平滑、顯示 ETA |
| P1-2 | workers>1 | 日誌可見即時翻譯行、無 Tk 錯誤 |
| P1-3 | 翻譯多批 | 動態 token cap 生效；截斷 / hang 風險下降 |
| P1-4 | 模型漏 id 或輸出 malformed YAML | 字幕不含 `- translation:` 等標記 |
| P1-5 | 高重複 SRT | 唯一句子各翻譯一次；時間軸與順序不變 |
| P2-x | preset / 自動釋放 / whisper GPU | 見各節驗收 |

### 6.3 回歸
- 完整測試：`python -m pytest -q`。
- 2026-07-27 實測結果：**384 passed in 24.62s**。
- 已覆蓋 server flags、動態 max_tokens、malformed YAML 清理、句內重複壓縮、跨字幕去重與 timestamp 回填。

---

## 7. 風險與緩解

| 風險 | 說明 | 緩解 |
|---|---|---|
| flag 版本不相容 | `--flash-attn` / `--cache-type` 形式依 build 而異 | §5.1 先 `--help` 實測；UI 提供關閉選項 |
| 併發 + 大 ctx 爆 VRAM | `-np N` 均分 KV，但同時開大 ctx 會超量 | preset 綁定合理組合；啟動失敗時 log 明確提示 |
| KV 量化品質風險 | q4_0 可能影響翻譯 | 預設 q8_0，q4_0 為選用 |
| 併發 log 執行緒安全 | 直接操作 Tk 會崩 | 一律經 log_bus + `after(0, ...)` |
| whisper 為 CPU build | 加 flag 無效甚至報錯 | 先確認 build，必要時重編 |

---

## 8. 實作狀態與後續順序

1. ✅ **P0-1 + P0-2**：server 併發、continuous batching、FlashAttention、KV q8_0。
2. ✅ **P1-1 + P1-2**：行級進度、速率 / ETA、併發即時 log。
3. ✅ **P1-3**：動態 max_tokens。
4. ✅ **P1-4 + P1-5**：LLM 標記清理、句內重複壓縮、跨字幕去重。
5. ⏳ **P2-1 / P2-2**：用途 preset / 切模型自動釋放，屬操作便利性改善。
6. ⏸️ **P2-3**：Whisper 目前速度足夠，暫不排期。
