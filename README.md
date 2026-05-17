# llama.cpp Manager 🚀

簡單易用的 GUI 管理器，用於管理 llama.cpp 服務器和模型。

## ✨ 功能特點

- ✅ **圖形化界面** - 簡潔直觀的 GUI
- ✅ **模型管理** - 自動掃描和添加 GGUF 模型
- ✅ **服務器控制** - 一鍵啟動/停止 llama-server
- ✅ **參數配置** - 可視化配置服務器參數
- ✅ **實時日誌** - 顯示服務器運行日誌
- ✅ **資源監控** - 監控 CPU/RAM 使用情況
- ✅ **配置保存** - 自動保存配置和模型列表

## 🚀 快速開始

### 方法 1: 雙擊啟動 (推薦)

直接雙擊 `start.bat` 即可啟動！

### 方法 2: 命令行啟動

```bash
# 安裝依賴
pip install -r requirements.txt

# 運行管理器
python llama_manager.py
```

## 📖 使用說明

### 1. 模型選擇

- **自動掃描**: 點擊 "🔄 掃描" 按鈕自動掃描 `llama-hip` 目錄中的 .gguf 文件
- **手動添加**: 點擊 "📂 添加" 按鈕手動選擇模型文件
- **模型信息**: 顯示模型大小和量化格式

### 2. 服務器設置

- **端口**: API 服務器端口 (默認 8080)
- **GPU 層數**: 卸載到 GPU 的層數 (0-99, 默認 99)
- **上下文大小**: 文本上下文長度 (512-16384, 默認 4096)
- **批次大小**: 批處理大小 (默認 512)

### 3. 啟動服務器

1. 選擇要使用的模型
2. 配置服務器參數
3. 點擊 "▶️ 啟動服務器"
4. 查看日誌輸出確認服務器正常啟動
5. 訪問 http://localhost:8080 使用 API

### 4. 停止服務器

- 點擊 "⏹️ 停止服務器" 按鈕
- 或直接關閉管理器窗口

## 📁 配置文件

### models.json
存儲模型列表：
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
存儲服務器配置：
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

服務器啟動後，可以通過以下方式使用：

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

### 服務器無法啟動

1. **檢查模型路徑**: 確保選擇的模型文件存在
2. **檢查端口占用**: 確保端口未被其他程序占用
3. **查看日誌**: 查看日誌輸出了解錯誤信息

### GPU 無法使用

- 檢查 `GPU 層數` 設置是否為 0 (應設置為 99)
- 確認 ROCm HIP 驅動正常安裝
- 查看 llama.cpp 目錄中的 DLL 文件是否完整

### 顯示編碼問題

如果中文顯示為亂碼，請確保：
- 系統默認編碼為 UTF-8
- 終端機支持中文字體

## 📊 系統要求

- **操作系統**: Windows 10/11
- **Python**: 3.7 或更高版本
- **GPU**: AMD Radeon RX 7900 XTX (或其他 AMD GPU)
- **RAM**: 建議 32GB 或更多
- **VRAM**: 建議 24GB 或更多

## 🔧 高級選項

### 打包為 .exe

如需打包為獨立可執行文件：

```bash
pip install pyinstaller
pyinstaller --onefile --windowed --name="llama-manager" llama_manager.py
```

生成的 `llama-manager.exe` 將在 `dist` 目錄中。

### 自動啟動服務器

在 `config.json` 中添加：
```json
{
  "ui": {
    "auto_start": true,
    "auto_start_model": "your-model-name"
  }
}
```

## 📝 更新日誌

### v1.0.0 (2026-04-28)
- ✅ 初始版本發布
- ✅ 支持模型管理
- ✅ 支持服務器控制
- ✅ 支持實時日誌顯示
- ✅ 支持資源監控

## 🤝 貢獻

歡迎提交 Issue 和 Pull Request！

## 📄 許可證

MIT License

## 🙏 鳴謝

- [llama.cpp](https://github.com/ggerganov/llama.cpp) - 核心推理引擎
- [llama-swap](https://github.com/mostlygeek/llama-swap) - 靈感來源

---

*Created with ❤️ by Claude Code*
