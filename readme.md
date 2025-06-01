# 夢境分析與視覺化系統

這是一個整合了多個AI模型的夢境分析系統，可以將使用者的夢境描述轉換成故事、圖像，並提供心理分析。

## 主要功能

1. **夢境故事生成**：使用Ollama的Qwen模型將零散的夢境元素編織成故事
2. **故事翻譯**：將生成的中文故事翻譯成英文
3. **圖像生成**：使用Stable Diffusion基於翻譯後的故事生成視覺圖像
4. **心理分析**：使用Ollama的Qwen模型對夢境進行心理解析

## 系統需求

- Python 3.10
- 至少8GB RAM

## 虛擬環境
1. conda create -n dream python=3.10
2. conda activate dream

### 手動安裝

1. 確保已安裝Python 3.8+
2. 克隆本儲存庫：
   ```
   git clone https://github.com/ws97109/dream-analyzer.git
   ```
3. 安裝依賴：
   ```
   pip install -r requirements.txt
   ```
4. 確保已安裝Ollama並下載Qwen模型：
   ```
   ollama pull qwen2.5:14b
   ```
5. 運行應用：
   ```
   python app.py
   ```
6. 在瀏覽器中訪問：http://localhost:5002

## 使用方法

1. 在文本框中輸入您的夢境描述（至少10個字）
2. 點擊「開始分析」按鈕
3. 等待系統處理（這可能需要幾分鐘，取決於您的硬件）
4. 查看生成的故事、圖像和心理分析

## 技術架構

- **前端**：HTML, CSS, JavaScript
- **後端**：Python Flask
- **AI模型**：
  - Ollama的Qwen模型：用於故事生成、翻譯和心理分析
  - Stable Diffusion：用於圖像生成
