# 🖼️ Image RAG (Colpali + LLaMA Vision)

A powerful Retrieval-Augmented Generation (RAG) system combining Colpali's ColQwen image embeddings with LLaMA Vision via Ollama.

## 🌟 Key Features

- 🧬 ColQwen model for generating powerful image embeddings via Colpali
- 🤖 LLaMA Vision integration through Ollama for image understanding
- 📥 Intelligent image indexing with duplicate detection
- 💬 Natural language image queries
- 📄 PDF document support
- 🔍 Semantic similarity search
- 📊 Efficient SQLite storage

## 🛠️ Technical Stack

- **Embedding Model**: ColQwen via Colpali
- **Vision Model**: LLaMA Vision via Ollama
- **Frontend**: Streamlit
- **Database**: SQLite
- **Image Processing**: Pillow, pdf2image
- **ML Framework**: PyTorch


## ⚡ Quick Start

1. Clone and setup environment:
```bash
git clone [repository-url]
python -m venv venv
source venv/bin/activate  # For Mac/Linux
pip install -r requirements.txt
```

2. Install Ollama from https://ollama.com

3. Launch application:
```bash
streamlit run app.py
```

## 💡 Usage

### 📤 Adding Images
1. Navigate to "➕ Add to Index"
2. Upload images/PDFs
3. System automatically:
   - Generates ColQwen embeddings
   - Checks for duplicates
   - Stores in SQLite

### 🔎 Querying
1. Go to "🔍 Query Index"
2. Enter natural language query
3. View similar images
4. Get LLaMA Vision analysis


## 💾 Database Schema

```sql
CREATE TABLE embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    image_base64 TEXT,
    image_hash TEXT UNIQUE,
    embedding BLOB
)
```
