# PDF-RAG

A lightweight **Retrieval-Augmented Generation (RAG) pipeline** for working with PDF documents.
This project extracts text from PDFs (with or without OCR), chunks the text, generates embeddings locally using `sentence-transformers`, and stores them in a **Postgres database with pgvector** for semantic search and retrieval.

---

## 🚀 Features

* Extract text from PDFs:
  * Direct text extraction (`PyMuPDF`)
  * OCR-based extraction (`ocrmypdf`) for scanned documents
* Text chunking for efficient embedding
* Local embeddings via `sentence-transformers`
* Postgres + pgvector integration for semantic search
* Configurable environment with `.env`

---

## 📂 Project Structure

```
.
├── main.py                # Entry point for the pipeline
├── requirements.txt       # Python dependencies
├── .gitignore
├── init.sql               # SQL schema/init file for Postgres
├── utils/
│   ├── dbLoader.py        # Handles DB connection and operations
│   ├── getTextFromPDF.py  # Extract text from digital PDFs
│   ├── getTextWithOCR.py  # Extract text from scanned PDFs using OCR
│   ├── LocalEmbeddingGenerator.py # Generate embeddings locally
│   └── textChunker.py     # Split text into chunks
```

---

## 🔧 Installation

### 1. Clone the repository

```bash
git clone https://github.com/skygig/pdf-rag.git
cd pdf-rag
```

### 2. Set up Python environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file in the root directory:

```ini
DB_NAME=rag_db
DB_USER=postgres
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5433
```

---

## 🐳 Setup Postgres with pgvector (Docker)

Run the following command to start a Postgres container with pgvector enabled:

```bash
docker run -d \
  --name rag-pg \
  -e POSTGRES_USER=postgres \
  -e POSTGRES_PASSWORD=your_password \
  -e POSTGRES_DB=rag_db \
  -p 5433:5432 \
  pgvector/pgvector:pg16
```

### Initialize database schema

Copy and execute the contents of `init.sql` inside the Postgres container:

```bash
docker cp init.sql rag-pg:/init.sql
docker exec -it rag-pg psql -U postgres -d rag_db -f //init.sql
```

---

## ▶️ Usage

Run the pipeline with the following commands:

### Process and store a PDF document

```bash
python main.py file.pdf
```

### Interactive search mode

```bash
python main.py --search
```

### Quick search with a query

```bash
python main.py --quick-search "query"
```

### Check embedding status in the database

```bash
python main.py --status
```

---

## 📦 Dependencies

* [PyMuPDF](https://pymupdf.readthedocs.io/)
* [psycopg2-binary](https://www.psycopg.org/)
* [python-dotenv](https://github.com/theskumar/python-dotenv)
* [ocrmypdf](https://ocrmypdf.readthedocs.io/)
* [sentence-transformers](https://www.sbert.net/)
* [torch](https://pytorch.org/)

---
