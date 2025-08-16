# Agentic RAG Question Paper (FastAPI + Milvus Lite)

## Setup

1. Create venv (optional) and install dependencies:

```bash
pip3 install -r requirements.txt
```

2. Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

3. Open the app at `http://localhost:8000`.

- Upload a PDF
- Enter prompt/guidelines and number of questions
- Get a shareable link to the generated paper
- Answer questions and receive a score

## Configuration

- `OPENAI_API_KEY`: If set, the app uses OpenAI for question generation; otherwise it falls back to a deterministic heuristic.
- `MILVUS_DB_PATH`: Path to Milvus Lite DB file (default `/workspace/data/milvus_lite.db`).
- `MILVUS_COLLECTION`: Collection name (default `rag_chunks`).

Data is stored in:
- `/workspace/data/uploads` (PDFs)
- `/workspace/data/milvus_lite.db` (Milvus Lite)
- `/workspace/data/app.sqlite3` (SQLite for papers and attempts)