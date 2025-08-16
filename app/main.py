import os
from typing import Dict, List, Optional

import shortuuid
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.pdf_utils import extract_text_from_pdf, split_text_into_chunks
from app.embedding_service import EmbeddingService
from app.milvus_service import MilvusService
from app.question_service import QuestionService
from app.storage import init_db, save_paper, get_paper, save_attempt


app = FastAPI(default_response_class=HTMLResponse)
app.mount("/static", StaticFiles(directory="/workspace/static"), name="static")
templates = Jinja2Templates(directory="/workspace/templates")

# Initialize services
init_db()
embedding_service = EmbeddingService()
milvus_db_path = os.environ.get("MILVUS_DB_PATH", "/workspace/data/milvus_lite.db")
milvus_service = MilvusService(
    db_path=milvus_db_path,
    collection_name=os.environ.get("MILVUS_COLLECTION", "rag_chunks"),
    dim=embedding_service.dimension or 384,
)
question_service = QuestionService()


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {"request": request},
    )


@app.post("/upload", response_class=HTMLResponse)
async def upload(
    request: Request,
    file: UploadFile = File(...),
    prompt: str = Form(""),
    num_questions: int = Form(10),
):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    upload_dir = "/workspace/data/uploads"
    os.makedirs(upload_dir, exist_ok=True)
    dataset_id = shortuuid.uuid()
    saved_path = os.path.join(upload_dir, f"{dataset_id}.pdf")
    with open(saved_path, "wb") as f:
        f.write(await file.read())

    # Parse and embed
    text = extract_text_from_pdf(saved_path)
    chunks = split_text_into_chunks(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No extractable text found in PDF")
    embeddings = embedding_service.embed_texts(chunks)
    milvus_service.upsert_chunks(dataset_id, chunks, embeddings)

    # Retrieve context for question generation
    query_embedding = embedding_service.embed_text(prompt or "Generate questions")
    hits = milvus_service.search(dataset_id, query_embedding, limit=8)
    context_snippets = [h.get("text", "") for h in hits]

    paper = question_service.generate_question_paper(prompt, context_snippets, num_questions=num_questions)
    title = paper.get("title") or "Question Paper"
    paper_id = shortuuid.uuid()
    save_paper(paper_id, dataset_id, title, paper)

    return RedirectResponse(url=f"/paper/{paper_id}", status_code=303)


@app.get("/paper/{paper_id}", response_class=HTMLResponse)
async def view_paper(request: Request, paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")
    return templates.TemplateResponse(
        "paper.html",
        {"request": request, "paper_id": paper_id, "paper": paper},
    )


@app.post("/paper/{paper_id}/submit", response_class=HTMLResponse)
async def submit_answers(request: Request, paper_id: str):
    paper = get_paper(paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found")

    form = await request.form()
    answers: Dict[str, str] = {}
    for q in paper.get("questions", []):
        qid = q["id"]
        answers[qid] = str(form.get(qid, "")).strip()

    obtained, total, breakdown = question_service.grade_answers(paper, answers)
    attempt_id = shortuuid.uuid()
    save_attempt(attempt_id, paper_id, answers, obtained, total, breakdown)

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "paper_id": paper_id,
            "score": obtained,
            "total": total,
            "breakdown": breakdown,
            "share_url": f"{request.url.scheme}://{request.url.netloc}/paper/{paper_id}",
        },
    )