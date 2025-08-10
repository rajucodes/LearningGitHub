import os
import io
import json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import streamlit as st
from dotenv import load_dotenv
from pypdf import PdfReader
from openai import OpenAI


# ---------------------- Config & Setup ----------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
DEFAULT_GENERATION_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


@dataclass
class RetrievedChunk:
    text: str
    metadata: Dict[str, Any]
    score: float


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set. Add it to your environment, .env file, or enter it in the sidebar.")
    return OpenAI(api_key=api_key)


def extract_text_from_pdf(file_bytes: bytes) -> List[Tuple[str, Dict[str, Any]]]:
    reader = PdfReader(io.BytesIO(file_bytes))
    docs: List[Tuple[str, Dict[str, Any]]] = []
    for page_index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if not text.strip():
            continue
        # Split by paragraphs for better granularity
        for para in [p.strip() for p in text.split("\n\n") if p.strip()]:
            docs.append((para, {"page": page_index + 1}))
    return docs


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1536), dtype=np.float32)
    # OpenAI supports batching; chunk to be safe on long docs
    batch_size = 128
    embeddings: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        vecs = np.array([item.embedding for item in resp.data], dtype=np.float32)
        embeddings.append(vecs)
    mat = np.vstack(embeddings)
    # Normalize for cosine similarity via dot product
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8
    return (mat / norms).astype(np.float32)


def build_vector_index(docs: List[Tuple[str, Dict[str, Any]]], client: OpenAI):
    if not docs:
        return None, None
    texts = [d[0] for d in docs]
    embeddings = embed_texts(client, texts)
    metadatas = [d[1] for d in docs]
    return embeddings, metadatas, texts


def search_index(embeddings: np.ndarray, texts: List[str], metadatas: List[Dict[str, Any]], client: OpenAI, query: str, k: int = 6) -> List[RetrievedChunk]:
    if embeddings is None or len(texts) == 0:
        return []
    q_resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    q_vec = np.array(q_resp.data[0].embedding, dtype=np.float32)
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    scores = embeddings @ q_vec  # cosine similarity since normalized
    topk_idx = np.argsort(-scores)[:k]
    results: List[RetrievedChunk] = []
    for idx in topk_idx:
        results.append(RetrievedChunk(text=texts[idx], metadata=metadatas[idx], score=float(scores[idx])))
    return results


def format_context(chunks: List[RetrievedChunk]) -> str:
    parts = []
    for ch in chunks:
        page_info = f"(page {ch.metadata.get('page')})" if ch.metadata else ""
        parts.append(f"{ch.text} {page_info}")
    return "\n\n".join(parts)


def generate_question_paper(context: str, user_guidelines: str, model: str = DEFAULT_GENERATION_MODEL, num_questions: int = 10) -> Dict[str, Any]:
    client = get_openai_client()
    system_prompt = (
        "You are an assessment generator. Create a clear, unambiguous question paper from the provided context. "
        "Follow the user's guidelines and ensure every question is answerable from the context. "
        "Return strict JSON with keys: title (string), questions (array of objects with: id, question, options (array) if MCQ, type ['mcq'|'short'], answer)."
    )

    user_prompt = (
        f"Context (from the study material):\n{context}\n\n"
        f"Guidelines: {user_guidelines}\n\n"
        f"Constraints: Make exactly {num_questions} questions unless the material limits it. Mix MCQ and short as appropriate."
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(content[start : end + 1])
        else:
            raise
    questions = data.get("questions", [])
    for i, q in enumerate(questions):
        q.setdefault("id", i + 1)
        q_type = q.get("type", "mcq").lower()
        if q_type not in ("mcq", "short"):
            q["type"] = "mcq"
        if q.get("type") == "mcq":
            q.setdefault("options", [])
    return {"title": data.get("title", "Question Paper"), "questions": questions}


def score_answers(questions: List[Dict[str, Any]], user_answers: Dict[str, str]) -> Tuple[int, int, List[Dict[str, Any]]]:
    total = len(questions)
    correct = 0
    details = []
    for q in questions:
        qid = str(q.get("id"))
        correct_answer = str(q.get("answer", "")).strip()
        user_answer = str(user_answers.get(qid, "")).strip()
        is_correct = False
        if q.get("type") == "mcq":
            is_correct = user_answer.lower() == correct_answer.lower()
        else:
            ca = correct_answer.lower()
            ua = user_answer.lower()
            is_correct = (ua == ca) or (ua in ca) or (ca in ua and len(ua) > 0)
        if is_correct:
            correct += 1
        details.append({
            "id": qid,
            "question": q.get("question"),
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct,
        })
    return correct, total, details


# ---------------------- UI ----------------------
st.set_page_config(page_title="Agentic RAG Exam App", page_icon="🧠", layout="wide")

st.title("🧠 Agentic RAG Exam App")

with st.sidebar:
    st.header("Settings")
    api_key_input = st.text_input("OpenAI API Key", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
    model = st.selectbox("Generation Model", [DEFAULT_GENERATION_MODEL, "gpt-4o-mini", "gpt-4o", "gpt-4.1-mini"], index=0)
    num_questions = st.slider("Number of questions", min_value=3, max_value=25, value=8)
    guidelines = st.text_area("Prompt guidelines (for the question paper)", "Focus on key concepts. Mix MCQs and short answers. Ensure difficulty is moderate.")

# Session state
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None  # np.ndarray normalized
    st.session_state.metadatas = None   # List[dict]
    st.session_state.texts = None       # List[str]
    st.session_state.question_paper = None
    st.session_state.answers = {}

st.subheader("1) Upload study material (PDF)")
uploaded = st.file_uploader("Upload a PDF", type=["pdf"])  # type: ignore

if uploaded is not None:
    file_bytes = uploaded.read()
    with st.spinner("Extracting text and computing embeddings..."):
        client = get_openai_client()
        doc_tuples = extract_text_from_pdf(file_bytes)
        embeddings, metadatas, texts = build_vector_index(doc_tuples, client)
        st.session_state.embeddings = embeddings
        st.session_state.metadatas = metadatas
        st.session_state.texts = texts
    st.success("Vector store built in-memory.")

st.subheader("2) Retrieve context and generate question paper")
query = st.text_input("Enter your prompt about what to focus on (e.g., 'chapters on photosynthesis, definitions, key dates')")

col1, col2 = st.columns(2)
with col1:
    if st.button("Retrieve Context", disabled=st.session_state.embeddings is None):
        with st.spinner("Retrieving relevant chunks..."):
            client = get_openai_client()
            chunks = search_index(
                st.session_state.embeddings,
                st.session_state.texts or [],
                st.session_state.metadatas or [],
                client,
                query if query.strip() else "overview",
                k=8,
            )
        if not chunks:
            st.warning("No content retrieved. Upload a PDF first.")
        else:
            st.session_state.retrieved_context = format_context(chunks)
            st.success("Context retrieved.")

with col2:
    if st.button("Generate Question Paper", disabled="retrieved_context" not in st.session_state):
        with st.spinner("Generating question paper via LLM..."):
            paper = generate_question_paper(st.session_state.get("retrieved_context", ""), guidelines, model=model, num_questions=num_questions)
        st.session_state.question_paper = paper
        st.session_state.answers = {}
        st.success("Question paper generated.")

if st.session_state.get("question_paper"):
    paper = st.session_state["question_paper"]
    st.subheader(f"3) {paper.get('title', 'Question Paper')}")
    questions = paper.get("questions", [])

    with st.form("answer_form", clear_on_submit=False):
        for q in questions:
            qid = str(q.get("id"))
            qtext = q.get("question", "")
            qtype = q.get("type", "mcq")
            st.markdown(f"**Q{qid}. {qtext}**")
            if qtype == "mcq":
                options = q.get("options", [])
                key = f"ans_{qid}"
                default_val = st.session_state.answers.get(qid)
                selected = st.radio("Select one:", options=options, index=options.index(default_val) if default_val in options else 0, key=key)
                st.session_state.answers[qid] = selected
            else:
                val = st.text_input("Your answer:", value=st.session_state.answers.get(qid, ""), key=f"ans_{qid}")
                st.session_state.answers[qid] = val
            st.divider()
        submitted = st.form_submit_button("Submit Answers")

    if submitted:
        correct, total, details = score_answers(questions, st.session_state.answers)
        st.subheader("4) Results")
        st.write(f"Score: {correct} / {total} ({(correct/total*100 if total else 0):.1f}%)")
        with st.expander("See detailed results"):
            for d in details:
                status = "✅" if d["is_correct"] else "❌"
                st.write(f"{status} Q{d['id']}: {d['question']}")
                st.write(f"Your answer: {d['user_answer']}")
                st.write(f"Correct answer: {d['correct_answer']}")
                st.markdown("---")

st.caption("Note: Enter your OpenAI API key in the sidebar or set OPENAI_API_KEY in the environment. Embeddings via OpenAI; retrieval is in-memory cosine similarity.")