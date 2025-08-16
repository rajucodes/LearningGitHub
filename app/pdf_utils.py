import re
from typing import List
from pypdf import PdfReader


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    texts: List[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        texts.append(text)
    return "\n".join(texts)


def split_text_into_chunks(text: str, chunk_size: int = 800, chunk_overlap: int = 200) -> List[str]:
    if not text:
        return []
    # Normalize whitespace
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    chunks: List[str] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end]
        chunks.append(chunk)
        if end == len(cleaned):
            break
        start = max(0, end - chunk_overlap)
    return chunks