"""
Knowledge-base document validation, parsing and chunking.
Supports PDF (PyMuPDF) and DOCX (python-docx) per SRS Step 3/5/6.
"""
from __future__ import annotations
import hashlib
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

MAX_FILE_SIZE_MB = 20
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}


@dataclass
class DocChunk:
    chunk_id: str
    document_id: str
    section: str
    heading: str
    page: Optional[int]
    content: str


@dataclass
class ValidationResult:
    valid: bool
    errors: list = field(default_factory=list)


def validate_file(path: Path) -> ValidationResult:
    errors = []
    if not path.exists():
        return ValidationResult(False, ["File does not exist"])
    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        errors.append(f"Unsupported file type: {path.suffix}")
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > MAX_FILE_SIZE_MB:
        errors.append(f"File too large: {size_mb:.1f}MB > {MAX_FILE_SIZE_MB}MB limit")
    if path.stat().st_size == 0:
        errors.append("File is empty")
    return ValidationResult(len(errors) == 0, errors)


def file_hash(path: Path) -> str:
    """Used for duplicate-document detection."""
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def extract_pdf(path: Path) -> list[dict]:
    """Returns [{page, text}] using PyMuPDF."""
    import fitz  # PyMuPDF
    pages = []
    with fitz.open(path) as doc:
        for i, page in enumerate(doc, start=1):
            pages.append({"page": i, "text": page.get_text()})
    return pages


def extract_docx(path: Path) -> list[dict]:
    """Returns [{page: None, text}] — DOCX has no reliable page boundaries,
    so chunking below falls back to heading-based sections."""
    import docx
    d = docx.Document(path)
    paragraphs = [p.text for p in d.paragraphs if p.text.strip()]
    return [{"page": None, "text": "\n".join(paragraphs)}]


def extract_text_file(path: Path) -> list[dict]:
    return [{"page": None, "text": path.read_text(encoding="utf-8", errors="ignore")}]


HEADING_PATTERN = re.compile(r"^(?:\d+(\.\d+)*\s+)?([A-Z][A-Za-z0-9 &/'-]{3,80})$", re.MULTILINE)


def chunk_text(document_id: str, raw_pages: list[dict], target_chars: int = 900) -> list[DocChunk]:
    """
    Splits extracted text into traceable chunks, trying to break on detected
    headings first, then falling back to paragraph-sized windows so every
    chunk stays under target_chars while keeping section/heading metadata.
    """
    chunks: list[DocChunk] = []
    current_heading = "General"
    section_counter = 0

    for pageinfo in raw_pages:
        page_num = pageinfo["page"]
        text = pageinfo["text"]
        paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

        buffer = ""
        for para in paragraphs:
            heading_match = HEADING_PATTERN.match(para.strip())
            if heading_match and len(para) < 90:
                # flush current buffer as a chunk before starting a new section
                if buffer.strip():
                    section_counter += 1
                    chunks.append(DocChunk(
                        chunk_id=f"CHK-{uuid.uuid4().hex[:8]}", document_id=document_id,
                        section=str(section_counter), heading=current_heading,
                        page=page_num, content=buffer.strip(),
                    ))
                    buffer = ""
                current_heading = para.strip()
                continue

            buffer += para + "\n\n"
            if len(buffer) >= target_chars:
                section_counter += 1
                chunks.append(DocChunk(
                    chunk_id=f"CHK-{uuid.uuid4().hex[:8]}", document_id=document_id,
                    section=str(section_counter), heading=current_heading,
                    page=page_num, content=buffer.strip(),
                ))
                buffer = ""

        if buffer.strip():
            section_counter += 1
            chunks.append(DocChunk(
                chunk_id=f"CHK-{uuid.uuid4().hex[:8]}", document_id=document_id,
                section=str(section_counter), heading=current_heading,
                page=page_num, content=buffer.strip(),
            ))
    return chunks


def process_document(path: Path, document_id: str) -> tuple[ValidationResult, list[DocChunk]]:
    validation = validate_file(path)
    if not validation.valid:
        return validation, []

    ext = path.suffix.lower()
    if ext == ".pdf":
        pages = extract_pdf(path)
    elif ext == ".docx":
        pages = extract_docx(path)
    else:
        pages = extract_text_file(path)

    chunks = chunk_text(document_id, pages)
    return validation, chunks


def retrieve_relevant_chunks(chunks: list[DocChunk], query: str, top_k: int = 3) -> list[DocChunk]:
    """
    Lightweight keyword-overlap retrieval (no external embedding service
    required). Swap for FAISS/ChromaDB embeddings in production — the
    interface (chunks in, ranked chunks out) stays the same.
    """
    query_words = set(re.findall(r"[a-z]{3,}", query.lower()))
    scored = []
    for c in chunks:
        content_words = set(re.findall(r"[a-z]{3,}", c.content.lower()))
        overlap = len(query_words & content_words)
        if overlap:
            scored.append((overlap, c))
    scored.sort(key=lambda x: -x[0])
    return [c for _, c in scored[:top_k]]
