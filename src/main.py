"""
SupportNova — FastAPI application entrypoint.

Run with:
    uvicorn src.main:app --reload

Endpoints:
    POST /api/complaints                 submit a new complaint (runs both pipelines)
    GET  /api/complaints                 list complaints (filter by status/category/department)
    GET  /api/complaints/{id}            full detail incl. GenAI + Python + comparison
    POST /api/complaints/{id}/review     reviewer action: approve / modify / reassign / escalate
    POST /api/knowledge-base/upload      upload + parse + chunk a policy document
    GET  /api/analytics/summary          dashboard aggregate numbers
    GET  /api/analytics/manual-review    manual review queue
"""
from __future__ import annotations
import json
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.append(str(Path(__file__).resolve().parent.parent))

from database import db
from src.preprocessing import preprocess_complaint, is_near_duplicate
from security.prompt_injection import scan_complaint_text
from genai_pipeline.pipeline import run_genai_pipeline
from python_validation.pipeline import validate_complaint
from comparison_engine.compare import compare
from document_processing.parser import process_document, validate_file, file_hash

app = FastAPI(title="SupportNova", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
db.init_db()


KB_UPLOAD_DIR = Path(__file__).resolve().parent.parent / "sample_documents" / "uploaded"
KB_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
def get_kb_excerpts(query_text: str, limit: int = 3) -> str:
    stop_words = {
        "the", "and", "that", "this", "with", "from",
        "have", "has", "had", "after", "before",
        "my", "your", "our", "for", "not", "but",
        "was", "were", "been", "into", "about"
    }
    query_words = [
        word
        for word in re.findall(r"\b[a-z0-9-]+\b", query_text.lower())
        if len(word) > 3 and word not in stop_words
    ]
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT content
            FROM kb_chunks
            """
        ).fetchall()

    scored = []
    for row in rows:
        content = row["content"]
        content_words = set(
            re.findall(r"\b[a-z0-9-]+\b", content.lower())
        )
        score = sum(
            1 for word in query_words
            if word in content_words
        )
        if score > 0:
            scored.append((score, content))

    scored.sort(key=lambda item: item[0], reverse=True)
    top_chunks = [
        content
        for score, content in scored[:limit]
    ]
    return "\n\n".join(top_chunks)


class ComplaintIn(BaseModel):
    title: str
    description: str
    customer_username: Optional[str] = "guest"
    customer_type: Optional[str] = "Standard"
    product: Optional[str] = None
    order_reference: Optional[str] = None
    channel: Optional[str] = "Web Form"
    is_repeat: Optional[bool] = False
    prior_complaint_id: Optional[str] = None
    prior_status: Optional[str] = None


class ReviewAction(BaseModel):
    actor: str
    action: str  # approve | modify | reassign | escalate | reject
    new_value: Optional[str] = None
    comment: Optional[str] = None


@app.post("/api/complaints")
def submit_complaint(payload: ComplaintIn):
    result = preprocess_complaint(payload.dict())
    if not result["valid"]:
        raise HTTPException(400, detail={"errors": result["errors"]})

    complaint = result["complaint"]
    complaint["complaint_id"] = f"CMP-{uuid.uuid4().hex[:5].upper()}"

    with db.get_conn() as conn:
        existing = [r["title"] + " " + r["description"] for r in
                    conn.execute("SELECT title, description FROM complaints").fetchall()]
    dup = is_near_duplicate(complaint["title"] + " " + complaint["description"], existing)
    is_duplicate = dup is not None

    security_scan = scan_complaint_text(complaint["title"] + " " + complaint["description"])

    # Pipeline 1 — GenAI
    query_text = f"{complaint.get('title', '')} {complaint.get('description', '')}"
    kb_excerpts = get_kb_excerpts(query_text)

    genai_result = run_genai_pipeline(
    complaint,
    kb_excerpts=kb_excerpts
)

    # Pipeline 2 — independent Python ground truth (never trusts Pipeline 1 blindly)
    validation_result = validate_complaint(complaint, genai_result.parsed)

    # Comparison + verification decision
    comparison_report = compare(genai_result, validation_result)

    with db.get_conn() as conn:
        db.insert_complaint(conn, complaint)
        db.log_analysis(conn, complaint["complaint_id"], genai_result)
        db.log_validation(conn, complaint["complaint_id"], validation_result)
        db.log_comparison(conn, complaint["complaint_id"], comparison_report)
        db.record_audit(conn, complaint["complaint_id"], actor="system",
                         action="submitted_and_analyzed")

    return {
        "complaint_id": complaint["complaint_id"],
        "duplicate_of": dup,
        "is_duplicate": is_duplicate,
        "security_scan": security_scan,
        "genai_output": genai_result.parsed,
        "genai_manual_review_required": genai_result.manual_review_required,
        "python_expected": validation_result.__dict__,
        "verification_status": comparison_report.verification_status,
        "verification_score": comparison_report.verification_score,
        "mismatch_reasons": comparison_report.mismatch_reasons,
    }


@app.get("/api/complaints")
def list_complaints(status: Optional[str] = None, category: Optional[str] = None):
    with db.get_conn() as conn:
        query = "SELECT * FROM complaints WHERE 1=1"
        params = []
        if status:
            query += " AND status=?"
            params.append(status)
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]


@app.get("/api/complaints/{complaint_id}")
def get_complaint(complaint_id: str):
    with db.get_conn() as conn:
        complaint = conn.execute(
            "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
        ).fetchone()
        if not complaint:
            raise HTTPException(404, "Complaint not found")
        analysis = conn.execute(
            "SELECT * FROM analysis_log WHERE complaint_id=? ORDER BY id DESC LIMIT 1",
            (complaint_id,),
        ).fetchone()
        validation = conn.execute(
            "SELECT * FROM validation_log WHERE complaint_id=? ORDER BY id DESC LIMIT 1",
            (complaint_id,),
        ).fetchone()
        comparison = conn.execute(
            "SELECT * FROM comparison_log WHERE complaint_id=? ORDER BY id DESC LIMIT 1",
            (complaint_id,),
        ).fetchone()
        audit = conn.execute(
            "SELECT * FROM audit_trail WHERE complaint_id=? ORDER BY timestamp", (complaint_id,)
        ).fetchall()
        return {
            "complaint": dict(complaint),
            "analysis": dict(analysis) if analysis else None,
            "validation": dict(validation) if validation else None,
            "comparison": dict(comparison) if comparison else None,
            "audit_trail": [dict(a) for a in audit],
        }


@app.post("/api/complaints/{complaint_id}/review")
def review_complaint(complaint_id: str, action: ReviewAction):
    with db.get_conn() as conn:
        complaint = conn.execute(
            "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
        ).fetchone()
        if not complaint:
            raise HTTPException(404, "Complaint not found")

        new_status_map = {
            "approve": "Resolved", "reject": "Reopened", "escalate": "Escalated",
            "modify": complaint["status"], "reassign": complaint["status"],
        }
        new_status = new_status_map.get(action.action, complaint["status"])
        conn.execute("UPDATE complaints SET status=?, updated_at=? WHERE complaint_id=?",
                     (new_status, time.time(), complaint_id))
        db.record_audit(conn, complaint_id, actor=action.actor, action=action.action,
                         original_value=complaint["status"], new_value=action.new_value or new_status)
    return {"complaint_id": complaint_id, "new_status": new_status}


@app.post("/api/knowledge-base/upload")
async def upload_kb_document(file: UploadFile = File(...), category: str = Form("General"),
                              version: str = Form("1.0"), effective_date: str = Form("")):
    dest = KB_UPLOAD_DIR / file.filename
    contents = await file.read()
    dest.write_bytes(contents)

    validation = validate_file(dest)
    if not validation.valid:
        raise HTTPException(400, detail={"errors": validation.errors})

    document_id = f"DOC-{uuid.uuid4().hex[:6].upper()}"
    _, chunks = process_document(dest, document_id)

    with db.get_conn() as conn:
        conn.execute(
            """INSERT INTO kb_documents
            (document_id, title, category, version, effective_date, status, file_path, uploaded_at)
            VALUES (?,?,?,?,?,?,?,?)""",
            (document_id, file.filename, category, version, effective_date, "Active",
             str(dest), time.time()),
        )
        for c in chunks:
            conn.execute(
                """INSERT INTO kb_chunks (chunk_id, document_id, section, heading, page, content)
                VALUES (?,?,?,?,?,?)""",
                (c.chunk_id, c.document_id, c.section, c.heading, c.page, c.content),
            )
    return {"document_id": document_id, "chunks_created": len(chunks)}


@app.get("/api/analytics/summary")
def analytics_summary():
    with db.get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM complaints").fetchone()["c"]
        by_status = conn.execute(
            "SELECT status, COUNT(*) c FROM complaints GROUP BY status"
        ).fetchall()
        manual_review = conn.execute(
            "SELECT COUNT(*) c FROM comparison_log WHERE verification_status='Manual Review'"
        ).fetchone()["c"]
        avg_score = conn.execute(
            "SELECT AVG(verification_score) a FROM comparison_log"
        ).fetchone()["a"]
    return {
        "total_complaints": total,
        "by_status": {r["status"]: r["c"] for r in by_status},
        "manual_review_count": manual_review,
        "avg_verification_score": round(avg_score, 3) if avg_score else None,
    }


@app.get("/api/analytics/manual-review")
def manual_review_queue():
    with db.get_conn() as conn:
        rows = conn.execute(
            """SELECT c.complaint_id, c.title, cl.mismatch_reasons, cl.verification_score
               FROM comparison_log cl JOIN complaints c ON c.complaint_id = cl.complaint_id
               WHERE cl.verification_status='Manual Review'
               ORDER BY cl.logged_at DESC"""
        ).fetchall()
        return [
            {**dict(r), "mismatch_reasons": json.loads(r["mismatch_reasons"])}
            for r in rows
        ]


@app.get("/health")
def health():
    return {"status": "ok"}
