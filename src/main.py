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
import bcrypt
import csv
import io
import json
import jwt
import os
import re
import sys
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import Depends, FastAPI, Header, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
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


def hash_password(password: str) -> str:
    return bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.checkpw(
        password.encode("utf-8"),
        password_hash.encode("utf-8")
    )


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
    actor: Optional[str] = None
    action: str
    new_value: Optional[str] = None
    comment: Optional[str] = None


class RegisterRequest(BaseModel):
    username: str
    password: str
    role: str = "customer"


class LoginRequest(BaseModel):
    username: str
    password: str


class AdminCreateUserRequest(BaseModel):
    username: str
    password: str
    role: str


class PolicyStatusUpdate(BaseModel):
    status: str


JWT_SECRET = os.environ.get("SUPPORTNOVA_JWT_SECRET", "dev-secret-change-me")
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_MINUTES = 60


def create_access_token(user_id: int, username: str, role: str) -> str:
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES)
    }
    return jwt.encode(
        payload,
        JWT_SECRET,
        algorithm=JWT_ALGORITHM
    )


def get_current_user(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail="Missing or invalid authorization header"
        )

    token = authorization.split(" ", 1)[1]
    try:
        payload = jwt.decode(
            token,
            JWT_SECRET,
            algorithms=[JWT_ALGORITHM]
        )
        return {
            "user_id": int(payload["sub"]),
            "username": payload["username"],
            "role": payload["role"]
        }
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token expired"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Invalid token"
        )


def require_roles(*allowed_roles):
    def checker(user=Depends(get_current_user)):
        if user["role"] not in allowed_roles:
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to access this resource"
            )
        return user
    return checker


@app.post("/api/auth/register")
def register_user(data: RegisterRequest):
    username = data.username.strip()
    password = data.password
    role = "customer"

    if not username:
        raise HTTPException(
            status_code=400,
            detail="Username is required"
        )
    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters"
        )

    password_hash = hash_password(password)
    try:
        with db.get_conn() as conn:
            conn.execute(
                """
                INSERT INTO users
                (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    username,
                    password_hash,
                    role,
                    time.time()
                )
            )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(
                status_code=409,
                detail="Username already exists"
            )
        raise

    return {
        "message": "User registered successfully",
        "username": username,
        "role": role
    }


@app.post("/api/admin/users")
def admin_create_user(
    data: AdminCreateUserRequest,
    user=Depends(require_roles("admin"))
):
    allowed_roles = {
        "customer",
        "agent",
        "reviewer",
        "manager",
        "admin"
    }
    role = data.role.strip().lower()
    username = data.username.strip()

    if not username:
        raise HTTPException(400, "Username is required")
    if role not in allowed_roles:
        raise HTTPException(400, "Invalid role")
    if len(data.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")

    password_hash = hash_password(data.password)
    try:
        with db.get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO users
                (username, password_hash, role, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (username, password_hash, role, time.time())
            )
    except Exception as exc:
        if "UNIQUE constraint failed" in str(exc):
            raise HTTPException(409, "Username already exists")
        raise

    return {
        "message": "User created successfully",
        "user_id": cursor.lastrowid,
        "username": username,
        "role": role
    }


@app.post("/api/auth/login")
def login_user(data: LoginRequest):
    username = data.username.strip()
    with db.get_conn() as conn:
        user = conn.execute(
            """
            SELECT id, username, password_hash, role
            FROM users
            WHERE username = ?
            """,
            (username,)
        ).fetchone()

    if not user:
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )
    if not verify_password(data.password, user["password_hash"]):
        raise HTTPException(
            status_code=401,
            detail="Invalid username or password"
        )

    token = create_access_token(
        user["id"],
        user["username"],
        user["role"]
    )
    return {
        "message": "Login successful",
        "access_token": token,
        "token_type": "bearer",
        "user_id": user["id"],
        "username": user["username"],
        "role": user["role"]
    }


@app.get("/api/auth/me")
def get_me(user=Depends(get_current_user)):
    return user


@app.get("/api/admin/test")
def admin_test(user=Depends(require_roles("admin"))):
    return {
        "message": "Admin access granted",
        "user": user
    }


SLA_HOURS = {
    "P0": 2,
    "P1": 8,
    "P2": 24,
    "P3": 72
}


def get_latest_priority(conn, complaint_id: str) -> str:
    row = conn.execute(
        """
        SELECT expected_json
        FROM validation_log
        WHERE complaint_id=?
        ORDER BY id DESC
        LIMIT 1
        """,
        (complaint_id,)
    ).fetchone()
    if not row or not row["expected_json"]:
        return "P2"
    try:
        data = json.loads(row["expected_json"])
        return data.get("expected_priority", "P2")
    except (TypeError, json.JSONDecodeError):
        return "P2"


@app.post("/api/complaints")
def submit_complaint(
    payload: ComplaintIn,
    user=Depends(require_roles(
        "customer",
        "agent",
        "reviewer",
        "manager",
        "admin"
    ))
):
    result = preprocess_complaint(payload.dict())
    if not result["valid"]:
        raise HTTPException(400, detail={"errors": result["errors"]})

    complaint = result["complaint"]
    complaint["customer_username"] = user["username"]
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
def list_complaints(
    status: Optional[str] = None,
    category: Optional[str] = None,
    q: Optional[str] = None,
    customer: Optional[str] = None,
    user=Depends(get_current_user)
):
    with db.get_conn() as conn:
        query = """
            SELECT *
            FROM complaints
            WHERE 1=1
        """
        params = []
        if user["role"] == "customer":
            query += " AND customer_username=?"
            params.append(user["username"])
        elif customer:
            query += " AND customer_username=?"
            params.append(customer)
        if status:
            query += " AND status=?"
            params.append(status)
        if q:
            query += """
                AND (
                    complaint_id LIKE ?
                    OR title LIKE ?
                    OR description LIKE ?
                    OR order_reference LIKE ?
                )
            """
            search = f"%{q}%"
            params.extend([search, search, search, search])
        query += " ORDER BY submitted_at DESC"
        rows = conn.execute(query, params).fetchall()
        complaints = [dict(row) for row in rows]

        if category:
            filtered = []
            for complaint in complaints:
                validation = conn.execute(
                    """
                    SELECT expected_json
                    FROM validation_log
                    WHERE complaint_id=?
                    ORDER BY id DESC
                    LIMIT 1
                    """,
                    (complaint["complaint_id"],)
                ).fetchone()
                found_category = None
                if validation and validation["expected_json"]:
                    try:
                        found_category = json.loads(
                            validation["expected_json"]
                        ).get("expected_category")
                    except (TypeError, json.JSONDecodeError):
                        pass
                if found_category is None:
                    analysis = conn.execute(
                        """
                        SELECT genai_parsed
                        FROM analysis_log
                        WHERE complaint_id=?
                        ORDER BY id DESC
                        LIMIT 1
                        """,
                        (complaint["complaint_id"],)
                    ).fetchone()
                    if analysis and analysis["genai_parsed"]:
                        try:
                            found_category = json.loads(
                                analysis["genai_parsed"]
                            ).get("issue_category")
                        except (TypeError, json.JSONDecodeError):
                            pass
                if found_category and found_category.casefold() == category.casefold():
                    filtered.append(complaint)
            complaints = filtered

        return complaints


@app.get("/api/complaints/{complaint_id}")
def get_complaint(
    complaint_id: str,
    user=Depends(get_current_user)
):
    with db.get_conn() as conn:
        complaint = conn.execute(
            "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
        ).fetchone()
        if not complaint:
            raise HTTPException(404, "Complaint not found")
        if (
            user["role"] == "customer"
            and complaint["customer_username"] != user["username"]
        ):
            raise HTTPException(
                status_code=403,
                detail="You do not have permission to view this complaint"
            )
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
def review_complaint(
    complaint_id: str,
    action: ReviewAction,
    user=Depends(require_roles(
        "reviewer",
        "manager",
        "admin"
    ))
):
    allowed_actions = {
        "approve",
        "reject",
        "modify",
        "reclassify",
        "reassign",
        "escalate",
        "comment"
    }
    if action.action not in allowed_actions:
        raise HTTPException(400, "Invalid review action")

    with db.get_conn() as conn:
        complaint = conn.execute(
            "SELECT * FROM complaints WHERE complaint_id=?", (complaint_id,)
        ).fetchone()
        if not complaint:
            raise HTTPException(404, "Complaint not found")

        new_status_map = {
            "approve": "Resolved",
            "reject": "Reopened",
            "modify": complaint["status"],
            "reclassify": complaint["status"],
            "reassign": "Assigned",
            "escalate": "Escalated",
            "comment": complaint["status"]
        }
        new_status = new_status_map[action.action]
        conn.execute("UPDATE complaints SET status=?, updated_at=? WHERE complaint_id=?",
                     (new_status, time.time(), complaint_id))
        db.record_audit(
            conn,
            complaint_id,
            actor=user["username"],
            action=action.action,
            original_value=complaint["status"],
            new_value=action.new_value or new_status
        )
    return {"complaint_id": complaint_id, "new_status": new_status}


@app.post("/api/knowledge-base/upload")
async def upload_kb_document(file: UploadFile = File(...), category: str = Form("General"),
                              version: str = Form("1.0"), effective_date: str = Form(""),
                              user=Depends(require_roles("manager", "admin"))):
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


@app.get("/api/knowledge-base/documents")
def list_kb_documents(
    user=Depends(require_roles("reviewer", "manager", "admin"))
):
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT document_id, title, category, version,
                   effective_date, status, uploaded_at
            FROM kb_documents
            ORDER BY uploaded_at DESC
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.patch("/api/knowledge-base/{document_id}/status")
def update_policy_status(
    document_id: str,
    payload: PolicyStatusUpdate,
    user=Depends(require_roles("manager", "admin"))
):
    allowed = {"Active", "Previous", "Superseded", "Draft"}
    if payload.status not in allowed:
        raise HTTPException(400, "Invalid policy status")
    with db.get_conn() as conn:
        row = conn.execute(
            """
            SELECT status
            FROM kb_documents
            WHERE document_id=?
            """,
            (document_id,)
        ).fetchone()
        if not row:
            raise HTTPException(404, "Document not found")
        conn.execute(
            """
            UPDATE kb_documents
            SET status=?
            WHERE document_id=?
            """,
            (payload.status, document_id)
        )
    return {
        "document_id": document_id,
        "previous_status": row["status"],
        "status": payload.status
    }


@app.get("/api/sla")
def sla_status(user=Depends(require_roles(
    "agent",
    "reviewer",
    "manager",
    "admin"
))):
    now = time.time()
    output = []
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM complaints
            WHERE status NOT IN ('Resolved', 'Closed')
            ORDER BY submitted_at
            """
        ).fetchall()
        for row in rows:
            priority = get_latest_priority(conn, row["complaint_id"])
            allowed_hours = SLA_HOURS.get(priority, 24)
            deadline = row["submitted_at"] + allowed_hours * 3600
            remaining = deadline - now
            if remaining <= 0:
                risk = "Breached"
            elif remaining <= allowed_hours * 3600 * 0.25:
                risk = "At Risk"
            else:
                risk = "On Track"
            output.append({
                "complaint_id": row["complaint_id"],
                "title": row["title"],
                "status": row["status"],
                "priority": priority,
                "deadline": deadline,
                "remaining_seconds": remaining,
                "risk": risk
            })
    return output


@app.get("/api/reports/summary")
def reports_summary(
    user=Depends(require_roles("reviewer", "manager", "admin"))
):
    with db.get_conn() as conn:
        complaints = conn.execute("SELECT * FROM complaints").fetchall()
        comparisons = conn.execute("SELECT * FROM comparison_log").fetchall()
        audits = conn.execute("SELECT * FROM audit_trail").fetchall()
        kb_count = conn.execute(
            "SELECT COUNT(*) AS c FROM kb_documents"
        ).fetchone()["c"]
    return {
        "complaint_count": len(complaints),
        "comparison_count": len(comparisons),
        "audit_events": len(audits),
        "knowledge_base_documents": kb_count,
        "manual_reviews": sum(
            1 for row in comparisons
            if row["verification_status"] == "Manual Review"
        )
    }


@app.get("/api/reports/complaints.csv")
def export_complaints_csv(
    user=Depends(require_roles("manager", "admin"))
):
    with db.get_conn() as conn:
        rows = conn.execute(
            """
            SELECT complaint_id, customer_username, title, status,
                   submitted_at, updated_at
            FROM complaints
            ORDER BY submitted_at DESC
            """
        ).fetchall()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "Complaint ID",
        "Customer",
        "Title",
        "Status",
        "Submitted At",
        "Updated At"
    ])
    for row in rows:
        writer.writerow([
            row["complaint_id"],
            row["customer_username"],
            row["title"],
            row["status"],
            row["submitted_at"],
            row["updated_at"]
        ])
    output.seek(0)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=complaints.csv"}
    )


@app.get("/api/analytics/summary")
def analytics_summary(user=Depends(require_roles(
    "agent",
    "reviewer",
    "manager",
    "admin"
))):
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
def manual_review_queue(user=Depends(require_roles(
    "reviewer",
    "manager",
    "admin"
))):
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
