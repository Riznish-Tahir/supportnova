"""
SQLite database layer for SupportNova.
Swap sqlite3 for psycopg2/PostgreSQL in production by changing connect().
"""
import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager

DB_PATH = Path(__file__).resolve().parent / "supportnova.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('customer','agent','reviewer','manager','admin')),
    created_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS complaints (
    complaint_id TEXT PRIMARY KEY,
    customer_username TEXT,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    customer_type TEXT DEFAULT 'Standard',
    product TEXT,
    order_reference TEXT,
    channel TEXT DEFAULT 'Web Form',
    is_repeat INTEGER DEFAULT 0,
    prior_complaint_id TEXT,
    prior_status TEXT,
    status TEXT DEFAULT 'New',
    submitted_at REAL NOT NULL,
    updated_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS analysis_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT NOT NULL,
    prompt_version TEXT,
    provider TEXT,
    model TEXT,
    policy_version TEXT,
    analysis_timestamp REAL,
    genai_raw TEXT,
    genai_parsed TEXT,
    schema_valid INTEGER,
    attempts INTEGER,
    FOREIGN KEY(complaint_id) REFERENCES complaints(complaint_id)
);

CREATE TABLE IF NOT EXISTS validation_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT NOT NULL,
    expected_json TEXT,
    prompt_injection_detected INTEGER,
    unsupported_promise_flags TEXT,
    hallucination_flags TEXT,
    notes TEXT,
    FOREIGN KEY(complaint_id) REFERENCES complaints(complaint_id)
);

CREATE TABLE IF NOT EXISTS comparison_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT NOT NULL,
    verification_status TEXT,
    verification_score REAL,
    comparisons_json TEXT,
    mismatch_reasons TEXT,
    logged_at REAL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(complaint_id)
);

CREATE TABLE IF NOT EXISTS audit_trail (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    complaint_id TEXT NOT NULL,
    actor TEXT,
    action TEXT,
    original_value TEXT,
    new_value TEXT,
    timestamp REAL,
    FOREIGN KEY(complaint_id) REFERENCES complaints(complaint_id)
);

CREATE TABLE IF NOT EXISTS kb_documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    category TEXT,
    version TEXT,
    effective_date TEXT,
    status TEXT DEFAULT 'Active',
    file_path TEXT,
    uploaded_at REAL
);

CREATE TABLE IF NOT EXISTS kb_chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    section TEXT,
    heading TEXT,
    page INTEGER,
    content TEXT,
    FOREIGN KEY(document_id) REFERENCES kb_documents(document_id)
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


def insert_complaint(conn, complaint: dict):
    conn.execute(
        """INSERT OR REPLACE INTO complaints
        (complaint_id, customer_username, title, description, customer_type, product,
         order_reference, channel, is_repeat, prior_complaint_id, prior_status, status,
         submitted_at, updated_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            complaint["complaint_id"], complaint.get("customer_username"), complaint["title"],
            complaint["description"], complaint.get("customer_type", "Standard"),
            complaint.get("product"), complaint.get("order_reference"),
            complaint.get("channel", "Web Form"), int(complaint.get("is_repeat", False)),
            complaint.get("prior_complaint_id"), complaint.get("prior_status"),
            complaint.get("status", "New"), time.time(), time.time(),
        ),
    )


def log_analysis(conn, complaint_id, genai_result):
    conn.execute(
        """INSERT INTO analysis_log
        (complaint_id, prompt_version, provider, model, policy_version, analysis_timestamp,
         genai_raw, genai_parsed, schema_valid, attempts)
        VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (
            complaint_id, genai_result.prompt_version, genai_result.provider, genai_result.model,
            "1.0", genai_result.analysis_timestamp, genai_result.raw_response,
            json.dumps(genai_result.parsed) if genai_result.parsed else None,
            int(genai_result.schema_valid), genai_result.attempts,
        ),
    )


def log_validation(conn, complaint_id, validation_result):
    conn.execute(
        """INSERT INTO validation_log
        (complaint_id, expected_json, prompt_injection_detected, unsupported_promise_flags,
         hallucination_flags, notes) VALUES (?,?,?,?,?,?)""",
        (
            complaint_id,
            json.dumps(validation_result.__dict__, default=str),
            int(validation_result.prompt_injection_detected),
            json.dumps(validation_result.unsupported_promise_flags),
            json.dumps(validation_result.hallucination_flags),
            json.dumps(validation_result.notes),
        ),
    )


def log_comparison(conn, complaint_id, comparison_report):
    conn.execute(
        """INSERT INTO comparison_log
        (complaint_id, verification_status, verification_score, comparisons_json,
         mismatch_reasons, logged_at) VALUES (?,?,?,?,?,?)""",
        (
            complaint_id, comparison_report.verification_status, comparison_report.verification_score,
            json.dumps([c.__dict__ for c in comparison_report.comparisons]),
            json.dumps(comparison_report.mismatch_reasons), time.time(),
        ),
    )
    status = "Escalated" if comparison_report.verification_status == "Manual Review" else "Analyzed"
    conn.execute("UPDATE complaints SET status=?, updated_at=? WHERE complaint_id=?",
                 (status, time.time(), complaint_id))


def record_audit(conn, complaint_id, actor, action, original_value=None, new_value=None):
    conn.execute(
        """INSERT INTO audit_trail (complaint_id, actor, action, original_value, new_value, timestamp)
        VALUES (?,?,?,?,?,?)""",
        (complaint_id, actor, action, original_value, new_value, time.time()),
    )
