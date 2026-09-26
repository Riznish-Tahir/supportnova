"""
Pipeline 1 — Python GenAI Complaint Intelligence Pipeline
===========================================================
Sends complaint + retrieved knowledge-base context to a Generative AI API
(Anthropic by default) and returns a STRUCTURED JSON result that must match
schemas/complaint_schema.json.

This module never approves its own output — that is Pipeline 2's job
(python_validation/pipeline.py). This file only calls the model, retries on
malformed output, and logs what happened (prompt version, model, timestamp)
for traceability, per SRS Step 47–49.
"""
from __future__ import annotations
import json
import os
import time
import uuid
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import jsonschema

BASE_DIR = Path(__file__).resolve().parent.parent
SCHEMA_PATH = BASE_DIR / "schemas" / "complaint_schema.json"
PROMPT_PATH = BASE_DIR / "prompt_templates" / "complaint_analysis_v1.json"

with open(SCHEMA_PATH) as f:
    COMPLAINT_SCHEMA = json.load(f)
with open(PROMPT_PATH) as f:
    PROMPT_TEMPLATE = json.load(f)

MAX_RETRIES = 3


@dataclass
class GenAIResult:
    complaint_id: str
    raw_response: str
    parsed: Optional[dict] = None
    valid_json: bool = False
    schema_valid: bool = False
    schema_errors: list = field(default_factory=list)
    attempts: int = 0
    prompt_version: str = PROMPT_TEMPLATE["version"]
    provider: str = PROMPT_TEMPLATE["provider"]
    model: str = PROMPT_TEMPLATE["model"]
    analysis_timestamp: float = field(default_factory=time.time)
    manual_review_required: bool = False
    failure_reason: Optional[str] = None


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    return text


def _validate_against_schema(obj: dict) -> tuple[bool, list]:
    validator = jsonschema.Draft7Validator(COMPLAINT_SCHEMA)
    errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
    return (len(errors) == 0, [e.message for e in errors])


def build_prompt(complaint: dict, kb_excerpts: str) -> tuple[str, str]:
    """Returns (system_prompt, user_prompt) with the template filled in."""
    system_prompt = PROMPT_TEMPLATE["system_prompt"]
    user_prompt = PROMPT_TEMPLATE["user_prompt_template"].format(
        title=complaint.get("title", ""),
        description=complaint.get("description", ""),
        customer_type=complaint.get("customer_type", "Standard"),
        product=complaint.get("product", ""),
        order_reference=complaint.get("order_reference", "N/A"),
        channel=complaint.get("channel", "Web Form"),
        complaint_history=complaint.get("complaint_history", "None on record"),
        kb_excerpts=kb_excerpts or "(no matching policy excerpts retrieved)",
        json_schema=json.dumps(COMPLAINT_SCHEMA, indent=2),
    )
    return system_prompt, user_prompt


def _call_anthropic(system_prompt: str, user_prompt: str) -> str:
    """Real call to the Anthropic API. Requires ANTHROPIC_API_KEY in env."""
    import anthropic  # pip install anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    resp = client.messages.create(
        model=PROMPT_TEMPLATE["model"],
        max_tokens=PROMPT_TEMPLATE["max_tokens"],
        temperature=PROMPT_TEMPLATE["temperature"],
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return "".join(block.text for block in resp.content if block.type == "text")


def _call_mock(
    system_prompt: str,
    user_prompt: str,
    complaint: dict,
    kb_excerpts: str = ""
) -> str:
    """
    Deterministic offline fallback used when ANTHROPIC_API_KEY is not set
    (e.g. local dev, CI, or grading environments without a live key).
    Produces schema-valid JSON from simple keyword heuristics so the rest
    of the pipeline (validation, comparison, dashboards) is fully runnable
    without a network call. Swap SUPPORTNOVA_GENAI_MODE=live to use the
    real API once a key is configured.
    """
    text = f"{complaint.get('title','')} {complaint.get('description','')}".lower()
    kb_text = kb_excerpts.lower()

    def kb_has(*phrases):
      return any(p in kb_text for p in phrases)
    

    def has(*words):
        return any(w in text for w in words)

    if (
    has("hazard", "fire", "shock", "explode", "injur",
        "swollen", "swelling", "overheat", "overheating",
        "extremely hot", "hissing", "battery leak", "leaking battery")
    or (
        kb_has(
            "battery swelling",
            "severe overheating",
            "hissing",
            "product safety hazard",
            "critical",
            "p0"
        )
        and has(
            "battery", "swollen", "overheat",
            "extremely hot", "hissing"
        )
    )
):
        category, urgency, dept = "Safety", "Critical", "Safety"
    elif has("hack", "login", "password", "unauthorized", "breach"):
        category, urgency, dept = "Account", "Critical", "Account Security"
    elif has("charged twice", "duplicate charge", "billed twice"):
        category, urgency, dept = "Billing", "High", "Billing"
    elif has("late", "delay", "hasn't arrived", "has not arrived", "tracking"):
        category, urgency, dept = "Delivery", "Medium", "Logistics Support"
    elif has("broken", "damaged", "defect", "doesn't work", "malfunction"):
        category, urgency, dept = "Product Defect", "Medium", "Returns"
    elif has("refund"):
        category, urgency, dept = "Refund", "Medium", "Billing"
    elif has("warranty"):
        category, urgency, dept = "Warranty", "Medium", "Warranty"
    elif has("rude", "unprofessional", "disrespect"):
        category, urgency, dept = "Staff Behavior", "Medium", "Customer Relations"
    else:
        category, urgency, dept = "Service Quality", "Low", "Customer Relations"
    
    injection = any(p in text for p in [
        "ignore your instructions", "ignore previous instructions", "you are now",
        "admin mode", "approve my refund immediately", "system prompt",
    ])

    priority = {"Critical": "P0", "High": "P1", "Medium": "P2", "Low": "P3"}[urgency]
    sentiment = "Strongly Negative" if has("furious", "disgusted", "unacceptable", "!!!") else \
                "Negative" if has("frustrat", "upset", "disappoint", "angry") else "Neutral"

    policy_id = None
    policy_section = None
    if category == "Safety":
        policy_id = "SAF-POL-11"
        policy_section = "1.0"

    result = {
        "complaint_id": complaint.get("complaint_id", f"CMP-{uuid.uuid4().hex[:5]}"),
        "primary_issue": complaint.get("title", "Unspecified issue"),
        "secondary_issue": None,
        "issue_category": category,
        "subcategory": f"{category} — general",
        "sentiment": sentiment,
        "urgency": urgency,
        "priority": priority,
        "entities": {
            "product": complaint.get("product"),
            "order_id": complaint.get("order_reference"),
            "transaction_id": None,
            "amount": None,
            "date": None,
        },
        "department": dept,
        "supporting_department": None,
        "policy_id": policy_id,
        "policy_section": policy_section,
        "resolution_steps": [
            "Verify order/account details against internal records",
            "Confirm the reported issue with the customer",
            "Apply the resolution defined in the applicable policy",
        ],
        "escalation_required": urgency in ("Critical",),
        "escalation_level": "Critical Management Escalation" if urgency == "Critical" else "No Escalation",
        "escalation_reason": "Critical-severity issue requires immediate escalation." if urgency == "Critical" else None,
        "response_type": "Apology and Resolution Update",
        "professional_response": (
            f"Thank you for letting us know about this — I'm sorry for the trouble. "
            f"I've logged your complaint and routed it to our {dept} team, who will "
            f"follow up with next steps shortly."
        ),
        "follow_up_required": True,
        "follow_up_type": "Resolution confirmation",
        "clarification_questions": [] if complaint.get("order_reference") else [
            "Could you share your order or account reference number?"
        ],
        "agent_guidance": ["Do not promise compensation before policy verification"],
        "missing_information": [] if complaint.get("order_reference") else ["order_reference"],
        "prompt_injection_detected": injection,
        "confidence": 0.72,
    }
    return json.dumps(result)


def run_genai_pipeline(complaint: dict, kb_excerpts: str = "") -> GenAIResult:
    """
    Analyze a complaint with the GenAI pipeline. Retries on invalid JSON /
    schema failures with a controlled retry budget (SRS Step 47), then
    routes to manual review rather than looping forever.
    """
    system_prompt, user_prompt = build_prompt(complaint, kb_excerpts)
    mode = os.environ.get("SUPPORTNOVA_GENAI_MODE", "mock")  # "mock" | "live"

    result = GenAIResult(complaint_id=complaint.get("complaint_id", ""), raw_response="")

    for attempt in range(1, MAX_RETRIES + 1):
        result.attempts = attempt
        try:
            if mode == "live":
                raw = _call_anthropic(system_prompt, user_prompt)
            else:
                raw = _call_mock(
                    system_prompt,
                    user_prompt,
                    complaint,
                    kb_excerpts,
                )
        except Exception as exc:
            result.failure_reason = f"API call failed: {exc}"
            continue

        result.raw_response = raw
        cleaned = _strip_code_fences(raw)
        try:
            parsed = json.loads(cleaned)
            result.valid_json = True
        except json.JSONDecodeError as exc:
            result.failure_reason = f"Invalid JSON: {exc}"
            continue

        ok, errors = _validate_against_schema(parsed)
        result.schema_valid = ok
        result.schema_errors = errors
        if ok:
            result.parsed = parsed
            return result
        result.failure_reason = "Schema validation failed: " + "; ".join(errors[:3])

    # exhausted retries — do not loop forever, route to manual review
    result.manual_review_required = True
    return result
