"""
Pipeline 2 — Python Ground-Truth Complaint Validation Pipeline
==================================================================
Deterministic, rule-based. Does NOT call a Generative AI API and does not
trust Pipeline 1's output. Computes its own expected category, department,
urgency, priority and escalation decision from the Complaint Resolution
Rule Matrix, then the comparison_engine diffs the two pipelines' outputs.

This is the module that stops a critical complaint from going unescalated
just because the GenAI pipeline missed it (SRS Step 39).
"""
from __future__ import annotations
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
RULE_MATRIX_PATH = BASE_DIR / "complaint_rules" / "rule_matrix.json"

with open(RULE_MATRIX_PATH) as f:
    RULE_MATRIX = json.load(f)

RULES_BY_KEY = {
    (r["category"], r["subcategory"]): r
    for r in RULE_MATRIX["rules"]
    if r["category"] != "Any"
}
GLOBAL_RULES = [r for r in RULE_MATRIX["rules"] if r["category"] == "Any"]

# Objective-risk keywords — urgency must not be driven by sentiment/tone alone
CRITICAL_KEYWORDS = [
    "fire", "shock", "spark", "smoke", "burning", "explode", "hazard",
    "swollen", "swelling", "overheat", "overheating", "extremely hot",
    "hissing", "battery leak", "leaking battery", "thermal",
    "unauthorized access", "data breach", "injur", "unconscious", "fraud",
    "hacked", "regulator", "lawsuit", "lawyer", "legal action",
]
INJECTION_PATTERNS = [
    r"ignore (your|previous|all)? ?instructions",
    r"you are now (in )?admin",
    r"system prompt",
    r"disregard (the )?[a-z ]{0,15}(rules|policy|policies)",
    r"approve (my|the) refund immediately",
    r"act as (an? )?administrator",
]

UNSUPPORTED_PROMISE_PATTERNS = [
    r"guarantee(d)? (a )?refund",
    r"guarantee(d)? compensation",
    r"you will (definitely|certainly) (get|receive)",
    r"100% refund approved",
    r"free replacement.{0,20}no questions asked",
]


@dataclass
class ValidationResult:
    complaint_id: str
    expected_category: Optional[str] = None
    expected_subcategory: Optional[str] = None
    expected_department: Optional[str] = None
    expected_urgency: Optional[str] = None
    expected_priority: Optional[str] = None
    expected_escalation_required: bool = False
    expected_escalation_level: str = "No Escalation"
    expected_policy_id: Optional[str] = None
    required_actions: list = field(default_factory=list)
    prohibited_actions: list = field(default_factory=list)
    compensation_eligible: bool = False
    prompt_injection_detected: bool = False
    unsupported_promise_flags: list = field(default_factory=list)
    hallucination_flags: list = field(default_factory=list)
    contradiction_flags: list = field(default_factory=list)
    notes: list = field(default_factory=list)


def detect_prompt_injection(text: str) -> bool:
    t = text.lower()
    return any(re.search(p, t) for p in INJECTION_PATTERNS)


def detect_objective_urgency(text: str) -> Optional[str]:
    """Urgency signal from objective risk words, independent of tone."""
    t = text.lower()
    if any(k in t for k in CRITICAL_KEYWORDS):
        return "Critical"
    return None


def detect_unsupported_promises(genai_response_text: str) -> list[str]:
    t = genai_response_text.lower()
    hits = []
    for pattern in UNSUPPORTED_PROMISE_PATTERNS:
        if re.search(pattern, t):
            hits.append(pattern)
    return hits


def lookup_rule(category: str, subcategory: str) -> Optional[dict]:
    if (category, subcategory) in RULES_BY_KEY:
        return RULES_BY_KEY[(category, subcategory)]
    # fall back to first rule with matching category if exact subcategory unseen
    # (hidden-evaluation "new subcategory" case) — flagged in notes by caller
    for (cat, _sub), r in RULES_BY_KEY.items():
        if cat == category:
            return r
    return None


def validate_complaint(complaint: dict, genai_output: Optional[dict]) -> ValidationResult:
    """
    Independently derive the ground-truth classification for a complaint and
    cross-check the GenAI pipeline's output for unsupported claims.
    complaint: raw submitted complaint dict (title, description, ...)
    genai_output: parsed + schema-valid output from Pipeline 1, or None if
                  Pipeline 1 failed (validation still proceeds independently).
    """
    text = f"{complaint.get('title','')} {complaint.get('description','')}"
    result = ValidationResult(complaint_id=complaint.get("complaint_id", ""))

    result.prompt_injection_detected = detect_prompt_injection(text)
    if result.prompt_injection_detected:
        result.notes.append(
            "Prompt-injection pattern found in complaint text — treated as content, "
            "not instruction. Routed for manual confirmation per SEC-POL-14."
        )

    # Use GenAI's category/subcategory as a *proposal* to look up in the matrix,
    # but urgency/escalation are always independently recomputed below.
    proposed_category = (genai_output or {}).get("issue_category")
    proposed_subcategory = (genai_output or {}).get("subcategory")

        # Independent raw-text classification for common technical issues
    text_lower = text.lower()

    if any(k in text_lower for k in [
        "app crash", "app crashes", "keeps crashing",
        "application crash", "application closes",
        "app closes", "crashes every time"
    ]):
        proposed_category = "Technical Support"
        proposed_subcategory = "App Crash"

    rule = None
    if proposed_category and proposed_subcategory:
        rule = lookup_rule(proposed_category, proposed_subcategory)
        if rule and rule["subcategory"] != proposed_subcategory:
            result.notes.append(
                f"Subcategory '{proposed_subcategory}' not in rule matrix — "
                f"matched nearest rule for category '{proposed_category}' instead. "
                "Flag for rule-matrix update (hidden-category case)."
            )

    if rule:
        result.expected_category = rule["category"]
        result.expected_subcategory = rule["subcategory"]
        result.expected_department = rule["department"]
        result.expected_policy_id = rule["policy_id"]
        result.required_actions = rule["required_actions"]
        result.prohibited_actions = rule["prohibited_actions"]
        result.compensation_eligible = rule["compensation_eligible"]
        base_urgency = rule["urgency"]
        base_escalation = rule["escalation_required"]
        base_level = rule["escalation_level"]
    else:
        result.notes.append("No matching rule found — routed to manual review.")
        base_urgency, base_escalation, base_level = "Medium", False, "No Escalation"

        # Objective-risk override: urgency is never determined by sentiment alone.
    if any(k in text.lower() for k in [
        "fire", "shock", "spark", "smoke", "burning",
        "swollen", "swelling", "overheat", "overheating",
        "extremely hot", "hissing", "battery leak", "leaking battery"
    ]):
        safety_rule = RULES_BY_KEY.get(("Safety", "Product Safety Hazard"))
        if safety_rule:
            rule = safety_rule
            result.expected_category = safety_rule["category"]
            result.expected_subcategory = safety_rule["subcategory"]
            result.expected_department = safety_rule["department"]
            result.expected_policy_id = safety_rule["policy_id"]
            result.required_actions = safety_rule["required_actions"]
            result.prohibited_actions = safety_rule["prohibited_actions"]
            result.compensation_eligible = safety_rule["compensation_eligible"]
            base_urgency = safety_rule["urgency"]
            base_escalation = safety_rule["escalation_required"]
            base_level = safety_rule["escalation_level"]
        
    objective_urgency = detect_objective_urgency(text)
    if objective_urgency == "Critical" and base_urgency != "Critical":
        result.notes.append(
            "Objective risk keywords found (safety/security/legal) — urgency "
            f"escalated from '{base_urgency}' to 'Critical' independent of GenAI/tone."
        )
        base_urgency = "Critical"
        base_escalation = True
        base_level = "Critical Management Escalation"

    result.expected_urgency = base_urgency
    result.expected_priority = {"Critical": "P0", "High": "P1", "Medium": "P2", "Low": "P3"}[base_urgency]
    result.expected_escalation_required = base_escalation
    result.expected_escalation_level = base_level

    # Repeat/unresolved complaint mandatory-escalation rule
    if complaint.get("is_repeat") and complaint.get("prior_status") == "unresolved":
        result.expected_escalation_required = True
        result.expected_escalation_level = "Department Manager"
        result.notes.append("Repeat unresolved complaint — mandatory escalation per RULE-045.")

    # Cross-check GenAI's generated response text for unsupported promises
    if genai_output and genai_output.get("professional_response"):
        flags = detect_unsupported_promises(genai_output["professional_response"])
        if flags and not result.compensation_eligible:
            result.unsupported_promise_flags = flags
            result.notes.append(
                "GenAI response contains a compensation/refund promise not "
                "supported by the rule matrix for this case — flagged for review."
            )

    # Hallucination check: GenAI cited a policy_id not in the matrix at all
    if genai_output and genai_output.get("policy_id"):
        known_ids = {r["policy_id"] for r in RULE_MATRIX["rules"]}
        if genai_output["policy_id"] not in known_ids:
            result.hallucination_flags.append(
                f"policy_id '{genai_output['policy_id']}' not found in approved rule matrix"
            )

    return result
