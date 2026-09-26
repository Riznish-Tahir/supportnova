"""
Comparison Engine
==================
Takes the GenAI pipeline's structured output and the Python validation
pipeline's independently-derived expectation, diffs the fields that matter,
and produces a verification decision: VERIFIED or MANUAL_REVIEW.

A case goes to manual review when GenAI and Python disagree on anything
safety-relevant (urgency/escalation), when Python found no matching rule,
when a hallucination/unsupported-promise flag was raised, or when GenAI
failed to return valid output at all.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FieldComparison:
    field: str
    genai_value: Optional[str]
    python_value: Optional[str]
    match: bool


@dataclass
class ComparisonReport:
    complaint_id: str
    verification_status: str  # "Verified" | "Manual Review"
    comparisons: list = field(default_factory=list)
    mismatch_reasons: list = field(default_factory=list)
    verification_score: float = 0.0


def compare(genai_result, validation_result) -> ComparisonReport:
    complaint_id = validation_result.complaint_id
    genai = getattr(genai_result, "parsed", None) or {}

    def cmp(name, g, p, critical=False):
        m = (g == p) if (g is not None and p is not None) else False
        fc = FieldComparison(field=name, genai_value=g, python_value=p, match=m)
        if not m:
            severity = "CRITICAL" if critical else "minor"
            report.mismatch_reasons.append(f"[{severity}] {name}: GenAI='{g}' vs Python='{p}'")
        return fc

    report = ComparisonReport(complaint_id=complaint_id, verification_status="Verified")

    report.comparisons = [
        cmp("category", genai.get("issue_category"), validation_result.expected_category),
        cmp("department", genai.get("department"), validation_result.expected_department),
        cmp("urgency", genai.get("urgency"), validation_result.expected_urgency, critical=True),
        cmp("priority", genai.get("priority"), validation_result.expected_priority),
        cmp(
            "escalation_required",
            genai.get("escalation_required"),
            validation_result.expected_escalation_required,
            critical=True,
        ),
        cmp("policy_id", genai.get("policy_id"), validation_result.expected_policy_id),
    ]

    matches = sum(1 for c in report.comparisons if c.match)
    report.verification_score = round(matches / len(report.comparisons), 3)

    needs_review = False
    if getattr(genai_result, "manual_review_required", False):
        needs_review = True
        report.mismatch_reasons.append("GenAI pipeline exhausted retries without valid output.")
    if validation_result.prompt_injection_detected:
        needs_review = True
    if validation_result.unsupported_promise_flags:
        needs_review = True
    if validation_result.hallucination_flags:
        needs_review = True
        report.mismatch_reasons.extend(validation_result.hallucination_flags)
    if not validation_result.expected_category:
        needs_review = True

    # Any CRITICAL-severity field mismatch forces manual review regardless of score
    if any("[CRITICAL]" in r for r in report.mismatch_reasons):
        needs_review = True

    report.verification_status = "Manual Review" if needs_review else "Verified"
    return report
