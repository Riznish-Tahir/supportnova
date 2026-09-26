"""
Runs both pipelines across a sample of the complaint dataset and produces
the GenAI/Python Comparison Report deliverable (SRS Project Deliverable #8).
Run:  python3 tests/generate_comparison_report.py
"""
import sys, csv, json, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from genai_pipeline.pipeline import run_genai_pipeline
from python_validation.pipeline import validate_complaint
from comparison_engine.compare import compare

random.seed(7)
DATA = Path(__file__).resolve().parent.parent / "sample_complaints" / "complaints.json"
OUT = Path(__file__).resolve().parent.parent / "sample_complaints" / "genai_python_comparison_report.csv"

with open(DATA) as f:
    complaints = json.load(f)

sample = random.sample(complaints, 120)  # "at least 100 unseen complaint cases"

rows = []
for c in sample:
    complaint = {
        "complaint_id": c["complaint_id"], "title": c["title"], "description": c["description"],
        "order_reference": c["order_reference"], "product": c["product"],
        "is_repeat": c["is_repeat"], "prior_status": c["prior_status"],
    }
    genai_result = run_genai_pipeline(complaint)
    validation_result = validate_complaint(complaint, genai_result.parsed)
    report = compare(genai_result, validation_result)
    g = genai_result.parsed or {}
    rows.append({
        "complaint_id": c["complaint_id"],
        "actual_category": c["category"],
        "genai_category": g.get("issue_category"),
        "python_expected_category": validation_result.expected_category,
        "genai_department": g.get("department"),
        "python_department": validation_result.expected_department,
        "genai_urgency": g.get("urgency"),
        "python_urgency": validation_result.expected_urgency,
        "genai_escalation": g.get("escalation_required"),
        "python_escalation": validation_result.expected_escalation_required,
        "policy_reference": validation_result.expected_policy_id,
        "verification_status": report.verification_status,
        "verification_score": report.verification_score,
        "explanation": "; ".join(report.mismatch_reasons) if report.mismatch_reasons else "Full agreement",
    })

with open(OUT, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)

verified = sum(1 for r in rows if r["verification_status"] == "Verified")
print(f"Compared {len(rows)} cases -> {OUT}")
print(f"Verified: {verified} ({verified/len(rows)*100:.1f}%)  |  Manual Review: {len(rows)-verified}")
