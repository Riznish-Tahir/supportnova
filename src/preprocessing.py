"""Complaint validation and pre-processing (SRS Steps 10-11, 52-54)."""
import re
import difflib
from html import escape

MIN_DESCRIPTION_LENGTH = 15


def sanitize(text: str) -> str:
    text = escape(text)  # neutralize any HTML/script injection
    text = re.sub(r"\s+", " ", text).strip()  # whitespace normalization
    text = text.replace("\u200b", "")  # strip zero-width characters
    return text


def validate_complaint_fields(payload: dict) -> list[str]:
    errors = []
    if not payload.get("title", "").strip():
        errors.append("Missing required field: title")
    desc = payload.get("description", "").strip()
    if not desc:
        errors.append("Missing required field: description")
    elif len(desc) < MIN_DESCRIPTION_LENGTH:
        errors.append(f"Description too short (min {MIN_DESCRIPTION_LENGTH} characters)")
    return errors


def is_near_duplicate(new_text: str, existing_texts: list[str], threshold: float = 0.85) -> str | None:
    """Returns the matching existing text if a near-duplicate is found."""
    for existing in existing_texts:
        ratio = difflib.SequenceMatcher(None, new_text.lower(), existing.lower()).ratio()
        if ratio >= threshold:
            return existing
    return None


def preprocess_complaint(payload: dict) -> dict:
    errors = validate_complaint_fields(payload)
    if errors:
        return {"valid": False, "errors": errors}

    cleaned = dict(payload)
    cleaned["title"] = sanitize(payload["title"])
    cleaned["description"] = sanitize(payload["description"])
    return {"valid": True, "errors": [], "complaint": cleaned}
