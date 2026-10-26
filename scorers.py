"""Each scorer compares a predicted value with the label and returns 0.0-1.0."""

import re


def norm(text):
    """Lowercase and collapse spaces: '  Software   Engineer ' -> 'software engineer'."""
    return None if text is None else " ".join(str(text).lower().split())


def words(text):
    """Set of lowercase words. Keeps c#, c++ intact."""
    return set(re.findall(r"[a-z0-9+#]+", str(text or "").lower()))


def exact(pred, exp):
    """title, company, location, is_remote: right or wrong."""
    return float(norm(pred) == norm(exp))


def overlap(pred, exp):
    """description, requirements: share of words in common (F1 over word sets)."""
    p, e = words(pred), words(exp)
    if not p and not e:
        return 1.0
    return 2 * len(p & e) / (len(p) + len(e))


def number(tol=0.05):
    """Within tol (relative). 0 and None both mean 'not present'."""

    def score(pred, exp):
        pred, exp = pred or None, exp or None
        if pred is None or exp is None:
            return float(pred == exp)
        return float(abs(pred - exp) <= tol * abs(exp))

    return score


def item_count(pred, exp):
    """1.0 when the counts match; partial credit for being close."""
    p, e = len(pred or []), len(exp or [])
    return 1.0 if p == e else min(p, e) / max(p, e, 1)


def item_amounts(pred, exp, tol=0.005):
    """Share of expected line amounts found among the predicted ones (each used once)."""
    remaining = [i["amount"] for i in (pred or [])]
    found = 0
    for item in exp or []:
        match = next(
            (
                a
                for a in remaining
                if abs(a - item["amount"]) <= tol * max(abs(item["amount"]), 1)
            ),
            None,
        )
        if match is not None:
            remaining.remove(match)
            found += 1
    return found / len(exp) if exp else float(not pred)


JOB_SCORERS = {
    "title": exact,
    "company": exact,
    "location": exact,
    "is_remote": exact,
    "description": overlap,
    "requirements": overlap,
    "min_salary": number(0.05),
    "max_salary": number(0.05),
}

INVOICE_SCORERS = {
    "invoice_number": exact,
    "vendor": exact,
    "customer": exact,
    "invoice_date": exact,
    "due_date": exact,
    "currency": exact,
    "subtotal": number(0.005),
    "discount": number(0.005),
    "tax": number(0.005),
    "total": number(0.005),
    "line_item_count": lambda p, e: item_count(p, e),
    "line_item_amounts": lambda p, e: item_amounts(p, e),
}


def score_fields(pred: dict | None, exp: dict, scorers: dict, rename=None) -> dict:
    """Score one record. rename maps a score name to the label field it reads."""
    rename = rename or {}
    if pred is None:
        return {name: 0.0 for name in scorers}
    return {
        name: fn(pred.get(rename.get(name, name)), exp.get(rename.get(name, name)))
        for name, fn in scorers.items()
    }


def score_posting(pred, exp):
    """pred and exp are {"jobs": [...]}. pred is None if extraction failed.
    Jobs are compared in order; a missing job scores 0."""
    pred_jobs = pred["jobs"] if pred else []
    exp_jobs = exp["jobs"]
    scores = {}
    for field, scorer in JOB_SCORERS.items():
        per_job = [
            scorer(pred_jobs[i].get(field), job.get(field))
            if i < len(pred_jobs)
            else 0.0
            for i, job in enumerate(exp_jobs)
        ]
        scores[field] = sum(per_job) / len(per_job)
    scores["job_count"] = float(len(pred_jobs) == len(exp_jobs))
    return scores


def score_invoice(pred, exp):
    rename = {"line_item_count": "line_items", "line_item_amounts": "line_items"}
    scores = score_fields(pred, exp, INVOICE_SCORERS, rename)
    # Label-free check: do the predicted items add up to the predicted subtotal?
    if pred and pred.get("line_items") and pred.get("subtotal"):
        s = sum(i["amount"] for i in pred["line_items"])
        scores["reconciles"] = float(
            abs(s - pred["subtotal"]) <= 0.01 * pred["subtotal"]
        )
    else:
        scores["reconciles"] = 0.0
    return scores
