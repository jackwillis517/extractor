import pymupdf
import pytest
from fastapi.testclient import TestClient

import api as api_module
from extract import ExtractionError
from schemas import Invoice, JobSearchResults

client = TestClient(api_module.app)
PARAMS = {"model": "test-model"}

JOB = {
    "jobs": [
        {
            "title": "SWE",
            "company": "Acme",
            "location": "remote",
            "is_remote": True,
            "description": "Build things",
            "min_salary": 0,
            "max_salary": 0,
            "requirements": "Python",
        }
    ]
}

INVOICE = {
    "invoice_number": "INV-1",
    "vendor": "Acme",
    "customer": "Beta",
    "invoice_date": "2026-10-01",
    "due_date": "2026-10-31",
    "currency": "USD",
    "subtotal": 100.0,
    "discount": 0.0,
    "tax": 8.0,
    "total": 108.0,
    "line_items": [],
}


def tiny_pdf() -> bytes:
    doc = pymupdf.open()
    doc.new_page().insert_text((72, 72), "Invoice INV-1  Total $108.00")
    return doc.tobytes()


@pytest.fixture
def fake_ok(monkeypatch):
    def fake(text, prompt, model, schema):
        result = (
            JobSearchResults.model_validate(JOB)
            if schema == "job_posting"
            else Invoice.model_validate(INVOICE)
        )
        return result, {
            "latency_s": 0.1,
            "tokens_in": 10,
            "tokens_out": 5,
            "attempts": 1,
        }

    monkeypatch.setattr(api_module, "extract", fake)


# Job posting extraction
def test_job_posting_text(fake_ok):
    r = client.post(
        "/extract/job_posting",
        params=PARAMS,
        files={"file": ("p.txt", b"SWE at Acme", "text/plain")},
    )
    assert r.status_code == 200
    assert r.json()["response"]["jobs"][0]["company"] == "Acme"


# Invoice extraction
def test_invoice_pdf(fake_ok):
    r = client.post(
        "/extract/invoice",
        params=PARAMS,
        files={"file": ("inv.pdf", tiny_pdf(), "application/pdf")},
    )
    assert r.status_code == 200
    assert r.json()["response"]["total"] == 108.0


# Unknown schemas get rejected
def test_unknown_schema_rejected(fake_ok):
    r = client.post(
        "/extract/receipt",
        params=PARAMS,
        files={"file": ("p.txt", b"x", "text/plain")},
    )
    assert r.status_code == 422  # Literal path parameter: FastAPI returns 422, not 404


# No file sent with extract request
def test_missing_file(fake_ok):
    assert client.post("/extract/invoice", params=PARAMS).status_code == 422


def test_extraction_failure_returns_502(monkeypatch):
    def boom(*args, **kwargs):
        raise ExtractionError("failed after 3 attempts")

    monkeypatch.setattr(api_module, "extract", boom)
    r = client.post(
        "/extract/invoice",
        params=PARAMS,
        files={"file": ("inv.pdf", tiny_pdf(), "application/pdf")},
    )
    assert r.status_code == 502


def test_unknown_prompt_returns_400(fake_ok):
    r = client.post(
        "/extract/invoice",
        params={**PARAMS, "prompt": "nope"},
        files={"file": ("inv.pdf", tiny_pdf(), "application/pdf")},
    )
    assert r.status_code == 400


def test_corrupt_pdf_returns_422(fake_ok):
    r = client.post(
        "/extract/invoice",
        params=PARAMS,
        files={"file": ("inv.pdf", b"not a pdf", "application/pdf")},
    )
    assert r.status_code == 422
