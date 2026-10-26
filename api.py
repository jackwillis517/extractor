import tempfile
from pathlib import Path
from typing import Literal

import pymupdf
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from core import get_text, load_prompt
from extract import ExtractionError, extract

app = FastAPI()


@app.post("/extract/{schema}")
async def extract_schema(
    schema: Literal["job_posting", "invoice"],
    file: UploadFile,
    model: str,
    prompt: str = "v1",
):
    try:
        prompt_text = load_prompt(prompt)
    except FileNotFoundError as e:
        raise HTTPException(400, str(e))

    data = await file.read()
    suffix = Path(file.filename or "").suffix.lower() or ".bin"

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        try:
            text = get_text(path, schema)
        except (UnicodeDecodeError, pymupdf.FileDataError) as e:
            raise HTTPException(422, f"Could not read uploaded file: {e}")

    try:
        # extract() is blocking (up to minutes against Ollama); keep it off the event loop
        result, stats = await run_in_threadpool(extract, text, prompt_text, model, schema)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except ExtractionError as e:
        raise HTTPException(502, str(e))
    return {"schema": schema, "response": result.model_dump(), "stats": stats}


# curl -X POST "http://127.0.0.1:8000/extract/invoice?model=qwen3.5:4b-32k&prompt=v1_invoice" -F "file=@data/invoices/inv_01.pdf"
