from pathlib import Path
from typing import cast

import pymupdf

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def load_prompt(version: str) -> str:
    path = PROMPTS_DIR / f"{version}.txt"
    if not path.exists():
        raise FileNotFoundError(f"No prompt '{version}' at {path}")
    return path.read_text(encoding="utf-8").strip()


def get_text(file: Path, schema: str) -> str:
    if schema == "job_posting":
        return file.read_text(encoding="utf-8")
    else:
        with pymupdf.open(file) as pdf:
            return "\n\n".join(cast(str, page.get_text()) for page in pdf)
