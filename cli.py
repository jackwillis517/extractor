# import logging
from pathlib import Path
from typing import Annotated, Literal

import typer

from core import get_text, load_prompt
from evals import ROOT, _save_csv, run_eval, summarize
from extract import ExtractionError, extract

# logging.basicConfig(level=logging.DEBUG)

app = typer.Typer()

Schema = Literal["job_posting", "invoice"]
COMPARE_MODELS = ["qwen3.5:2b-32k", "qwen3.5:4b-32k"]
COMPARE_PROMPTS = {
    "job_posting": ["v1", "v2", "v3", "v4"],
    "invoice": ["v1_invoice", "v2_invoice", "v3_invoice"],
}


@app.command()
def run(
    file: Path,
    schema: Annotated[Schema, typer.Option()],
    prompt: str = "v1",
    model: str = typer.Option(...),
):
    text = get_text(file, schema)
    try:
        result, stats = extract(text, load_prompt(prompt), model, schema)
        print(result.model_dump_json(indent=2))
        print(stats)
    except FileNotFoundError as e:
        typer.secho(f"File not found: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)
    except ExtractionError as e:
        typer.secho(f"Extraction failed: {e}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)


@app.command("eval")
def eval_cmd(
    schema: Annotated[Schema, typer.Option()],
    prompt: str | None = None,
    model: str = typer.Option(...),
):
    rows = run_eval(schema, model, prompt)
    for field, score in summarize(rows).items():
        print(f"{field:<14} {score:.2f}")


@app.command()
def compare(schema: Annotated[Schema, typer.Option()]):
    table = []
    for model in COMPARE_MODELS:
        for prompt in COMPARE_PROMPTS[schema]:
            rows = run_eval(schema, model, prompt)
            table.append({"model": model, "prompt": prompt, **summarize(rows)})
    for row in table:
        print(row)
        print("=================")
    _save_csv(table, ROOT / "results" / f"compare_{schema}.csv")


if __name__ == "__main__":
    app()

# uv run python cli.py run data/postings/01.txt --prompt v1 --model qwen3.5:4b --schema job_posting
# uv run python cli.py run data/invoices/inv_01.pdf --prompt v1_invoice --model qwen3.5:4b-32k --schema invoice
# uv run python cli.py eval --schema job_posting --prompt v1 --model qwen3.5:4b
# uv run python cli.py eval --schema invoice --prompt v1_invoice --model qwen3.5:4b-32k
# uv run python cli.py compare --schema job_posting
# uv run python cli.py compare --schema invoice
