# Local Structured Extraction with Small LLMs

Turn messy documents into validated JSON with a small model running entirely on your own machine.
Two schemas are supported out of the box:

- **Job postings** (plain-text job ads) → title, company, location type, salary range, description, requirements
- **Invoices** (multi-page PDFs) → header fields, totals and every line item

Every extraction is typed with Pydantic, validated (and retried) by Instructor, and scored against hand-labelled data, so prompt and model changes can be compared with numbers instead of vibes.

## Stack

| Layer | Tool | Role |
|---|---|---|
| Model runtime | [Ollama](https://ollama.com) | Serves the model locally through an OpenAI-compatible API on `localhost:11434` |
| Model | `qwen3.5:2b` (plus `qwen3.5:4b` for comparison) | Small open-weights model; `Modelfile.2b` / `Modelfile` raise the context window to 32k |
| Structured output | [Instructor](https://python.useinstructor.com) (`Mode.JSON_SCHEMA`) | Sends the Pydantic schema as a JSON-schema constraint, parses the reply and retries on validation errors |
| Schemas | [Pydantic v2](https://docs.pydantic.dev) | `JobPosting` / `JobSearchResults` and `Invoice` / `LineItem` in `schemas.py` |
| CLI | [Typer](https://typer.tiangolo.com) | `run`, `eval` and `compare` commands in `cli.py` |
| API | [FastAPI](https://fastapi.tiangolo.com) | `POST /extract/{schema}` file upload in `api.py` |
| PDF text | [PyMuPDF](https://pymupdf.readthedocs.io) | Pulls the text layer out of invoice PDFs |
| Tooling | Python 3.12, [uv](https://docs.astral.sh/uv/), pytest | Dependency management and tests |

## Setup

```bash
# 1. Model: pull the base model, then build the 32k-context variant
ollama pull qwen3.5:2b
ollama create qwen3.5:2b-32k -f Modelfile.2b

# optional, for comparisons
ollama pull qwen3.5:4b
ollama create qwen3.5:4b-32k -f Modelfile

# 2. Python deps
uv sync
```

> **Why the `-32k` variant?** Ollama's default context window is far too small for this workload. qwen3.5 "thinks" before
> answering and a 3-page invoice is ~3k tokens in, so on the stock `qwen3.5:2b` most runs died with
> `The output is incomplete due to a max_tokens length limit` (see `results/v2_qwen3.5:2b.csv`: 17/20 failures).

## Usage

### CLI

```bash
# Extract one document
uv run python cli.py run data/postings/01.txt  --schema job_posting --prompt v4         --model qwen3.5:2b-32k
uv run python cli.py run data/invoices/inv_01.pdf --schema invoice  --prompt v2_invoice --model qwen3.5:2b-32k

# Score one prompt/model over the labelled set (writes results/<schema>_<prompt>_<model>.csv + .preds.jsonl)
uv run python cli.py eval --schema job_posting --prompt v4         --model qwen3.5:2b-32k
uv run python cli.py eval --schema invoice     --prompt v2_invoice --model qwen3.5:2b-32k

# Grid every prompt x model (writes results/compare_<schema>.csv)
uv run python cli.py compare --schema job_posting
uv run python cli.py compare --schema invoice
```

`--prompt` is the file name (without `.txt`) of a prompt in `prompts/`. `compare` runs the models in `COMPARE_MODELS`
against the prompts in `COMPARE_PROMPTS` (both in `cli.py`).

### API

```bash
uv run fastapi dev api.py

curl -X POST "http://127.0.0.1:8000/extract/job_posting?model=qwen3.5:2b-32k&prompt=v4" \
     -F "file=@data/postings/06.txt"

curl -X POST "http://127.0.0.1:8000/extract/invoice?model=qwen3.5:2b-32k&prompt=v2_invoice" \
     -F "file=@data/invoices/inv_01.pdf"
```

The response is `{"schema": ..., "response": {...extracted JSON...}, "stats": {latency_s, tokens_in, tokens_out, tok_per_s, attempts, retried, errors}}`.
Errors: `400` for an unknown prompt, `422` for an unreadable or empty file, and `502` when the model fails. Interactive docs are at `/docs`.

### Tests

```bash
uv run pytest
```

The tests cover the scorers and the API (with the model call mocked), so they run without Ollama.

## Project layout

```
cli.py         Typer CLI: run / eval / compare
api.py         FastAPI app: POST /extract/{schema}
extract.py     Instructor + Ollama call, stats (latency, tokens, retries), error mapping
core.py        load_prompt(), get_text() (txt read / PyMuPDF for PDFs)
schemas.py     Pydantic models for job postings and invoices
evals.py       run_eval() over a labelled set, summarize(), CSV/JSONL writers
scorers.py     Field-level scorers (exact, word-F1 overlap, numeric tolerance, line-item matching)
prompts/       Prompt versions: v1-v4 (jobs), v1_invoice-v3_invoice (invoices)
data/          20 job postings + 5 invoice PDFs, with *_labels.jsonl ground truth
results/       Per-run CSV + predictions JSONL, compare_*.csv summary tables
Modelfile*     Ollama model definitions with num_ctx 32768
```

## Evaluation

Each labelled document is extracted and scored field by field (0–1), and the document score is the mean of its fields.

| Scorer | Used for | Rule |
|---|---|---|
| `exact` | title, company, location, is_remote, invoice_number, vendor, customer, dates, currency | Case- and whitespace-insensitive equality |
| `overlap` | description, requirements | F1 over the sets of words |
| `number(tol)` | salaries (5%), money fields (0.5%) | Within relative tolerance; 0 and missing both count as "not present" |
| `item_count` / `item_amounts` | invoice line items | Count ratio; share of labelled line amounts found in the prediction |
| `reconciles` | invoices (label-free) | Predicted line amounts add up to the predicted subtotal (±1%) |
| `job_count` | job postings | The number of jobs returned matches the label |

A failed extraction scores 0 on every field instead of stopping the run. `summarize()` reports
`accuracy` (mean document score), `fail_rate`, and latency / tokens / throughput averaged over the successful documents.

## Results

All runs are on an RTX 3080 (10 GB) with Ollama, 20 job postings and 5 invoices. Both tables were produced by `compare`
(`results/compare_job_posting.csv`, `results/compare_invoice.csv`). Per-document scores and raw predictions are in the matching
`results/<schema>_<prompt>_<model>.csv` / `.preds.jsonl` files.

### Job postings: `uv run python cli.py compare --schema job_posting`

| model | prompt | accuracy | fail_rate | latency_s | tokens/doc | tok/s |
|---|---|---|---|---|---|---|
| qwen3.5:2b-32k | v1 | 0.602 | 0.00 | 14.0 | 3,398 | 169 |
| qwen3.5:2b-32k | v2 | 0.553 | 0.00 | 20.9 | 4,714 | 177 |
| qwen3.5:2b-32k | v3 | 0.713 | 0.00 | 15.3 | 3,613 | 171 |
| **qwen3.5:2b-32k** | **v4** | **0.779** | 0.00 | 40.5 | 8,555 | 178 |
| qwen3.5:4b-32k | v1 | 0.759 | 0.00 | 22.2 | 2,390 | 63 |
| qwen3.5:4b-32k | v2 | **0.800** | 0.00 | 48.7 | 4,058 | 63 |
| qwen3.5:4b-32k | v3 | 0.785 | 0.00 | 34.3 | 3,140 | 62 |
| qwen3.5:4b-32k | v4 | 0.786 | 0.00 | 83.1 | 6,482 | 63 |

Per-field means, v1 → v4:

| field | 2b v1 | 2b v4 | 4b v1 | 4b v4 |
|---|---|---|---|---|
| title | 0.60 | 0.80 | 0.80 | 0.80 |
| company | 0.90 | 0.90 | 0.90 | 0.85 |
| location | 0.85 | 0.80 | 0.85 | 0.90 |
| is_remote | 0.60 | 0.80 | 0.95 | 0.85 |
| **description** | **0.20** | **0.33** | **0.26** | **0.34** |
| requirements | 0.42 | 0.53 | 0.47 | 0.48 |
| min_salary | 0.80 | 0.95 | 0.85 | 0.95 |
| max_salary | 0.75 | 0.90 | 0.80 | 0.95 |
| job_count | 0.30 | 1.00 | 0.95 | 0.95 |

**Takeaway:** the new v4 prompt brings the 2b model (0.60 → 0.78) level with the 4b model (0.76–0.80), at about 2.5× the 2b latency.

### Invoices: `uv run python cli.py compare --schema invoice`

| model | prompt | accuracy | fail_rate | latency_s | tokens/doc | tok/s |
|---|---|---|---|---|---|---|
| qwen3.5:2b-32k | v1_invoice | 0.448 | 0.20 | 49.4 | 11,730 | 175 |
| **qwen3.5:2b-32k** | **v2_invoice** | **0.630** | 0.20 | 86.3 | 18,978 | 182 |
| qwen3.5:2b-32k | v3_invoice | 0.597 | 0.20 | 119.1 | 25,426 | 183 |
| qwen3.5:4b-32k | v1_invoice | 0.595 | 0.20 | 121.3 | 10,363 | 61 |
| qwen3.5:4b-32k | v2_invoice | 0.562 | 0.40 | 252.4 | 20,126 | 62 |
| qwen3.5:4b-32k | v3_invoice | 0.356 | 0.60 | 215.6 | 17,078 | 61 |

The best possible score is **0.80**, not 1.0: `inv_05.pdf` is a scanned image with no text layer and always fails (see below).

Per-field means for the 2b model:

| field | v1_invoice | v2_invoice | v3_invoice |
|---|---|---|---|
| invoice_date / due_date | 0.20 | 0.80 | 0.80 |
| currency | 0.40 | 0.80 | 0.80 |
| discount | 0.40 | 0.80 | 0.80 |
| tax | 0.60 | 0.80 | 0.80 |
| subtotal | 0.60 | 0.40 | 0.60 |
| line_item_count | 0.23 | 0.34 | 0.18 |
| line_item_amounts | 0.20 | 0.26 | 0.17 |
| reconciles | 0.00 | 0.00 | 0.00 |

**Takeaway:** the field rules in v2_invoice fix almost every *header* field (dates, currency, sign of the discount, tax amount). The 2b model
still can't copy 80–96 line items: it stops after 3–47 of them on every invoice. The 4b model copies them far better
(84/84, 96/96, 86/86 with v1) but times out on this hardware (see below).

> **Noise warning:** with 20 and 5 documents, scores move by a few points between identical runs (2b/v1 on job postings scored
> 0.671 on an earlier run and 0.602 on this one). Treat differences under ~0.05 as ties.

## Error analysis

### Job postings: why `description` is so bad

`description` is the worst field for every model and prompt (0.20–0.37). There are three separate causes.

**1. The schema has nowhere to put the location text, so it leaks into `description`.**
`location` is `Literal["remote", "hybrid", "onsite"]`, but every posting also has a city or address ("Austin, TX",
"Manhattan, NY • On-site", "Tucson - AZ, United States of America. Located at 1151 E Hermans Rd ~ BLDG 805"). With no slot
for it, and `description` coming right after `location`/`is_remote` in the field order, the model writes the location there.
With v1, **9/20 (2b) and 11/20 (4b) descriptions were just a location string**:

| file | expected description | predicted (4b, v1) |
|---|---|---|
| 13.txt | Consultant will join a team of 20 engineers and product managers, focusing on improving the rider experience… | `Hybrid work (3 days onsite / 2 Remote) in Manhattan, NY 10004` |
| 11.txt | We are looking for Full-Stack engineers to develop and maintain both front-end and back-end components… | `You can work from our offices in London, New York, San Francisco, and Warsaw.` |

v4 says outright that city and address text goes nowhere, and that cut location leaks to **0/20 for both models**.

**2. The labels don't follow one definition of "description".** Some labels describe the role ("Develop, test, and improve
software applications as part of an Agile scrum team…"), some the company mission (02.txt: "Fixing the youth mental health
crisis by redefining what mental healthcare looks like in the AI era"; 14.txt: "At Capital One, we are creating…"), and one is
just a sentence about the job type (03.txt: "This is a full-time remote role for Software Engineer (Ads)"). A prompt that asks for
"what the role does" is penalised on exactly those documents. 02, 03 and 14 are among the lowest v4 description scores for both models.

**3. The scorer punishes good paraphrases.** `overlap` is word-set F1. The model summarises in its own words, so a correct
description of 30 words can share fewer than 20% of its words with a correct 30-word label. Example, 19.txt (4b/v4, score 0.17):
*label* "Join Qaurs Techno Systems LLC as a Java Full Stack Developer with Python Scripting…" vs *predicted* "You will develop
responsive UI components using React.js, build backend services with Java and Spring Boot microservices…". Both are reasonable.

### Job postings: other errors

| issue | example | fixed by v4? |
|---|---|---|
| One posting split into several jobs | 2b/v1 returned extra jobs on **14/20** postings (job_count 0.30) | ✅ 0/20 |
| `is_remote` contradicts `location` | `location: "remote", is_remote: false` (8/20 for 2b/v1) | ✅ 1/20 |
| Title padded or trimmed | 09: "Software Engineer I - 2027" → "Campus Undergraduate Full-Time Engineer - 2027 Software Engineer I…"; 12: "Founding Web Full-Stack Software Engineer" → "Software Engineer" | partly |
| Company name shortened | "Raytheon / RTX" → "Raytheon", "Center for Data Science at NYU (Initiative: Polymathic AI)" → "NYU" | partly |
| Salary units | `100`/`150` for "$100k–$150k"; an invented salary on a posting that lists none | ✅ |

**A lesson from writing v4:** the first draft told the model to annualise hourly pay (×2080). The labels keep hourly rates hourly
(10.txt: `20–30`, 13.txt: `50–52`), so that rule *added* errors, and the 2b model applied it to an annual salary as well
(16.txt: `90,000 → 187,200,000`). The released v4 says "pay exactly as stated, do not convert". Prompt rules have to match
the labelling guide, and small models apply "if X then do Y" rules unreliably.

### Invoices

**1. Header fields are copied verbatim instead of normalised** (v1_invoice). These are easy wins, and v2_invoice fixes most of them:

| field | expected | predicted (v1_invoice, across runs) |
|---|---|---|
| invoice_date | `2026-09-06` | `06.09.2026`, `26/09/2026`, `September 5, 2026` |
| currency | `EUR` / `GBP` / `USD` | `€`, `GBP (£)`, `USD (implied)` |
| discount | `3976.65` | `-3976.65` (copied with its printed minus sign) |
| tax | `13715.19` | `19.0` (the VAT *rate*) or `1374.0` (misread German `13.715,19`) |
| invoice_number | `INV-2026-0412` | `[INV-2026-0412]`, `No. 10388`, `Rechnungsnr.: RE-2026-00731` |

**2. Line items get dropped, and this is the biggest gap.** Each invoice has 82–96 rows spread over 3 pages. The 2b model returns a
*prefix* of the list and stops: 3/84, 28/82, 41/96, 28/86 rows with v1_invoice, and never more than 47 with any prompt. Because the
items are missing, `reconciles` is 0.00 on every 2b run, and some subtotals are then made up to fit the partial list
(inv_02: 79,532.92 → 14,736.87). The 4b model copies the full table on most documents (84/84, 96/96, 86/86) but gave up early on
inv_02 (29/82).

**3. The worked example in v3_invoice hurt the small model.** v3_invoice adds a short example invoice and a "check that it adds up" step.
On 2b it improved subtotal (0.40 → 0.60) but made line-item recall *worse* (0.34 → 0.18) and once swapped vendor and customer
(inv_01). On 4b the extra reasoning pushed calls past the timeout (2 of the 4 readable invoices failed). The extra instructions use up the small model's
attention and thinking budget instead of helping it.

**4. Scanned PDFs have no text.** `inv_05.pdf` is an image. PyMuPDF returns an empty string and the run fails with
`Input text is empty`, which costs 20% of the invoice score on every run.

**5. Thinking tokens dominate cost and latency.** qwen3.5 reasons before answering, and the reasoning counts towards output tokens.
Measured directly against Ollama: a one-line job-title extraction took **4,653 completion tokens with thinking on and 14 with
`reasoning_effort: "none"`**. That is why a job posting costs 3–8k tokens for ~400 tokens of JSON, and an invoice costs 11–25k.
It is also what caused the `max_tokens` truncation on the default-context model.

**6. Infrastructure failures are mislabelled.** The 4b failures on invoices are client **timeouts** (`timeout=300` in `extract.py`):
4b-32k doesn't fit entirely in 10 GB of VRAM at 32k context (Ollama put 18–22% of it on the CPU, so it ran at ~62 tok/s vs ~180 for 2b), and its
successful v2 invoices already averaged 252 s. They show up in the CSV as *"Output failed validation after 1 attempts. Last error: unknown"*
because Instructor's retry loop catches **every** exception, including `APITimeoutError` and `NotFoundError`, and re-raises it as
`InstructorRetryException`. The `except APITimeoutError` / `APIConnectionError` / `NotFoundError` branches in `extract.py`
never run. (Reproduce: `cli.py run … --model does-not-exist` prints the "validation" message, not "Model not found".)

## Proposed fixes

A bigger model is the obvious fix: 4b already beats 2b on line items, and a 7–14B model or a hosted API would close most of the
remaining gap. These fixes help without changing the model:

### Schema and validation (largest effect for the least work)
- **Add a `location_text: str` field** (city/address as written) next to the `location` enum. Giving the leaked text its own slot is
  more reliable than telling the model not to write it.
- **Put `description` first** (after `title`/`company`) so it is generated before the location fields, and add
  `Field(description=...)` to every field. Instructor sends field descriptions inside the JSON schema, so the rules sit next to the
  field instead of in the prompt.
- **Stricter types:** `datetime.date` for invoice dates, `Literal["USD","EUR","GBP",...]` or a 3-letter regex for currency,
  `NonNegativeFloat` for discount and tax, and `requirements: list[str]`. Bad values then fail validation and Instructor retries with the
  error message, which `max_retries=3` already allows.
- **Business-rule validators:** a `model_validator` on `Invoice` that checks `sum(line amounts) ≈ subtotal` and
  `subtotal − discount + tax ≈ total`, and on `JobPosting` that `is_remote == (location == "remote")` and `min_salary ≤ max_salary`.
  A failure triggers a targeted retry ("you returned 28 line items summing to 14,736.87 but the subtotal is 79,532.92").

### Pipeline
- **Turn thinking off** for extraction (`reasoning_effort="none"`, or `think: false` in the native Ollama API). This is
  mostly copying, not reasoning. It should cut tokens/latency by an order of magnitude and remove the truncation and timeout failures;
  re-run `compare` to check its effect on accuracy.
- **Split invoices in two:** extract the header and totals in one call, then line items in another, **one page (or N rows) at a time**,
  and concatenate. A small model can copy 30 rows reliably even when it can't copy 96.
- **Skip the LLM for tables:** PyMuPDF's `page.find_tables()` can read line-item tables as cells. Parse them
  deterministically and use the model only for header fields, and as a fallback for layouts that aren't tables.
- **OCR fallback for scanned PDFs:** if `get_text()` comes back empty, run `page.get_textpage_ocr()` (Tesseract) or send the page
  image to a vision model (e.g. Qwen-VL via Ollama). This recovers inv_05.
- **Deterministic post-processing:** normalise dates (`dateutil`), map currency symbols to ISO codes, `abs()` the discount, and
  strip brackets and "No." prefixes. This is cheap and makes results independent of the prompt.
- **Fix the error mapping in `extract.py`:** in the `InstructorRetryException` handler, check `e.__cause__` (it holds the original
  `APITimeoutError` / `NotFoundError` / `APIConnectionError`) and raise the matching message. Raise the client timeout,
  or make it scale with document length, for long invoices.
- **`temperature=0`** (and a fixed seed) to make eval runs repeatable, so 0.02 differences mean something.

### Evaluation and data
- **Write a labelling guide** and re-label `description` consistently ("1–2 sentences on what the role does"). Several current labels
  are company-mission text, which caps the score of any prompt that follows the rule.
- **A better description scorer:** embedding cosine similarity or a small LLM-as-judge ("does the prediction describe the same role?")
  in place of word-set F1, which punishes good paraphrases.
- **Normalise before `exact`** (e.g. strip "(Ads)" / punctuation in titles) or allow partial credit, and report *error categories*
  (format vs. missing vs. hallucinated) as well as a single number.
- **More data:** 5 invoices and 20 postings are too few to separate prompts reliably (run-to-run swings of ±0.07). Aim for 50–100
  per schema, including scanned and non-English documents.

### Prompting
- Keep prompts **short and specific to each field** (v4, v2_invoice). Long worked examples (v3_invoice) hurt the 2b model.
- Put any few-shot example in the **system message** and keep it shorter than the real input, or pick examples by similarity.
- Every rule must match the label conventions (see the hourly-pay lesson above).

## Prompts

| prompt | schema | idea |
|---|---|---|
| `v1` | job | Bare "Extract job info from this text" |
| `v2` | job | + "description is the job, not the company, salary or location" |
| `v3` | job | + "ensure the description is of the job itself" |
| **`v4`** | job | **New.** One rule per field: exact title, full company name, location as an enum only with city text going nowhere, is_remote tied to location, 1–2 sentence role-only description, pay as stated (no conversion), one job unless the posting clearly lists several |
| `v1_invoice` | invoice | Bare "Extract the info from the following invoice" |
| **`v2_invoice`** | invoice | **New.** One rule per field: ISO dates, ISO currency, plain numbers, positive discount, tax as an amount not a rate, copy every line item |
| **`v3_invoice`** | invoice | **New.** v2 rules + a worked German-invoice example + a self-check that the line items, subtotal and total reconcile |

Recommended defaults: **`v4`** for job postings and **`v2_invoice`** for invoices.

## Potential upgrades

- **Async and batching:** use `instructor.from_openai(AsyncOpenAI(...))` with a concurrency limit, and turn `/extract` into a job queue
  (`POST` returns an id, `GET` polls) so 2-minute invoices don't hold HTTP connections open.
- **Schema registry:** load schemas, prompts and scorers from a single config so a new document type (receipts, resumes) is one file,
  not edits to `cli.py`, `api.py`, `extract.py` and `evals.py`.
- **Confidence and human review:** route documents that fail a validator, need retries, or don't reconcile to a review queue, and
  feed corrections back into the labelled set.
- **Vision input:** send page images to a multimodal model, so scanned documents, stamps, handwriting and layout-heavy invoices work.
- **Fine-tuning:** a LoRA on a few hundred labelled examples would likely let the 2b model match much larger models on these fixed schemas.
- **CI eval gate:** run `eval` on a small fixed subset in CI and fail the build if accuracy drops by more than a threshold.
- **Caching and observability:** cache by `(file hash, prompt, model)`, and log prompts, raw outputs, retries and token counts
  (e.g. OpenTelemetry or Logfire, both of which integrate with Instructor).
- **Pluggable backends:** the OpenAI-compatible client already works with vLLM, llama.cpp server, LM Studio or a hosted API.
  Make `base_url` a setting so the same eval can compare local and hosted models.

## Broader applications

The pattern (local model + typed schema + validation + labelled eval) carries over to most document-to-data problems,
especially where data **can't leave the building**:

- **Accounts payable:** invoices, receipts and purchase orders → ERP entries, with three-way matching against POs
- **Expense management:** receipt photos → categorised expense lines
- **Recruiting:** resumes and job ads → searchable candidate/role databases; salary-transparency analytics across job boards
- **Legal and procurement:** contracts → parties, dates, renewal terms, liability caps, governing law
- **Healthcare and insurance:** claims forms, referrals and lab reports → structured records (on-prem helps with HIPAA/GDPR)
- **Logistics:** bills of lading, customs declarations, packing lists
- **Finance and research:** earnings releases, filings and broker notes → tabular metrics
- **Support and ops:** emails and tickets → intent, product, urgency and entities for routing
- **Web data:** scraped listings (real estate, products, events) → clean, deduplicated rows
- **Public sector and research:** digitising archives and forms where documents are sensitive and budgets rule out per-token APIs
