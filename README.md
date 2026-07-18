# Prompt Optimizer Prompt Optimizer

Prompt Optimizer is a local-first system for improving and serving task-specific LLM prompts with evidence instead of guesswork. Its browser experience is a **Prompt Architect**: describe an outcome and it returns a detailed, portable prompt you can paste into a capable LLM. The project also includes a separate structured-extraction benchmark used to exercise the optimization machinery.

It is not a foundation model or a training pipeline. It is the measurement, optimization, and serving layer around a model you choose. By default it uses **local Ollama with `llama3.1:latest`**; a user can temporarily override that with their own OpenAI-compatible API key in the browser UI.

## What it does

1. Stores a prompt as independently versioned modules: role, context rules, output format, examples, and constraints.
2. Evaluates each prompt against individually stored benchmark cases, not only a single aggregate score.
3. Uses a stratified train / holdout / vault dataset split to avoid silently optimizing against the test set.
4. Generates constrained prompt variants that can modify only one module at a time.
5. Lets only the strongest training candidate access holdout evaluation.
6. Requires a sequential confidence bound, quality fences, and a human review before a candidate can become the serving champion.
7. Routes browser requests to the current champion and saves private traces plus optional user feedback.

## System at a glance

| Layer | Purpose |
| --- | --- |
| Browser UI | Aesthetic chat interface for sending requests, seeing confidence/trace data, and leaving feedback. |
| Serving plane | FastAPI endpoint that compiles the current champion prompt and calls the configured model backend. |
| Evaluation plane | Deterministic per-case metrics, latency capture, run manifests, and idempotent SQLite persistence. |
| Optimization plane | Train-only failure analysis, constrained prompt mutations, and a two-stage promotion gate. |
| Safety plane | Holdout isolation, vault confirmation, hard quality fences, and manual deployment. |

## Project layout

```text
optimizer/
├── serve.py       # FastAPI app, model selection, browser UI endpoints
├── static/         # Prompt Optimizer UI: HTML, CSS, and browser-side JavaScript
├── compiler.py     # Strict prompt rendering and version hashes
├── runner.py       # Async, idempotent benchmark execution
├── evaluators.py   # Deterministic JSON/schema/length metrics
├── generator.py    # Single-module prompt mutations and deduplication
├── gate.py         # Train margin, fences, and sequential promotion decision
├── cycle.py        # End-to-end optimization cycle and human promotion helper
├── seed.py         # Extraction benchmark plus the serving Prompt Architect champion
├── backends.py     # Demo backend and OpenAI-compatible HTTP backend
└── cli.py          # Commands for seed, run, review, promote, and vault checks
```

## Requirements

- Windows PowerShell instructions are provided below; Python 3.11+ is required.
- An internet connection is needed only to install Python packages or call an external model provider.
- Install and run Ollama locally for the default model path, or use a compatible API key from the browser UI.

## Run it locally

From the project folder, create an isolated environment and install the application:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Seed the local SQLite database with the extraction benchmark and the Prompt Architect champion:

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli seed
```

Start the web application:

```powershell
.\.venv\Scripts\python.exe -m uvicorn optimizer.serve:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). On first use, the app can also bootstrap its Prompt Architect champion automatically.

### Default local model: Ollama

Prompt Optimizer uses Ollama at `http://127.0.0.1:11434` by default and requests `llama3.1:latest`. Install and start Ollama, then make the model available before launching Prompt Optimizer:

```powershell
ollama pull llama3.1:latest
ollama serve
```

In another PowerShell window, start Prompt Optimizer with the Uvicorn command above. If you use another Ollama model, override it before starting Uvicorn:

```powershell
$env:OLLAMA_MODEL = "your-local-model:latest"
```

## Connect a real model

The browser never receives model credentials. It calls FastAPI, and FastAPI calls your provider from the server process.

For an application-wide server provider, set these variables in the same PowerShell window **before** starting Uvicorn:

```powershell
$env:OPTIMIZER_BACKEND = "http"
$env:OPTIMIZER_BASE_URL = "http://localhost:11434"
$env:OPTIMIZER_MODEL = "your-model-name"
$env:OPTIMIZER_API_KEY = "x"
.\.venv\Scripts\python.exe -m uvicorn optimizer.serve:app --reload
```

The configured provider must expose the OpenAI-compatible route:

```text
POST {OPTIMIZER_BASE_URL}/v1/chat/completions
```

For a provider that does not require authentication, leave `OPTIMIZER_API_KEY` as `x`. For a provider that does, replace it with its key. Do not put this key in `optimizer/static/app.js` or in the browser.

### Let a user use their own API key

In the UI, choose **Model** in the top right. The panel defaults to Ollama. To use another provider for the current browser tab, enter its compatible base URL, model name, and API key, then select **Use this API key**.

That selection automatically overrides Ollama for all new messages in that tab. The key is held only in page memory and is sent to the local FastAPI server only when making a request to the selected provider. Prompt Optimizer does not store the key in SQLite, browser storage, traces, or logs. Choosing **Use Ollama instead** immediately clears the in-memory key and returns to `llama3.1:latest`.

### Example: Ollama

Use an Ollama installation that exposes an OpenAI-compatible endpoint, then adapt the model name to one installed locally:

```powershell
$env:OPTIMIZER_BACKEND = "http"
$env:OPTIMIZER_BASE_URL = "http://localhost:11434"
$env:OPTIMIZER_MODEL = "llama3.1:latest"
$env:OPTIMIZER_API_KEY = "x"
.\.venv\Scripts\python.exe -m uvicorn optimizer.serve:app --reload
```

If your local provider uses a different route or request schema, add a small backend adapter in `optimizer/backends.py`; keep credentials and network access in that server-side adapter.

## Use the browser UI

The Prompt Optimizer interface includes:

- A focused Prompt Architect chat with sample requests to get started.
- Live backend status (`Demo model` or `Model connected`).
- Per-request category, confidence, latency, and trace identifier.
- Thumbs-up / thumbs-down feedback, stored locally for later analysis.
- A “New conversation” action that clears the visual session without deleting stored traces.

The seeded champion is designed to create robust prompts. Describe your desired end result, audience, constraints, and output when you know them; it will turn that into a copy-ready prompt. For example:

```text
Create an original multi-vendor marketplace website with a polished shopping experience, seller onboarding, search, cart, checkout, order management, and an admin panel. I need a complete full-stack build plan and production-ready implementation guidance.
```

## Run the optimizer

The following commands use the configured backend. With a running local Ollama model, they exercise the live local inference path; use the server-level provider settings above when you need another backend.

### 1. Run one optimization cycle

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli run
```

The cycle evaluates the champion on the train split, analyzes **only train failures**, generates novel one-module variants, evaluates them on train, and permits only the best candidate to touch holdout data.

If no candidate clears the 3-point train margin, the process stops before holdout evaluation. That is expected behavior, not an error.

### 2. Review proposed candidates

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli review
```

Review candidate mutation notes and any proposal that passed the statistical gate. A proposal is not automatically deployed.

### 3. Promote a reviewed candidate

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli promote extraction <prompt-id>
```

This retires the previous champion for the category, promotes the reviewed candidate, and clears the serving cache. Substitute `<prompt-id>` with an ID shown by `review`.

### 4. Run a vault regression check

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli vault-check extraction <old-champion-id> <candidate-id>
```

Vault data is never used to choose a candidate. It is a post-review regression guardrail. If it detects a regression, roll back by promoting the prior champion:

```powershell
.\.venv\Scripts\python.exe -m optimizer.cli rollback extraction <old-champion-id>
```

## Verify the system

Run the test suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Run the built-in statistical checks:

```powershell
.\.venv\Scripts\python.exe -m optimizer.selftest
```

The self-test performs two useful checks:

- **A/A null:** champion versus itself should not produce promotions.
- **Planted winner:** a deliberately superior mock prompt should become eligible for human review.

## HTTP API

The browser UI calls these same local endpoints.

| Method | Route | Description |
| --- | --- | --- |
| `GET` | `/` | Prompt Optimizer browser interface. |
| `GET` | `/health` | Active Ollama model and availability of the API-key override. |
| `POST` | `/query` | Runs a user request through the current champion. |
| `POST` | `/feedback` | Saves feedback against a prior response trace. |

Example request from PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/query `
  -ContentType "application/json" `
  -Body '{"text":"Create a portable prompt for an LLM to design a complete customer-feedback analysis workflow."}'
```

## Data and safety notes

- The default database is `optimizer.db` in the project directory. Set `OPTIMIZER_DB` before starting the app to use another location.
- Prompt versions, benchmark cases, evaluation results, traces, and feedback are stored in SQLite locally by default.
- Dataset split discipline matters: keep train, holdout, and vault examples separate. Do not put holdout/vault cases into prompt examples or failure analysis.
- Use only non-sensitive data with third-party model providers unless you have confirmed their retention and processing terms.
- The confidence score shown in the UI is routing confidence, not a guarantee that the extracted content is correct. Verify consequential outputs.

## Extending beyond extraction

To add a new task category, define its cases and deterministic evaluator(s), seed a baseline champion, then update routing. The extension points are:

1. Add category-specific metrics to `optimizer/evaluators.py`.
2. Add benchmark cases with stratified difficulty and split assignments.
3. Seed a category champion in `optimizer/seed.py`.
4. Extend `classify()` in `optimizer/serve.py`.
5. Run the self-tests and add task-specific regression tests before trusting promotions.

The crucial principle is unchanged: optimize prompts only against evidence you are allowed to inspect, and promote only with a reviewable record of improvement.
