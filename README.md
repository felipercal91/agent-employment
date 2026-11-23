# Job Search Assistant

A personal, multi-agent job search assistant that automates **job discovery, deduplication, fit scoring, tailored 2-page PDF resume generation, application Q&A drafting, and follow-up tracking**.

> **Human-in-the-loop by design:** This tool automates preparation, **never submission**. Every generated resume, answer, and follow-up message stays pending your review and approval.

---

## Prerequisites

- **Python 3.11+**
- **Git**
- **Google Chrome or Microsoft Edge** (used by Playwright to render 2-page PDFs; alternatively run `playwright install chromium`)
- An API key from **DeepSeek** (recommended for low cost, ~\$0.01/CV) or **Anthropic (Claude)**

---

## Quick Start (Windows / PowerShell)

### 1. Clone and install dependencies

```powershell
git clone https://github.com/felipercal91/agent-employment.git
cd agent-employment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure your `.env`

```powershell
Copy-Item .env.example .env
```

Open `.env` and add your API key:
- **Option A — DeepSeek (cheapest):** Set `DEEPSEEK_API_KEY=...`, `WRITER_PROVIDER=deepseek`, `REVIEW_PROVIDER=deepseek`, and optionally `ANTHROPIC_ENABLED=false`.
- **Option B — Anthropic (Claude):** Set `ANTHROPIC_API_KEY=...` (default provider).

> `.env` is gitignored and will never be committed.

### 3. Copy the example templates into `data/`

All personal files live inside `data/` (gitignored). Copy the templates from `examples/` and fill them in with your real information:

```powershell
Copy-Item examples\cv_master.example.toml data\cv_master.toml
Copy-Item examples\profile.example.toml   data\profile.toml
Copy-Item examples\discovery.example.toml data\discovery.toml
Copy-Item examples\roles.example.toml     data\roles.toml
Copy-Item examples\tracker.example.xlsx   data\tracker.xlsx
```

*(Alternatively, you can generate a fresh blank workbook anytime with `python -m job_assistant tracker init`.)*

Edit each file with your own details:
- **`data/cv_master.toml`**: Your real work history, dates, achievements, skills, certifications, and education. The resume writer is strictly grounded in this file and will never invent titles, dates, or technologies.
- **`data/profile.toml`**: Your target title, years of experience, verified skills, work authorization, target locations, and `[situation]` block (availability, salary expectation, relocation preference).
- **`data/discovery.toml`**: Your target ATS boards (Greenhouse, Lever, Ashby), location patterns, title exclusions, `max_age_days` (default `14`), and `salary_floor_cad` (default `85000`).
- **`data/roles.toml`**: Your target role families (`title_keywords`, `ats_keywords`, `verify_terms`, and tailoring focus).

### 4. Customize code-level filters (important for new users)

Before running your first search, review two files for your own profile:
- **`job_assistant/discovery/unified_search.py`**: Update `ACTIVE_INTERVIEWS_EXCLUDE`, `EXCLUDED_CORE_STACK_REGEX`, and `SALARY_FLOOR_CAD` to match your own target stack and active interviews.
- **`job_assistant/agents/resume_writer.py`**: Review the system prompt constraints to ensure tenure/headline rules match your background.

### 5. Verify installation

```powershell
python -m pytest                 # Local unit tests (no API calls)
python -m job_assistant check    # Verifies .env and API connectivity
```

*(For DeepSeek: `python -m job_assistant check --provider deepseek`)*

---

## Daily Workflow

### 1. Discover new postings

```powershell
python -m job_assistant search
```

This runs the unified search across:
- Configured **Greenhouse, Lever, and Ashby** company boards (`data/discovery.toml`)
- **Job Bank Canada**
- **GetOnBoard** (100% remote LATAM/Chile roles -> `Postular Chile` sheet)
- **Manual inbox** (`data/inbox/` — drop `.txt` postings there and they are parsed and archived to `data/inbox/processed/`)

Built-in rules:
- **Deduplication**: Skips postings already in `data/tracker.xlsx` by normalized URL or company + title.
- **Salary filter & `Salary` column**: Logs the posted salary range (or `Not posted`) in the `Salary` column. Roles overlapping `salary_floor_cad` (default 85k+ CAD) or with no posted salary get `Estado = "Nuevo"`. Roles below the floor, hourly, or contract are **not dropped silently** — they are logged with `Estado = "Revisar"` so you can decide case by case.
- **Freshness**: Skips postings older than `max_age_days` (default 14 days).

### 2. Add manual postings (LinkedIn, Indeed, Bank portals)

Open `data/tracker.xlsx` → **`Postular`** sheet:
- Add a row with at least the **Link** and **Description** (plus Company, Role, Location, and Salary if available).
- Leave `Estado` as `Nuevo` (or change any `Revisar` row to `Nuevo` if you want a CV generated for it).
- Save and close Excel before running the CLI.

### 3. Generate tailored 2-page PDF resumes in parallel

```powershell
python -m job_assistant postular -w 5          # Process all Nuevo / Regenerar / Forzar rows with 5 workers
python -m job_assistant postular --row 12      # Process a single specific row
```

For each posting, the pipeline:
1. Runs a **Requirements Check** against `data/profile.toml` (stops on hard mismatches unless `Estado = "Forzar"`).
2. Runs the **Resume Writer Agent** grounded strictly in `data/cv_master.toml`.
3. Runs deterministic guardrails (no invented numbers/technologies, no protected titles in Quebec, no AI clichés, exact 2-page check) + **HR Reviewer Agent** (up to 2 rounds).
4. Saves outputs to `output/Resumenes <M-D>/<Company>_<Role>/`:
   - Tailored 2-page PDF + HTML + `resume_content.json`
   - `job_description.txt` and `revision_hr.md`
5. Updates `data/tracker.xlsx` with the status, PDF filename, and HR score.

### 4. Draft application form answers

```powershell
python -m job_assistant answer examples/forms/sample_form.txt --company "Acme" --role "Senior Java Developer"
```

Uses `[situation]` from `data/profile.toml` and past answers in `data/qa_history/`. Flags false premises (e.g., "current employer" when unemployed) and leaves legal/consent decisions for you.

### 5. Track applications, recruiters & follow-ups

Update `Estado` in `data/tracker.xlsx` to `Aplicado`, `Entrevista`, or `Descartado` as you progress. Use the built-in CRM and follow-up commands:

```powershell
python -m job_assistant contacts add --name "Jane Doe" --agency "Recruiting Co" --client "Bank" --role "Senior Java Dev"
python -m job_assistant contacts list
python -m job_assistant followup due
python -m job_assistant followup draft --company "Acme"
python -m job_assistant followup sent --to "Jane Doe"
```

---

## Optional Commands

| Command | Description |
|---------|-------------|
| `python -m job_assistant index` | Index your local documents in `data/` into the local Chroma vector store (`vectorstore/`). |
| `python -m job_assistant ask "Have I used Kafka?"` | Query your indexed profile/documents with source citations. |
| `python -m job_assistant masters` | Generate a master resume PDF per role defined in `data/roles.toml` in `output/master_resumes/`. |
| `python -m job_assistant evaluate posting.txt` | Run deterministic fit score + LLM requirements check on a single job posting. |
| `python -m job_assistant audit` | Re-audit previously generated resumes against the HR reviewer. |
| `python -m job_assistant tracker list` | Print detected sheets, columns, and row counts from `data/tracker.xlsx`. |

---

## Multiple Users on One Machine (Profiles)

You can run separate isolated profiles without duplicating the codebase using `jobs.cmd` (or `--profile <name>`). Each profile lives in `profiles/<name>/` (gitignored) with its own `data/`, `output/`, and optional `.env`:

```powershell
.\jobs.cmd friend setup
.\jobs.cmd friend index
.\jobs.cmd friend search
.\jobs.cmd friend postular -w 5
```

---

## Project Structure

```text
job_assistant/
├── __main__.py        # CLI entry point (`python -m job_assistant <command>`)
├── config.py          # Environment & profile settings
├── llm.py             # LLM provider factory (Anthropic / DeepSeek)
├── agents/            # Resume writer, HR reviewer, requirements checker, Q&A, follow-up
├── cv/                # Master CV loader, deterministic checks, HTML/PDF renderer, postular pipeline
├── discovery/         # Unified search, Greenhouse/Lever/Ashby boards, Job Bank, GetOnBoard
├── profile/           # Candidate profile & local Chroma RAG store
├── scoring/           # Deterministic keyword fit scoring
├── tailoring/         # Market/language detection (EN/FR/ES, Quebec OIQ guardrails)
└── tracker/           # Excel tracker reader/writer, deduplication, backups, templates
examples/              # Clean templates (cv_master, profile, discovery, roles, tracker.example.xlsx)
data/                  # Your personal data (gitignored — see data/README.md)
output/                # Generated resumes & reports (gitignored)
tests/                 # Automated pytest suite
```
