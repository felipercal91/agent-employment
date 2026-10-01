# CLAUDE.md - Job Search Assistant

Personal multi-agent assistant (Python + LangChain/LangGraph + Claude) that supports a job search for Senior Java Backend roles in Toronto, Montreal and remote Canada. It **prepares** material; it never **sends** anything.

## Non-negotiable rules (override everything else)

No code in this repo may:
1. Create accounts on job portals (Indeed, LinkedIn, etc.).
2. Fill in or submit application forms automatically.
3. Send messages/emails/replies to recruiters without explicit human approval.
4. Drive a job portal's web UI (no aggressive scraping, no browser automation that clicks "Apply"/"Send").

Every generated artifact (CV, cover letter, message, form answer) ends in a `pending_approval` state for manual review.

## Language rules

- Code, identifiers, comments: English.
- Employer-facing documents: English by default; French **only** for Quebec postings.
- In French, the title is "Développeur", never "Ingénieur" (protected by the OIQ).
- Never invent experience: every claim must be grounded in the documents under `data/`.

## Conventions

- Secrets only via `.env` (see `.env.example`); `data/`, `output/`, vector stores are gitignored.
- All Python functions have type hints.
- Build proceeds phase by phase; do not start the next phase until the user confirms the current one.
- Run tests: `.venv\Scripts\python -m pytest`
