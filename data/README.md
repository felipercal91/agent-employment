# data/

Local directory for your personal job search data. **Everything placed here (except this README) is gitignored** and will never be committed to GitHub.

## Required Files

Copy the starter templates from `examples/` into this folder and remove `.example` from the filenames:

```text
data/
├── cv_master.toml          # Your master resume facts (jobs, dates, achievements, skills, education)
├── cv_master_es.toml       # Optional: Spanish version of your master CV (for LATAM/Chile remote roles)
├── profile.toml            # Your structured skills, work authorization, locations, and [situation]
├── discovery.toml          # Target ATS boards (Greenhouse/Lever/Ashby), filters, and salary floor
├── roles.toml              # Target role families, title keywords, and ATS keywords
├── tracker.xlsx            # Your job tracker workbook (copy from examples/tracker.example.xlsx)
├── backups/                # Automatic Excel backups created before every write
├── inbox/                  # Optional: drop manual .txt job postings here to ingest on `search`
├── cv/                     # Optional: existing CV files (pdf/docx/md) for RAG indexing
├── projects/               # Optional: project notes (md/txt) for RAG indexing
├── feedback/               # Optional: interview feedback notes (md/txt)
└── qa_history/             # Optional: past application form answers (used as tone examples)
```
