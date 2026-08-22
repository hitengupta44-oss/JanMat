# JanMat

Free, non-commercial MVP. Ingests PRS Legislative Research stakeholder
analysis, scores sentiment, clusters feedback into themes, and generates
plain-language summaries — on a schedule, no manual data entry.

**Architecture:** FastAPI backend on Hugging Face Spaces (Docker SDK) +
Next.js frontend on Vercel + a GitHub Actions cron that pings the backend
on a schedule.

Deliberately minimal: **1 backend file, 3 frontend files.** Only PRS is
implemented — MyGov and MCA aren't half-built-and-disabled, they're just
not here, because neither can legally be scraped right now (see "What's
not here" below).

Files are merged wherever the tooling allows it, and split apart only
where a specific tool requires an exact filename:
- `config.yaml`, `db.py`, `scraper.py`, `nlp.py` → all merged into
  `app.py` (config dict, SQLite schema/queries, the PRS scraper, the
  Groq/clustering/summarization pipeline, and the FastAPI routes —
  clearly section-commented, top to bottom).
- `tailwind.config.ts` → merged into `postcss.config.js` as an inline
  config object.
- `next.config.js` → removed — its only setting was already Next's default.
- **TypeScript → plain JavaScript.** This project's data shapes are
  simple enough that type-checking wasn't earning its keep, so
  `tsconfig.json` and the `.tsx` files are gone; the frontend is now
  `.jsx` + `package.json` only.
- `package.json` stays on its own — npm won't find it under any other
  name.

## Folder structure

```
consultation-pulse/
├── backend/                  # FastAPI app → deploy as an HF Space (Docker SDK)
│   ├── app.py                 # EVERYTHING: config, db, scraper, NLP pipeline, FastAPI routes
│   ├── Dockerfile              # HF Spaces container (listens on :7860)
│   ├── requirements.txt
│   ├── .env.example
│   ├── .gitignore
│   └── tests/
│       └── test_scraper.py     # 5 unit tests — logic only, not live-site
│
├── frontend/                  # Next.js (App Router, plain JS, Tailwind) → deploy to Vercel
│   ├── app/
│   │   ├── layout.jsx           # fonts + metadata (Next.js requires this separate)
│   │   ├── page.jsx              # the ENTIRE UI — fetch, register, rows, theme cards, sentiment stamp
│   │   └── globals.css           # required separate — plain CSS, not a component
│   ├── package.json               # required by name — npm won't find it otherwise
│   ├── postcss.config.js          # Tailwind config inlined here
│   ├── .env.example
│   └── .gitignore
│
└── .github/workflows/scrape.yml # cron → POST /internal/run-pipeline
```

## Deploying

### 1. Backend → Hugging Face Spaces
1. Create a new Space → SDK: **Docker**.
2. Push the contents of `backend/` to the Space repo root (Dockerfile at root).
3. In Space **Settings → Repository secrets**, add:
   - `GROQ_API_KEY` — free key from console.groq.com
   - `CRON_SECRET` — any long random string
4. Space boots on `:7860` and serves `/health`, `/bills`, `/bills/{id}/themes`, `/bills/{id}/sentiment-summary`, `/sources`, `POST /internal/run-pipeline`.

⚠️ **Honest caveat:** free HF Spaces disk is **ephemeral** — it resets on
rebuild/sleep unless you enable persistent storage (paid). Fine for a
demo; for anything you don't want to lose, enable persistent storage or
point the `storage.sqlite_path` value in `app.py`'s `CONFIG` dict at a
mounted volume.

### 2. Frontend → Vercel
1. Import `frontend/` as a Vercel project (framework: Next.js, auto-detected — it doesn't require TypeScript).
2. Set env var `NEXT_PUBLIC_API_BASE_URL` = your Space's URL (e.g. `https://you-consultation-pulse.hf.space`).
3. Deploy.

`page.jsx` is a client component (fetches in the browser, not at
build/server time) so the whole UI — API calls, sentiment stamp, theme
cards, expandable rows — fits in one file without fighting Next.js's
server/client component split.

### 3. Cron → GitHub Actions
1. In the GitHub repo, add secrets `BACKEND_URL` (your Space URL) and
   `CRON_SECRET` (must match the Space's).
2. `.github/workflows/scrape.yml` runs every 6 hours (edit the cron
   expression to taste) and calls `POST /internal/run-pipeline`.
3. You can also trigger it manually from the Actions tab (`workflow_dispatch`).

## Local dev

```bash
# backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY, CRON_SECRET
uvicorn app:app --reload --port 7860

# frontend (separate terminal)
cd frontend
cp .env.example .env.local   # point at http://localhost:7860 for local dev
npm install
npm run dev
```

Run tests: `cd backend && pytest tests/ -v` (5/5 passing — logic-only,
not against the live PRS site).

## What's not here, and why

MyGov and MCA (both e-Consultation and Reports/Library) were dropped
entirely rather than kept as disabled stubs:
- **MyGov.in** — Terms & Conditions explicitly prohibit bots/scrapers
  "unless expressly authorized by MyGov in writing." Not conditioned on
  commercial use.
- **MCA e-Consultation** — submitted comments are only visible to the
  submitter after login (no public listing to fetch), and MCA's
  copyright policy separately bans storing/reproducing content without
  written consent.
- **MCA Reports/Library** — blocked by that same copyright policy.

If either becomes legally available (written authorization, or a
confirmed public dataset via data.gov.in), the cleanest path is a new
scraper class in `app.py` modeled on `PRSScraper`, not resurrecting old
disabled code.

## Known gaps (carried over honestly)

- **PRS selectors are unverified** against live markup — confirm the
  `selectors` block in `CONFIG["source"]` (top of `app.py`) before
  trusting a real run.
- **Single-source volume** — themes reflect PRS's stakeholder analysis,
  not raw high-volume citizen sentiment.
- **Groq free-tier rate limits** — fine for MVP demo volume, not
  production-scale comment counts without a paid tier.
- **HF Spaces free-tier disk is ephemeral** — see deploy note above.
