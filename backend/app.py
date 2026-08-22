"""
Consultation Pulse — the entire backend in one file.

Sections below, top to bottom: config, database, scraper, NLP pipeline,
FastAPI app + routes. Deployed as an HF Space (Docker SDK); the Vercel
frontend reads from it, and a GitHub Actions cron calls
POST /internal/run-pipeline on a schedule.
"""
import hashlib
import json
import logging
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("consultation_pulse")

load_dotenv()


# =====================================================================
# CONFIG
# =====================================================================

CONFIG = {
    "source": {
        "name": "PRS Legislative Research",
        "base_url": "https://prsindia.org",
        "listing_url": "https://prsindia.org/billtrack",
        # NOTE (honest caveat): these selectors are structurally reasonable
        # placeholders, unverified against the live DOM. This sandbox cannot
        # reach prsindia.org (network restricted to package registries), so
        # the scraper is only tested for logic correctness (config loading,
        # dedup, parsing) — not against the live site. A human must confirm
        # these selectors before the first real run.
        "selectors": {
            "bill_card": "div.bill-item",
            "bill_title": "h3.bill-title a",
            "bill_link_attr": "href",
            "bill_status": "span.bill-status",
            "analysis_section": "div.stakeholder-analysis",
            "analysis_paragraph": "p",
        },
        "rate_limit_seconds": 2,
        # PRS's disclaimer permits reproduction "for non-commercial purposes...
        # with due acknowledgement of PRS Legislative Research." This deployment
        # is free/non-commercial and always attributes PRS in the UI.
        "license_note": (
            "PRS's disclaimer permits reproduction \"for non-commercial purposes... "
            "with due acknowledgement of PRS Legislative Research.\" This deployment "
            "is free/non-commercial and always attributes PRS in the UI."
        ),
    },
    "storage": {
        "sqlite_path": "data/consultation_pulse.db",
    },
    "nlp": {
        "provider": "groq",
        "model": "llama-3.1-8b-instant",
        "max_comments_per_batch": 20,
        "clustering": {
            "min_k": 3,
            "max_k": 10,
        },
    },
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
CRON_SECRET = os.getenv("CRON_SECRET", "")  # shared secret for the cron call


# =====================================================================
# DATABASE (SQLite — stands in for the graph/vector DB in the original
# design; Neo4j/pgvector add hosting complexity this MVP doesn't need yet)
# =====================================================================

DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS bills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    title TEXT NOT NULL,
    url TEXT UNIQUE NOT NULL,
    status TEXT,
    scraped_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS comments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bill_id INTEGER NOT NULL REFERENCES bills(id),
    source TEXT NOT NULL,
    author TEXT,
    body TEXT NOT NULL,
    content_hash TEXT UNIQUE NOT NULL,
    scraped_at TEXT NOT NULL,
    sentiment_label TEXT,
    sentiment_score REAL,
    theme_id INTEGER
);

CREATE TABLE IF NOT EXISTS themes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    bill_id INTEGER NOT NULL REFERENCES bills(id),
    label TEXT NOT NULL,
    summary TEXT NOT NULL,
    comment_count INTEGER NOT NULL,
    avg_sentiment REAL,
    generated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    detail TEXT
);

CREATE INDEX IF NOT EXISTS idx_comments_bill ON comments(bill_id);
CREATE INDEX IF NOT EXISTS idx_themes_bill ON themes(bill_id);
"""


def get_db_path() -> Path:
    path = Path(CONFIG["storage"]["sqlite_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def get_connection():
    conn = sqlite3.connect(get_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_connection() as conn:
        conn.executescript(DB_SCHEMA)


def save_bill(conn, source: str, title: str, url: str, status: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO bills (source, title, url, status, scraped_at)
           VALUES (?, ?, ?, ?, ?)
           ON CONFLICT(url) DO UPDATE SET status = excluded.status""",
        (source, title, url, status, now),
    )
    return conn.execute("SELECT id FROM bills WHERE url = ?", (url,)).fetchone()["id"]


def save_comment(conn, bill_id: int, source: str, author: str, body: str, content_hash_value: str) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT OR IGNORE INTO comments (bill_id, source, author, body, content_hash, scraped_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (bill_id, source, author, body, content_hash_value, now),
    )
    if cur.lastrowid:
        return cur.lastrowid
    return conn.execute(
        "SELECT id FROM comments WHERE content_hash = ?", (content_hash_value,)
    ).fetchone()["id"]


def is_duplicate_comment(conn, content_hash_value: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM comments WHERE content_hash = ? LIMIT 1", (content_hash_value,)
    ).fetchone() is not None


def save_sentiment(conn, comment_id: int, label: str, score: float):
    conn.execute(
        "UPDATE comments SET sentiment_label = ?, sentiment_score = ? WHERE id = ?",
        (label, score, comment_id),
    )


def save_theme(conn, bill_id: int, label: str, summary: str, comment_ids: list, avg_sentiment: float) -> int:
    now = datetime.now(timezone.utc).isoformat()
    cur = conn.execute(
        """INSERT INTO themes (bill_id, label, summary, comment_count, avg_sentiment, generated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (bill_id, label, summary, len(comment_ids), avg_sentiment, now),
    )
    theme_id = cur.lastrowid
    if comment_ids:
        conn.executemany(
            "UPDATE comments SET theme_id = ? WHERE id = ?",
            [(theme_id, cid) for cid in comment_ids],
        )
    return theme_id


def clear_themes_for_bill(conn, bill_id: int):
    conn.execute("DELETE FROM themes WHERE bill_id = ?", (bill_id,))
    conn.execute("UPDATE comments SET theme_id = NULL WHERE bill_id = ?", (bill_id,))


def list_bills(conn) -> list:
    rows = conn.execute(
        """
        SELECT b.id, b.source, b.title, b.url, b.status, b.scraped_at,
               (SELECT COUNT(*) FROM themes t WHERE t.bill_id = b.id) AS theme_count,
               (SELECT COUNT(*) FROM comments c WHERE c.bill_id = b.id) AS comment_count
        FROM bills b
        ORDER BY b.scraped_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def get_bill_themes(conn, bill_id: int) -> list:
    rows = conn.execute(
        """SELECT id, label, summary, comment_count, avg_sentiment, generated_at
           FROM themes WHERE bill_id = ? ORDER BY comment_count DESC""",
        (bill_id,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_bill_sentiment_summary(conn, bill_id: int) -> dict:
    row = conn.execute(
        """SELECT
             COUNT(*) AS total,
             AVG(sentiment_score) AS avg_score,
             SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive,
             SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative,
             SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) AS neutral,
             SUM(CASE WHEN sentiment_label = 'mixed' THEN 1 ELSE 0 END) AS mixed
           FROM comments WHERE bill_id = ?""",
        (bill_id,),
    ).fetchone()
    return dict(row)


def bill_exists(conn, bill_id: int) -> bool:
    return conn.execute("SELECT 1 FROM bills WHERE id = ?", (bill_id,)).fetchone() is not None


# =====================================================================
# SCRAPER — PRS Legislative Research (the only source this project
# ingests). Legally clean because PRS's disclaimer permits reproduction
# "for non-commercial purposes... with due acknowledgement." MyGov and
# MCA are not implemented at all: MyGov's ToS bans bots/scrapers outright,
# and MCA either has no public comment listing or a copyright policy
# blocking storage/reproduction. See README.md for the full rationale.
# =====================================================================

@dataclass
class ScrapedBill:
    title: str
    url: str
    status: Optional[str] = None


@dataclass
class ScrapedComment:
    bill_url: str
    author: Optional[str]
    body: str
    scraped_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


def content_hash(unique_text: str) -> str:
    return hashlib.sha256(f"prs::{unique_text.strip()}".encode("utf-8")).hexdigest()


class PRSScraper:
    source_key = "prs"

    def __init__(self, source_config: dict):
        self.config = source_config
        self.rate_limit_seconds = source_config.get("rate_limit_seconds", 2)
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "ConsultationPulseBot/0.1 (+non-commercial civic-tech MVP)"}
        )

    def _get(self, url: str) -> requests.Response:
        resp = self.session.get(url, timeout=15)
        resp.raise_for_status()
        time.sleep(self.rate_limit_seconds)
        return resp

    def fetch_bills(self) -> List[ScrapedBill]:
        sel = self.config["selectors"]
        resp = self._get(self.config["listing_url"])
        soup = BeautifulSoup(resp.text, "lxml")

        bills = []
        for card in soup.select(sel["bill_card"]):
            title_el = card.select_one(sel["bill_title"])
            if not title_el:
                continue
            href = title_el.get(sel["bill_link_attr"], "")
            url = href if href.startswith("http") else self.config["base_url"] + href
            status_el = card.select_one(sel["bill_status"])
            bills.append(
                ScrapedBill(
                    title=title_el.get_text(strip=True),
                    url=url,
                    status=status_el.get_text(strip=True) if status_el else None,
                )
            )
        return bills

    def fetch_comments(self, bill: ScrapedBill) -> List[ScrapedComment]:
        """
        PRS doesn't host raw citizen comments — it hosts structured
        stakeholder analysis. Each analysis paragraph becomes one
        "comment" unit so it flows through the same sentiment/clustering
        pipeline a real comment would.
        """
        sel = self.config["selectors"]
        resp = self._get(bill.url)
        soup = BeautifulSoup(resp.text, "lxml")

        section = soup.select_one(sel["analysis_section"])
        if not section:
            return []

        comments = []
        for para in section.select(sel["analysis_paragraph"]):
            text = para.get_text(strip=True)
            if len(text) < 40:  # skip stray short fragments/captions
                continue
            comments.append(
                ScrapedComment(
                    bill_url=bill.url,
                    author="PRS Legislative Research (stakeholder analysis)",
                    body=text,
                )
            )
        return comments

    def run(self) -> List[tuple]:
        """Returns [(ScrapedBill, [ScrapedComment, ...]), ...]."""
        results = []
        for bill in self.fetch_bills():
            try:
                comments = self.fetch_comments(bill)
            except Exception as exc:  # one bad bill shouldn't kill the run
                logger.warning("Failed fetching comments for %s: %s", bill.url, exc)
                comments = []
            results.append((bill, comments))
        return results


# =====================================================================
# NLP — Groq client, sentiment, TF-IDF/KMeans clustering, summarization.
# Free-tier stack on purpose: Groq (no GPU/hosting) for sentiment + theme
# text, scikit-learn TF-IDF+KMeans (no model download) for clustering
# instead of an embeddings model + vector DB.
# =====================================================================

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
DEFAULT_MODEL = "llama-3.1-8b-instant"


class GroqError(RuntimeError):
    pass


def call_groq_json(system_prompt: str, user_prompt: str, model: str = DEFAULT_MODEL,
                    max_retries: int = 3, temperature: float = 0.2) -> dict:
    if not GROQ_API_KEY:
        raise GroqError(
            "GROQ_API_KEY is not set. Add it as an environment secret "
            "(HF Space settings -> Repository secrets, or .env locally)."
        )

    headers = {"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "temperature": temperature,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                wait = 2 ** attempt
                logger.warning("Groq rate-limited, backing off %ss", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (requests.RequestException, KeyError, json.JSONDecodeError) as exc:
            last_error = exc
            logger.warning("Groq call failed (attempt %s/%s): %s", attempt, max_retries, exc)
            time.sleep(1.5 * attempt)

    raise GroqError(f"Groq call failed after {max_retries} attempts: {last_error}")


SENTIMENT_SYSTEM_PROMPT = """You are a sentiment classifier for public policy consultation
comments. For each numbered comment, return its sentiment toward the policy/bill being
discussed. Respond ONLY with JSON of the form:
{"results": [{"id": <int>, "label": "positive"|"negative"|"neutral"|"mixed", "score": <float -1..1>}]}
No extra commentary, no markdown fences."""


def score_batch(comments: List[str], model: str = DEFAULT_MODEL) -> List[Dict]:
    if not comments:
        return []
    numbered = "\n".join(f"{i+1}. {c}" for i, c in enumerate(comments))
    data = call_groq_json(SENTIMENT_SYSTEM_PROMPT, f"Comments:\n{numbered}", model=model)
    by_id = {r["id"]: r for r in data.get("results", []) if "id" in r}
    return [
        {
            "label": by_id.get(i + 1, {}).get("label", "neutral"),
            "score": float(by_id.get(i + 1, {}).get("score", 0.0)),
        }
        for i in range(len(comments))
    ]


def score_all(comments: List[str], batch_size: int = 20) -> List[Dict]:
    results = []
    for start in range(0, len(comments), batch_size):
        chunk = comments[start:start + batch_size]
        try:
            results.extend(score_batch(chunk))
        except Exception as exc:
            logger.error("Sentiment batch failed, defaulting to neutral: %s", exc)
            results.extend([{"label": "neutral", "score": 0.0} for _ in chunk])
    return results


def cluster_comments(comments: List[str], min_k: int = 3, max_k: int = 10) -> Dict[int, List[int]]:
    """Returns {cluster_id: [comment_index, ...]}."""
    n = len(comments)
    if n == 0:
        return {}
    if n <= min_k:
        return {i: [i] for i in range(n)}

    k = min(max(min_k, n // 4 or 1), max_k, n)
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(comments)
    labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(matrix)

    clusters: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(int(label), []).append(idx)
    return clusters


def top_terms_per_cluster(comments: List[str], clusters: Dict[int, List[int]], top_n: int = 6) -> Dict[int, List[str]]:
    """Cheap keyword extraction, used as a fallback theme label if summarization fails."""
    vectorizer = TfidfVectorizer(max_df=0.9, min_df=1, stop_words="english")
    matrix = vectorizer.fit_transform(comments)
    terms = np.array(vectorizer.get_feature_names_out())

    out = {}
    for cluster_id, idxs in clusters.items():
        scores = np.asarray(matrix[idxs].sum(axis=0)).ravel()
        top_idx = scores.argsort()[::-1][:top_n]
        out[cluster_id] = [t for t in terms[top_idx] if t.strip()]
    return out


SUMMARY_SYSTEM_PROMPT = """You summarize a cluster of public consultation comments that all
raised a similar theme about a policy or bill. Write for a busy government official with no
time to read raw comments. Respond ONLY with JSON:
{"label": "<3-6 word theme name>", "summary": "<2-3 plain-language sentences describing the
concern/support and why it matters>"}
No markdown, no extra keys."""


def summarize_cluster(comments: List[str], fallback_keywords: List[str], model: str = DEFAULT_MODEL) -> Dict[str, str]:
    if not comments:
        return {"label": "Uncategorized", "summary": "No comments in this cluster."}

    joined = "\n".join(f"- {c}" for c in comments[:15])  # cap prompt size
    try:
        data = call_groq_json(SUMMARY_SYSTEM_PROMPT, f"Comments:\n{joined}", model=model)
        return {
            "label": data.get("label") or (", ".join(fallback_keywords[:3]) or "Theme"),
            "summary": data.get("summary", ""),
        }
    except Exception as exc:
        logger.error("Summarization failed, using keyword fallback: %s", exc)
        label = ", ".join(fallback_keywords[:3]) if fallback_keywords else "Theme"
        return {
            "label": label.title(),
            "summary": (
                f"{len(comments)} comments touched on: {label}. "
                "(Auto-summary unavailable this run — showing extracted keywords.)"
            ),
        }


def run_pipeline() -> dict:
    """
    One full run: scrape PRS -> dedup-insert into SQLite -> score sentiment
    -> cluster into themes -> summarize -> store. Called on a schedule by
    the GitHub Actions cron via /internal/run-pipeline.
    """
    init_db()
    started_at = datetime.now(timezone.utc).isoformat()
    summary = {"bills_seen": 0, "comments_new": 0, "themes_generated": 0, "errors": []}

    with get_connection() as conn:
        run_id = conn.execute(
            "INSERT INTO pipeline_runs (started_at, status) VALUES (?, ?)",
            (started_at, "running"),
        ).lastrowid

        try:
            scraper = PRSScraper(CONFIG["source"])
            for bill, comments in scraper.run():
                summary["bills_seen"] += 1
                bill_id = save_bill(conn, "prs", bill.title, bill.url, bill.status)

                new_ids, new_bodies = [], []
                for c in comments:
                    h = content_hash(c.body)
                    if is_duplicate_comment(conn, h):
                        continue
                    cid = save_comment(conn, bill_id, "prs", c.author, c.body, h)
                    new_ids.append(cid)
                    new_bodies.append(c.body)

                summary["comments_new"] += len(new_ids)
                if not new_bodies:
                    continue

                for cid, s in zip(new_ids, score_all(new_bodies)):
                    save_sentiment(conn, cid, s["label"], s["score"])

                # Re-cluster ALL of this bill's comments so themes stay
                # coherent as volume grows, not just the newly-added ones.
                rows = conn.execute(
                    "SELECT id, body, sentiment_score FROM comments WHERE bill_id = ?",
                    (bill_id,),
                ).fetchall()
                bodies = [r["body"] for r in rows]
                ids = [r["id"] for r in rows]
                scores_by_id = {r["id"]: (r["sentiment_score"] or 0.0) for r in rows}

                clusters = cluster_comments(bodies, **CONFIG["nlp"]["clustering"])
                keywords = top_terms_per_cluster(bodies, clusters)

                clear_themes_for_bill(conn, bill_id)
                for cluster_id, idxs in clusters.items():
                    cluster_ids = [ids[i] for i in idxs]
                    theme = summarize_cluster([bodies[i] for i in idxs], keywords.get(cluster_id, []))
                    avg_sent = sum(scores_by_id[i] for i in cluster_ids) / len(cluster_ids)
                    save_theme(conn, bill_id, theme["label"], theme["summary"], cluster_ids, avg_sent)
                    summary["themes_generated"] += 1

            conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = ?, detail = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), "success", str(summary), run_id),
            )
        except Exception as exc:
            logger.exception("Pipeline run failed")
            summary["errors"].append(str(exc))
            conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, status = ?, detail = ? WHERE id = ?",
                (datetime.now(timezone.utc).isoformat(), "failed", str(exc), run_id),
            )

    return summary


# =====================================================================
# FASTAPI APP — deployed as an HF Space (Docker SDK)
# =====================================================================

init_db()

app = FastAPI(
    title="Consultation Pulse API",
    description=(
        "Free, non-commercial MVP. Ingests PRS Legislative Research "
        "stakeholder analysis only — see /sources for status."
    ),
    version="0.3.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # replace with ["https://your-app.vercel.app"] once deployed
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Bill(BaseModel):
    id: int
    source: str
    title: str
    url: str
    status: Optional[str]
    scraped_at: str
    theme_count: int
    comment_count: int


class Theme(BaseModel):
    id: int
    label: str
    summary: str
    comment_count: int
    avg_sentiment: Optional[float]
    generated_at: str


class PipelineRunResult(BaseModel):
    bills_seen: int
    comments_new: int
    themes_generated: int
    errors: List[str]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/sources")
def sources():
    src = CONFIG["source"]
    return [{"key": "prs", "name": src["name"], "enabled": True, "reason": src.get("license_note", "").strip()}]


@app.get("/bills", response_model=List[Bill])
def list_bills_route():
    with get_connection() as conn:
        return list_bills(conn)


@app.get("/bills/{bill_id}/themes", response_model=List[Theme])
def bill_themes(bill_id: int):
    with get_connection() as conn:
        if not bill_exists(conn, bill_id):
            raise HTTPException(status_code=404, detail="Bill not found")
        return get_bill_themes(conn, bill_id)


@app.get("/bills/{bill_id}/sentiment-summary")
def bill_sentiment_summary(bill_id: int):
    with get_connection() as conn:
        if not bill_exists(conn, bill_id):
            raise HTTPException(status_code=404, detail="Bill not found")
        return get_bill_sentiment_summary(conn, bill_id)


@app.post("/internal/run-pipeline", response_model=PipelineRunResult)
def trigger_pipeline(x_cron_secret: str = Header(default="")):
    """
    Called by the GitHub Actions cron job (see .github/workflows/scrape.yml).
    Protected by a shared secret so random internet traffic can't trigger
    scrapes/LLM spend.
    """
    if not CRON_SECRET or x_cron_secret != CRON_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Cron-Secret header")
    return run_pipeline()
