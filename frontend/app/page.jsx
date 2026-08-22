"use client";

/**
 * Consultation Pulse — the entire UI in one file.
 *
 * Fetches from the FastAPI backend (NEXT_PUBLIC_API_BASE_URL), renders the
 * bill register, and expands each row into its themes on click. The stamp
 * badge is the signature visual — styled like the ink seals on official
 * government paperwork.
 *
 * This is a client component (not server-rendered) so all the fetching,
 * expand/collapse state, and rendering can live in a single file instead
 * of being split across server/client component boundaries.
 */
import { useEffect, useState } from "react";

// ---------- api ----------

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || "";

class ApiError extends Error {}

async function getJson(path) {
  if (!API_BASE) {
    throw new ApiError(
      "NEXT_PUBLIC_API_BASE_URL is not set. Add it in .env.local (dev) or Vercel project settings (prod)."
    );
  }
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) throw new ApiError(`Request to ${path} failed with ${res.status}`);
  return res.json();
}

const fetchBills = () => getJson("/bills");
const fetchThemes = (billId) => getJson(`/bills/${billId}/themes`);
const fetchSources = () => getJson("/sources");

// ---------- sentiment stamp (signature visual) ----------

function classifySentiment(score) {
  if (score === null) return { label: "UNSCORED", color: "text-ink-faint", ring: "border-ink-faint" };
  if (score > 0.25) return { label: "SUPPORTIVE", color: "text-moss", ring: "border-moss" };
  if (score < -0.25) return { label: "OPPOSED", color: "text-seal", ring: "border-seal" };
  return { label: "MIXED", color: "text-slate", ring: "border-slate" };
}

function SealStamp({ score }) {
  const { label, color, ring } = classifySentiment(score);
  return (
    <div
      className="relative w-16 h-16 shrink-0 -rotate-6 select-none"
      role="img"
      aria-label={`Sentiment: ${label}${score !== null ? `, score ${score.toFixed(2)}` : ""}`}
    >
      <div
        className={`absolute inset-0 rounded-full border-2 ${ring} opacity-80`}
        style={{ boxShadow: "inset 0 0 0 3px rgba(237,230,211,1)" }}
      />
      <div className={`absolute inset-[3px] rounded-full border ${ring} opacity-60`} />
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-1">
        <span className={`font-mono text-[7px] tracking-widest ${color} opacity-90`}>PULSE</span>
        <span className={`font-display font-semibold text-sm ${color} leading-none mt-0.5`}>
          {score !== null ? score.toFixed(2) : "—"}
        </span>
        <span className={`font-mono text-[7px] tracking-wider ${color} mt-0.5`}>{label}</span>
      </div>
    </div>
  );
}

// ---------- theme card ----------

function ThemeCard({ theme }) {
  return (
    <div className="flex gap-4 border border-paper-line/80 bg-paper/60 p-4 rounded-sm">
      <SealStamp score={theme.avg_sentiment} />
      <div className="flex-1 min-w-0">
        <div className="flex items-baseline justify-between gap-2">
          <h4 className="font-display text-lg text-ink leading-snug">{theme.label}</h4>
          <span className="font-mono text-xs text-ink-faint whitespace-nowrap">
            {theme.comment_count} {theme.comment_count === 1 ? "entry" : "entries"}
          </span>
        </div>
        <p className="mt-1.5 text-sm text-ink-faint leading-relaxed">{theme.summary}</p>
      </div>
    </div>
  );
}

// ---------- empty / error state ----------

function EmptyState({ title, detail }) {
  return (
    <div className="border border-dashed border-paper-line rounded-sm py-14 px-6 text-center">
      <p className="font-display text-xl text-ink">{title}</p>
      <p className="font-mono text-xs text-ink-faint mt-2 max-w-md mx-auto leading-relaxed">{detail}</p>
    </div>
  );
}

// ---------- bill row (expandable, lazy-loads its own themes) ----------

function BillRow({ bill, index }) {
  const [open, setOpen] = useState(false);
  const [themes, setThemes] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  async function toggle() {
    const next = !open;
    setOpen(next);
    if (next && themes === null) {
      setLoading(true);
      setError(null);
      try {
        setThemes(await fetchThemes(bill.id));
      } catch (e) {
        setError(e instanceof ApiError ? e.message : "Couldn't load themes for this bill.");
      } finally {
        setLoading(false);
      }
    }
  }

  return (
    <div className="border-b border-paper-line last:border-b-0">
      <button onClick={toggle} aria-expanded={open} className="w-full flex items-center gap-4 py-4 text-left group">
        <span className="font-mono text-xs text-ink-faint w-8 shrink-0">
          {String(index + 1).padStart(2, "0")}
        </span>
        <span className="font-display text-lg text-ink flex-1 group-hover:text-marigold transition-colors">
          {bill.title}
        </span>
        {bill.status && (
          <span className="font-mono text-[10px] uppercase tracking-wider text-ink-faint border border-paper-line px-2 py-1 rounded-sm shrink-0">
            {bill.status}
          </span>
        )}
        <span className="font-mono text-xs text-ink-faint shrink-0 w-24 text-right">
          {bill.theme_count} {bill.theme_count === 1 ? "theme" : "themes"}
        </span>
        <span className={`font-mono text-ink-faint shrink-0 transition-transform ${open ? "rotate-90" : ""}`} aria-hidden>
          →
        </span>
      </button>

      {open && (
        <div className="pb-5 pl-12 space-y-3">
          {loading && <p className="font-mono text-xs text-ink-faint">Reading stakeholder analysis…</p>}
          {error && <p className="font-mono text-xs text-seal">{error}</p>}
          {themes && themes.length === 0 && !loading && (
            <p className="font-mono text-xs text-ink-faint">
              No themes generated yet for this bill — the pipeline hasn&apos;t processed it, or PRS
              hasn&apos;t published stakeholder analysis for it yet.
            </p>
          )}
          {themes?.map((t) => (
            <ThemeCard key={t.id} theme={t} />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------- page ----------

export default function Page() {
  const [bills, setBills] = useState(null);
  const [sources, setSources] = useState(null);
  const [loadError, setLoadError] = useState(null);

  useEffect(() => {
    Promise.all([fetchBills(), fetchSources()])
      .then(([b, s]) => {
        setBills(b);
        setSources(s);
      })
      .catch((e) => setLoadError(e instanceof ApiError ? e.message : "Couldn't reach the register."));
  }, []);

  return (
    <main className="max-w-3xl mx-auto px-6 py-14">
      {/* Masthead */}
      <header className="mb-10 border-b-2 border-ink pb-6">
        <p className="font-mono text-xs tracking-[0.2em] text-marigold uppercase mb-2">
          Public Consultation Register
        </p>
        <h1 className="font-display text-4xl sm:text-5xl text-ink font-semibold leading-tight">
          Consultation Pulse
        </h1>
        <p className="font-body text-sm text-ink-faint mt-3 max-w-xl leading-relaxed">
          A plain-language snapshot of public feedback on tracked bills — sentiment scored and
          clustered into themes so the volume is readable, not just archived.
        </p>
      </header>

      {loadError && (
        <EmptyState
          title="The register can't be reached right now."
          detail={`${loadError} If you're running this locally, check NEXT_PUBLIC_API_BASE_URL points at your live HF Space and that the Space is awake.`}
        />
      )}

      {!loadError && bills === null && (
        <p className="font-mono text-xs text-ink-faint">Loading the register…</p>
      )}

      {!loadError && bills && bills.length === 0 && (
        <EmptyState
          title="No entries yet."
          detail="The scraper hasn't run, or hasn't found any tracked bills yet. Trigger a pipeline run (see .github/workflows/scrape.yml) to populate the register."
        />
      )}

      {!loadError && bills && bills.length > 0 && (
        <section aria-label="Tracked bills">
          {bills.map((bill, i) => (
            <BillRow key={bill.id} bill={bill} index={i} />
          ))}
        </section>
      )}

      {/* Source attribution — required by PRS's non-commercial reproduction terms */}
      {sources && (
        <footer className="mt-14 pt-6 border-t border-paper-line">
          <p className="font-mono text-[10px] uppercase tracking-widest text-ink-faint mb-3">Data sources</p>
          <ul className="space-y-1.5">
            {sources.map((s) => (
              <li key={s.key} className="font-mono text-xs text-ink-faint flex gap-2">
                <span className={s.enabled ? "text-moss" : "text-seal"}>{s.enabled ? "●" : "○"}</span>
                <span>
                  {s.name} — {s.enabled ? "live" : "not yet enabled"}
                </span>
              </li>
            ))}
          </ul>
          <p className="font-mono text-[10px] text-ink-faint mt-4 leading-relaxed">
            Content reproduced from PRS Legislative Research for non-commercial purposes, with due
            acknowledgement, per their reproduction policy. This is a free, non-commercial demonstration.
          </p>
        </footer>
      )}
    </main>
  );
}
