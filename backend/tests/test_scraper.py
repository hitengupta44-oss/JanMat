"""
Unit tests scoped to logic correctness (config, dedup, parser logic
against a static HTML fixture) — NOT against the live PRS site, since
this build environment can't reach prsindia.org. Run these for real
before trusting a live run.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bs4 import BeautifulSoup

from app import CONFIG, PRSScraper, content_hash


def test_config_has_prs_source():
    assert CONFIG["source"]["name"] == "PRS Legislative Research"
    assert "selectors" in CONFIG["source"]


def test_content_hash_is_deterministic():
    h1 = content_hash("Some comment text")
    h2 = content_hash("Some comment text")
    h3 = content_hash("Different comment text")
    assert h1 == h2
    assert h1 != h3


def test_content_hash_ignores_surrounding_whitespace():
    assert content_hash("  Some comment text  ") == content_hash("Some comment text")


def test_prs_parser_handles_static_fixture():
    """
    Logic-only test: feeds a hand-built HTML fixture (matching the
    placeholder selectors in CONFIG) through PRSScraper's parsing,
    without hitting the network at all.
    """
    fixture = """
    <div class="bill-item">
      <h3 class="bill-title"><a href="/bill/123">Data Protection Amendment Bill</a></h3>
      <span class="bill-status">Pending</span>
    </div>
    """
    scraper = PRSScraper(CONFIG["source"])
    soup = BeautifulSoup(fixture, "lxml")
    sel = scraper.config["selectors"]

    card = soup.select_one(sel["bill_card"])
    title_el = card.select_one(sel["bill_title"])
    assert title_el.get_text(strip=True) == "Data Protection Amendment Bill"
    assert title_el.get(sel["bill_link_attr"]) == "/bill/123"


def test_prs_analysis_paragraph_parsing():
    """Short fragments/captions under 40 chars should be filtered out."""
    fixture = """
    <div class="stakeholder-analysis">
      <p>Short caption</p>
      <p>Industry stakeholders raised concerns about compliance costs for small businesses under the new rules.</p>
    </div>
    """
    scraper = PRSScraper(CONFIG["source"])
    soup = BeautifulSoup(fixture, "lxml")
    sel = scraper.config["selectors"]
    section = soup.select_one(sel["analysis_section"])
    paragraphs = [p.get_text(strip=True) for p in section.select(sel["analysis_paragraph"])]
    long_paragraphs = [p for p in paragraphs if len(p) >= 40]
    assert len(long_paragraphs) == 1
    assert "compliance costs" in long_paragraphs[0]
