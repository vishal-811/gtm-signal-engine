"""Small pure string helpers shared across stages. No I/O, easy to unit test."""

from __future__ import annotations

import re
from urllib.parse import urlparse, urlunparse

# Words that carry no identity — stripped before comparing company names so
# "Acme, Inc." and "Acme" collapse to the same key.
_COMPANY_SUFFIXES = {
    "inc",
    "inc.",
    "llc",
    "ltd",
    "limited",
    "corp",
    "corporation",
    "co",
    "gmbh",
    "bv",
    "plc",
    "pvt",
    "private",
    "technologies",
    "technology",
    "labs",
}

# Deliberately NOT suffixes: "ai" and "io". They read like noise but are part of
# the actual name for a whole generation of companies — Scale AI is not Scale,
# and stripping them collapsed "Scale AI", "Scale Labs" and "Scale" onto one
# key. That key feeds the 30-day suppression ledger, so the cost was a real new
# company being silently held back because an unrelated one shared a stem.
#
# The reverse error — the same company appearing under two spellings — is now
# caught downstream by collapse_duplicate_boards(), which matches on the
# resolved ATS board. A visible duplicate is also cheaper than an invisible
# suppression.

# Tracking params that differ between feeds carrying the same story.
_TRACKING_PARAM_PREFIXES = ("utm_", "ref", "fbclid", "gclid", "mc_cid", "mc_eid")


def normalize_title(title: str) -> str:
    """Collapse a headline to a comparable key.

    Four outlets covering one round produce four different headlines; this
    strips punctuation, case, and filler so near-identical ones match.
    """
    lowered = title.lower()
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)
    tokens = [t for t in lowered.split() if t not in {"the", "a", "an", "to", "in"}]
    return " ".join(tokens)


def normalize_company(name: str) -> str:
    """Collapse a company name to a comparable key."""
    lowered = name.lower().strip()
    lowered = re.sub(r"[^a-z0-9\s.]", " ", lowered)
    tokens = [t for t in lowered.split() if t not in _COMPANY_SUFFIXES]
    return " ".join(tokens) or lowered.strip()


def company_slug(name: str) -> str:
    """Best-guess ATS board token for a company name, e.g. 'Acme Labs' -> 'acmelabs'.

    ATS tokens are usually the company name lowercased with separators removed.
    Used only as a fallback after careers-page discovery fails.
    """
    return re.sub(r"[^a-z0-9]", "", name.lower())


def canonical_url(url: str) -> str:
    """Strip tracking params and fragments so the same article dedupes cleanly."""
    parsed = urlparse(url)
    kept = [
        pair
        for pair in parsed.query.split("&")
        if pair
        and not pair.split("=", 1)[0]
        .lower()
        .startswith(_TRACKING_PARAM_PREFIXES)
    ]
    netloc = parsed.netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((parsed.scheme, netloc, path, "", "&".join(kept), ""))


def domain_from_url(url: str) -> str | None:
    """Extract a bare domain ('acme.com') from a URL, or None if unparseable."""
    if not url:
        return None
    if "://" not in url:
        url = f"https://{url}"
    netloc = urlparse(url).netloc.lower()
    if netloc.startswith("www."):
        netloc = netloc[4:]
    netloc = netloc.split(":", 1)[0]
    return netloc or None


def truncate_words(text: str, limit: int) -> str:
    """Cut text to at most `limit` words, appending an ellipsis if cut."""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + "…"
