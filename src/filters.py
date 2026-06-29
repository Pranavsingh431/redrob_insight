"""
src/filters.py — Layer 1: Hard Filter

Fast O(n) pass/fail screen over all 100K candidates.
Goal: eliminate obviously irrelevant candidates in seconds, before
any expensive operations (embeddings, detailed scoring).

Pass criteria (any ONE is sufficient):
  1. Candidate has ≥ 1 skill that matches a Tier1 or Tier2 keyword.
  2. Career history descriptions contain ≥ 2 ML/search keywords.
  3. Professional headline or summary contains ≥ 2 ML keywords.
  4. Current title matches a technical or ML title fragment.

This is intentionally permissive — we'd rather do extra work on a
borderline candidate than eliminate a good one early.
"""

from typing import Dict, Any

import re

from src.config import (
    TIER1_SKILLS,
    TIER2_SKILLS,
    CAREER_ML_KEYWORDS,
    ML_TITLE_FRAGMENTS,
    TECHNICAL_TITLE_FRAGMENTS,
    NON_TECHNICAL_TITLE_FRAGMENTS,
)


def _keyword_match(text: str, keyword: str) -> bool:
    """Word-boundary-safe containment match.

    For short single-token keywords (<=4 chars, no space) we require a
    word-boundary regex match so 'rag' does not match inside 'storage',
    'ltr' inside 'filter', 'ann' inside 'channel', 'ada' inside 'adaptation'.
    """
    if " " in keyword or len(keyword) > 4:
        return keyword in text
    try:
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    except re.error:
        return keyword in text


def _skill_name_matches_tier(name: str, tier: frozenset) -> bool:
    """Case-insensitive, word-boundary-safe match for skill names vs a tier set."""
    name_lower = name.lower()
    for kw in tier:
        if _keyword_match(name_lower, kw) or _keyword_match(kw, name_lower):
            return True
    return False


def _count_career_keywords(candidate: Dict[str, Any]) -> int:
    """Count how many ML keywords appear in all career history descriptions."""
    career_text = " ".join(
        job.get("description", "").lower()
        for job in candidate.get("career_history", [])
    )
    return sum(1 for kw in CAREER_ML_KEYWORDS if _keyword_match(career_text, kw))


def _count_profile_keywords(candidate: Dict[str, Any]) -> int:
    """Count ML keywords in headline + summary."""
    profile = candidate.get("profile", {})
    text = (
        profile.get("headline", "") + " " + profile.get("summary", "")
    ).lower()
    return sum(1 for kw in CAREER_ML_KEYWORDS if _keyword_match(text, kw))


def _title_is_technical(title: str) -> bool:
    """Return True if the current title looks like a technical role."""
    title_lower = title.lower()
    return any(frag in title_lower for frag in ML_TITLE_FRAGMENTS | TECHNICAL_TITLE_FRAGMENTS)


def _title_is_non_technical(title: str) -> bool:
    """Return True if the current title is clearly non-technical for this JD."""
    title_lower = title.lower()
    return any(frag in title_lower for frag in NON_TECHNICAL_TITLE_FRAGMENTS)


def passes_hard_filter(candidate: Dict[str, Any]) -> bool:
    """
    Return True if the candidate should proceed to detailed scoring.

    A candidate is eliminated (returns False) only if ALL of the following
    are true simultaneously:
      - No Tier1/Tier2 skill matches
      - Career history has < 2 ML keywords
      - Profile text has < 2 ML keywords
      - Current title is not technical

    We also fast-reject candidates with explicitly non-technical current
    titles AND no ML keyword evidence anywhere — these are the 'Marketing
    Manager with AI keywords' traps the JD warned about.
    """
    profile = candidate.get("profile", {})
    current_title = profile.get("current_title", "")

    # ── Fast-reject: non-technical title with zero ML evidence ───────────────
    if _title_is_non_technical(current_title):
        # Still allow if they have strong career/skill evidence
        # (career changer who held a non-tech title but has ML skills)
        career_kw = _count_career_keywords(candidate)
        if career_kw < 3:
            return False

    # ── Check 1: any Tier1 or Tier2 skill ─────────────────────────────────
    skills = candidate.get("skills", [])
    for skill in skills:
        name = skill.get("name", "")
        if _skill_name_matches_tier(name, TIER1_SKILLS):
            return True
        if _skill_name_matches_tier(name, TIER2_SKILLS):
            return True

    # ── Check 2: career history has ≥ 2 ML keywords ───────────────────────
    if _count_career_keywords(candidate) >= 2:
        return True

    # ── Check 3: headline/summary has ≥ 2 ML keywords ─────────────────────
    if _count_profile_keywords(candidate) >= 2:
        return True

    # ── Check 4: current title is explicitly technical ─────────────────────
    if _title_is_technical(current_title):
        return True

    return False
