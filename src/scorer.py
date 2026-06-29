"""
src/scorer.py — Layer 2: Multi-Signal Scoring Engine

Six independent sub-scores, each normalised to [0, 1]:

  2B  skills_score()            — weighted skill match with anti-stuffing penalty
  2C  career_score()            — coherence, company quality, production signals,
                                  + career progression trajectory (Day 2)
  2D  behavioral_score()        — availability, recency, engagement signals
  2E  location_score()          — JD-specified city proximity (Day 2)
  2F  education_score()         — pre-labeled institution tier (Day 2)

Layer 2A (semantic embedding) lives in embeddings.py and is computed
separately after scores from 2B/2C/2D/2E/2F are used to narrow the field.

Final composite (weights from config.py):
  composite = 0.38×semantic + 0.22×skills + 0.20×career
            + 0.10×behavioral + 0.05×location + 0.05×education
"""

import math
import re
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

from src.config import (
    CAREER_ML_KEYWORDS,
    COMPANY_SIZE_SCORE,
    CONSULTING_COMPANIES,
    EDUCATION_TIER_SCORES,
    LOCATION_TIER1,
    LOCATION_TIER2,
    ML_TITLE_FRAGMENTS,
    NON_TECHNICAL_TITLE_FRAGMENTS,
    PRODUCTION_SIGNALS,
    PROFICIENCY_SCORES,
    SENIORITY_LEVELS,
    TECHNICAL_TITLE_FRAGMENTS,
    TIER1_SKILLS,
    TIER2_SKILLS,
    TIER3_SKILLS,
    TIER_WEIGHTS,
    NOTICE_IDEAL_DAYS,
    NOTICE_MAX_DAYS,
    RECENCY_HALF_LIFE_DAYS,
    RESPONSE_TIME_IDEAL_H,
    RESPONSE_TIME_MAX_H,
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _today() -> date:
    return date.today()


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    try:
        return datetime.strptime(s[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _skill_tier(name: str) -> int:
    """Return the tier (1/2/3/0) for a skill name using word-boundary matching.

    Uses regex word boundaries instead of plain substring matching so that
    short tokens like 'rag', 'ann', 'ltr', 'ada', 'e5', 'gpt' don't
    false-match inside unrelated words ('storage' contains 'rag', 'filter'
    contains 'ltr', 'channel' contains 'ann', 'adaptation' contains 'ada').
    Multi-word tier keywords (e.g. 'vector search', 'learning to rank') use
    bounded substring matching since spaces already imply word separation.
    """
    n = name.lower()
    for kw in TIER1_SKILLS:
        if _keyword_match(n, kw):
            return 1
    for kw in TIER2_SKILLS:
        if _keyword_match(n, kw):
            return 2
    for kw in TIER3_SKILLS:
        if _keyword_match(n, kw):
            return 3
    return 0


def _keyword_match(text: str, keyword: str) -> bool:
    """Match a keyword against text safely.

    For short single-token keywords (<=4 chars, no space) we require a
    word-boundary regex match to avoid substring false positives
    ('rag' inside 'storage', 'ann' inside 'channel', 'ltr' inside 'filter').
    Multi-word or longer keywords use plain substring containment.
    """
    if " " in keyword or len(keyword) > 4:
        return keyword in text or text in keyword
    try:
        return re.search(rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])", text) is not None
    except re.error:
        return keyword in text


def _career_text(candidate: Dict[str, Any]) -> str:
    """Concatenated lower-case text of all career history descriptions."""
    return " ".join(
        job.get("description", "").lower()
        for job in candidate.get("career_history", [])
    )


def _is_skill_verified_in_career(skill_name: str, career_text_lower: str) -> bool:
    """
    Return True if the skill appears to be genuinely used, not just listed.
    Checks the skill name itself (with word boundaries to avoid 'rag' matching
    'storage') and common aliases against career text.
    """
    name = skill_name.lower()

    # Direct name match — word-boundary safe
    if _keyword_match(career_text_lower, name):
        return True

    # Alias / keyword match for common skills
    aliases = {
        "sentence-transformers": ["sentence transformer", "sbert", "embedding model"],
        "faiss":                  ["vector index", "similarity search", "ann index"],
        "elasticsearch":          ["elastic search", "es cluster", "kibana"],
        "bm25":                   ["bm 25", "okapi", "keyword search", "tf-idf"],
        "rag":                    ["retrieval augmented", "retrieval-augmented"],
        "ndcg":                   ["ranking metric", "relevance metric", "offline eval"],
        "lora":                   ["low-rank adaptation", "fine-tun", "peft"],
        "pytorch":                ["torch.", "nn.module", "dataloader"],
    }
    for canonical, alias_list in aliases.items():
        if canonical in name:
            if any(a in career_text_lower for a in alias_list):
                return True

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2B — Skills Score
# ─────────────────────────────────────────────────────────────────────────────

def skills_score(candidate: Dict[str, Any]) -> Tuple[float, List[str]]:
    """
    Compute a skills match score weighted by:
      - Skill tier (1 = must-have, 2 = strong, 3 = nice-to-have)
      - Proficiency level
      - Endorsements (log-scaled, max 1.5×)
      - Duration months (more = more trustworthy, max 1.25×)
      - Assessment score if available (max 1.3×)
      - Penalty if skill not mentioned anywhere in career text (keyword stuffing)

    Returns:
      (raw_score, verified_skill_names)
      raw_score is NOT yet normalised to [0,1]; normalisation happens in rank.py
      verified_skill_names: list of tier-1/2 skills confirmed in career history
    """
    career = _career_text(candidate)
    skills = candidate.get("skills", [])
    assessment_map = candidate.get("redrob_signals", {}).get("skill_assessment_scores", {})

    raw = 0.0
    verified_skills: List[str] = []

    for skill in skills:
        name  = skill.get("name", "")
        tier  = _skill_tier(name)
        if tier == 0:
            continue  # Not relevant to this JD

        prof   = PROFICIENCY_SCORES.get(skill.get("proficiency", "").lower(), 0.28)
        endorse = skill.get("endorsements", 0)
        dur_mo  = skill.get("duration_months", 0)

        # Endorsement multiplier: log-scaled, cap at 1.50×
        endorse_mult = 1.0 + 0.10 * min(5.0, math.log1p(max(0, endorse)))

        # Duration trust: longer use = more credible, cap at 1.25×
        dur_trust = min(1.25, 1.0 + 0.005 * dur_mo)

        # Assessment score bonus: 0-100 → 1.0–1.30×
        assess = assessment_map.get(name, None)
        assess_mult = 1.0 + 0.003 * float(assess) if assess is not None else 1.0

        # Anti-keyword-stuffing: verify skill appears in career history
        verified = _is_skill_verified_in_career(name, career)
        stuff_penalty = 1.0 if verified else 0.65

        contrib = TIER_WEIGHTS[tier] * prof * endorse_mult * dur_trust * assess_mult * stuff_penalty
        raw += contrib

        if verified and tier in (1, 2):
            verified_skills.append(name)

    return raw, verified_skills[:6]  # return at most 6 for reasoning string


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2C — Career Coherence Score
# ─────────────────────────────────────────────────────────────────────────────

def _title_relevance(title: str) -> float:
    t = title.lower()
    if any(f in t for f in ML_TITLE_FRAGMENTS):
        return 1.0
    if any(f in t for f in TECHNICAL_TITLE_FRAGMENTS):
        return 0.70
    if any(f in t for f in NON_TECHNICAL_TITLE_FRAGMENTS):
        return 0.05
    return 0.45  # unknown / ambiguous title


def _experience_relevance(years: float) -> float:
    """Score based on how well years_of_experience matches the JD's 5-9 range."""
    if 5 <= years <= 9:
        return 1.0
    elif 4 <= years < 5:
        return 0.82
    elif 9 < years <= 12:
        return 0.78
    elif 3 <= years < 4:
        return 0.55
    elif 12 < years <= 15:
        return 0.65
    elif years < 3:
        return 0.25
    else:
        return 0.50  # 15+ years: over-qualified risk


def _company_quality_score(career_history: List[Dict]) -> Tuple[float, bool]:
    """
    Score the quality of company experience.
    Returns (score 0-1, is_any_product_company).

    Per JD: "only worked at consulting firms in entire career" = disqualifier.
    Mixed history (some consulting + some product) = fine.
    """
    if not career_history:
        return 0.3, False

    total_months    = 0
    consulting_mos  = 0
    product_mos     = 0
    quality_sum     = 0.0
    any_product     = False

    for job in career_history:
        dur = job.get("duration_months", 0)
        if dur == 0:
            continue

        company = job.get("company", "").lower()
        size    = job.get("company_size", "1-10")

        is_consulting = any(c in company for c in CONSULTING_COMPANIES)
        size_score    = COMPANY_SIZE_SCORE.get(size, 0.6)

        # Consulting companies at 10001+ are definitely consulting giants
        if is_consulting:
            consulting_mos += dur
            quality_sum += dur * size_score * 0.45  # consulting penalty
        else:
            product_mos += dur
            any_product  = True
            quality_sum += dur * size_score

        total_months += dur

    if total_months == 0:
        return 0.3, False

    # Penalise if entire career is consulting
    consulting_ratio = consulting_mos / total_months
    consulting_penalty = 1.0 - 0.6 * consulting_ratio  # all-consulting → 0.4×

    avg_quality = quality_sum / total_months
    return min(1.0, avg_quality * consulting_penalty), any_product


def _production_signal_score(career_history: List[Dict]) -> float:
    """
    Score how much of the career history demonstrates production-level work
    (deployed systems, real users, at scale) vs research/academic work.
    """
    career_text = " ".join(
        job.get("description", "").lower() for job in career_history
    )
    hits = sum(1 for sig in PRODUCTION_SIGNALS if sig in career_text)
    # 3+ signals = confident production background
    return min(1.0, hits / 3.0)


def _ml_career_depth(career_history: List[Dict]) -> float:
    """
    Score how much ML/search work appears across career history.
    """
    career_text = " ".join(
        job.get("description", "").lower() for job in career_history
    )
    hits = sum(1 for kw in CAREER_ML_KEYWORDS if kw in career_text)
    # 8+ unique ML keywords in career = strong ML background
    return min(1.0, hits / 8.0)


def _title_multiplier(title: str) -> float:
    """
    Return a multiplier applied to the ENTIRE career score based on title.

    Using title as a multiplier (not just a weighted component) ensures that
    a non-technical title fundamentally limits how high a candidate can score
    on career coherence, regardless of their company quality or depth signals.

    This directly addresses the 'Marketing Manager with perfect AI keywords'
    trap described in the JD.

    Calibrated to the JD's Tier-5 hint: a candidate who does NOT carry an
    explicit ML title but who actually shipped recommendation / ranking /
    search systems at a product company is still a fit. So:
      • ML-title           → 1.00  (ideal)
      • Technical title     → 0.78  (backend/software/data engineer who
                                     may have shipped the system; kept high
                                     so a Tier-5 product engineer can still
                                     rank in the top 100)
      • Ambiguous / unknown → 0.55  (mildly capped, not killed)
      • Non-technical title → 0.12  (hard cap — keyword stuffer trap)
    """
    t = title.lower()
    if any(f in t for f in ML_TITLE_FRAGMENTS):
        return 1.00   # Ideal: ML Engineer, Data Scientist, etc.
    if any(f in t for f in TECHNICAL_TITLE_FRAGMENTS):
        return 0.78   # Technical but not explicitly ML
    if any(f in t for f in NON_TECHNICAL_TITLE_FRAGMENTS):
        return 0.12   # Non-technical: hard cap on career score
    return 0.55       # Ambiguous / unknown title


def career_score(candidate: Dict[str, Any]) -> Tuple[float, bool]:
    """
    Compute career coherence score (0–1).
    Returns (score, is_any_product_company) for reasoning generation.

    Architecture change vs naive approach:
      Title is used as a MULTIPLIER on the base score, not as one weighted
      component. This ensures a Project Manager or Marketing Manager cannot
      score > 0.12 × base_max on career coherence, regardless of company
      quality or ML keyword density in their descriptions.

    Base score weights (applied before title multiplier):
      30%  experience years fit (5-9yr ideal)
      40%  company quality (product vs consulting history)
      15%  production deployment signals in descriptions
      15%  ML/search depth in career descriptions
    """
    profile  = candidate.get("profile", {})
    history  = candidate.get("career_history", [])

    exp_sc       = _experience_relevance(profile.get("years_of_experience", 0))
    company_sc, any_product = _company_quality_score(history)
    prod_sc      = _production_signal_score(history)
    ml_depth_sc  = _ml_career_depth(history)
    progress_sc  = _career_progression_score(history)  # Day 2

    # Base score weights (applied before title multiplier):
    #   25% experience years fit (5-9yr ideal per JD)
    #   35% company quality (product vs consulting history)
    #   15% production deployment signals
    #   15% ML/search depth in career descriptions
    #   10% career progression trajectory (Day 2)
    base_score = (
        0.25 * exp_sc
        + 0.35 * company_sc
        + 0.15 * prod_sc
        + 0.15 * ml_depth_sc
        + 0.10 * progress_sc
    )

    # Title multiplier: non-technical title = hard cap at 12% of base
    title_mult = _title_multiplier(profile.get("current_title", ""))
    score = base_score * title_mult

    return min(1.0, score), any_product


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2D — Behavioral / Availability Score
# ─────────────────────────────────────────────────────────────────────────────

def behavioral_score(candidate: Dict[str, Any]) -> Tuple[float, Dict]:
    """
    Compute behavioral availability score (0–1) from Redrob signals.

    Per the JD: "a perfect-on-paper candidate who hasn't logged in for 6 months
    and has a 5% response rate is, for hiring purposes, not actually available."

    The open_to_work_flag acts as a global multiplier (1.3× if true, 0.65× if false)
    applied to the entire score, reflecting its dominant importance.

    Returns (score, signal_summary_dict) for reasoning generation.
    """
    sigs = candidate.get("redrob_signals", {})
    today = _today()

    # ── Recency ──────────────────────────────────────────────────────────────
    last_active = _parse_date(sigs.get("last_active_date"))
    if last_active:
        days_ago = max(0, (today - last_active).days)
        # Exponential decay: halves every RECENCY_HALF_LIFE_DAYS days
        recency_sc = math.exp(-math.log(2) * days_ago / RECENCY_HALF_LIFE_DAYS)
    else:
        recency_sc = 0.20
        days_ago = 999

    # ── Response quality ─────────────────────────────────────────────────────
    rr = float(sigs.get("recruiter_response_rate", 0.5))
    avg_resp_h = float(sigs.get("avg_response_time_hours", 48))
    # Normalise response time: ideal <24h, full penalty at 96h+
    resp_time_sc = max(0.0, 1.0 - (avg_resp_h - RESPONSE_TIME_IDEAL_H) /
                       (RESPONSE_TIME_MAX_H - RESPONSE_TIME_IDEAL_H))
    resp_time_sc = max(0.0, min(1.0, resp_time_sc))
    response_sc = 0.6 * rr + 0.4 * resp_time_sc

    # ── Interview quality ─────────────────────────────────────────────────────
    interview_sc = float(sigs.get("interview_completion_rate", 0.5))
    offer_acc    = sigs.get("offer_acceptance_rate", -1)
    if offer_acc == -1:
        offer_sc = 0.5   # no prior offers → neutral
    else:
        offer_sc = float(offer_acc)

    # ── Notice period ─────────────────────────────────────────────────────────
    notice = int(sigs.get("notice_period_days", 60))
    if notice <= NOTICE_IDEAL_DAYS:
        notice_sc = 1.0
    elif notice >= NOTICE_MAX_DAYS:
        notice_sc = 0.20
    else:
        notice_sc = 1.0 - 0.80 * (notice - NOTICE_IDEAL_DAYS) / (NOTICE_MAX_DAYS - NOTICE_IDEAL_DAYS)

    # ── GitHub activity (strong signal for engineering roles) ─────────────────
    gh = sigs.get("github_activity_score", -1)
    github_sc = max(0.0, float(gh)) / 100.0 if gh != -1 else 0.25

    # ── Profile completeness ───────────────────────────────────────────────────
    completeness_sc = float(sigs.get("profile_completeness_score", 50)) / 100.0

    # ── Verification bonus ────────────────────────────────────────────────────
    verify_sc = (
        0.5 * int(sigs.get("verified_email", False))
        + 0.3 * int(sigs.get("verified_phone", False))
        + 0.2 * int(sigs.get("linkedin_connected", False))
    )

    # ── Composite (pre-multiplier) ────────────────────────────────────────────
    composite = (
        0.28 * recency_sc
        + 0.22 * response_sc
        + 0.15 * interview_sc
        + 0.12 * notice_sc
        + 0.12 * github_sc
        + 0.06 * completeness_sc
        + 0.05 * verify_sc
    )

    # ── open_to_work_flag global multiplier ───────────────────────────────────
    open_to_work = bool(sigs.get("open_to_work_flag", False))
    multiplier   = 1.30 if open_to_work else 0.65
    final_sc     = min(1.0, composite * multiplier)

    summary = {
        "days_since_active":       days_ago,
        "recruiter_response_rate": rr,
        "notice_period_days":      notice,
        "open_to_work":            open_to_work,
        "github_activity_score":   gh,
    }
    return final_sc, summary


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2E — Location Score  (Day 2)
# ─────────────────────────────────────────────────────────────────────────────

def location_score(candidate: Dict[str, Any]) -> Tuple[float, str]:
    """
    Score how well the candidate's location matches the JD's city requirements.

    JD: 'located in or willing to relocate to Pune, Noida, Hyderabad, Mumbai,
    or Delhi NCR'

    Returns:
        (score 0–1, location_label) for reasoning generation.
    """
    profile = candidate.get("profile", {})
    sigs    = candidate.get("redrob_signals", {})

    location        = profile.get("location", "").lower()
    country         = profile.get("country", "").lower()
    willing_relocate = bool(sigs.get("willing_to_relocate", False))

    # Check tier-1 cities (JD-specified + Bengaluru)
    if any(city in location for city in LOCATION_TIER1):
        return 0.95, "tier1-city"

    # Check tier-2 cities (other Indian metros)
    if any(city in location for city in LOCATION_TIER2):
        return 0.75, "indian-metro"

    # India, not in known metro
    if "india" in country or "india" in location:
        return 0.60, "india-other"

    # Outside India
    if willing_relocate:
        return 0.45, "abroad-open-reloc"
    else:
        return 0.10, "abroad-no-reloc"


# ─────────────────────────────────────────────────────────────────────────────
# Layer 2F — Education Score  (Day 2)
# ─────────────────────────────────────────────────────────────────────────────

def education_score(candidate: Dict[str, Any]) -> float:
    """
    Score based on the highest education tier in the candidate's profile.
    The dataset pre-labels institutions as tier_1 through tier_4.

    We take the BEST (highest) tier across all education entries — a candidate
    who has a tier-1 degree plus a tier-4 degree is scored on their tier-1.
    Returns 0.35 as a neutral default when no education data is present.
    """
    education = candidate.get("education", []) or []
    if not education:
        return 0.35  # neutral fallback

    best_score = 0.0
    for edu in education:
        tier  = edu.get("tier", "tier_4")
        score = EDUCATION_TIER_SCORES.get(tier, 0.25)
        best_score = max(best_score, score)

    return best_score


# ─────────────────────────────────────────────────────────────────────────────
# Career Progression Score  (Day 2 — integrated into career_score)
# ─────────────────────────────────────────────────────────────────────────────

def _get_seniority(title: str) -> int:
    """Return numeric seniority level for a job title (0=intern, 5=exec)."""
    t = title.lower()
    for fragment, level in SENIORITY_LEVELS.items():
        if fragment in t:
            return level
    return 2  # default: mid-level


def _career_progression_score(career_history: List[Dict]) -> float:
    """
    Score upward career trajectory.

    Compares the seniority of the most recent role vs the earliest role.
    A candidate who progressed Junior → Senior → Staff scores higher than
    one who has been 'Engineer' at one place for 8 years.

    Returns a score in [0, 1]:
      0.5  — flat career (no change in seniority)
      >0.5 — upward trajectory (positive progression)
      <0.5 — downward trajectory (unusual, mild penalty)
    """
    if len(career_history) < 2:
        return 0.5  # not enough data — neutral

    sorted_hist = sorted(
        career_history,
        key=lambda j: j.get("start_date", "1900-01-01"),
    )

    earliest_seniority = _get_seniority(sorted_hist[0].get("title", ""))
    latest_seniority   = _get_seniority(sorted_hist[-1].get("title", ""))

    delta = latest_seniority - earliest_seniority
    # Map delta to [0, 1]: delta of +2 or more = excellent (1.0)
    # delta of 0 = neutral (0.5), delta of -1 = mild penalty (0.3)
    if delta >= 2:
        return 1.0
    elif delta == 1:
        return 0.78
    elif delta == 0:
        return 0.50
    elif delta == -1:
        return 0.30
    else:
        return 0.10  # unusual downward shift


# ─────────────────────────────────────────────────────────────────────────────
# Composite score (L2 only, before semantic embedding)
# ─────────────────────────────────────────────────────────────────────────────

def compute_l2_composite(
    sk_score: float,
    ca_score: float,
    beh_score: float,
    loc_score: float,
    edu_score: float,
    sk_norm_factor: float = 1.0,
) -> float:
    """
    Compute a partial composite using Layer 2B/C/D/E/F scores (no semantic).
    Used to select the top-N candidates for expensive embedding.

    sk_norm_factor normalises skills_score to [0, 1] (set after seeing the pool).
    Weights rescaled to sum to 1 without the 0.38 semantic component.
    Non-semantic weights: skills=0.22, career=0.20, beh=0.10, loc=0.05, edu=0.05
    Sum = 0.62  → rescale by /0.62
    """
    sk_norm = min(1.0, sk_score * sk_norm_factor)
    return (
        (0.22 / 0.62) * sk_norm
        + (0.20 / 0.62) * ca_score
        + (0.10 / 0.62) * beh_score
        + (0.05 / 0.62) * loc_score
        + (0.05 / 0.62) * edu_score
    )
