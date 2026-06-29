"""
src/honeypot.py — Layer 3: Honeypot & Trap Detection

The dataset contains ~80 honeypot candidates with subtly impossible profiles.
Submissions with honeypot rate > 10% in top 100 are DISQUALIFIED.

Three classes of traps (per JD + signals doc):

  CLASS A — Temporal impossibility
    A skill's duration_months exceeds the candidate's total career duration.
    A single employer's claimed tenure exceeds the total experience.

  CLASS B — Expert-with-no-evidence
    "Expert" proficiency + 0 endorsements + 0 skill assessment score +
    no mention in career history = almost certainly fabricated.

  CLASS C — Title-skills mismatch
    Non-technical current title (Marketing Manager, HR Manager) with an
    otherwise high-ranking skill list.  This is the "keyword stuffer trap"
    the JD explicitly warns about.

Each detected class increments a flag count.
Flag count ≥ 2 → honeypot_multiplier = 0.0  (forces candidate out of top 100)
Flag count == 1 → honeypot_multiplier = 0.4  (heavy penalty, rarely top-100)
Flag count == 0 → honeypot_multiplier = 1.0  (clean, no penalty)
"""

from datetime import date, datetime
from typing import Any, Dict

from src.config import NON_TECHNICAL_TITLE_FRAGMENTS


def _parse_date(s: Any):
    if not s:
        return None
    try:
        return datetime.strptime(str(s)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None


def _months_between(d1, d2) -> int:
    """Whole months from d1 to d2 (d2 >= d1)."""
    return max(0, (d2.year - d1.year) * 12 + (d2.month - d1.month))


# ─────────────────────────────────────────────────────────────────────────────
# Class A: Temporal Impossibility
# ─────────────────────────────────────────────────────────────────────────────

def _flag_temporal_impossibility(candidate: Dict[str, Any]) -> int:
    """
    Check for impossible time relationships:
      - Any skill's duration_months > total career experience months
      - Any single job's duration_months > total experience months
      - Any job's claimed duration_months exceeds the wall-clock time
        between its start date and today (catches the JD's canonical
        honeypot: "8 years of experience at a company founded 3 years
        ago"). For past jobs, also checks duration vs start->end window.
      - Any career_history date is in the future.

    Returns 2 (not 1) because a clear temporal impossibility is a definitive
    honeypot signal — it should trigger the 0.0 multiplier directly.
    """
    years_exp    = candidate.get("profile", {}).get("years_of_experience", 0)
    total_months = int(years_exp * 12)
    today        = date.today()

    # Check individual skill durations
    for skill in candidate.get("skills", []):
        dur = skill.get("duration_months", 0)
        if dur > total_months + 6:   # +6 month buffer for rounding
            return 2  # Definitive honeypot — force 0.0 multiplier

    # Check any single job duration vs total experience AND vs calendar time
    for job in candidate.get("career_history", []):
        job_dur = job.get("duration_months", 0)
        if job_dur > total_months + 6:
            return 2

        start = _parse_date(job.get("start_date"))
        end   = _parse_date(job.get("end_date"))
        # Future-dated role is a clear fabrication
        if start and start > today:
            return 2

        if start:
            ref_end = end if end else today
            if ref_end < start:
                return 2  # end before start
            calendar_months = _months_between(start, ref_end)
            # +1 month tolerance for partial-month rounding in the dataset
            if job_dur > calendar_months + 1:
                return 2

    return 0


# ─────────────────────────────────────────────────────────────────────────────
# Class B: Expert-With-No-Evidence
# ─────────────────────────────────────────────────────────────────────────────

def _flag_expert_without_evidence(candidate: Dict[str, Any]) -> int:
    """
    Detect skills listed as 'expert' with zero supporting evidence:
      - endorsements == 0
      - no skill assessment score (or score == 0)
      - skill name not in any career history description
      - duration_months == 0  (claimed mastery but never used it — implausible)

    Threshold: >= 3 such "ghost expert" skills = 1 flag.
    Also flags if there are > 9 "expert" skills total (implausibly broad mastery).
    """
    sigs         = candidate.get("redrob_signals", {})
    assessment   = sigs.get("skill_assessment_scores", {})
    career_text  = " ".join(
        j.get("description", "").lower() for j in candidate.get("career_history", [])
    )

    ghost_experts = 0
    total_experts = 0

    for skill in candidate.get("skills", []):
        proficiency = skill.get("proficiency", "").lower()
        if proficiency != "expert":
            continue
        total_experts += 1

        name       = skill.get("name", "")
        endorses   = skill.get("endorsements", 0)
        dur        = skill.get("duration_months", 0)
        assess_sc  = assessment.get(name, None)
        in_career  = name.lower() in career_text

        no_endorsements = endorses == 0
        no_assessment   = assess_sc is None or assess_sc == 0
        no_career_ref   = not in_career
        no_duration     = dur == 0

        # Expert with zero duration is almost always fabrication, regardless
        # of the other signals — weight it heavily in the ghost count.
        if no_endorsements and no_assessment and no_career_ref:
            ghost_experts += 1
            continue
        if no_duration and no_endorsements and no_assessment:
            ghost_experts += 1

    flags = 0
    if ghost_experts >= 3:
        flags += 1
    if total_experts > 9:
        flags += 1

    return min(flags, 1)  # cap at 1 for this class


# ─────────────────────────────────────────────────────────────────────────────
# Class C: Title-Skills Mismatch (Keyword Stuffer Trap)
# ─────────────────────────────────────────────────────────────────────────────

def _flag_title_skills_mismatch(candidate: Dict[str, Any]) -> int:
    """
    Detect the 'Marketing Manager with perfect AI keywords' trap.

    A non-technical current title combined with ≥ 4 advanced/expert-level
    technical skills in the Tier-1 list = almost certainly a honeypot or
    irrelevant candidate that keyword-stuffed their profile.
    """
    from src.config import TIER1_SKILLS

    title = candidate.get("profile", {}).get("current_title", "").lower()

    is_non_technical = any(frag in title for frag in NON_TECHNICAL_TITLE_FRAGMENTS)
    if not is_non_technical:
        return 0

    # Count tier-1 skills with advanced/expert proficiency
    advanced_tier1 = 0
    for skill in candidate.get("skills", []):
        prof = skill.get("proficiency", "").lower()
        name = skill.get("name", "").lower()
        if prof in ("expert", "advanced"):
            if any(kw in name or name in kw for kw in TIER1_SKILLS):
                advanced_tier1 += 1

    # ≥ 2 advanced tier-1 AI skills + non-technical title = very suspicious.
    # (Lowered from 4: even 2 'expert' retrieval skills on a Marketing Manager
    # profile is a clear keyword-stuffer signal, per the JD's explicit warning.)
    return 1 if advanced_tier1 >= 2 else 0


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def compute_honeypot_multiplier(candidate: Dict[str, Any]) -> float:
    """
    Return a multiplier to apply to the final composite score:
      0.0  — strong honeypot signal (≥ 2 flags); forces out of top 100
      0.4  — weak honeypot signal (1 flag); heavy penalty
      1.0  — clean profile

    Design note: we use a multiplier (not an additive penalty) so that
    even a legitimately great candidate with one suspicious signal gets
    heavily demoted, while a great clean candidate is unaffected.
    """
    flags = 0
    flags += _flag_temporal_impossibility(candidate)
    flags += _flag_expert_without_evidence(candidate)
    flags += _flag_title_skills_mismatch(candidate)

    if flags >= 2:
        return 0.0
    if flags == 1:
        return 0.40
    return 1.0
