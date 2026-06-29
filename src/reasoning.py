"""
src/reasoning.py — Reasoning String Generator

Generates the 'reasoning' column for the submission CSV.

From the submission spec, Stage 4 evaluation criteria:
  ✅ GOOD:  specific, references actual profile data, honest
  ❌ BAD:   empty, identical across rows, templated (just inserts name),
            mentions skills NOT in the profile (hallucination),
            contradicts the rank

Our approach: build the reasoning string entirely from the candidate's
actual data fields, never from assumptions. Every fact stated in the
reasoning string is directly readable from the candidate object and
the scoring metadata. Zero hallucination risk.

Format (pipe-separated for readability):
  {title} | {years}yr | {verified_skills} | {career_signal} | {behavioral_summary}

Example output:
  "ML Engineer | 6.4yr | embeddings+FAISS+hybrid-search (career-verified) |
   product co (Swiggy, 501-1000) | active 8d ago, RR:0.87, notice:30d, open-to-work"
"""

from typing import Any, Dict, List


def generate_reasoning(
    candidate: Dict[str, Any],
    verified_skills: List[str],
    is_product_company: bool,
    behavioral_summary: Dict[str, Any],
    loc_label: str = "",
) -> str:
    """
    Build a specific, honest, non-templated reasoning string.

    Stage 4 manual review explicitly rewards: specific facts, JD connection,
    honest concerns, no hallucination, variation across rows, rank-consistent
    tone. We append a short 'concerns' clause when availability / notice /
    engagement signals warrant it — this keeps the reasoning honest rather
    than uniformly glowing.
    """
    profile = candidate.get("profile", {})
    sigs    = candidate.get("redrob_signals", {})

    title   = profile.get("current_title", "Unknown Title")
    years   = profile.get("years_of_experience", 0)
    company = profile.get("current_company", "")
    co_size = profile.get("current_company_size", "")

    days_ago    = behavioral_summary.get("days_since_active", 999)
    rr          = behavioral_summary.get("recruiter_response_rate", 0.0)
    notice      = behavioral_summary.get("notice_period_days", 90)
    open_flag   = behavioral_summary.get("open_to_work", False)
    github      = behavioral_summary.get("github_activity_score", -1)

    # ── Part 1: Core identity ─────────────────────────────────────────────────
    parts: List[str] = [f"{title} | {years:.1f}yr exp"]

    # ── Part 2: Verified skills (anti-keyword-stuffer evidence) ──────────────
    if verified_skills:
        skill_str = "+".join(s for s in verified_skills[:4])
        parts.append(f"career-verified skills: {skill_str}")
    else:
        # Still mention their skills even if not career-verified
        skills = candidate.get("skills", [])
        top_sk = [s["name"] for s in skills if s.get("proficiency") in ("expert", "advanced")][:3]
        if top_sk:
            parts.append(f"skills (profile only): {'+'.join(top_sk)}")

    # ── Part 3: Career quality signal ────────────────────────────────────────
    if is_product_company:
        if company and co_size:
            parts.append(f"product-co: {company} ({co_size})")
        else:
            parts.append("product-co experience")
    else:
        parts.append("services/consulting background")

    # ── Part 4: Behavioral summary ────────────────────────────────────────────
    beh_parts: List[str] = []

    if days_ago < 999:
        beh_parts.append(f"active {days_ago}d ago")

    if rr >= 0.0:
        beh_parts.append(f"RR:{rr:.2f}")

    if notice < 999:
        beh_parts.append(f"notice:{notice}d")

    if open_flag:
        beh_parts.append("open-to-work")

    if github >= 0:
        beh_parts.append(f"gh:{int(github)}")

    if beh_parts:
        parts.append(" | ".join(beh_parts))

    # ── Part 5: Location signal ───────────────────────────────────────────────
    if loc_label == "tier1-city":
        loc = candidate.get("profile", {}).get("location", "")
        if loc:
            parts.append(f"loc:{loc.split(',')[0].strip()}")
    elif loc_label == "abroad-no-reloc":
        parts.append("⚠ abroad/no-reloc")

    # ── Part 6: Honest concerns (Stage 4 rewards acknowledgment of gaps) ─────
    concerns: List[str] = []
    if not open_flag:
        concerns.append("not actively open-to-work")
    if days_ago is not None and days_ago >= 90:
        concerns.append(f"inactive {days_ago}d")
    if rr is not None and rr < 0.40:
        concerns.append("low recruiter response")
    if notice is not None and notice > 60:
        concerns.append(f"long {notice}d notice")
    if years is not None and (years < 5 or years > 9):
        concerns.append(f"{years:.1f}yr outside JD 5-9 band")

    if concerns:
        parts.append("concerns: " + ", ".join(concerns[:3]))

    return "; ".join(parts) + "."
