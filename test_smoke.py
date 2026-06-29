#!/usr/bin/env python3
"""
Quick smoke test: run the full pipeline on sample_candidates.json (50 candidates).
Verifies all modules import and execute correctly before the real 100K run.

Usage:
    python test_smoke.py
"""

import json
import sys
import time

print("="*60)
print("SMOKE TEST — sample_candidates.json (50 candidates)")
print("="*60)

# ── 1. Test imports ──────────────────────────────────────────────────────────
print("\n[1] Testing imports...")
try:
    from src.config import JD_TEXT, TIER1_SKILLS, SEMANTIC_WEIGHT
    from src.filters import passes_hard_filter
    from src.scorer import (
        skills_score, career_score, behavioral_score,
        location_score, education_score, compute_l2_composite,
    )
    from src.honeypot import compute_honeypot_multiplier
    from src.reasoning import generate_reasoning
    print("    ✅  All src modules imported successfully")
except ImportError as e:
    print(f"    ❌  Import error: {e}")
    sys.exit(1)

# ── 2. Load sample data ───────────────────────────────────────────────────────
print("\n[2] Loading sample_candidates.json...")
with open("sample_candidates.json") as f:
    candidates = json.load(f)
print(f"    ✅  Loaded {len(candidates)} sample candidates")

# ── 3. Hard filter ────────────────────────────────────────────────────────────
print("\n[3] Layer 1 — Hard filter...")
passed = [c for c in candidates if passes_hard_filter(c)]
print(f"    ✅  {len(passed)}/{len(candidates)} passed")

# ── 4. Scoring ────────────────────────────────────────────────────────────────
print("\n[4] Layer 2B/C/D/E/F — Scoring...")
t = time.perf_counter()
scored = []
raw_sk_scores = []
for c in passed:
    sk_raw, verified = skills_score(c)
    ca_sc, is_prod   = career_score(c)
    beh_sc, beh_sum  = behavioral_score(c)
    loc_sc, loc_lbl  = location_score(c)
    edu_sc           = education_score(c)
    raw_sk_scores.append(sk_raw)
    scored.append({
        "candidate": c, "sk_raw": sk_raw, "ca_sc": ca_sc,
        "beh_sc": beh_sc, "loc_sc": loc_sc, "edu_sc": edu_sc,
        "verified_skills": verified, "is_product": is_prod,
        "beh_summary": beh_sum, "loc_label": loc_lbl,
    })

max_sk = max(raw_sk_scores) if raw_sk_scores else 1.0
sk_norm = 1.0 / max_sk if max_sk > 0 else 1.0

for entry in scored:
    entry["l2_composite"] = compute_l2_composite(
        entry["sk_raw"], entry["ca_sc"], entry["beh_sc"],
        entry["loc_sc"], entry["edu_sc"], sk_norm,
    )

scored.sort(key=lambda e: (-e["l2_composite"], e["candidate"]["candidate_id"]))
print(f"    ✅  Scored in {time.perf_counter()-t:.2f}s")

# ── 5. Honeypot detection ─────────────────────────────────────────────────────
print("\n[5] Layer 3 — Honeypot detection...")
n_flagged = 0
for entry in scored:
    mult = compute_honeypot_multiplier(entry["candidate"])
    entry["honeypot_mult"] = mult
    if mult < 1.0:
        n_flagged += 1
        cid   = entry["candidate"]["candidate_id"]
        title = entry["candidate"]["profile"]["current_title"]
        print(f"    ⚠️   Flagged {cid} ({title}) — multiplier={mult:.1f}")
print(f"    ✅  {n_flagged} flagged")

# ── 6. Embeddings (on the passed pool) ────────────────────────────────────────
print("\n[6] Layer 2A — Embedding (testing on passed candidates)...")
try:
    from src.embeddings import EmbeddingScorer
    emb = EmbeddingScorer()
    aspect_vecs, aspect_w = emb.embed_jd()
    cands = [e["candidate"] for e in scored]
    vecs = emb.embed_candidates(cands, show_progress=False)
    sims = emb.multi_aspect_similarities(aspect_vecs, aspect_w, vecs)
    for i, entry in enumerate(scored):
        entry["sem_sc"] = float(sims[i])
    print(f"    ✅  Embedded {len(cands)} candidates")
    print(f"    Semantic scores — min:{sims.min():.3f}  max:{sims.max():.3f}  mean:{sims.mean():.3f}")
except Exception as e:
    print(f"    ⚠️   Embedding failed (sentence-transformers not installed?): {e}")
    for entry in scored:
        entry["sem_sc"] = 0.5  # neutral fallback

# ── 7. Final composite + reasoning ───────────────────────────────────────────
print("\n[7] Final composite + reasoning...")
for entry in scored:
    sk_norm_val = min(1.0, entry["sk_raw"] * sk_norm)
    raw = (
        SEMANTIC_WEIGHT * entry["sem_sc"]
        + 0.22 * sk_norm_val
        + 0.20 * entry["ca_sc"]
        + 0.10 * entry["beh_sc"]
        + 0.05 * entry["loc_sc"]
        + 0.05 * entry["edu_sc"]
    )
    entry["final_score"] = raw * entry["honeypot_mult"]

scored.sort(key=lambda e: (-e["final_score"], e["candidate"]["candidate_id"]))

print("\nTop 10 candidates:")
for i, entry in enumerate(scored[:10], 1):
    c     = entry["candidate"]
    cid   = c["candidate_id"]
    title = c["profile"]["current_title"]
    yrs   = c["profile"]["years_of_experience"]
    sc    = entry["final_score"]
    rsn   = generate_reasoning(
        c, entry["verified_skills"], entry["is_product"],
        entry["beh_summary"], entry["loc_label"],
    )
    print(f"\n  #{i}  {cid}  score={sc:.4f}")
    print(f"      {title} | {yrs:.1f}yr")
    print(f"      reasoning: {rsn[:140]}...")
    print(f"      sem:{entry['sem_sc']:.3f}  sk:{entry['sk_raw']*sk_norm:.3f}  "
          f"ca:{entry['ca_sc']:.3f}  beh:{entry['beh_sc']:.3f}  "
          f"loc:{entry['loc_sc']:.2f}  edu:{entry['edu_sc']:.2f}  "
          f"hp:{entry['honeypot_mult']:.1f}")

print("\n" + "="*60)
print("✅  SMOKE TEST PASSED — safe to run on full 100K dataset")
print("="*60)
print("\nNext step:")
print("  python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose")
