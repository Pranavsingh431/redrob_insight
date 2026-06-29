#!/usr/bin/env python3
"""
rank.py — Intelligent Candidate Discovery & Ranking
Redrob India Runs Hackathon — Data & AI Challenge

Day 2 improvements over Day 1:
  • Multi-aspect JD embedding (3 focused queries → weighted avg semantic score)
  • Location scoring (JD-specified cities: Pune/Noida/Hyderabad/Mumbai/Delhi NCR)
  • Education tier scoring (pre-labeled tier_1–4 in dataset)
  • Career progression trajectory (seniority delta: Junior→Senior = good signal)
  • Expanded embedding pool: 2K → 4K (better edge-case coverage)
  • Rebalanced weights: semantic=38%, skills=22%, career=20%,
                        behavioral=10%, location=5%, education=5%

Produces a ranked CSV of the top-100 candidates from the 100K pool,
optimised for the competition metric:
  Score = 0.50×NDCG@10 + 0.30×NDCG@50 + 0.15×MAP + 0.05×P@10

Architecture (five layers):
  Layer 1    Hard filter            — keyword/title screen of all 100K
  Layer 2B   Skills scoring         — weighted match with anti-stuffing
  Layer 2C   Career coherence       — title×(experience+company+production+ML+progression)
  Layer 2D   Behavioral score       — recency, engagement, availability
  Layer 2E   Location score         — JD-specified city proximity
  Layer 2F   Education tier         — pre-labeled institution quality
  Layer 2A   Multi-aspect embedding — 3-query weighted semantic similarity
              (run only on top-4K from L2BCDEF to stay within time budget)
  Layer 3    Honeypot detection     — temporal, expert-without-evidence, title-mismatch
  Output     Top-100 ranked CSV     — with specific per-candidate reasoning

Usage:
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl.gz --out ./submission.csv
    python rank.py --candidates ./candidates.jsonl --out ./submission.csv --verbose
"""

import argparse
import csv
import gzip
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List

import numpy as np
from tqdm import tqdm

from src.config import (
    BEHAVIORAL_WEIGHT,
    CAREER_WEIGHT,
    EDUCATION_WEIGHT,
    EMBEDDING_TOPK,
    LOCATION_WEIGHT,
    MAX_OUTPUT,
    SEMANTIC_WEIGHT,
    SKILLS_WEIGHT,
)
from src.embeddings import EmbeddingScorer
from src.filters import passes_hard_filter
from src.honeypot import compute_honeypot_multiplier
from src.reasoning import generate_reasoning
from src.scorer import (
    behavioral_score,
    career_score,
    compute_l2_composite,
    education_score,
    location_score,
    skills_score,
)

# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def stream_jsonl(path: str) -> Iterator[Dict[str, Any]]:
    """Stream candidates one-by-one from JSONL or gzipped JSONL."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Candidates file not found: {path}")

    open_fn = gzip.open if p.suffix == ".gz" else open
    mode    = "rt" if p.suffix == ".gz" else "r"

    with open_fn(p, mode, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  [warn] Skipping malformed line {lineno}: {e}", file=sys.stderr)


def write_csv(rows: List[Dict[str, Any]], output_path: str) -> None:
    """Write the final submission CSV with exactly 100 rows."""
    fieldnames = ["candidate_id", "rank", "score", "reasoning"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n✅  Submission written: {output_path}  ({len(rows)} rows)")


def write_scores_sidecar(top100: List[Dict[str, Any]], csv_path: str) -> None:
    """Write the per-candidate sub-scores to a JSON sidecar next to the CSV.

    The submission CSV only carries (candidate_id, rank, score, reasoning) per
    the competition spec — but the Streamlit demo needs the *actual* pipeline
    sub-scores to render the dashboard truthfully (semantic, skills, career,
    behavioral, location, education + honeypot + verified skills). Emitting a
    sidecar avoids re-running the ranker inside the demo and keeps the demo's
    metrics grounded in real computation rather than heuristics.
    """
    sidecar_path = Path(csv_path).with_suffix(".scores.json")
    payload = []
    for entry in top100:
        cand = entry["candidate"]
        payload.append({
            "candidate_id":     cand["candidate_id"],
            "rank":             entry["rank"],
            "final_score":      entry["final_score"],
            "normalized_score": entry["norm_score"],
            "sub_scores": {
                "semantic":   round(float(entry["sem_sc"]), 4),
                "skills":     round(float(entry["sk_norm"]), 4),
                "career":     round(float(entry["ca_sc"]), 4),
                "behavioral": round(float(entry["beh_sc"]), 4),
                "location":   round(float(entry["loc_sc"]), 4),
                "education":  round(float(entry["edu_sc"]), 4),
            },
            "weights": {
                "semantic":   0.38, "skills": 0.22, "career": 0.20,
                "behavioral": 0.10, "location": 0.05, "education": 0.05,
            },
            "honeypot_mult":       round(float(entry["honeypot_mult"]), 2),
            "verified_skills":      entry["verified_skills"],
            "is_product_company":  entry["is_product"],
            "behavioral_summary":  entry["beh_summary"],
            "location_label":      entry["loc_label"],
        })
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"✅  Sub-scores sidecar written: {sidecar_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run(candidates_path: str, output_path: str, verbose: bool = False) -> None:
    t_start = time.perf_counter()

    # ── 0. Initialise embedding model ─────────────────────────────────────────
    print("Loading embedding model (first run downloads ~90 MB, cached after)...")
    embedder = EmbeddingScorer()
    aspect_vecs, aspect_weights = embedder.embed_jd()
    print(f"  Model ready  ({time.perf_counter() - t_start:.1f}s)")

    # ── Layer 1: Hard filter ───────────────────────────────────────────────────
    print("\n[Layer 1] Streaming & filtering 100K candidates...")
    t1 = time.perf_counter()

    passed: List[Dict[str, Any]] = []
    for candidate in tqdm(stream_jsonl(candidates_path), total=100_000,
                          desc="  Filtering", disable=not verbose, unit="cand"):
        if passes_hard_filter(candidate):
            passed.append(candidate)

    n_passed = len(passed)
    print(f"  {n_passed:,} candidates passed hard filter  ({time.perf_counter()-t1:.1f}s)")

    if n_passed == 0:
        print("ERROR: No candidates passed. Check candidates file.")
        sys.exit(1)

    # ── Layer 2B / 2C / 2D / 2E / 2F: Non-embedding scoring ─────────────────
    print("\n[Layer 2B–F] Computing skills, career, behavioral, location, education scores...")
    t2 = time.perf_counter()

    scored: List[Dict[str, Any]] = []
    raw_skill_scores: List[float] = []

    for cand in tqdm(passed, desc="  Scoring", disable=not verbose):
        sk_raw, verified_skills = skills_score(cand)
        ca_sc,  is_product      = career_score(cand)
        beh_sc, beh_summary     = behavioral_score(cand)
        loc_sc, loc_label       = location_score(cand)
        edu_sc                  = education_score(cand)

        raw_skill_scores.append(sk_raw)
        scored.append({
            "candidate":       cand,
            "sk_raw":          sk_raw,
            "ca_sc":           ca_sc,
            "beh_sc":          beh_sc,
            "loc_sc":          loc_sc,
            "edu_sc":          edu_sc,
            "verified_skills": verified_skills,
            "is_product":      is_product,
            "beh_summary":     beh_summary,
            "loc_label":       loc_label,
        })

    # Normalise skills score across the pool
    max_sk = max(raw_skill_scores) if raw_skill_scores else 1.0
    sk_norm_factor = 1.0 / max_sk if max_sk > 0 else 1.0

    for entry in scored:
        entry["l2_composite"] = compute_l2_composite(
            entry["sk_raw"], entry["ca_sc"], entry["beh_sc"],
            entry["loc_sc"], entry["edu_sc"], sk_norm_factor,
        )

    print(f"  Done  ({time.perf_counter()-t2:.1f}s)")

    # ── Select top-N for embedding ─────────────────────────────────────────────
    scored.sort(key=lambda e: (-e["l2_composite"], e["candidate"]["candidate_id"]))
    embed_pool = scored[:EMBEDDING_TOPK]

    print(f"\n[Layer 2A] Embedding top-{len(embed_pool):,} candidates (multi-aspect)...")
    t3 = time.perf_counter()

    candidate_objs = [e["candidate"] for e in embed_pool]
    cand_vecs      = embedder.embed_candidates(candidate_objs, show_progress=verbose)
    sem_scores     = embedder.multi_aspect_similarities(aspect_vecs, aspect_weights, cand_vecs)

    for i, entry in enumerate(embed_pool):
        entry["sem_sc"] = float(sem_scores[i])

    print(f"  Done  ({time.perf_counter()-t3:.1f}s)")

    # ── Layer 3: Honeypot detection ────────────────────────────────────────────
    print("\n[Layer 3] Honeypot & trap detection...")
    t4 = time.perf_counter()
    n_honeypots = 0

    for entry in embed_pool:
        mult = compute_honeypot_multiplier(entry["candidate"])
        entry["honeypot_mult"] = mult
        if mult < 1.0:
            n_honeypots += 1

    print(f"  {n_honeypots} candidates flagged (multiplier < 1.0)  ({time.perf_counter()-t4:.1f}s)")

    # ── Final composite score ──────────────────────────────────────────────────
    for entry in embed_pool:
        sk_norm = min(1.0, entry["sk_raw"] * sk_norm_factor)
        raw_composite = (
            SEMANTIC_WEIGHT    * entry["sem_sc"]
            + SKILLS_WEIGHT    * sk_norm
            + CAREER_WEIGHT    * entry["ca_sc"]
            + BEHAVIORAL_WEIGHT * entry["beh_sc"]
            + LOCATION_WEIGHT  * entry["loc_sc"]
            + EDUCATION_WEIGHT * entry["edu_sc"]
        )
        entry["final_score"] = raw_composite * entry["honeypot_mult"]

    # Sort: descending score, tie-break by candidate_id ascending
    embed_pool.sort(key=lambda e: (-e["final_score"], e["candidate"]["candidate_id"]))

    # ── Build top-100 output ───────────────────────────────────────────────────
    print("\n[Output] Building top-100 ranked list...")
    top100 = embed_pool[:MAX_OUTPUT]

    max_score   = top100[0]["final_score"]  if top100 else 1.0
    min_score   = top100[-1]["final_score"] if top100 else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    output_rows: List[Dict[str, Any]] = []
    prev_score  = float("inf")
    EPSILON     = 1e-6  # Guarantees strictly non-increasing after 6-decimal formatting

    for rank, entry in enumerate(top100, start=1):
        cand = entry["candidate"]
        cid  = cand["candidate_id"]

        norm_sc    = 0.01 + 0.99 * (entry["final_score"] - min_score) / score_range
        norm_sc    = min(norm_sc, prev_score - EPSILON)
        prev_score = norm_sc

        reasoning = generate_reasoning(
            candidate          = cand,
            verified_skills    = entry["verified_skills"],
            is_product_company = entry["is_product"],
            behavioral_summary = entry["beh_summary"],
            loc_label          = entry["loc_label"],
        )

        entry["rank"]        = rank
        entry["norm_score"]  = norm_sc
        entry["sk_norm"]     = min(1.0, entry["sk_raw"] * sk_norm_factor)

        output_rows.append({
            "candidate_id": cid,
            "rank":         rank,
            "score":        f"{norm_sc:.6f}",
            "reasoning":    reasoning,
        })

        if verbose:
            print(
                f"  #{rank:3d}  {cid}  score={norm_sc:.4f}  "
                f"sem={entry['sem_sc']:.3f}  "
                f"loc={entry['loc_sc']:.2f}  "
                f"edu={entry['edu_sc']:.2f}  "
                f"{cand['profile']['current_title']}"
            )

    assert len(output_rows) == MAX_OUTPUT, \
        f"Expected {MAX_OUTPUT} rows, got {len(output_rows)}"

    write_csv(output_rows, output_path)
    write_scores_sidecar(top100, output_path)

    elapsed = time.perf_counter() - t_start
    print(f"\n⏱  Total runtime: {elapsed:.1f}s  ({elapsed/60:.1f} min)")
    print("   Run `python validate_submission.py <your_file>.csv` to verify format.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Redrob Hackathon — Candidate Ranker (Day 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--candidates", required=True,
                   help="Path to candidates.jsonl or candidates.jsonl.gz")
    p.add_argument("--out", required=True,
                   help="Output CSV path (e.g. ./submission.csv)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-candidate scoring details and progress bars")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(candidates_path=args.candidates, output_path=args.out, verbose=args.verbose)
