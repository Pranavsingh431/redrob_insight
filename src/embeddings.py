"""
src/embeddings.py — Layer 2A: Semantic Embedding Scorer

Day 2 upgrade: Multi-aspect JD embedding.

Instead of one monolithic JD embedding, we embed three focused aspect queries
and compute a weighted average similarity. This is more robust than a single
query because:

  1. A single embedding averages across all JD requirements, making it harder
     to distinguish candidates who excel at one specific dimension.
  2. Different candidates may be described using different vocabulary; multiple
     queries improve recall across linguistic variations.
  3. The weighted average lets us control which JD dimensions matter most
     (technical stack 50%, experience profile 30%, role context 20%).

All aspect queries + the full JD text are embedded at startup and cached.
No additional network calls; everything runs fully offline after first download.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from src.config import BATCH_SIZE_EMBED, EMBEDDING_MODEL, JD_ASPECT_QUERIES, JD_TEXT


class EmbeddingScorer:
    """Wraps sentence-transformers for multi-aspect JD-candidate semantic scoring."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer  # type: ignore
        self._model = SentenceTransformer(EMBEDDING_MODEL)
        # Cache: (aspect_vecs, aspect_weights) — set by embed_jd()
        self._aspect_vecs: np.ndarray | None = None   # shape: (n_aspects, D)
        self._aspect_weights: np.ndarray | None = None  # shape: (n_aspects,)

    # ── Candidate context blob ────────────────────────────────────────────────

    @staticmethod
    def build_context_blob(candidate: Dict[str, Any]) -> str:
        """
        Build the text representation of a candidate for embedding.

        Day 2 improvement: also includes certifications (if relevant) and
        slightly expanded top-skill list to help candidates with sparse
        career descriptions.

        Includes (in order of recency/relevance):
          - Headline (high signal — written by the candidate as their pitch)
          - Summary (medium signal — prose description of expertise)
          - Top-3 most recent career job descriptions (high signal)
          - Certifications (medium signal for ML certs like GCP ML Engineer)
          - Top-6 skills by proficiency (low signal — for sparse profiles)
        """
        profile = candidate.get("profile", {})
        parts: List[str] = []

        # Headline + summary
        headline = profile.get("headline", "")
        summary  = profile.get("summary", "")
        if headline:
            parts.append(headline)
        if summary:
            parts.append(summary[:500])

        # Top-3 career descriptions (most recent first, weighted by recency)
        history = candidate.get("career_history", [])
        sorted_history = sorted(
            history,
            key=lambda j: j.get("start_date", "1900-01-01"),
            reverse=True,
        )
        for i, job in enumerate(sorted_history[:3]):
            desc = job.get("description", "").strip()
            if desc:
                max_len = 400 if i == 0 else (300 if i == 1 else 200)
                parts.append(
                    f"{job.get('title', '')} at {job.get('company', '')}: {desc[:max_len]}"
                )

        # Certifications (Day 2 addition — ML certs are meaningful signals)
        certs = candidate.get("certifications") or []
        ml_cert_keywords = {
            "machine learning", "deep learning", "nlp", "tensorflow", "pytorch",
            "aws ml", "gcp ml", "azure ml", "data science", "ai", "bert",
        }
        ml_certs = [
            c["name"] for c in certs
            if any(kw in c.get("name", "").lower() for kw in ml_cert_keywords)
        ]
        if ml_certs:
            parts.append(f"Certifications: {', '.join(ml_certs[:3])}")

        # Top-6 skills (expanded from 5 — helps sparse profile candidates)
        skills = candidate.get("skills", [])
        top_skills = sorted(
            skills,
            key=lambda s: {"expert": 4, "advanced": 3, "intermediate": 2, "beginner": 1}.get(
                s.get("proficiency", "").lower(), 0
            ),
            reverse=True,
        )[:6]
        if top_skills:
            skill_names = ", ".join(s.get("name", "") for s in top_skills)
            parts.append(f"Skills: {skill_names}")

        return " | ".join(p for p in parts if p)

    # ── JD embedding (multi-aspect) ───────────────────────────────────────────

    def embed_jd(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Embed all JD aspect queries and return (aspect_vecs, aspect_weights).

        Returns:
            aspect_vecs:    np.ndarray shape (n_aspects, D), L2-normalised
            aspect_weights: np.ndarray shape (n_aspects,), sums to 1.0
        """
        if self._aspect_vecs is None:
            queries  = list(JD_ASPECT_QUERIES.keys())
            weights  = list(JD_ASPECT_QUERIES.values())

            self._aspect_vecs = self._model.encode(
                queries,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            )  # (n_aspects, D)

            w = np.array(weights, dtype=np.float32)
            self._aspect_weights = w / w.sum()  # normalise to sum=1

        return self._aspect_vecs, self._aspect_weights

    # ── Candidate embedding ───────────────────────────────────────────────────

    def embed_candidates(
        self,
        candidates: List[Dict[str, Any]],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Embed a list of candidates.

        Returns:
            np.ndarray of shape (len(candidates), D), L2-normalised.
        """
        blobs = [self.build_context_blob(c) for c in candidates]
        return self._model.encode(
            blobs,
            batch_size=BATCH_SIZE_EMBED,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )

    # ── Multi-aspect similarity ───────────────────────────────────────────────

    def multi_aspect_similarities(
        self,
        aspect_vecs: np.ndarray,
        aspect_weights: np.ndarray,
        candidate_vecs: np.ndarray,
    ) -> np.ndarray:
        """
        Compute weighted-average cosine similarities across all JD aspects.

        All vectors are L2-normalised, so cosine similarity = dot product.

        Args:
            aspect_vecs:     (n_aspects, D) — JD aspect embeddings
            aspect_weights:  (n_aspects,)   — weights summing to 1
            candidate_vecs:  (N, D)         — candidate embeddings

        Returns:
            np.ndarray of shape (N,) — weighted average similarity in [0, 1]
        """
        # candidate_vecs @ aspect_vecs.T → (N, n_aspects)
        per_aspect = candidate_vecs @ aspect_vecs.T

        # Clip negatives to 0 (meaningless for text similarity)
        per_aspect = np.clip(per_aspect, 0.0, 1.0)

        # Weighted average across aspects → (N,)
        return per_aspect @ aspect_weights

    # ── Legacy single-vector interface (for backward compatibility) ───────────

    @staticmethod
    def cosine_similarities(
        jd_vec: np.ndarray,
        candidate_vecs: np.ndarray,
    ) -> np.ndarray:
        """Single-vector cosine similarity. Kept for test_smoke.py compatibility."""
        return np.clip(candidate_vecs @ jd_vec, 0.0, 1.0)
