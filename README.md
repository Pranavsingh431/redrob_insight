# Redrob India Runs Hackathon — Data & AI Challenge
## Intelligent Candidate Discovery & Ranking

---

## Reproduce Command

```bash
python rank.py --candidates ./candidates.jsonl --out ./submission.csv
```

Runs in ~3.5 minutes on CPU with 8 GB RAM (well under the 5-minute limit). No network access required during ranking.

---

## Architecture

```
100K candidates (JSONL stream)
        │
        ▼
┌──────────────────────────────────────────┐
│  LAYER 1  Hard Filter                    │
│  Word-boundary-safe title/keyword screen │
│  (prevents 'rag' matching 'storage' etc) │
│  Passes: ~5-27K candidates                │
└──────────────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  LAYER 2B  Skills Score (22%)           │
│  Tier-weighted match + anti-stuffing    │
│  Endorsement × duration × assessment   │
│                                         │
│  LAYER 2C  Career Coherence (20%)      │
│  Title MULTIPLIER (caps non-tech titles)│
│  Company quality (product vs consulting)│
│  Production signals + ML depth          │
│  Career progression trajectory          │
│                                         │
│  LAYER 2D  Behavioral Score (10%)      │
│  Recency decay (60d half-life)          │
│  Response rate × time × notice          │
│  open_to_work global multiplier         │
│                                         │
│  LAYER 2E  Location Score (5%)          │
│  JD cities (Pune/Noida/Hyd/Mum/Del NCR) │
│                                         │
│  LAYER 2F  Education Tier (5%)          │
│  Pre-labeled tier_1–4 institution       │
└──────────────────────┬───────────────────┘
                       │  Top 4,000 by L2 composite
                       ▼
┌──────────────────────────────────────────┐
│  LAYER 2A  Semantic Embedding (38%)     │
│  Multi-aspect JD queries (3, weighted)  │
│  sentence-transformers all-MiniLM-L6-v2 │
│  Cosine similarity (offline, no API)     │
└──────────────────────┬───────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────┐
│  LAYER 3  Honeypot Detection            │
│  Class A: Temporal impossibility        │
│           (incl. job dur vs calendar)    │
│  Class B: Expert-without-evidence       │
│           (incl. expert w/ 0 duration)   │
│  Class C: Title-skills mismatch         │
│  2+ flags → multiplier = 0.0             │
└──────────────────────┬───────────────────┘
                       │
                       ▼
        Top-100 Ranked CSV (submission.csv)
   + submission.scores.json sub-score sidecar
   with specific, profile-grounded reasoning
```

**Final composite weights (sum = 1.0):**
`0.38×semantic + 0.22×skills + 0.20×career + 0.10×behavioral + 0.05×location + 0.05×education`

---

## Design Decisions

### Why multi-layer instead of pure embedding similarity?

The JD explicitly warns: *"The right answer to this JD is not 'find candidates whose skills section contains the most AI keywords.'"* A pure cosine similarity on skills text would fall directly into this trap. Our system cross-validates every skill claim against career history descriptions before counting it as evidence, and uses the current title as a hard **multiplier** (non-technical titles cap the career score at 12%) so the "Marketing Manager with perfect AI keywords" trap cannot rank.

### Why word-boundary-safe keyword matching?

Plain substring matching makes short tokens like `rag`, `ltr`, `ann`, `ada` false-match inside unrelated words (`rag` ⊂ `storage`/`paragraph`, `ltr` ⊂ `filter`, `ann` ⊂ `channel`). That bug silently verified keyword-stuffer skills and inflated their scores. We use word-boundary regex for any single-token keyword ≤ 4 chars.

### Why sentence-transformers over keyword BM25?

A candidate who "built a retrieval pipeline using approximate nearest neighbor search" is semantically equivalent to "implemented FAISS-based vector search" — keyword matching misses this. Sentence-transformers captures it. The model is downloaded once (~90 MB) and then runs fully offline. We use **multi-aspect** embedding: 3 focused JD queries (technical core 50%, experience profile 30%, role context 20%) averaged, which is more robust than a single monolithic JD embedding.

### Why embed only the top-4K, not all 100K?

Embedding 100K candidates would take ~20+ minutes on CPU. Instead, we use the fast (no-embedding) Layer 2B–F composite to identify the top 4K most promising candidates, then embed only those. This keeps total runtime under 4 minutes while losing virtually no quality — anyone who would rank in the top 100 is almost certainly in the top 4K by the grounded signals.

### Why behavioral signals as a multiplier?

The JD says: *"A perfect-on-paper candidate who hasn't logged in for 6 months and has a 5% response rate is, for hiring purposes, not actually available."* We treat `open_to_work_flag` as a 1.3× (open) / 0.65× (not open) global multiplier on behavioral score. Inactivity decays with an exponential half-life of 60 days.

### Anti-keyword-stuffing

For each skill claimed, we check if any related keyword appears (word-boundary safe) in the candidate's career history descriptions. Skills that appear only in the skills section (never in any job description) receive a 0.65× penalty. This is the primary defense against the "keyword stuffer trap" the JD warns about.

### Honeypot detection beyond duration checks

Beyond the obvious "skill duration > total experience" check, we also verify each job's claimed `duration_months` against the wall-clock time between its `start_date` and today (or `end_date`) — catching the JD's canonical honeypot "8 years of experience at a company founded 3 years ago." We also flag "expert" proficiency with `duration_months == 0` as fabrication evidence.

### Honest reasoning

The reasoning column is built entirely from the candidate's actual data fields. Where availability, notice, or experience signals are weak, the reasoning appends a `concerns:` clause (e.g. `concerns: long 120d notice, not actively open-to-work`) — Stage-4 review rewards honest acknowledgment of gaps over uniformly glowing text.

---

## File Structure

```
├── rank.py                     # Main entry point (the submission script)
├── src/
│   ├── config.py               # All constants, weights, skill taxonomies, JD text
│   ├── filters.py              # Layer 1: word-boundary-safe keyword/title pre-screen
│   ├── scorer.py               # Layers 2B/2C/2D/2E/2F: skills, career, behavioral, location, education
│   ├── embeddings.py           # Layer 2A: multi-aspect sentence-transformers wrapper
│   ├── honeypot.py             # Layer 3: impossible-profile + keyword-stuffer detection
│   └── reasoning.py            # Per-candidate, profile-grounded reasoning string generator
├── streamlit_demo.py           # Interactive demo (uses OpenRouter, network OK)
├── requirements.txt            # rank.py dependencies (CPU only, offline)
├── requirements-demo.txt       # streamlit_demo.py dependencies
├── submission.csv              # Final submission (100 rows)
├── submission.scores.json      # Real per-candidate sub-scores sidecar (drives the demo UI)
├── submission_metadata.yaml    # Competition metadata
└── validate_submission.py      # Official format validator (provided by org)
```

---

## Setup

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the ranker (downloads embedding model ~90MB on first run)
python rank.py --candidates ./candidates.jsonl --out ./submission.csv

# 4. Validate output format
python validate_submission.py submission.csv
```

### For the Streamlit demo:

```bash
pip install -r requirements-demo.txt
OPENROUTER_API_KEY=your_key streamlit run streamlit_demo.py
```

---

## Compute Environment

- Platform: MacBook (Apple Silicon / Intel)
- RAM: 8 GB
- CPU only (no GPU)
- No network during ranking
- Python 3.11+

---

## AI Tools Declaration

- **Antigravity (Claude)**: Architecture discussion, code review
- **No candidate data was fed to any external LLM**
- Ranking is 100% local computation

---

## Scoring Metric

```
Final = 0.50 × NDCG@10 + 0.30 × NDCG@50 + 0.15 × MAP + 0.05 × P@10
```

Our system prioritises NDCG@10 (highest weight) by being conservative about the top 10 — only candidates with strong signals across all four dimensions (semantic fit, skills, career coherence, behavioral availability) rank in the top 10.
