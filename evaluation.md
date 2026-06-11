# Stress-Test Evaluation — HR Policies RAG (Hybrid + Re-rank)

The assignment write-up: 15 questions across four difficulty buckets, with where
retrieval **succeeds**, where it **fails**, and **why**. **These are actual results
from running `hr_policy_rag.ipynb` end-to-end** with the final two-stage retrieval.

**Configuration under test:** chunk 500 / overlap 100 · embeddings **Nebius
`Qwen/Qwen3-Embedding-8B` (4096-dim)** · Pinecone cosine · **hybrid retrieval
(BM25 + dense via `EnsembleRetriever`/RRF) → FlashRank cross-encoder re-rank → top-4** ·
generation Claude `claude-opus-4-8` with an *answer-only-from-context* prompt.

**Headline result:** **14 / 15 successes, 1 honest partial (Q11)**, and **all 3
unanswerable questions correctly declined** with zero hallucination. Re-ranking
improved rank-1 precision on the factual questions and fixed the Q11 *retrieval* gap;
what remains on Q11 is a genuine knowledge-base gap, not a retrieval one.

---

## Results table

| # | Bucket | Question | Outcome | What happened / why |
|---|--------|----------|---------|---------------------|
| 1 | Direct factual | PTO days accrued per year? | ✅ | "20 days / 1.67 per month." Re-ranking put `PTO Accrual` at **rank 1** (it was rank 2 under dense-only). |
| 2 | Direct factual | 401(k) match? | ✅ | "100% up to 4%, immediately vested." `benefits-03` rank 1. |
| 3 | Direct factual | Weeks of paid parental leave (primary)? | ✅ | "16 weeks." `leave-04` rank 1. |
| 4 | Direct factual | How often are employees paid? | ✅ | "Bi-weekly, 26 periods." `benefits-06` rank 1. |
| 5 | Direct factual | Home office equipment stipend? | ✅ | "$1,000 + $50/mo internet," with receipt rule. `remote-02` rank 1. |
| 6 | Direct factual | How do I report harassment? | ✅ | Manager / HR / hotline 1-800-555-0142. `conduct-03` rank 1. |
| 7 | Ambiguous | What's the policy on time off? | ✅ | Synthesized PTO + carryover/request + bereavement + parental from 4 leave chunks; offered to narrow down. |
| 8 | Ambiguous | Can I work from home? | ✅ | Role designation + 90-day eligibility + manager approval + hybrid 3-day rule. |
| 9 | Ambiguous | What's the deadline? | ✅ | Recognized the ambiguity, **listed the candidate deadlines (benefits 30-day, PTO carryover Mar 31) and asked which was meant** — didn't guess. |
| 10 | Multi-document | New remote hire — setup + remote rules? | ✅ | Combined **Onboarding** (`onboard-02`, `onboard-01`, `onboard-03`) **and Remote Work** (`faq-02`) into setup + rules + first-week items. |
| 11 | Multi-document | Parental leave → health insurance & 401(k)? | ⚠️ Partial (retrieval fixed; KB gap remains) | Hybrid + rerank **now retrieves the 401(k) chunk** (`benefits-03`, rank 4) — the retrieval failure is fixed. Claude answers the insurance part ("benefits continue unchanged") and **precisely declines the 401(k)-during-leave specifics, because the KB describes the 401(k) *plan* but never states its treatment *during leave*.** The residual gap is in the knowledge base, not retrieval. |
| 12 | Multi-document | Reimbursements as a remote employee? | ✅ (narrower) | Returned the remote-specific items (stipend + internet). Note: re-ranking **focused** the answer on "remote-employee" reimbursements and dropped the general wellness/expense items the dense-only run had surfaced — a precision-over-recall trade-off (see below). |
| 13 | Unanswerable | Relocation reimbursement? | ✅ decline | Not in KB; returned the exact "not covered" reply. No fabrication. |
| 14 | Unanswerable | Pet insurance? | ✅ decline | Retrieved health/dental/life insurance by proximity; did not invent a pet plan. |
| 15 | Unanswerable | Sabbatical after 5 years? | ✅ decline | No sabbatical content; declined rather than inventing a tenure perk. |

Legend: ✅ success · ⚠️ partial (correct, honest behavior).

---

## What hybrid retrieval + re-ranking changed (vs. plain vector search)

We ran the same 15 questions on dense-only retrieval first, then on hybrid + rerank.
Three concrete differences:

1. **Better rank-1 precision (re-ranking).** On the factual questions the cross-encoder
   pushed the exact answer chunk to rank 1 (e.g. Q1 `PTO Accrual` moved rank 2 → rank 1).
2. **Fixed the Q11 retrieval gap (hybrid + tokenizer).** Dense-only never retrieved the
   401(k) chunk for Q11. BM25 keyword matching pulls it into the pool **once the BM25
   tokenizer is fixed** — the default tokenizer kept punctuation, so the query token
   `401(k)?` failed to match the document's `401(k)`. With a lowercase word-split
   tokenizer, `benefits-03` enters the pool and survives re-ranking into the top-4.
   See §5b of the notebook for the side-by-side.
3. **A precision/recall trade-off on broad queries (Q12).** Re-ranking tightened "what
   can I get reimbursed for as a remote employee?" to the *remote-specific* items and
   dropped the general wellness/expense reimbursements. More focused, but lower recall —
   worth knowing that aggressive re-ranking can trim legitimately-relevant breadth.

---

## Where retrieval succeeds

- **Specific factual questions (Q1–Q6)** — exact chunk retrieved and ranked first; correct, cited answers.
- **Ambiguous questions (Q7–Q9)** — the *answer-from-context* prompt makes Claude
  surface multiple interpretations or ask for clarification (Q9) rather than guessing.
- **Cross-document questions (Q10, and Q11's retrieval)** — hybrid retrieval pulls chunks
  from more than one source document; the keyword stage is what rescues exact-term chunks
  (401(k)) that pure embeddings rank too low.
- **Unanswerable questions (Q13–Q15)** — the most important behavior: omitted topics are
  declined every time, never fabricated.

## Where it still falls short (and why)

- **Q11 — a knowledge gap, not a retrieval gap (now).** The right chunk is retrieved, but
  the KB simply doesn't state what happens to a 401(k) *during* parental leave, so the
  honest answer is a partial decline. Fixing this means **adding the fact to the KB**, not
  changing retrieval.
- **Q12 — re-ranking can over-narrow.** A broad "what can I get reimbursed for" question
  has several equally-valid answers across documents; a precision-tuned re-ranker may keep
  only the most on-topic few. Tunable via `top_n` / fusion weights.

## Lessons & next steps

1. **Hybrid retrieval is only as good as its tokenizer.** Domain terms with punctuation
   (`401(k)`, `PTO`) silently defeat the default BM25 splitter; a lowercase word-split
   tokenizer was the actual fix for Q11's retrieval.
2. **Re-ranking trades recall for precision** — great for factual lookups, but tune
   `top_n` / weights for broad "list everything" questions (Q12).
3. **Distinguish retrieval gaps from knowledge gaps.** Once Q11's chunk is retrieved, the
   remaining limit is content coverage — the fix is data, not retrieval.
4. **Query decomposition** would further help multi-aspect questions (Q11): split
   "insurance and 401(k)" into two sub-queries so each aspect retrieves its own best chunk.
5. **Metadata filtering** (by `policy_area`) would make ambiguous queries fan out
   deterministically rather than relying on similarity.
