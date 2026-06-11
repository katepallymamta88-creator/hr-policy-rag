# Design Decisions & Learnings

A narrative of how this HR RAG assistant was built — the design choices, the problems
we hit along the way, and how each was fixed. The most important thread is the
**retrieval failure on Q11** and the two-step fix (hybrid retrieval + a tokenizer
correction) that resolved it.

---

## Goal

Build a Q&A assistant over enterprise HR documents that (a) answers from the official
policy knowledge base with cited sources, and (b) **declines instead of hallucinating**
when the answer isn't there — then stress-test it with 15 questions and document where
retrieval succeeds, fails, and why.

## Stack (and the one hard constraint)

| Concern | Choice | Why |
|---|---|---|
| Embeddings | **Nebius Token Factory** — `Qwen/Qwen3-Embedding-8B` (4096-dim) | Assignment requires ≥1 Nebius model call. (Anthropic has no embeddings API.) |
| Vector store | **Pinecone** serverless (cosine) | Mirrors the course reference stack. |
| Retrieval | **Hybrid (BM25 + dense) → cross-encoder re-rank** | See the Q11 story below. |
| Generation | **Claude `claude-opus-4-8`** | Strong reasoning + careful, citable, low-hallucination answers. |
| Orchestration | **LangChain** | Standard RAG plumbing. |

---

## Design evolution

### v1 — Basic RAG (single retriever)
`load → chunk (500/100) → embed (Nebius) → index (Pinecone) → top-k=4 dense retrieval → Claude`.
This passed 14 of 15 stress questions. But it exposed one real retrieval failure (Q11)
and several engineering problems that had to be solved to make it run at all.

### v2 — Hybrid retrieval + re-ranking (current)
Retrieval became two-stage:
1. **Hybrid retrieve** — BM25 keyword search + dense vector search, fused with
   `EnsembleRetriever` (Reciprocal Rank Fusion), to cast a wide net.
2. **Re-rank** — a FlashRank cross-encoder scores each candidate against the query and
   keeps the best top-4, which is far more precise than first-stage similarity scores.

Everything lives in **one notebook** (`hr_policy_rag.ipynb`) and is mirrored in the
Streamlit app (`company_kb_viewer.py`).

---

## Problems encountered & how we fixed them

### 1. The embedding step hung for ~12 minutes
- **Symptom:** the notebook froze on the embed/index cell; it hit the execution timeout (600s).
- **Root cause:** `langchain-nebius`'s `embed_documents` issued **one request per chunk**, which tripped Nebius rate limits and triggered exponential-backoff retries — **727 seconds** for 30 chunks. (A single embedding call alone was only 2.2s, which is what made it confusing.)
- **Fix:** a thin wrapper (`nebius_embeddings.py`) that sends **all chunks in one batched request** to the same Nebius OpenAI-compatible endpoint → **~6.5s**. Still a genuine Nebius call (same endpoint, key, and model).
- **Lesson:** batch your embedding requests; per-item requests against a rate-limited API are silently ~100x slower, and the failure looks like a hang, not an error.

### 2. A "documented" Nebius model didn't exist
- **Symptom:** `BAAI/bge-en-icl` returned `404 - model does not exist`.
- **Root cause:** Nebius rebranded "AI Studio" → "Token Factory" and the docs example was stale.
- **Fix:** verified the model catalog against the live API and stayed on `Qwen/Qwen3-Embedding-8B`; pinned the base URL to `api.tokenfactory.nebius.com`.
- **Lesson:** verify model IDs against the live endpoint, not the docs.

### 3. Bad keys failed late and cryptically
- **Symptom:** with `.env` keys left as the placeholder `...`, the run failed deep inside the embedding cell with an auth error.
- **Root cause:** the setup check only tested that a variable was *set*; `"..."` is a non-empty string, so it passed.
- **Fix:** the setup cell now rejects known placeholder values up front with a clear "edit your .env" message.
- **Lesson:** validate config for *placeholder* values, not just presence — fail fast with a readable message.

### 4. Upserting to a brand-new index could hang
- **Root cause:** a freshly created Pinecone serverless index isn't immediately writable; upserting too early can hang.
- **Fix:** a "wait until `status.ready`" loop before the first upsert.
- **Lesson:** treat resource creation as async — poll for readiness before using it.

### 5. ⭐ The retrieval failure — Q11 (the headline)

**Question 11 (multi-document):** *"If I take parental leave, what happens to my
health insurance and 401(k)?"* — the answer needs the **Parental Leave** chunk *and*
the **401(k) Retirement Plan** chunk.

**Stage A — pure vector retrieval missed it.** Top-4 dense retrieval returned
parental-leave + life-insurance + sick-leave chunks. The **401(k) chunk was never
retrieved**, because "401(k) retirement match" is semantically distant from "parental
leave," so it ranked just outside the top-4. The system degraded gracefully (Claude
answered the insurance part and *correctly declined* the 401(k) part rather than
inventing it), but the underlying retrieval gap was real.

**Stage B — we added hybrid retrieval + re-ranking… and it *still* missed it.** This
was the key surprise. The expectation was that BM25 keyword search would catch the
literal token "401(k)". It didn't — the 401(k) chunk still wasn't in the pool.

**Stage C — root cause: BM25 tokenization.** BM25's default tokenizer keeps
punctuation, so the query token `401(k)?` never matched the document's `401(k)`. The
keyword advantage was silently nullified by a question mark and parentheses.

**Stage D — the fix.** A proper BM25 tokenizer — lowercase and split on word
characters (`re.findall(r"\w+", text.lower())`) — so `401(k)?` and `401(k)` both
tokenize to `["401", "k"]` and match. After this, BM25 ranks the 401(k) chunk into the
pool, and **hybrid + re-rank now includes "401(k) Retirement Plan" in the final top-4**
(verified against the live index). **The retrieval failure is fixed.**

**Stage E — the honest epilogue: a retrieval gap vs. a knowledge gap.** With the 401(k)
chunk now in context, Claude *still* gives a partial answer on Q11 — and correctly so.
The retrieved chunk describes the 401(k) *plan and match*; it does **not** say what
happens to a 401(k) *during* parental leave, which the knowledge base never states. So
Claude answers the health-insurance part ("benefits continue unchanged") and precisely
declines the 401(k)-during-leave specifics. The lesson: **we fixed the retrieval gap,
and doing so revealed that the residual limitation is a knowledge-base gap** — the fix
for that is adding the fact to the data, not changing retrieval.

**Lessons:**
- Adding hybrid retrieval is not a magic fix — **verify** it actually surfaces the
  target chunk; don't assume. (Our first hybrid attempt still missed it.)
- Keyword/BM25 retrieval is only as good as its tokenizer. Punctuation in domain terms
  (`401(k)`, `PTO`, phone numbers) will defeat the default splitter.
- Cross-encoder re-ranking can only re-order what the first stage retrieves — if the
  chunk isn't in the candidate pool, no re-ranker can save it.
- **Separate retrieval failures from knowledge failures.** Once the right chunk is
  retrieved, a remaining wrong/partial answer is a *content* problem, not a *retrieval*
  problem — and the two have completely different fixes.

### 6. The 15-question run "timed out" (an ops note, not a bug)
- **Symptom:** the stress-test cell hit the per-cell execution timeout.
- **Root cause:** 15 sequential Claude Opus calls (several with long answers) exceeded
  the default nbconvert timeout — not a code error.
- **Fix:** raised the per-cell timeout for the headless run. Interactive (`Run All`) is unaffected.
- **Lesson:** size automation timeouts to the slowest legitimate cell (LLM loops are slow).

---

## Final design (what runs today)

```
hr_knowledge_base.json
  → load (LangChain Documents + metadata)
  → chunk (RecursiveCharacterTextSplitter, 500/100)
  → embed (Nebius Qwen3-Embedding-8B, 4096-dim, BATCHED)      ← required Nebius call
  → index (Pinecone serverless, cosine; wait-until-ready)
  → HYBRID retrieve: BM25 (custom tokenizer) + dense, fused via EnsembleRetriever/RRF
  → RE-RANK: FlashRank cross-encoder → top-4
  → generate (Claude claude-opus-4-8, answer-from-context-only + citations)
```

## Headline takeaways

1. **The guardrail is the product.** "Answer only from context, else decline" turns
   every retrieval gap (Q11, the unanswerable questions, ambiguous queries) into safe,
   transparent behavior instead of confident fabrication.
2. **Most of the hard bugs were operational, not conceptual** — embedding throughput,
   index readiness, tokenization, config validation, timeouts. RAG is as much plumbing
   as it is models.
3. **Hybrid + re-rank is the right architecture, but the details decide whether it
   works** — the tokenizer fix, not the architecture alone, is what actually retrieved
   the 401(k) chunk.
4. **Verify against the live system at every step** — model availability, embedding
   speed, and retrieval results all differed from what the docs/assumptions implied.

See [`evaluation.md`](./evaluation.md) for the full 15-question results table and
[`rag_pipeline_infographic.html`](./rag_pipeline_infographic.html) for the visual flow.
