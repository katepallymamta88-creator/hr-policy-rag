# HR Policies RAG — Q&A System

Week 2 Assignment · **Retrieval-Augmented Generation over enterprise HR documents.**

A RAG pipeline that ingests a knowledge base of HR policy documents, chunks and
embeds them, indexes them in Pinecone, and answers natural-language questions with
**cited sources** — then is stress-tested with 15 questions (including ambiguous,
multi-document, and unanswerable edge cases).

The file structure mirrors the course reference repo
[`Mastering-Agentic-AI-Week2 / Week2-Session1`](https://github.com/The-Gen-Academy/Mastering-Agentic-AI-Week2/tree/main/Week2-Session1).
Two intentional changes from the course: **embeddings run on Nebius Token Factory**
(`Qwen/Qwen3-Embedding-8B` via `langchain-nebius`) instead of OpenAI — satisfying the
assignment's "use Nebius for at least one model call" requirement — and **generation
runs on Claude `claude-opus-4-8`** (via `langchain-anthropic`) instead of the course's
`gpt-4.1-mini`. The vector store (Pinecone) is unchanged. OpenAI is no longer used.

## Architecture

```
hr_knowledge_base.json
        │  load → LangChain Documents (content + metadata)
        ▼
   chunk  (RecursiveCharacterTextSplitter: 500 chars / 100 overlap)
        ▼
   embed  (Nebius Token Factory — Qwen/Qwen3-Embedding-8B, 4096-dim)   ← Nebius model call
        ▼
   index  (Pinecone, serverless, cosine)
        ▼
  hybrid retrieve  (BM25 keyword + dense vector, fused via EnsembleRetriever / RRF)
        ▼
  re-rank  (FlashRank cross-encoder) → keep top-4
        ▼
generate  (Claude claude-opus-4-8, answer-from-context-only + citations)
```

**Retrieval is two-stage:** a hybrid first stage casts a wide net (semantic *and*
keyword matching), then a cross-encoder re-ranker keeps only the most relevant chunks
— more precise than either dense or keyword search alone.

## Files

| File | Purpose |
|------|---------|
| `hr_knowledge_base.json` | The HR documents — structured `{id, content, metadata}` entries. |
| `hr_policy_rag.ipynb` | **Main deliverable.** Full pipeline end-to-end + the 15-question stress test, run inline. |
| `nebius_embeddings.py` | Thin batched embeddings wrapper for the Nebius Token Factory OpenAI-compatible endpoint (one request instead of one-per-chunk — ~100x faster). Imported by the notebook and the app. |
| `evaluation.md` | Qualitative write-up: each test question, retrieved sources, outcome, and *why* retrieval succeeded/failed. |
| `design_and_learnings.md` | The design journey — decisions, the problems hit (incl. the Q11 retrieval failure), and how each was fixed. |
| `company_kb_viewer.py` | Streamlit app — KB browser + HR assistant chat (reuses the Pinecone index). |
| `rag_pipeline_infographic.html` | Visual step-by-step infographic of the whole pipeline (open in a browser; print to PDF). |
| `.env.demo` | Template for the required API keys. |

## Setup

**1. Install dependencies** (Python ≥ 3.12, via [`uv`](https://docs.astral.sh/uv/)):

```bash
uv sync
```

(or with pip: `python -m venv .venv && source .venv/bin/activate && pip install -e .`)

**2. Add your API keys:**

```bash
cp .env.demo .env
# then edit .env and fill in the three keys
```

You need **three** keys:
- `NEBIUS_API_KEY` — embeddings (`Qwen/Qwen3-Embedding-8B`, via Nebius Token Factory)
- `ANTHROPIC_API_KEY` — generation (`claude-opus-4-8`)
- `PINECONE_API_KEY` — vector store

The Pinecone index (`PINECONE_INDEX_NAME`, default `hr-policies-rag`) is created
automatically by the notebook as a **4096-dim, cosine** index if it doesn't exist
(4096 = the native dimension of `Qwen/Qwen3-Embedding-8B`). If you previously created
the index at a different dimension, delete it or use a new `PINECONE_INDEX_NAME`.

## Run

**1. The notebook (build + validate first):**

```bash
uv run jupyter notebook
```

Open `hr_policy_rag.ipynb` and run all cells top to bottom. It will:
1. load + chunk + embed + index the HR knowledge base into Pinecone,
2. answer a sanity-check question with sources,
3. run all 15 stress-test questions and print question → answer → retrieved sources.

**2. The Streamlit app (after the notebook works):**

```bash
uv run streamlit run company_kb_viewer.py
```

The app reuses the existing Pinecone index — no re-indexing.

## Stress test & findings

The 15 questions span four buckets — **direct factual**, **ambiguous**,
**multi-document**, and **unanswerable** (topics deliberately omitted from the KB).
See [`evaluation.md`](./evaluation.md) for the full results table and the analysis
of where retrieval succeeds, where it fails, and why.
