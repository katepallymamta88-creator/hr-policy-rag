"""HR Policies RAG — Streamlit viewer.

Two tabs:
  1. Knowledge Base — browse / search the HR documents.
  2. HR Assistant   — chat that answers using hybrid retrieval (BM25 + dense) with
     cross-encoder re-ranking, then Claude generation — the same flow as the notebook.

The app REUSES the Pinecone index built by `hr_policy_rag.ipynb` — it does not
re-index. Run the notebook first, then:

    uv run streamlit run company_kb_viewer.py
"""

import json
import os
import re

import streamlit as st
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from flashrank import Ranker, RerankRequest
from langchain_core.embeddings import Embeddings
from openai import OpenAI

load_dotenv()

INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "hr-policies-rag")
EMBED_MODEL = "Qwen/Qwen3-Embedding-8B"   # Nebius Token Factory embedding model (4096-dim)
GEN_MODEL = "claude-opus-4-8"
RERANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # FlashRank cross-encoder re-ranker


# --- Nebius Token Factory embeddings (batched) --------------------------------
# Sends all texts in ONE request to Nebius's OpenAI-compatible endpoint (fast).
# Kept inline so the app is self-contained (same class as in the notebook).
class NebiusBatchEmbeddings(Embeddings):
    def __init__(self, model=EMBED_MODEL, api_key=None,
                 base_url="https://api.tokenfactory.nebius.com/v1/", batch_size=100):
        self.model = model
        self.batch_size = batch_size
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ["NEBIUS_API_KEY"])

    def embed_documents(self, texts):
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            resp = self.client.embeddings.create(model=self.model, input=texts[start:start + self.batch_size])
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    def embed_query(self, text):
        return self.client.embeddings.create(model=self.model, input=[text]).data[0].embedding

SYSTEM = (
    "You are an HR policy assistant for Northwind Labs. Answer the employee's "
    "question using ONLY the context below, retrieved from the official HR "
    "knowledge base.\n\n"
    "Rules:\n"
    "- Use only facts present in the context. Do not use outside knowledge or guess.\n"
    "- If the context does not contain the answer, reply EXACTLY: \"I'm sorry, that "
    "topic isn't covered in the HR knowledge base. Please contact HR for help.\" "
    "Do not invent a policy.\n"
    "- If the question is ambiguous, briefly note the interpretations the context supports.\n"
    "- Be concise and name the policy titles you used.\n\n"
    "Context:\n{context}"
)


# --------------------------------------------------------------------------- #
# Cached resources (built once per session)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Connecting to the knowledge base (hybrid + re-ranker)...")
def get_chain():
    embeddings = NebiusBatchEmbeddings(model=EMBED_MODEL)
    vectorstore = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME, embedding=embeddings
    )
    dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 8})

    # BM25 keyword retriever needs the chunk texts in memory -> rebuild them from the KB.
    kb = load_kb()
    docs = [
        Document(page_content=d["content"], metadata={**d["metadata"], "id": d["id"]})
        for d in kb["documents"]
    ]
    chunks = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100).split_documents(docs)
    # Custom tokenizer so terms like "401(k)" match regardless of punctuation.
    bm25_retriever = BM25Retriever.from_documents(
        chunks, preprocess_func=lambda t: re.findall(r"\w+", t.lower())
    )
    bm25_retriever.k = 8

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, dense_retriever], weights=[0.4, 0.6]
    )
    reranker = Ranker(model_name=RERANK_MODEL)

    def retrieve(query, fetch_k=10, top_n=4):
        candidates = hybrid_retriever.invoke(query)[:fetch_k]
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(candidates)]
        ranked = reranker.rerank(RerankRequest(query=query, passages=passages))
        return [candidates[r["id"]] for r in ranked[:top_n]]

    llm = ChatAnthropic(model=GEN_MODEL, max_tokens=1024)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{question}")])
    return retrieve, llm, prompt


@st.cache_data
def load_kb():
    with open("hr_knowledge_base.json") as f:
        return json.load(f)


def format_docs(retrieved):
    return "\n\n".join(
        f"[{d.metadata.get('title', '?')} -- {d.metadata.get('source_document', '?')}]\n"
        f"{d.page_content}"
        for d in retrieved
    )


def answer(question):
    retrieve, llm, prompt = get_chain()
    retrieved = retrieve(question)
    messages = prompt.format_messages(context=format_docs(retrieved), question=question)
    return llm.invoke(messages).content, retrieved


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="Northwind HR Assistant", page_icon="📋", layout="wide")
st.title("📋 Northwind Labs — HR Policies")

# Fail fast with a friendly message if keys are missing.
missing = [k for k in ("NEBIUS_API_KEY", "ANTHROPIC_API_KEY", "PINECONE_API_KEY") if not os.environ.get(k)]
if missing:
    st.error(f"Missing environment variables: {', '.join(missing)}. Copy .env.demo to .env and fill in your keys.")
    st.stop()

kb = load_kb()
docs = kb["documents"]

tab_kb, tab_chat = st.tabs(["📚 Knowledge Base", "💬 HR Assistant"])

# ---- Tab 1: Knowledge Base browser ---------------------------------------- #
with tab_kb:
    st.caption(f"{len(docs)} documents · {kb['company']}")
    col1, col2 = st.columns([2, 1])
    with col1:
        query = st.text_input("Search the knowledge base", placeholder="e.g. parental leave, 401k, remote work")
    with col2:
        areas = sorted({d["metadata"]["policy_area"] for d in docs})
        area = st.selectbox("Filter by policy area", ["(all)"] + areas)

    results = docs
    if area != "(all)":
        results = [d for d in results if d["metadata"]["policy_area"] == area]
    if query:
        q = query.lower()
        results = [d for d in results if q in d["content"].lower() or q in d["metadata"]["title"].lower()]

    st.write(f"**{len(results)}** matching document(s)")
    for d in results:
        m = d["metadata"]
        with st.expander(f"{m['title']}  —  {m['source_document']}  ·  [{m['policy_area']}]"):
            st.write(d["content"])
            st.caption(f"id: {d['id']} · doc_type: {m['doc_type']}")

# ---- Tab 2: HR Assistant chat --------------------------------------------- #
with tab_chat:
    st.caption("Answers come only from the indexed HR knowledge base. Unknown topics are declined, not guessed.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("Sources"):
                    for s in msg["sources"]:
                        st.markdown(f"- **{s['title']}** ({s['source_document']}) · `{s['id']}`")

    if user_q := st.chat_input("Ask an HR question..."):
        st.session_state.messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving and answering..."):
                reply, retrieved = answer(user_q)
            st.markdown(reply)
            sources = [
                {
                    "title": d.metadata.get("title", "?"),
                    "source_document": d.metadata.get("source_document", "?"),
                    "id": d.metadata.get("id", "?"),
                }
                for d in retrieved
            ]
            with st.expander("Sources"):
                for s in sources:
                    st.markdown(f"- **{s['title']}** ({s['source_document']}) · `{s['id']}`")
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": sources})
