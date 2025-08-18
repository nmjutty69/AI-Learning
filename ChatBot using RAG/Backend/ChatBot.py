import os
import json
import time
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import numpy as np
from sentence_transformers import SentenceTransformer
from langdetect import detect
import chromadb 

import requests

# HF Transformers (encoder-only intent, encoder–decoder summarization)
from transformers import pipeline

# -----------------------
# Environment & Constants
# -----------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "llama3-8b-8192")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
DOCS_PATH = os.path.join(DATA_DIR, "docs.jsonl")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", 4))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 6))  # user-assistant pairs

# -----------------------
# App & CORS
# -----------------------
app = FastAPI(title="Tour Guide Chatbot — RAG + Transformers")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------
# Models: Embeddings, Intent, Summarizer
# -----------------------
# Embeddings
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBED_MODEL_NAME)
EMBED_DIM = embedder.get_sentence_embedding_dimension()

# Init chroma client + collection
chroma_client = chromadb.Client()
collection = chroma_client.get_or_create_collection(
    name="tour_docs",
    metadata={"hnsw:space": "cosine"}
)

# Intent (encoder-only). Zero-shot so we can define labels without training
INTENT_LABELS = [
    "recommendation", "history", "itinerary", "directions",
    "costs_pricing", "visa_rules", "faq", "safety"
]
intent_clf = pipeline(
    "zero-shot-classification",
    model="facebook/bart-large-mnli"
)

# Summarizer (encoder–decoder)
summarizer = pipeline(
    "summarization",
    model="t5-small",  # light & fast
    tokenizer="t5-small"
)

# -----------------------
# Chroma Store (Chroma) + Docs
# -----------------------
class ChromaStore:
    def __init__(self, collection):
        self.collection = collection

    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]
        embeddings = embedder.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        ids = [str(uuid.uuid4()) for _ in texts]

        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

    def search(self, query: str, top_k: int = 4) -> List[Dict[str, Any]]:
        qemb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).tolist()[0]
        results = self.collection.query(query_embeddings=[qemb], n_results=top_k)

        out = []
        for i in range(len(results["ids"][0])):
            out.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "meta": results["metadatas"][0][i],
                "score": results["distances"][0][i]
            })
        return out


store = ChromaStore(collection)

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# -----------------------
# Pydantic Schemas
# -----------------------
class IngestItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = None

class IngestRequest(BaseModel):
    items: List[IngestItem]

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None  # for memory
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    retrieved: Optional[List[Dict[str, Any]]] = None
    session_id: str

# -----------------------
# Memory (short-term)
# -----------------------
MEMORY: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str) -> List[Dict[str, str]]:
    return MEMORY.get(session_id, [])[-MAX_HISTORY_TURNS:]

def append_history(session_id: str, role: str, content: str):
    MEMORY.setdefault(session_id, []).append({"role": role, "content": content})
    # Trim
    if len(MEMORY[session_id]) > 2 * MAX_HISTORY_TURNS:
        MEMORY[session_id] = MEMORY[session_id][-2*MAX_HISTORY_TURNS:]

# -----------------------
# Helpers: Intent, Summarize, Build Prompt, LLM Call
# -----------------------

def classify_intent(text: str) -> str:
    try:
        res = intent_clf(text, INTENT_LABELS, multi_label=False)
        return res["labels"][0]
    except Exception:
        return "faq"


def summarize_chunks(chunks: List[str], max_chars: int = 1200) -> str:
    if not chunks:
        return ""
    # Light summarization to compress retrieved context
    joined = "\n\n".join(chunks)
    if len(joined) <= max_chars:
        return joined
    # T5 expects "summarize: " prefix
    try:
        summary = summarizer("summarize: " + joined, max_length=220, min_length=60, do_sample=False)
        return summary[0]["summary_text"]
    except Exception:
        # Fallback: take head
        return joined[:max_chars]


def build_prompt(user_msg: str, history: List[Dict[str, str]], intent: str, context_snippet: str) -> str:
    system_prompt = (
        "You are a precise, hype yet honest tour-guide assistant for tourists. "
        "Answer ONLY from the provided context. "
        "If info is missing, say 'I don’t know from the data provided.'\n\n"
    )

    developer_prompt = (
        f"STYLE & RULES:\n"
        f"- Voice: friendly, concise, professional.\n"
        f"- Answer structure:Essentials → Itinerary → Costs & Tips → Closing Q.\n"
        f"- Current intent: {intent}\n\n"
    )

    context_block = f"CONTEXT:\n{context_snippet}\n\n"

    history_block = ""
    for turn in history:
        history_block += f"{turn['role'].upper()}: {turn['content']}\n"

    return f"{system_prompt}{developer_prompt}{context_block}{history_block}USER: {user_msg}\nASSISTANT:"


# -----------------------
# LLM Call
# -----------------------
def call_local_llm(prompt: str, max_new_tokens: int = 250) -> str:
    try:
        output = generator(prompt, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.6)
        return output[0]["generated_text"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Local LLM error: {str(e)}")



# -----------------------
# Routes
# -----------------------

@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    intent = classify_intent(req.message)

    retrieved = []
    context_snippet = ""
    if req.use_rag:
        retrieved = store.search(req.message, top_k=RAG_TOP_K)
        context_snippet = summarize_chunks([r["text"] for r in retrieved])

    history = get_history(session_id)

    # build local prompt
    prompt = build_prompt(req.message, history, intent, context_snippet)

    # call local HF model
    reply = call_local_llm(prompt)

    append_history(session_id, "user", req.message)
    append_history(session_id, "assistant", reply)

    return ChatResponse(
        reply=reply,
        intent=intent,
        retrieved=retrieved,
        session_id=session_id
    )

    session_id = req.session_id or str(uuid.uuid4())

    # classify intent
    intent = classify_intent(req.message)

    # retrieve docs if RAG enabled
    retrieved = []
    context_snippet = ""
    if req.use_rag:
        retrieved = store.search(req.message, top_k=RAG_TOP_K)
        context_snippet = summarize_chunks([r["text"] for r in retrieved])

    # build history
    history = get_history(session_id)

    # build messages for LLM
    messages = build_messages(req.message, history, intent, context_snippet)

    # call Groq
    reply = call_groq(messages)

    # update memory
    append_history(session_id, "user", req.message)
    append_history(session_id, "assistant", reply)

    return ChatResponse(
        reply=reply,
        intent=intent,
        retrieved=retrieved,
        session_id=session_id
    )


@app.post("/reset_memory")
def reset_memory(session_id: Optional[str] = None):
    """Clear memory for one session, or all if no session_id given."""
    if session_id:
        MEMORY.pop(session_id, None)
        return {"status": "ok", "cleared": session_id}
    else:
        MEMORY.clear()
        return {"status": "ok", "cleared": "all sessions"}


@app.post("/reset_store")
def reset_store():
    """⚠️ Completely wipe the ChromaDB collection (all ingested docs)."""
    global collection, store
    chroma_client.delete_collection("tour_docs")
    collection = chroma_client.get_or_create_collection(
        name="tour_docs", metadata={"hnsw:space": "cosine"}
    )
    store = ChromaStore(collection)
    return {"status": "ok", "cleared": "vector store"}


