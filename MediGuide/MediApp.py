import os
import time
import json
import uuid
import glob
import textwrap
import requests
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from dotenv import load_dotenv

# ---- Optional: Only used to show progress in the UI
from tqdm import tqdm

# ---- Vector DB
import chromadb
from chromadb.config import Settings

# ---- OpenAI/Groq for LLM calls
from groq import Groq
# from openai import OpenAI

############################################################
# App Constants
############################################################
APP_TITLE = "MediGuide RAG API: Cloud-Powered Patient Education"
VECTOR_DIR = "./vectorstore"
KB_DIR = "./knowledge_base"
COLLECTION_NAME = "mediguide_kb"
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
HF_EMBEDDING_URL = (
    f"https://api-inference.huggingface.co/pipeline/feature-extraction/{HF_EMBEDDING_MODEL}"
)

# Chunking
CHUNK_SIZE = 300  # characters
CHUNK_OVERLAP = 25  # characters

# Retrieval
TOP_K = 3

############################################################
# Secrets / Keys Loading
############################################################

def load_keys() -> Tuple[str, str]:
    """Load keys from (priority): st.secrets -> environment -> empty string.
    Also allow user sidebar overrides (highest priority).
    """
    # Load .env for local dev
    load_dotenv(override=False)

    hf_key_env = os.getenv("HUGGING_FACE_API_KEY", "")
    # openai_key_env = os.getenv("OPENAI_API_KEY", "")

    # Streamlit Secrets (preferred for production)
    # try:
    #     hf_key_secret = st.secrets.get("HUGGING_FACE_API_KEY", "")
    #     openai_key_secret = st.secrets.get("OPENAI_API_KEY", "")
    # except Exception:
    #     hf_key_secret, openai_key_secret = "", ""

    groq_key_env = os.getenv("GROQ_API_KEY", "")
    try:
        hf_key_secret = st.secrets.get("HUGGING_FACE_API_KEY", "")
        groq_key_secret = st.secrets.get("GROQ_API_KEY", "")
    except Exception:
        hf_key_secret, groq_key_secret = "", ""

    # Base keys (secret > env)
    hf_key = hf_key_secret or hf_key_env
    groq_key = groq_key_secret or groq_key_env
    # openai_key = openai_key_secret or openai_key_env

    # Sidebar manual override
    with st.sidebar:
        st.subheader("🔐 API Keys")
        st.caption("Keys are read from st.secrets (prod) or .env (dev). You can override below.")
        hf_key_input = st.text_input("HUGGING_FACE_API_KEY", value=hf_key, type="password")
        groq_key_input = st.text_input("GROQ_API_KEY", value=groq_key, type="password")
        # openai_key_input = st.text_input("GROQ_API_KEY", value=openai_key, type="password")

    return hf_key_input.strip(), groq_key_input.strip()

############################################################
# Hugging Face Embeddings via Inference API
############################################################

def _hf_feature_extract(batch_texts: List[str], hf_api_key: str, retries: int = 4) -> List[List[float]]:
    headers = {"Authorization": f"Bearer {hf_api_key}"}
    payload = {"inputs": batch_texts, "options": {"wait_for_model": True}}

    for attempt in range(retries):
        resp = requests.post(HF_EMBEDDING_URL, headers=headers, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            # The API returns list[embedding] if we send list[str]
            if isinstance(data, list) and all(isinstance(v, list) for v in data):
                # If a single item returns nested list (token-level), average
                processed = []
                for item in data:
                    # Some endpoints return [seq_len x dim]; average over seq_len
                    if item and isinstance(item[0], list):
                        dim = len(item[0])
                        # mean across tokens
                        vec = [0.0] * dim
                        for token_vec in item:
                            for i, val in enumerate(token_vec):
                                vec[i] += float(val)
                        vec = [v / len(item) for v in vec]
                        processed.append(vec)
                    else:
                        # Already a vector
                        processed.append([float(x) for x in item])
                return processed
            else:
                raise RuntimeError(f"Unexpected HF response format: {type(data)}")
        elif resp.status_code in (429, 503):
            # Rate limited or model loading — backoff
            sleep_s = 2 ** attempt
            time.sleep(sleep_s)
            continue
        else:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"HF API error {resp.status_code}: {err} URL={HF_EMBEDDING_URL}")

    raise TimeoutError("HF Inference API: retries exhausted")


def get_embeddings(texts: List[str], hf_api_key: str, batch_size: int = 1) -> List[List[float]]:
    """Batch texts to HF Inference API. Returns one embedding per input text."""
    all_vecs: List[List[float]] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = _hf_feature_extract(batch, hf_api_key)
        all_vecs.extend(vecs)
    return all_vecs

############################################################
# ChromaDB Helpers
############################################################

def get_chroma_client() -> chromadb.PersistentClient:
    Path(VECTOR_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=VECTOR_DIR, settings=Settings(anonymized_telemetry=False))
    return client


def get_or_create_collection(client: chromadb.PersistentClient):
    try:
        col = client.get_collection(COLLECTION_NAME)
    except Exception:
        col = client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    return col

############################################################
# Knowledge Base Ingestion
############################################################

def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(text):
            break
    return [c.strip() for c in chunks if c.strip()]


def load_kb_files(kb_dir: str = KB_DIR) -> List[Tuple[str, str]]:
    files = sorted(glob.glob(os.path.join(kb_dir, "**/*.txt"), recursive=True))
    data = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data.append((fp, f.read()))
        except Exception as e:
            st.warning(f"Could not read {fp}: {e}")
    return data

def ingest_documents_streaming(hf_api_key: str, chunk_size: int = 300, chunk_overlap: int = 25, batch_size: int = 1) -> Dict:
    """
    Memory-friendly ingestion:
    - chunk_size: max chars per chunk
    - chunk_overlap: overlap between chunks
    - batch_size: number of chunks to embed at once (keep 1 for lowest memory)
    """
    client = get_chroma_client()
    col = get_or_create_collection(client)

    stats = {"files": 0, "chunks": 0, "added": 0}

    # Skip if collection has vectors
    if col.count() > 0:
        return {"skipped": True, "existing": col.count(), **stats}

    kb_items = load_kb_files()
    if not kb_items:
        raise FileNotFoundError(f"No .txt files found under {KB_DIR}.")

    st.info("Streaming KB ingestion (low memory)…")

    for path, text in tqdm(kb_items, desc="Chunking KB"):
        stats["files"] += 1
        chunks = chunk_text(text, size=chunk_size, overlap=chunk_overlap)
        stats["chunks"] += len(chunks)

        # Embed and add in tiny batches
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings = get_embeddings(batch, hf_api_key, batch_size=batch_size)

            ids = [str(uuid.uuid4()) for _ in batch]
            metadatas = [{"source": path, "chunk": i + j} for j in range(len(batch))]

            col.add(ids=ids, documents=batch, metadatas=metadatas, embeddings=embeddings)
            stats["added"] += len(batch)

    return {"skipped": False, "existing": 0, **stats}


def ingest_documents(hf_api_key: str) -> Dict:
    """If collection empty, embed and add all text chunks from KB to ChromaDB."""
    client = get_chroma_client()
    col = get_or_create_collection(client)

    stats = {"files": 0, "chunks": 0, "added": 0}

    # Quick check: if collection already populated, skip unless user forces re-ingest
    existing_count = col.count()
    if existing_count > 0:
        return {"skipped": True, "existing": existing_count, **stats}

    kb_items = load_kb_files()
    if not kb_items:
        raise FileNotFoundError(
            f"No .txt files found under {KB_DIR}. Add trusted sources before ingesting."
        )

    all_chunks, all_metadatas, all_ids = [], [], []

    for path, text in tqdm(kb_items, desc="Chunking KB"):
        stats["files"] += 1
        chunks = chunk_text(text)
        stats["chunks"] += len(chunks)
        for idx, ch in enumerate(chunks):
            all_chunks.append(ch)
            all_metadatas.append({"source": path, "chunk": idx})
            all_ids.append(str(uuid.uuid4()))

    # Embed in batches via HF API
    st.info("Embedding knowledge base via Hugging Face Inference API…")
    embeddings = get_embeddings(all_chunks, hf_api_key)

    # Add to Chroma
    col.add(ids=all_ids, documents=all_chunks, metadatas=all_metadatas, embeddings=embeddings)
    stats["added"] = len(all_ids)

    return {"skipped": False, "existing": 0, **stats}

############################################################
# Retrieval + LLM Generation
############################################################

def build_patient_query(fields: Dict) -> str:
    parts = [
        f"Age: {fields.get('age')}",
        f"Weight(kg): {fields.get('weight_kg')}",
        f"Height(cm): {fields.get('height_cm')}",
        f"Diagnosis: {fields.get('diagnosis')}",
        f"Current Medications: {fields.get('current_medications')}",
        f"Allergies: {fields.get('allergies')}",
    ]
    return " | ".join([p for p in parts if p and str(p).strip() != "None"]).strip()


def retrieve_context(query: str, hf_api_key: str, top_k: int = TOP_K) -> List[Dict]:
    client = get_chroma_client()
    col = get_or_create_collection(client)
    if col.count() == 0:
        raise RuntimeError("Vector collection is empty. Please run 'Ingest Knowledge Base' first.")

    q_vec = get_embeddings([query], hf_api_key)[0]

    res = col.query(query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"])

    contexts = []
    for i in range(len(res["ids"][0])):
        contexts.append(
            {
                "text": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i],
            }
        )
    return contexts


def format_context(contexts: List[Dict]) -> str:
    blocks = []
    for c in contexts:
        src = c["metadata"].get("source", "unknown")
        ch = c["metadata"].get("chunk", "?")
        blocks.append(f"[Source: {src} | chunk {ch}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


def build_prompt(context_str: str, patient_query: str) -> str:
    system_rules = f"""
You are MediGuide, a medical patient-education assistant. Your job is to explain in **plain, empathetic language** while preserving clinical accuracy. Use only the provided context and general, non-controversial medical knowledge.

Hard rules:
- If context is insufficient, say what’s missing and ask for clinicians to confirm.
- NEVER invent citations. Cite only as [source: FILENAME | chunk N].
- Keep advice educational, not prescriptive. Encourage consulting a clinician.
- Prefer bullet points and short paragraphs.
- Tone: calm, supportive, culturally neutral.
    """.strip()

    user_prompt = f"""
PATIENT DATA (summarize, then tailor advice):
{patient_query}

CONTEXT (top {TOP_K} chunks):
{context_str}

TASK:
1) Summarize the patient info.
2) Provide an easy-to-understand explanation of the diagnosis and typical care pathways.
3) Call out red-flag symptoms that need urgent care.
4) Provide a **sample** daily plan (education, diet pointers, activity) appropriate for age/weight/height.
5) List sources using the given [source: ...] format only.
6) End with a short disclaimer.

Output should be concise, structured, and **grounded in the provided context**.
    """.strip()

    full = f"System:\n{system_rules}\n\nUser:\n{user_prompt}"
    return full


def call_llm(prompt: str, groq_api_key: str) -> str:
    client = Groq(api_key=groq_api_key)
    resp = client.chat.completions.create(
        model="llama3-8b-8192",  # Simulating Med-Gemma-4
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
    )
    return resp.choices[0].message.content

############################################################
# Simple Diet Visualization (placeholder demo)
############################################################

def diet_keyword_counts(text: str) -> Dict[str, int]:
    keys = {
        "carbohydrates": ["carb", "carbohydrate", "carbohydrates"],
        "protein": ["protein", "proteins"],
        "fats": ["fat", "fats", "lipid", "lipids"],
        "fiber": ["fiber", "fibre"],
    }
    text_l = text.lower()
    counts = {}
    for k, variants in keys.items():
        counts[k] = sum(text_l.count(v) for v in variants)
    return counts

############################################################
# Streamlit UI
############################################################
# --- Custom iPhone-style CSS ---
st.markdown("""
<style>
/* ----------------------
   General App Styling
   ---------------------- */
body, .stApp {
    font-family: -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Helvetica, Arial, sans-serif;
    background-color: #f5f5f7 !important;
    color: #111 !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    font-weight: 600 !important;
    color: #111 !important;
}

/* ----------------------
   Card-like containers
   ---------------------- */
.card {
    background-color: #ffffff !important;
    border-radius: 20px !important;
    padding: 20px !important;
    margin-bottom: 20px !important;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.08) !important;
    color: #111 !important;
}

.card div, .stMarkdown {
    color: #111 !important;
}

.card a {
    color: #007aff !important;
    text-decoration: none !important;
}

/* ----------------------
   Buttons
   ---------------------- */
.stButton>button {
    border-radius: 12px !important;
    padding: 10px 20px !important;
    font-size: 16px !important;
    background-color: #007aff !important;
    color: white !important;
    font-weight: 600 !important;
    border: none !important;
}

.stButton>button:hover,
.stButton>button:focus,
.stButton>button:active {
    background-color: #0051a8 !important;
    color: white !important;
}

/* ----------------------
   Sidebar
   ---------------------- */
.sidebar .sidebar-content {
    border-radius: 20px !important;
    background-color: #ffffff !important;
    color: #111 !important;
    padding: 15px !important;
}

/* ----------------------
   Expander
   ---------------------- */
.st-expander,
.st-expander .st-expanderHeader,
.st-expander .st-expanderContent {
    background-color: #ffffff !important;
    color: #111 !important;
    border-radius: 15px !important;
    padding: 10px !important;
    box-shadow: 0px 2px 10px rgba(0,0,0,0.05) !important;
}

.st-expander .st-expanderHeader:hover,
.st-expander .st-expanderHeader:focus {
    background-color: #f0f0f0 !important;
    color: #111 !important;
}

/* ----------------------
   Markdown inside cards or expanders
   ---------------------- */
.stMarkdown,
.stExpanderContent,
.card div {
    color: #111 !important;
    line-height: 1.5 !important;
}

/* ----------------------
   Tables & Metrics
   ---------------------- */
.stMetricLabel, .stMetricValue {
    color: #111 !important;
}

/* ----------------------
   Bar chart text (optional)
   ---------------------- */
svg text {
    fill: #111 !important;
}

</style>
""", unsafe_allow_html=True)

# --- App UI ---

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    
    # Header
    st.markdown('<div class="card"><h1>🩺 MediGuide</h1>'
                '<p>Cloud-powered RAG: HF Embeddings + ChromaDB + Groq LLaMA 3.1</p></div>', unsafe_allow_html=True)

    # Keys
    hf_key, groq_key = load_keys()

    # About demo
    with st.expander("ℹ️ About this demo", expanded=False):
        st.markdown("""
        **Architecture**: No local models. Embeddings via Hugging Face Inference API. Vectors in local ChromaDB. Generation via Groq LLaMA 3.1.
        
        **Security**:
        - Keys in **st.secrets** (prod) or **.env** (dev)
        - Sidebar override available
        
        **Workflow**:
        1. Add trusted `.txt` files into `./knowledge_base/`.
        2. Click **Ingest Knowledge Base**.
        3. Enter patient info → Click **Run RAG**
        """)

    # Knowledge Base ingestion card
    st.markdown('<div class="card"><h3>1️⃣ Knowledge Base</h3></div>', unsafe_allow_html=True)
    col_ingest, col_status = st.columns([1, 1])
    with col_ingest:
        if st.button("📥 Ingest Knowledge Base", use_container_width=True):
            if not hf_key:
                st.error("HUGGING_FACE_API_KEY is required to embed documents.")
            else:
                with st.spinner("Ingesting…"):
                    stats = ingest_documents_streaming(hf_key)
                    if stats.get("skipped"):
                        st.success(f"Collection already populated ({stats.get('existing',0)} vectors). Skipped.")
                    else:
                        st.success(f"Added {stats['added']} chunks from {stats['files']} files.")

    with col_status:
        try:
            client = get_chroma_client()
            col = get_or_create_collection(client)
            st.metric("Vector count", value=col.count())
        except Exception as e:
            st.warning(f"Vector DB not initialized: {e}")

    # Patient input card
    st.markdown('<div class="card"><h3>2️⃣ Patient Inputs</h3></div>', unsafe_allow_html=True)
    with st.form("patient_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 0, 120, 30)
            weight_kg = st.number_input("Weight (kg)", 0.0, 500.0, 70.0, step=0.5)
        with c2:
            height_cm = st.number_input("Height (cm)", 0.0, 250.0, 170.0, step=0.5)
            diagnosis = st.text_input("Diagnosis", "Type 2 Diabetes")
        with c3:
            current_medications = st.text_input("Current Medications", "Metformin 500mg BID")
            allergies = st.text_input("Allergies", "None known")
        submitted = st.form_submit_button("🚀 Run RAG", use_container_width=True)

    # RAG execution
    if submitted:
        if not hf_key or not groq_key:
            st.error("API keys missing. Add in sidebar.")
            st.stop()

        patient_fields = {
            "age": age, "weight_kg": weight_kg, "height_cm": height_cm,
            "diagnosis": diagnosis, "current_medications": current_medications,
            "allergies": allergies
        }

        try:
            with st.spinner("Retrieving context…"):
                query = build_patient_query(patient_fields)
                contexts = retrieve_context(query, hf_key)
                context_str = format_context(contexts)

            prompt = build_prompt(context_str, query)

            with st.spinner("Generating guidance…"):
                output = call_llm(prompt, groq_key)

            # Output card
            st.markdown(f'<div class="card"><h3>3️⃣ Output</h3><div>{output}</div></div>', unsafe_allow_html=True)

            # Diet chart
            counts = diet_keyword_counts(output)
            st.markdown('<div class="card"><h4>🍎 Keyword Hits (diet placeholders)</h4></div>', unsafe_allow_html=True)
            st.bar_chart(counts)

            # Retrieved chunks
            with st.expander("📄 Show retrieved context chunks"):
                for i, c in enumerate(contexts, start=1):
                    st.markdown(f"**Top {i}** — distance: {c['distance']:.4f}")
                    st.code(c['text'])
                    st.caption(f"{c['metadata']}")
        except Exception as e:
            st.exception(e)

    # Disclaimer
    st.markdown('<div class="card"><p>⚠️ Disclaimer: General education only. Consult a licensed clinician for treatment.</p></div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()