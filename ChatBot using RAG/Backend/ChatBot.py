import os
import uuid
import io
import tempfile
from typing import List, Optional, Dict, Any, Tuple

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq
import requests

# -----------------------
# Load Environment
# -----------------------
load_dotenv()

TOP_K = int(os.getenv("RAG_TOP_K", 4))
MAX_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 6))
CHROMA_DB_PATH = os.getenv("CHROMA_DB", "chroma_db")
EMB_MODEL = os.getenv("EMB_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Groq API setup
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEN_MODEL = os.getenv("GEN_MODEL", "llama3-8b-8192")  # fallback
groq_client = Groq(api_key=GROQ_API_KEY)

# Eleven Labs Setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")
ELEVENLABS_API_URL = os.getenv("ELEVENLABS_API_URL", "https://api.elevenlabs.io/v1/text-to-speech")
ELEVENLABS_MODEL_ID = os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1")
ELEVENLABS_STABILITY = float(os.getenv("ELEVENLABS_STABILITY", 0.4))
ELEVENLABS_SIMILARITY = float(os.getenv("ELEVENLABS_SIMILARITY", 0.8))

# -----------------------
# App Setup
# -----------------------
app = FastAPI(title="Tourist AI ChatBot")
app.add_middleware(
    CORSMiddleware,

    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding model
embedder = SentenceTransformer(EMB_MODEL)

# Vector store (Chroma)
chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="tour_docs")

class ChromaStore:
    def __init__(self, collection):
        self.collection = collection

    def search(self, query: str, top_k: int = 4):
        qemb = embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).tolist()[0]

        results = self.collection.query(
            query_embeddings=[qemb],
            n_results=top_k
        )

        docs = results.get("documents", [[]])[0]
        dists = results.get("distances", [[]])[0]
        return [
            {"text": docs[i], "score": dists[i]}
            for i in range(len(docs))
        ]

store = ChromaStore(collection)

# -----------------------
# Memory
# -----------------------
MEMORY: Dict[str, List[Dict[str, str]]] = {}

def get_history(session_id: str):
    return MEMORY.get(session_id, [])[-MAX_TURNS:]

def append_history(session_id: str, role: str, content: str):
    MEMORY.setdefault(session_id, []).append({"role": role, "content": content})

# -----------------------
# Schemas
# -----------------------
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    use_rag: Optional[bool] = True

class ChatResponse(BaseModel):
    reply: str
    retrieved: Optional[List[Dict[str, Any]]] = None
    session_id: str

class VoiceRequest(BaseModel):
    text: str

class VoiceChatResponse(ChatResponse):
    transcript: str

# -----------------------
# Helper: generate reply
# -----------------------
def generate_reply(message: str, session_id: str, use_rag: bool = True) -> Tuple[str, List[Dict[str, Any]]]:
    # RAG retrieval
    retrieved, context_snippet = [], ""
    if use_rag:
        retrieved = store.search(message, top_k=TOP_K)
        context_snippet = "\n".join([r["text"] for r in retrieved])

    # Build prompt
    history = get_history(session_id)
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])
    prompt = (
        f"Context:\n{context_snippet}\n\n"
        f"History:\n{history_text}\n\n"
        f"User: {message}\nAssistant:"
    )

    # Generate response via Groq
    try:
        completion = groq_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "developer", "content": "Use short sentences. Max ~100 words."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=200,
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    # Save memory
    append_history(session_id, "user", message)
    append_history(session_id, "assistant", reply)
    return reply, retrieved

# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    reply, retrieved = generate_reply(req.message, session_id, use_rag=req.use_rag)
    return ChatResponse(reply=reply, retrieved=retrieved, session_id=session_id)


@app.post("/speak")
def voice(req: VoiceRequest):
    if not ELEVENLABS_API_KEY or not ELEVENLABS_VOICE_ID:
        raise HTTPException(status_code=500, detail="ElevenLabs API key or voice ID not configured.")

    try:
        sts_api = f"{ELEVENLABS_API_URL}/{ELEVENLABS_VOICE_ID}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": ELEVENLABS_API_KEY,
        }
        payload = {
            "text": req.text,
            "model_id": os.getenv("ELEVENLABS_MODEL_ID", "eleven_monolingual_v1"),
            "voice_settings": {
                "stability": float(os.getenv("ELEVENLABS_STABILITY", 0.4)),
                "similarity_boost": float(os.getenv("ELEVENLABS_SIMILARITY", 0.8)),
            },
        }

        response = requests.post(sts_api, headers=headers, json=payload)
        response.raise_for_status()

        return StreamingResponse(io.BytesIO(response.content), media_type="audio/mpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")
