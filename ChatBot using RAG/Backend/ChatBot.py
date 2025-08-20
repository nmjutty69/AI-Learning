import os
import uuid
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from groq import Groq

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
GEN_MODEL = os.getenv("GEN_MODEL")
groq_client = Groq(api_key=GROQ_API_KEY)

# Eleven Labs Setup
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID")

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

        return [
            {"text": results["documents"][0][i], "score": results["distances"][0][i]}
            for i in range(len(results["ids"][0]))
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

# -----------------------
# Routes
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "message": "Backend is running 🚀"}

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())

    # RAG retrieval
    retrieved, context_snippet = [], ""
    if req.use_rag:
        retrieved = store.search(req.message, top_k=TOP_K)
        context_snippet = "\n".join([r["text"] for r in retrieved])

    # Build prompt
    history = get_history(session_id)
    history_text = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])
    prompt = (
        f"Context:\n{context_snippet}\n\n"
        f"History:\n{history_text}\n\n"
        f"User: {req.message}\nAssistant:"
    )

    # Generate response via Groq
    try:
        completion = groq_client.chat.completions.create(
            model=GEN_MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "developer", "content": "In Sentences. Max words should be 100"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            max_tokens=200,
        )
        reply = completion.choices[0].message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Groq API error: {str(e)}")

    # Save memory
    append_history(session_id, "user", req.message)
    append_history(session_id, "assistant", reply)

    return ChatResponse(reply=reply, retrieved=retrieved, session_id=session_id)


@app.post("/speak", response_model=ChatResponse)
async def speak(file: UploadFile = File(...), session_id: Optional[str] = None, use_rag: Optional[bool] = True):
    session_id = session_id or str(uuid.uuid4())

    # 1️⃣ Convert audio to text
    recognizer = sr.Recognizer()
    audio_text = ""
    try:
        with sr.AudioFile(file.file) as source:
            audio_data = recognizer.record(source)
            audio_text = recognizer.recognize_google(audio_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Speech recognition failed: {str(e)}")

    # 2️⃣ Pass text to your existing chat function logic
    chat_req = ChatRequest(message=audio_text, session_id=session_id, use_rag=use_rag)
    chat_resp = chat(chat_req)

    # 3️⃣ Convert assistant reply to speech using ElevenLabs
    try:
        tts_response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{ELEVENLABS_VOICE_ID}",
            headers={
                "xi-api-key": ELEVENLABS_API_KEY,
                "Content-Type": "application/json",
            },
            json={"text": chat_resp.reply, "voice_settings": {"stability": 0.5, "similarity_boost": 0.75}},
        )
        tts_response.raise_for_status()
        audio_bytes = tts_response.content
        # Save audio to file (optional)
        with open(f"output_{session_id}.mp3", "wb") as f:
            f.write(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ElevenLabs TTS error: {str(e)}")

    # 4️⃣ Return chat reply + retrieved context
    return ChatResponse(reply=chat_resp.reply, retrieved=chat_resp.retrieved, session_id=session_id)