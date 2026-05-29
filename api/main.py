import json
import os
from pathlib import Path
from typing import Optional

from google import genai as google_genai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

from pipeline.loader import load_messages
from pipeline.checkpoints import build_checkpoints, load_checkpoints
from pipeline.retriever import Retriever
from pipeline.persona import build_persona, load_persona, persona_to_paragraph

BASE_DIR         = Path(__file__).parent.parent
DATA_CSV         = BASE_DIR / "data" / "conversations.csv"
CHECKPOINTS_JSON = BASE_DIR / "checkpoints.json"
PERSONA_JSON     = BASE_DIR / "persona.json"
INDEX_DIR        = BASE_DIR / "indices"

GEMINI_CHAT_MODEL = "gemini-2.0-flash"

app = FastAPI(
    title="KaStack RAG + Persona API",
    description="RAG system with persona extraction over conversation data.",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

_retriever: Optional[Retriever] = None
_persona:   Optional[dict]      = None
_checkpoints_data: Optional[dict] = None
_messages                         = None
_build_in_progress                = False

_PERSONA_KEYWORDS = [
    "what kind of person","personality","habits","behavior","behaviour",
    "how do they talk","communication style","how does this person",
    "character","trait","emoji","slang","routines","typical","usually",
    "what are they like","who is","describe them","describe the person",
]

def _is_persona_query(query: str) -> bool:
    q = query.lower()
    return any(kw in q for kw in _PERSONA_KEYWORDS)

@app.on_event("startup")
async def startup_event():
    global _retriever, _persona, _checkpoints_data, _messages

    print("[startup] Loading messages...")
    _messages = load_messages(DATA_CSV, limit=1000)

    if PERSONA_JSON.exists():
        _persona = load_persona(PERSONA_JSON)
        print("[startup] persona.json loaded.")
    else:
        print("[startup] persona.json not found — run POST /build.")

    _retriever = Retriever()
    if _retriever.load(INDEX_DIR):
        print("[startup] FAISS indices loaded.")
    else:
        print("[startup] No indices found — run POST /build.")
        _retriever = None

    if CHECKPOINTS_JSON.exists():
        _checkpoints_data = load_checkpoints(CHECKPOINTS_JSON)
        print("[startup] checkpoints.json loaded.")
    else:
        print("[startup] checkpoints.json not found — run POST /build.")

    print("[startup] Ready.")

class ChatRequest(BaseModel):
    query: str
    include_sources: bool = False

class ChatResponse(BaseModel):
    answer: str
    mode: str
    sources: Optional[dict] = None

class BuildRequest(BaseModel):
    api_key: Optional[str] = None
    rebuild: bool = False

def _generate_offline_answer(query: str, context: str, mode: str, persona_summary: str) -> str:
    lines = []
    lines.append("📢 **Offline Demonstration Mode** *(Gemini API Key Quota Exceeded)*\n")
    
    q_words = [w.lower().strip("?,.!") for w in query.split() if len(w) > 3]
    
    if mode == "persona":
        lines.append("Based on the analyzed multi-pass **Persona Data**, here are the details related to your query:\n")
        try:
            persona_dict = json.loads(context)
            facts = persona_dict.get("facts", {})
            habits = persona_dict.get("habits", {})
            traits = persona_dict.get("personality", [])
            comm = persona_dict.get("communication_style", {})
            
            matched = False
            for k, val in facts.items():
                if any(w in k.lower() or any(w in str(v).lower() for v in val) for w in q_words):
                    lines.append(f"• **{k.capitalize()}**: {', '.join(val)}")
                    matched = True
            for k, val in habits.items():
                if any(w in k.lower() or w in str(val).lower() for w in q_words):
                    if isinstance(val, list):
                        lines.append(f"• **{k.capitalize()}**: {', '.join(val)}")
                    else:
                        lines.append(f"• **{k.capitalize()}**: {val}")
                    matched = True
            
            if any(w in "personality traits" for w in q_words):
                lines.append(f"• **Personality Traits**: {', '.join(traits)}")
                matched = True
            if any(w in "talk communication style tone" for w in q_words):
                lines.append(f"• **Communication Style**: {comm.get('tone', '')} (average length: {comm.get('avg_message_length', '')} chars, emoji usage: {comm.get('emoji_usage', '')})")
                matched = True
                
            if not matched:
                lines.append(f"• **Extracted Facts**: {', '.join(facts.get('locations', []))}")
                lines.append(f"• **Food & Habits**: {', '.join(habits.get('food', []))}")
                lines.append(f"• **Routine Activities**: {', '.join(habits.get('routines', []))}")
        except Exception:
            lines.append(persona_summary)
    else:
        lines.append("Successfully retrieved matching segments from the **FAISS Vector Index**:\n")
        lines.append(context)
        lines.append("\n*Note: To restore full AI-generated synthesis, please configure a Gemini API key with active quota in your `.env` file.*")
        
    return "\n".join(lines)

def _generate_answer(query: str, context: str, mode: str, persona_summary: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    try:
        client = google_genai.Client(api_key=api_key)

        system_instruction = (
            "You are an assistant helping the user understand conversation data. "
            f"Background on the participants: {persona_summary}\n\n"
            "Answer questions based on the provided context. Be concise and accurate. "
            "If the context doesn't contain enough information, say so clearly."
        )

        if mode == "persona":
            user_content = f"Using the persona information below, answer: {query}\n\nPERSONA DATA:\n{context}"
        else:
            user_content = f"Using the context below, answer: {query}\n\n{context}"

        from google.genai import types
        response = client.models.generate_content(
            model=GEMINI_CHAT_MODEL,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                max_output_tokens=1024,
            ),
        )
        return response.text.strip()
    except Exception as e:
        print(f"[chat] Gemini API call failed: {e}. Falling back to Offline Simulation Mode.")
        return _generate_offline_answer(query, context, mode, persona_summary)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not _persona and not _retriever:
        raise HTTPException(status_code=503, detail="Pipeline not built. POST /build first.")

    persona_summary = persona_to_paragraph(_persona) if _persona else "No persona data."

    if _is_persona_query(req.query):
        context = json.dumps(_persona, indent=2) if _persona else "No persona data."
        answer  = _generate_answer(req.query, context, "persona", persona_summary)
        return ChatResponse(answer=answer, mode="persona")
    else:
        if not _retriever:
            raise HTTPException(status_code=503, detail="RAG index not built. POST /build first.")
        context_str, topic_hits, chunk_hits = _retriever.query(req.query)
        answer = _generate_answer(req.query, context_str, "rag", persona_summary)
        sources = None
        if req.include_sources:
            sources = {"topic_summaries": topic_hits, "raw_chunks": chunk_hits}
        return ChatResponse(answer=answer, mode="rag", sources=sources)

@app.get("/persona")
async def get_persona():
    if not _persona:
        raise HTTPException(status_code=404, detail="No persona. POST /build first.")
    return _persona

@app.get("/checkpoints")
async def get_checkpoints():
    if not _checkpoints_data:
        raise HTTPException(status_code=404, detail="No checkpoints. POST /build first.")
    return {
        "topic_checkpoints":      _checkpoints_data.get("topic_checkpoints", []),
        "structural_checkpoints": _checkpoints_data.get("structural_checkpoints", []),
        "stats":                  _checkpoints_data.get("stats", {}),
    }

@app.post("/build")
async def build_pipeline(req: BuildRequest, background_tasks: BackgroundTasks):
    global _build_in_progress
    if _build_in_progress:
        return {"status": "Build already running. Monitor server logs."}

    api_key = req.api_key or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY required.")

    async def _run():
        global _retriever, _persona, _checkpoints_data, _build_in_progress
        _build_in_progress = True
        try:
            messages = _messages or load_messages(DATA_CSV, limit=1000)

            if not req.rebuild and CHECKPOINTS_JSON.exists():
                cp_data = load_checkpoints(CHECKPOINTS_JSON)
            else:
                cp_data = build_checkpoints(messages, CHECKPOINTS_JSON, api_key)
            _checkpoints_data = cp_data

            retriever = Retriever()
            if not req.rebuild and (INDEX_DIR / "topic.index").exists():
                retriever.load(INDEX_DIR)
            else:
                retriever.build_topic_index(cp_data.get("topic_checkpoints", []))
                retriever.build_chunk_index(messages)
                retriever.save(INDEX_DIR)
            _retriever = retriever

            if not req.rebuild and PERSONA_JSON.exists():
                _persona = load_persona(PERSONA_JSON)
            else:
                _persona = build_persona(messages, PERSONA_JSON, api_key)

            print("[build] Complete.")
        except Exception as e:
            print(f"[build] ERROR: {e}")
            raise
        finally:
            _build_in_progress = False

    background_tasks.add_task(_run)
    return {"status": "Build started in background. Monitor server logs."}

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "indices_loaded":      _retriever is not None,
        "persona_loaded":      _persona is not None,
        "checkpoints_loaded":  _checkpoints_data is not None,
        "build_in_progress":   _build_in_progress,
    }

frontend_dir = BASE_DIR / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(str(frontend_dir / "index.html"))
