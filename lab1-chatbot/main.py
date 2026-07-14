from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from models import ChatRequest, ChatResponse, HistoryResponse, HistoryMessage, SessionListResponse
from memory import memory
from llm_client import get_chat_completion, stream_chat_completion

app = FastAPI(title="Chatbot API", version="1.0")

@app.get("/")
def root():
    return {"status": "ok", "message": "Chatbot API is running"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    """Main chat endpoint. Adds user message, gets reply, stores it, returns response."""
    if request.stream:
        raise HTTPException(400, "Use /chat/stream for streaming responses")
    # 1. Save user message to memory
    memory.add_message(request.session_id, "user", request.message)
    # 2. Build history for the API call
    history = memory.get_history_for_api(request.session_id)
    # 3. Call the LLM
    result = get_chat_completion(history)
    # 4. Save assistant reply to memory
    memory.add_message(request.session_id, "assistant", result["reply"])

    return ChatResponse(
        session_id=request.session_id,
        reply=result["reply"],
        turn_count=memory.turn_count(request.session_id),
        tokens_used=result["tokens_used"]
    )

@app.post("/chat/stream")
def chat_stream(request: ChatRequest):
    """Streaming version — yields tokens as they arrive."""
    memory.add_message(request.session_id, "user", request.message)
    history = memory.get_history_for_api(request.session_id)

    def generate():
        full_reply = ""
        for token in stream_chat_completion(history):
            full_reply += token
            yield token
        memory.add_message(request.session_id, "assistant", full_reply)

    return StreamingResponse(generate(), media_type="text/plain")

@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    raw = memory.get_history(session_id)
    if not raw:
        raise HTTPException(404, "Session not found")
    return HistoryResponse(
        session_id=session_id,
        messages=[HistoryMessage(**m) for m in raw],
        total_turns=memory.turn_count(session_id)
    )

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    memory.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}

@app.get("/sessions", response_model=SessionListResponse)
def list_sessions():
    sessions = memory.list_sessions()
    return SessionListResponse(active_sessions=sessions, count=len(sessions))
