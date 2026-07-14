from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime

class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique ID for the conversation session")
    message: str = Field(..., min_length=1, description="User's message")
    stream: bool = Field(default=False, description="Whether to stream the response")

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turn_count: int
    tokens_used: Optional[int] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())

class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class HistoryResponse(BaseModel):
    session_id: str
    messages: List[HistoryMessage]
    total_turns: int

class SessionListResponse(BaseModel):
    active_sessions: List[str]
    count: int
