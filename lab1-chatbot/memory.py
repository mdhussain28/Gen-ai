from collections import defaultdict
from datetime import datetime
from typing import List, Dict
import os

MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", 10))


class ConversationMemory:
    """
    In-memory store for chat sessions.
    Each session_id maps to a list of {role, content, timestamp} dicts.

    NOTE: This resets when the server restarts. For production,
    swap this out for Redis, Postgres, or SQLite (see notes at bottom).
    """

    def __init__(self, max_turns: int = MAX_HISTORY_TURNS):
        self.sessions: Dict[str, List[dict]] = defaultdict(list)
        self.max_turns = max_turns  # max USER+ASSISTANT pairs kept

    def add_message(self, session_id: str, role: str, content: str):
        self.sessions[session_id].append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat()
        })
        self._trim(session_id)

    def _trim(self, session_id: str):
        """Keep only the last N turns to control token usage / cost."""
        max_messages = self.max_turns * 2  # user + assistant per turn
        if len(self.sessions[session_id]) > max_messages:
            self.sessions[session_id] = self.sessions[session_id][-max_messages:]

    def get_history(self, session_id: str) -> List[dict]:
        return self.sessions.get(session_id, [])

    def get_history_for_api(self, session_id: str) -> List[dict]:
        """Return history in the {role, content} format LLM APIs expect."""
        return [
            {"role": m["role"], "content": m["content"]}
            for m in self.sessions.get(session_id, [])
        ]

    def clear_session(self, session_id: str):
        if session_id in self.sessions:
            del self.sessions[session_id]

    def list_sessions(self) -> List[str]:
        return list(self.sessions.keys())

    def turn_count(self, session_id: str) -> int:
        return len(self.sessions.get(session_id, [])) // 2


# Singleton instance shared across the app
memory = ConversationMemory()
