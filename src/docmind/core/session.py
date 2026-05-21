"""In-memory session store for multi-turn conversation memory.

Each session holds a rolling window of (question, answer) turns. When a
session_id is provided on /api/ask, the last N turns are injected into the
LLM prompt so follow-up questions can reference prior context.

Why in-memory? The spec does not require persistence across server restarts
for this iteration, and avoiding a database dependency keeps the deployment
simple. Sessions are naturally ephemeral -- they expire when the server
restarts or when the user clears them.

Thread safety: the store uses a simple dict. For multi-worker deployments,
replace with Redis-backed storage.
"""

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime


@dataclass
class Turn:
    """A single question/answer exchange in a session."""

    question: str
    answer: str


@dataclass
class Session:
    """A conversation session with a rolling history of turns."""

    id: str
    created_at: str
    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append(Turn(question=question, answer=answer))

    def recent_turns(self, window: int) -> list[Turn]:
        """Return the last `window` turns."""
        return self.turns[-window:] if window > 0 else []


class SessionStore:
    """In-memory registry of active sessions."""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    def create(self) -> Session:
        session = Session(
            id=str(uuid.uuid4()),
            created_at=datetime.now(tz=UTC).isoformat(),
        )
        self._sessions[session.id] = session
        return session

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> Session:
        """Return an existing session or create a new one.

        If session_id is None, always creates a new session.
        If session_id refers to an unknown session, creates a new session
        with that id (allows clients to name their own sessions).
        """
        if session_id is None:
            return self.create()
        if session_id not in self._sessions:
            session = Session(
                id=session_id,
                created_at=datetime.now(tz=UTC).isoformat(),
            )
            self._sessions[session_id] = session
        return self._sessions[session_id]


session_store = SessionStore()
