"""Interface web do NotesId (backend FastAPI + WebSocket e frontend em pt-BR)."""

from guitar_string_identifier.web.app import app, create_app, run_server

__all__ = ["app", "create_app", "run_server"]
