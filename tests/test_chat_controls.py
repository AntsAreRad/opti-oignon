#!/usr/bin/env python3
"""Tests pour les controles de chat S42 (think, web_search)."""

import pytest

from opti_oignon.api.schemas import ChatRequest, ChatToken


def test_chat_request_think_field():
    """Verifie que le champ think est accepte dans ChatRequest."""
    req = ChatRequest(message="test", think=True)
    assert req.think is True


def test_chat_request_web_search_field():
    """Verifie que le champ web_search est accepte dans ChatRequest."""
    req = ChatRequest(message="test", web_search=True)
    assert req.web_search is True


def test_chat_request_defaults():
    """Verifie les valeurs par defaut des nouveaux champs."""
    req = ChatRequest(message="test")
    assert req.think is None
    assert req.web_search is None


def test_chat_request_backward_compatible():
    """Verifie que les requetes existantes fonctionnent toujours."""
    req = ChatRequest(
        conversation_id="abc",
        message="hello",
        model="qwen3:32b",
        preset="code",
        temperature=0.5,
        use_presets=True,
    )
    assert req.message == "hello"
    assert req.model == "qwen3:32b"
    assert req.think is None
    assert req.web_search is None


def test_chat_request_all_fields():
    """Verifie que tous les champs fonctionnent ensemble."""
    req = ChatRequest(
        conversation_id="conv123",
        message="Explain this R code",
        model="qwen3-coder:30b",
        preset="code",
        temperature=0.2,
        use_presets=False,
        think=True,
        web_search=False,
    )
    assert req.think is True
    assert req.web_search is False
    assert req.use_presets is False


def test_chat_request_think_false():
    """Verifie que think=False est distinct de None."""
    req = ChatRequest(message="test", think=False)
    assert req.think is False


def test_chat_request_web_search_false():
    """Verifie que web_search=False est distinct de None."""
    req = ChatRequest(message="test", web_search=False)
    assert req.web_search is False


def test_chat_token_thinking_type():
    """Verifie que ChatToken accepte le type 'thinking' (S42)."""
    token = ChatToken(type="thinking", content="Let me reason about this...")
    assert token.type == "thinking"
    assert token.content == "Let me reason about this..."


def test_chat_token_standard_types():
    """Verifie que les types standard fonctionnent toujours."""
    for token_type in ["token", "done", "error", "metadata"]:
        token = ChatToken(type=token_type, content="test")
        assert token.type == token_type


def test_chat_request_serialization():
    """Verifie la serialisation JSON avec les nouveaux champs."""
    req = ChatRequest(message="test", think=True, web_search=True)
    data = req.model_dump()
    assert data["think"] is True
    assert data["web_search"] is True
    assert data["message"] == "test"


def test_chat_request_from_dict():
    """Verifie la construction depuis un dict (simule WebSocket)."""
    raw = {
        "message": "hello",
        "conversation_id": "abc",
        "think": True,
        "web_search": False,
    }
    req = ChatRequest(**raw)
    assert req.think is True
    assert req.web_search is False


def test_chat_request_from_dict_without_new_fields():
    """Verifie la construction depuis un dict sans les nouveaux champs."""
    raw = {
        "message": "hello",
        "conversation_id": "abc",
    }
    req = ChatRequest(**raw)
    assert req.think is None
    assert req.web_search is None
