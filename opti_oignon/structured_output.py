#!/usr/bin/env python3
"""
STRUCTURED OUTPUT ENGINE - OPTI-OIGNON v1.5.0
==============================================

Moteur de generation de sorties structurees via Ollama.

Utilise le parametre format= d'Ollama pour contraindre la sortie
au schema JSON defini par un modele Pydantic, avec validation
automatique et boucle de retry en cas d'echec.

Author: Leon
"""

import json
import logging
import time
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

# The client library's presence, kept for callers that read the flag. The
# request goes through the registry; no client is imported here.
try:
    import importlib.util as _importlib_util
    OLLAMA_AVAILABLE = _importlib_util.find_spec("ollama") is not None
except Exception:  # noqa: BLE001 - presence is a fact, not a dependency
    OLLAMA_AVAILABLE = False

try:
    from .config import config  # noqa: F401
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False

logger = logging.getLogger(__name__)

T = TypeVar('T', bound=BaseModel)


def _resolve_backend(model: str):
    """The registry's backend for ``model``, or None when there is none.

    Structured output used to call the client library directly, which is
    where the schema channel on the registry came from in the first place.
    It goes through the registry now, and the schema travels as the engine
    option the registry already understands.
    """
    try:
        from opti_oignon.inference_backend import get_backend_registry
    except Exception as exc:  # noqa: BLE001 - absence is an answer here
        logger.debug("Inference registry unavailable: %s", exc)
        return None
    try:
        return get_backend_registry().resolve_backend(model)
    except Exception as exc:  # noqa: BLE001 - a broken registry is absence
        logger.debug("Inference registry could not resolve %s: %s", model, exc)
        return None


def _message_field(response: Any, field: str) -> str:
    """Pull a message field from an ollama chat response, dict or object form.

    The engine assumed the object form
    (response.message.content); a dict-form ollama-python response raised an
    AttributeError caught by the generic handler, which broke the retry
    loop. The inverse of the RSN-01 / SCR-01 dict-only sites -- both
    conventions coexisted in the codebase (the dict-vs-object
    class).
    """
    if response is None:
        return ""
    if isinstance(response, dict):
        message = response.get("message")
    else:
        message = getattr(response, "message", None)
    if message is None:
        return ""
    if isinstance(message, dict):
        return str(message.get(field) or "")
    return str(getattr(message, field, "") or "")


class StructuredResult(BaseModel):
    """Result of a structured generation."""
    success: bool
    data: Any | None = None      # Le modele Pydantic valide
    raw_response: str = ""           # Reponse brute du LLM
    attempts: int = 1                # Nombre de tentatives
    errors: list[str] = []           # Erreurs rencontrees
    model: str = ""
    duration_ms: int = 0
    thinking: str = ""               # Contenu de reflexion si think=True


class StructuredOutputEngine:
    """Moteur de sortie structuree avec validation et retry.

    Utilise Ollama format= pour contraindre la sortie JSON,
    then validates with Pydantic. Retry with error feedback
    si la validation echoue.
    """

    def __init__(
        self,
        default_model: str = "qwen3:32b",
        max_retries: int = 3,
        default_temperature: float = 0.0,
    ):
        self.default_model = default_model
        self.max_retries = max_retries
        self.default_temperature = default_temperature

    def generate_structured(
        self,
        messages: list[dict],
        schema: type[T],
        model: str | None = None,
        max_retries: int | None = None,
        temperature: float | None = None,
        think: bool = False,
        extra_system_prompt: str = "",
    ) -> StructuredResult:
        """Generate a structured response conforming to the Pydantic schema.

        Args:
            messages: Historique de conversation (format Ollama)
            schema: Classe Pydantic definissant le schema attendu
            model: Modele Ollama (defaut: self.default_model)
            max_retries: Nombre max de tentatives (defaut: self.max_retries)
            temperature: Temperature (defaut: self.default_temperature)
            think: Activer le mode reflexion (chain-of-thought)
            extra_system_prompt: Instructions supplementaires pour le prompt

        Returns:
            StructuredResult with the validated Pydantic model or errors
        """
        _model = model or self.default_model
        backend = _resolve_backend(_model)
        if backend is None:
            return StructuredResult(
                success=False,
                errors=["no inference backend is registered"],
            )
        _max_retries = max_retries if max_retries is not None else self.max_retries
        _temperature = temperature if temperature is not None else self.default_temperature

        # Preparer le schema JSON pour le prompt
        json_schema = schema.model_json_schema()
        schema_description = self._format_schema_description(schema, json_schema)

        # Ajouter les instructions de schema au prompt
        enhanced_messages = self._inject_schema_instructions(
            messages, schema_description, extra_system_prompt
        )

        errors = []
        start_time = time.time()
        thinking_content = ""
        raw_content = ""

        for attempt in range(1, _max_retries + 1):
            try:
                # Through the registry, with the schema as an engine option.
                response = backend.generate(
                    model=_model,
                    messages=enhanced_messages,
                    options={"temperature": _temperature, "schema": json_schema},
                    think=think,
                )

                raw_content = response.content or ""
                thinking_value = response.thinking or ""
                if thinking_value:
                    thinking_content = thinking_value

                # Validation Pydantic
                validated = schema.model_validate_json(raw_content)

                duration = int((time.time() - start_time) * 1000)
                return StructuredResult(
                    success=True,
                    data=validated,
                    raw_response=raw_content,
                    attempts=attempt,
                    errors=errors,
                    model=_model,
                    duration_ms=duration,
                    thinking=thinking_content,
                )

            except ValidationError as e:
                error_msg = f"Tentative {attempt}: Erreur de validation Pydantic: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

                # Ajouter le feedback d'erreur pour le retry
                if attempt < _max_retries:
                    enhanced_messages = self._add_error_feedback(
                        enhanced_messages, raw_content, str(e), schema_description
                    )

            except json.JSONDecodeError as e:
                error_msg = f"Tentative {attempt}: JSON invalide: {str(e)}"
                errors.append(error_msg)
                logger.warning(error_msg)

                if attempt < _max_retries:
                    enhanced_messages = self._add_json_error_feedback(
                        enhanced_messages, str(e), schema_description
                    )

            except Exception as e:
                error_msg = f"Tentative {attempt}: Erreur inattendue: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
                break  # Erreur critique, pas de retry

        # Toutes les tentatives echouees
        duration = int((time.time() - start_time) * 1000)
        return StructuredResult(
            success=False,
            raw_response=raw_content,
            attempts=_max_retries,
            errors=errors,
            model=_model,
            duration_ms=duration,
            thinking=thinking_content,
        )

    def _format_schema_description(self, schema: type[BaseModel], json_schema: dict) -> str:
        """Generate a readable description of the schema for the prompt."""
        fields = []
        properties = json_schema.get("properties", {})
        required = json_schema.get("required", [])

        for name, prop in properties.items():
            field_type = prop.get("type", "unknown")
            description = prop.get("description", "")
            is_required = name in required
            req_marker = " (REQUIRED)" if is_required else " (optional)"

            # Gerer les enums
            if "enum" in prop:
                field_type = f"one of: {prop['enum']}"

            fields.append(f"  - {name}: {field_type}{req_marker}"
                         + (f" -- {description}" if description else ""))

        return (
            f"Respond with a JSON object matching this schema:\n"
            f"Schema: {schema.__name__}\n"
            + "\n".join(fields)
        )

    def _inject_schema_instructions(
        self, messages: list[dict], schema_description: str, extra: str
    ) -> list[dict]:
        """Injecte les instructions de schema dans les messages."""
        enhanced = list(messages)

        # Trouver ou creer le message systeme
        system_instruction = (
            "You MUST respond with valid JSON only. No markdown, no explanation, no preamble.\n"
            f"{schema_description}"
        )
        if extra:
            system_instruction += f"\n\nAdditional instructions:\n{extra}"

        if enhanced and enhanced[0].get("role") == "system":
            enhanced[0] = {
                "role": "system",
                "content": enhanced[0]["content"] + "\n\n" + system_instruction,
            }
        else:
            enhanced.insert(0, {"role": "system", "content": system_instruction})

        return enhanced

    def _add_error_feedback(
        self, messages: list[dict], raw_response: str, error: str,
        schema_description: str
    ) -> list[dict]:
        """Ajoute le feedback d'erreur pour la prochaine tentative."""
        enhanced = list(messages)
        enhanced.append({
            "role": "assistant",
            "content": raw_response,
        })
        enhanced.append({
            "role": "user",
            "content": (
                f"Your previous response had validation errors:\n{error}\n\n"
                f"Please fix the errors and respond with valid JSON matching the schema.\n"
                f"{schema_description}"
            ),
        })
        return enhanced

    def _add_json_error_feedback(
        self, messages: list[dict], error: str, schema_description: str
    ) -> list[dict]:
        """Ajoute le feedback pour une erreur JSON."""
        enhanced = list(messages)
        enhanced.append({
            "role": "user",
            "content": (
                f"Your response was not valid JSON: {error}\n\n"
                f"Respond ONLY with valid JSON. No text before or after.\n"
                f"{schema_description}"
            ),
        })
        return enhanced


# -- Schemas internes pour usage par les sessions suivantes --

class TaskAnalysis(BaseModel):
    """Structured analysis of a user request."""
    task_type: str  # "code_r", "code_python", "debug", "explanation", etc.
    complexity: str  # "simple", "moderate", "complex"
    requires_tools: list[str] = []  # "web_search", "code_execution", "rag"
    requires_thinking: bool = False
    language: str = "auto"  # "en", "fr", "auto"
    confidence: float = 0.8


class ToolCallRequest(BaseModel):
    """Tool-call request issued by the LLM."""
    tool_name: str
    arguments: dict[str, Any] = {}
    reasoning: str = ""


class CodeVerification(BaseModel):
    """Resultat de verification de code."""
    has_errors: bool
    error_type: str | None = None  # "syntax", "runtime", "logic", "import"
    error_message: str = ""
    suggested_fix: str = ""
    confidence: float = 0.0


# -- Singleton --
structured_output_engine = StructuredOutputEngine()
