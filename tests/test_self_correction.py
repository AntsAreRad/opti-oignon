#!/usr/bin/env python3
"""
Tests pour le module d'auto-correction (S51).

Coverage:
1. Extraction d'instructions
2. Verification de conformite (heuristique)
3. Evaluation de qualite (heuristique)
4. Configuration
5. SelfCorrectionEngine -- checks individuels
6. Boucle d'auto-correction (mock)
7. Integration avec agentic_executor
8. Edge cases et backward compatibility
"""

import json
import os
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import du module principal
from opti_oignon.self_correction import (
    _FORMAT_KEYWORDS,
    _HEDGING_MARKERS,
    _LENGTH_KEYWORDS,
    _OVERCONFIDENCE_MARKERS,
    _TONE_KEYWORDS,
    ComplianceResult,
    CorrectionIteration,
    FactCheckResult,
    FactualFlag,
    InstructionCheck,
    QualityResult,
    SelfCorrectionConfig,
    SelfCorrectionEngine,
    SelfCorrectionResult,
    compute_heuristic_compliance,
    compute_heuristic_quality,
    extract_instructions,
)

# =============================================================================
# 1. TESTS: EXTRACTION D'INSTRUCTIONS
# =============================================================================

class TestExtractInstructions:
    """Tests pour extract_instructions."""

    def test_empty_message(self):
        """Message vide -> aucune instruction."""
        result = extract_instructions("")
        assert result == []

    def test_simple_message_no_instructions(self):
        """Message simple sans instructions detectables."""
        result = extract_instructions("What is the capital of France?")
        assert len(result) == 0

    def test_format_list_detection(self):
        """Detecte une demande de format liste."""
        result = extract_instructions("Give me a list of top 5 movies")
        assert any("format:" in i for i in result)

    def test_format_table_detection(self):
        """Detecte une demande de format tableau."""
        result = extract_instructions("Present the data as a table please")
        assert any("format:" in i for i in result)

    def test_format_json_detection(self):
        """Detecte une demande de format JSON."""
        result = extract_instructions("Return the result in json format")
        assert any("format:" in i for i in result)

    def test_format_markdown_detection(self):
        """Detecte une demande de format markdown."""
        result = extract_instructions("Write it in markdown please")
        assert any("format:" in i for i in result)

    def test_length_short_detection(self):
        """Detecte une demande de reponse courte."""
        result = extract_instructions("Give me a short summary")
        assert any("length:" in i for i in result)

    def test_length_detailed_detection(self):
        """Detecte une demande de reponse detaillee."""
        result = extract_instructions("Explain this in detail please")
        assert any("length:" in i for i in result)

    def test_tone_formal_detection(self):
        """Detecte une demande de ton formel."""
        result = extract_instructions("Write a formal response")
        assert any("tone:" in i for i in result)

    def test_tone_professional_detection(self):
        """Detecte une demande de ton professionnel."""
        result = extract_instructions("Be professional in your answer")
        assert any("tone:" in i for i in result)

    def test_language_french_detection(self):
        """Detecte une demande en francais."""
        result = extract_instructions("Explain this en francais")
        assert "language: french" in result

    def test_language_english_detection(self):
        """Detecte une demande en anglais."""
        result = extract_instructions("Reponds en anglais s'il te plait")
        assert "language: english" in result

    def test_multiple_questions_completeness(self):
        """Detecte les questions multiples."""
        result = extract_instructions(
            "What is Python? How is it used? Why is it popular?"
        )
        assert any("completeness:" in i for i in result)

    def test_constraint_do_not_mention(self):
        """Detecte une contrainte negative."""
        result = extract_instructions("Do not mention any brand names")
        assert any("constraint:" in i for i in result)

    def test_multiple_instructions(self):
        """Detecte plusieurs types d'instructions simultanement."""
        result = extract_instructions(
            "Give me a short list of tips in french en francais"
        )
        # Au moins format et longueur et langue
        assert len(result) >= 2


# =============================================================================
# 2. TESTS: CONFORMITE HEURISTIQUE
# =============================================================================

class TestHeuristicCompliance:
    """Tests pour compute_heuristic_compliance."""

    def test_no_instructions(self):
        """Pas d'instructions -> score parfait."""
        result = compute_heuristic_compliance("hello", "world", [])
        assert result.score == 1.0
        assert result.total_count == 0

    def test_list_format_satisfied(self):
        """Format liste respecte."""
        result = compute_heuristic_compliance(
            "as a list",
            "- Item 1\n- Item 2\n- Item 3",
            ["format: as a list"],
        )
        assert result.score == 1.0
        assert result.satisfied_count == 1

    def test_list_format_not_satisfied(self):
        """Format liste non respecte."""
        result = compute_heuristic_compliance(
            "as a list",
            "Here is a paragraph without any list formatting.",
            ["format: as a list"],
        )
        assert result.score == 0.0
        assert result.satisfied_count == 0

    def test_table_format_satisfied(self):
        """Format tableau respecte."""
        result = compute_heuristic_compliance(
            "as a table",
            "| Col1 | Col2 |\n| --- | --- |\n| val1 | val2 |",
            ["format: as a table"],
        )
        assert result.score == 1.0

    def test_json_format_satisfied(self):
        """Format JSON respecte."""
        result = compute_heuristic_compliance(
            "json",
            '{"key": "value", "count": 42}',
            ["format: json"],
        )
        assert result.score == 1.0

    def test_short_length_satisfied(self):
        """Longueur courte respectee."""
        short_response = " ".join(["word"] * 50)
        result = compute_heuristic_compliance(
            "short", short_response, ["length: short"],
        )
        assert result.score == 1.0

    def test_short_length_not_satisfied(self):
        """Longueur courte non respectee (reponse trop longue)."""
        long_response = " ".join(["word"] * 300)
        result = compute_heuristic_compliance(
            "short", long_response, ["length: short"],
        )
        assert result.score == 0.0

    def test_detailed_length_satisfied(self):
        """Longueur detaillee respectee."""
        detailed_response = " ".join(["word"] * 200)
        result = compute_heuristic_compliance(
            "detailed", detailed_response, ["length: detailed"],
        )
        assert result.score == 1.0

    def test_french_language_satisfied(self):
        """Langue francaise detectee."""
        result = compute_heuristic_compliance(
            "en francais",
            "Voici la reponse. Les resultats sont dans le tableau. "
            "La France est un pays de l'Europe de l'Ouest.",
            ["language: french"],
        )
        assert result.score == 1.0

    def test_english_language_satisfied(self):
        """Langue anglaise detectee."""
        result = compute_heuristic_compliance(
            "in english",
            "Here is the answer. The results are in the table. "
            "France is a country in Western Europe.",
            ["language: english"],
        )
        assert result.score == 1.0

    def test_completeness_sufficient(self):
        """Completude suffisante pour 2 questions."""
        long_answer = " ".join(["word"] * 100)
        result = compute_heuristic_compliance(
            "Q1? Q2?", long_answer,
            ["completeness: 2 questions to answer"],
        )
        assert result.score == 1.0

    def test_completeness_insufficient(self):
        """Completude insuffisante pour 3 questions."""
        short_answer = "Yes."
        result = compute_heuristic_compliance(
            "Q1? Q2? Q3?", short_answer,
            ["completeness: 3 questions to answer"],
        )
        assert result.score == 0.0

    def test_multiple_checks_mixed(self):
        """Plusieurs checks avec resultats mixtes."""
        result = compute_heuristic_compliance(
            "give me a short list",
            "Here is a paragraph with no formatting.",
            ["format: as a list", "length: short"],
        )
        # Un satisfait (longueur), un echoue (format)
        assert 0.0 < result.score < 1.0


# =============================================================================
# 3. TESTS: QUALITE HEURISTIQUE
# =============================================================================

class TestHeuristicQuality:
    """Tests pour compute_heuristic_quality."""

    def test_normal_response(self):
        """Reponse normale -> bonne qualite."""
        result = compute_heuristic_quality(
            "What is Python?",
            "Python is a high-level programming language. "
            "It is widely used for web development, data science, "
            "and artificial intelligence. Python was created by "
            "Guido van Rossum and first released in 1991.",
        )
        assert result.overall_score > 0.6
        assert result.completeness_score > 0.5

    def test_very_short_response(self):
        """Reponse tres courte -> mauvaise completude."""
        result = compute_heuristic_quality(
            "Explain the theory of relativity in detail",
            "It is about space.",
        )
        assert result.completeness_score < 0.5
        assert result.overall_score < 0.7

    def test_extremely_short_response(self):
        """Reponse extremement courte."""
        result = compute_heuristic_quality("question?", "ok")
        assert result.completeness_score <= 0.3

    def test_no_sentences(self):
        """Pas de phrases completes."""
        result = compute_heuristic_quality(
            "question?",
            "",
        )
        assert result.coherence_score <= 0.3

    def test_high_repetition(self):
        """Repetition excessive."""
        result = compute_heuristic_quality(
            "explain",
            " ".join(["the the the the the word"] * 20),
        )
        assert result.coherence_score < 0.8

    def test_excessive_hedging(self):
        """Langage hedging excessif."""
        result = compute_heuristic_quality(
            "question?",
            "I believe this might be true. I think probably it could be "
            "approximately correct. It seems like possibly it might be "
            "the case that I believe this is probably right.",
        )
        assert result.hallucination_risk > 0.0

    def test_overconfident_language(self):
        """Langage trop confiant."""
        result = compute_heuristic_quality(
            "question?",
            "It is certain without a doubt that this is absolutely "
            "guaranteed to be correct. There is no question about it.",
        )
        assert result.hallucination_risk > 0.0

    def test_good_quality_no_issues(self):
        """Bonne reponse sans problemes."""
        result = compute_heuristic_quality(
            "What is 2+2?",
            "The sum of 2 and 2 equals 4. This is a basic arithmetic operation.",
        )
        assert result.overall_score > 0.5
        assert len(result.issues) == 0 or result.coherence_score >= 0.8


# =============================================================================
# 4. TESTS: CONFIGURATION
# =============================================================================

class TestSelfCorrectionConfig:
    """Tests pour la configuration du moteur."""

    def test_default_config(self):
        """Configuration par defaut."""
        config = SelfCorrectionConfig()
        assert config.enable_auto is False
        assert config.max_iterations == 2
        assert config.compliance_threshold == 0.7
        assert config.quality_threshold == 0.6
        assert config.check_instructions is True
        assert config.check_facts is True
        assert config.check_quality is True
        assert config.correction_model is None
        assert config.temperature == 0.2

    def test_custom_config(self):
        """Configuration personnalisee."""
        config = SelfCorrectionConfig(
            enable_auto=True,
            max_iterations=3,
            compliance_threshold=0.8,
            quality_threshold=0.7,
        )
        assert config.enable_auto is True
        assert config.max_iterations == 3
        assert config.compliance_threshold == 0.8
        assert config.quality_threshold == 0.7

    def test_yaml_config_loading(self):
        """Chargement de config depuis YAML."""
        yaml_path = (
            Path(__file__).parent.parent
            / "opti_oignon" / "config" / "self_correction.yaml"
        )
        if yaml_path.exists():
            engine = SelfCorrectionEngine(config_path=str(yaml_path))
            assert engine.config.max_iterations > 0
            assert 0 <= engine.config.compliance_threshold <= 1.0
            assert 0 <= engine.config.quality_threshold <= 1.0

    def test_missing_yaml_fallback(self):
        """Config introuvable -> defaut."""
        engine = SelfCorrectionEngine(
            config_path="/nonexistent/path/config.yaml"
        )
        assert engine.config.max_iterations == 2

    def test_engine_with_explicit_config(self):
        """Engine avec config explicite (prioritaire)."""
        config = SelfCorrectionConfig(max_iterations=5)
        engine = SelfCorrectionEngine(config=config)
        assert engine.config.max_iterations == 5


# =============================================================================
# 5. TESTS: SELF-CORRECTION ENGINE -- CHECKS INDIVIDUELS
# =============================================================================

class TestSelfCorrectionEngineChecks:
    """Tests pour les checks individuels du moteur."""

    def setup_method(self):
        """Config de base pour chaque test."""
        self.engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(
                check_instructions=True,
                check_facts=True,
                check_quality=True,
            )
        )

    def test_compliance_no_instructions(self):
        """Compliance sans instructions -> parfait."""
        result = self.engine.check_compliance(
            "What is 2+2?",
            "The answer is 4.",
            use_llm=False,
        )
        assert result.score == 1.0

    def test_compliance_with_format(self):
        """Compliance avec format demande."""
        result = self.engine.check_compliance(
            "Give me a list of colors",
            "- Red\n- Blue\n- Green",
            use_llm=False,
        )
        # Devrait detecter "as a list" ou equivalent
        assert isinstance(result, ComplianceResult)
        assert isinstance(result.score, float)

    def test_quality_good_response(self):
        """Qualite d'une bonne reponse."""
        result = self.engine.check_quality(
            "Explain Python",
            "Python is a versatile programming language. "
            "It supports multiple paradigms including OOP and functional. "
            "Python is widely used in data science and web development.",
            use_llm=False,
        )
        assert result.overall_score > 0.5
        assert isinstance(result, QualityResult)

    def test_quality_poor_response(self):
        """Qualite d'une mauvaise reponse."""
        result = self.engine.check_quality(
            "Explain the entire history of computing in detail",
            "Computers exist.",
            use_llm=False,
        )
        assert result.completeness_score < 0.5

    def test_engine_available_property(self):
        """La propriete available depend d'Ollama."""
        assert isinstance(self.engine.available, bool)


# =============================================================================
# 6. TESTS: BOUCLE D'AUTO-CORRECTION (MOCK)
# =============================================================================

class TestSelfCorrectionLoop:
    """Tests pour la boucle de correction (avec mocks)."""

    def setup_method(self):
        """Config avec seuils bas pour forcer la correction."""
        self.engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(
                max_iterations=2,
                compliance_threshold=0.9,
                quality_threshold=0.9,
                check_instructions=True,
                check_facts=False,  # Desactiver facts pour eviter les appels LLM
                check_quality=True,
            )
        )

    def test_no_correction_needed(self):
        """Pas de correction si la qualite est bonne."""
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(
                compliance_threshold=0.1,
                quality_threshold=0.1,
            )
        )
        result = engine.correct(
            "What is 2+2?",
            "The answer is 4. This is basic arithmetic.",
            use_llm=False,
        )
        assert result.was_corrected is False
        assert result.iterations_performed == 0
        assert result.corrected_response == "The answer is 4. This is basic arithmetic."

    def test_correction_result_structure(self):
        """Le resultat a la bonne structure."""
        result = self.engine.correct(
            "What is Python?",
            "Python is a language.",
            use_llm=False,
        )
        assert isinstance(result, SelfCorrectionResult)
        assert hasattr(result, 'original_response')
        assert hasattr(result, 'corrected_response')
        assert hasattr(result, 'was_corrected')
        assert hasattr(result, 'iterations_performed')
        assert hasattr(result, 'compliance_before')
        assert hasattr(result, 'quality_before')
        assert hasattr(result, 'total_duration_ms')

    def test_correction_without_llm(self):
        """Sans LLM, pas de correction meme si scores bas."""
        result = self.engine.correct(
            "Give me a detailed list of all programming languages",
            "Python.",
            use_llm=False,
        )
        # Sans LLM, la correction ne peut pas etre generee
        assert result.was_corrected is False

    @patch("opti_oignon.self_correction.OLLAMA_AVAILABLE", True)
    @patch("opti_oignon.self_correction.ollama")
    def test_correction_with_mock_llm(self, mock_ollama):
        """Correction avec LLM mocke."""
        mock_ollama.generate.return_value = {
            "response": "Python is a high-level programming language "
                       "created by Guido van Rossum. It supports OOP, "
                       "functional programming, and procedural styles."
        }
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(
                compliance_threshold=0.95,
                quality_threshold=0.95,
                check_facts=False,
                max_iterations=1,
            )
        )
        result = engine.correct(
            "Give me a detailed explanation of Python",
            "Python.",
            use_llm=True,
        )
        assert isinstance(result, SelfCorrectionResult)
        assert result.total_duration_ms >= 0

    def test_correction_preserves_original(self):
        """L'original est toujours preserve."""
        original = "This is the original response."
        result = self.engine.correct(
            "Question?",
            original,
            use_llm=False,
        )
        assert result.original_response == original

    def test_streaming_execution(self):
        """Execute en mode streaming."""
        engine = SelfCorrectionEngine(
            config=SelfCorrectionConfig(
                compliance_threshold=0.1,
                quality_threshold=0.1,
                check_facts=False,
            )
        )
        chunks = list(engine.execute_self_correction(
            "What is 2+2?",
            "The answer is 4.",
        ))
        # Devrait avoir au moins le texte et le resultat final
        has_correction_done = any(
            isinstance(c, tuple) and c[0] == "correction_done"
            for c in chunks
        )
        assert has_correction_done

        # Le texte doit etre present
        text_chunks = [c for c in chunks if isinstance(c, str)]
        full_text = "".join(text_chunks)
        assert "4" in full_text


# =============================================================================
# 7. TESTS: INTEGRATION AGENTIC EXECUTOR
# =============================================================================

class TestAgenticExecutorIntegration:
    """Tests d'integration avec l'executeur agentique."""

    def test_pipeline_constant_exists(self):
        """La constante PIPELINE_SELF_CORRECT existe."""
        from opti_oignon.agentic_executor import PIPELINE_SELF_CORRECT
        assert PIPELINE_SELF_CORRECT == "self_correct"

    def test_import_self_correction_available(self):
        """Le flag SELF_CORRECTION_AVAILABLE est defini."""
        from opti_oignon.agentic_executor import SELF_CORRECTION_AVAILABLE
        assert isinstance(SELF_CORRECTION_AVAILABLE, bool)

    def test_agentic_executor_has_self_correction(self):
        """L'AgenticExecutor a la propriete self_correction_available."""
        from opti_oignon.agentic_executor import AgenticExecutor
        ae = AgenticExecutor()
        assert hasattr(ae, 'self_correction_available')
        assert isinstance(ae.self_correction_available, bool)

    def test_agentic_executor_has_correction_result(self):
        """L'AgenticExecutor a la propriete last_correction_result."""
        from opti_oignon.agentic_executor import AgenticExecutor
        ae = AgenticExecutor()
        assert hasattr(ae, 'last_correction_result')
        assert ae.last_correction_result is None

    def test_agentic_executor_accepts_self_correct(self):
        """La methode execute accepte self_correct."""
        import inspect

        from opti_oignon.agentic_executor import AgenticExecutor
        sig = inspect.signature(AgenticExecutor.execute)
        assert 'self_correct' in sig.parameters

    def test_agentic_executor_accepts_on_correction_step(self):
        """La methode execute accepte on_correction_step."""
        import inspect

        from opti_oignon.agentic_executor import AgenticExecutor
        sig = inspect.signature(AgenticExecutor.execute)
        assert 'on_correction_step' in sig.parameters

    def test_reset_clears_correction(self):
        """reset() nettoie le resultat de correction."""
        from opti_oignon.agentic_executor import AgenticExecutor
        ae = AgenticExecutor()
        ae._last_correction_result = "something"
        ae.reset()
        assert ae._last_correction_result is None

    def test_self_correction_engine_constructor(self):
        """Le constructeur accepte self_correction_engine."""
        from opti_oignon.agentic_executor import AgenticExecutor
        mock_engine = MagicMock()
        mock_engine.available = True
        ae = AgenticExecutor(self_correction_engine=mock_engine)
        assert ae._self_correction_engine == mock_engine

    def test_self_correct_fallback_without_engine(self):
        """Sans engine disponible, le pipeline tombe en fallback direct."""
        from opti_oignon.agentic_executor import AgenticExecutor
        mock_engine = MagicMock()
        mock_engine.available = False
        ae = AgenticExecutor(self_correction_engine=mock_engine)
        assert ae.self_correction_available is False


# =============================================================================
# 8. TESTS: EDGE CASES ET BACKWARD COMPATIBILITY
# =============================================================================

class TestEdgeCases:
    """Tests pour les cas limites et la compatibilite."""

    def test_empty_response(self):
        """Reponse vide."""
        result = compute_heuristic_quality("question?", "")
        assert result.coherence_score <= 0.3

    def test_empty_user_message(self):
        """Message utilisateur vide."""
        instructions = extract_instructions("")
        assert instructions == []

    def test_very_long_response(self):
        """Reponse tres longue."""
        long_resp = " ".join(["word"] * 5000)
        result = compute_heuristic_quality("explain", long_resp)
        assert result.completeness_score > 0.5

    def test_unicode_content(self):
        """Contenu Unicode."""
        result = compute_heuristic_quality(
            "Traduis en japonais",
            "Python est un langage de programmation.",
        )
        assert isinstance(result.overall_score, float)

    def test_special_characters(self):
        """Caracteres speciaux dans la reponse."""
        result = compute_heuristic_compliance(
            "json", '{"key": "value\twith\ttabs"}',
            ["format: json"],
        )
        assert result.score == 1.0

    def test_compliance_result_dataclass(self):
        """ComplianceResult est bien un dataclass."""
        result = ComplianceResult()
        assert result.score == 1.0
        assert result.instructions_found == []
        assert result.checks == []

    def test_quality_result_dataclass(self):
        """QualityResult est bien un dataclass."""
        result = QualityResult()
        assert result.completeness_score == 1.0
        assert result.coherence_score == 1.0
        assert result.hallucination_risk == 0.0
        assert result.overall_score == 1.0
        assert result.issues == []

    def test_fact_check_result_dataclass(self):
        """FactCheckResult est bien un dataclass."""
        result = FactCheckResult()
        assert result.flags == []
        assert result.flag_count == 0
        assert result.confidence == 1.0

    def test_correction_iteration_dataclass(self):
        """CorrectionIteration est bien un dataclass."""
        iteration = CorrectionIteration(
            iteration=1,
            compliance_score=0.8,
            quality_score=0.7,
            response_text="corrected",
        )
        assert iteration.iteration == 1
        assert iteration.improvements == []
        assert iteration.duration_ms == 0

    def test_self_correction_result_dataclass(self):
        """SelfCorrectionResult est bien un dataclass."""
        result = SelfCorrectionResult(
            original_response="original",
            corrected_response="corrected",
        )
        assert result.was_corrected is False
        assert result.iterations_performed == 0

    def test_json_parse_helper_valid(self):
        """Parse JSON valide."""
        engine = SelfCorrectionEngine(config=SelfCorrectionConfig())
        result = engine._parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_json_parse_helper_markdown(self):
        """Parse JSON entoure de markdown."""
        engine = SelfCorrectionEngine(config=SelfCorrectionConfig())
        result = engine._parse_json_response(
            'Here is the result:\n```json\n{"key": "value"}\n```'
        )
        assert result == {"key": "value"}

    def test_json_parse_helper_noisy(self):
        """Parse JSON dans du texte bruite."""
        engine = SelfCorrectionEngine(config=SelfCorrectionConfig())
        result = engine._parse_json_response(
            'Some preamble text {"key": "value"} and trailing text'
        )
        assert result == {"key": "value"}

    def test_json_parse_helper_invalid(self):
        """Parse JSON invalide -> None."""
        engine = SelfCorrectionEngine(config=SelfCorrectionConfig())
        result = engine._parse_json_response("not json at all")
        assert result is None

    def test_json_parse_helper_empty(self):
        """Parse texte vide -> None."""
        engine = SelfCorrectionEngine(config=SelfCorrectionConfig())
        result = engine._parse_json_response("")
        assert result is None


class TestSchemaBackwardCompat:
    """Tests de compatibilite backward pour les schemas."""

    def test_chat_request_has_self_correct(self):
        """ChatRequest a le champ self_correct."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(message="hello")
        assert hasattr(req, 'self_correct')
        assert req.self_correct is None

    def test_chat_request_backward_compat(self):
        """ChatRequest fonctionne sans self_correct."""
        from opti_oignon.api.schemas import ChatRequest
        req = ChatRequest(message="hello", model="qwen3:32b")
        assert req.message == "hello"
        assert req.model == "qwen3:32b"
        assert req.consensus is None
        assert req.self_correct is None

    def test_correction_schemas_exist(self):
        """Les schemas de correction existent."""
        from opti_oignon.api.schemas import (
            CorrectionConfigResponse,
            CorrectionIterationSchema,
            CorrectionResultSchema,
        )
        cr = CorrectionResultSchema()
        assert cr.was_corrected is False
        cc = CorrectionConfigResponse()
        assert cc.max_iterations == 2
        ci = CorrectionIterationSchema()
        assert ci.iteration == 0

    def test_existing_pipelines_still_work(self):
        """Les pipelines existants sont toujours la."""
        from opti_oignon.agentic_executor import (
            PIPELINE_CODE_VERIFY,
            PIPELINE_CONSENSUS,
            PIPELINE_DIRECT,
            PIPELINE_REASONING,
            PIPELINE_SELF_CORRECT,
            PIPELINE_THINK,
            PIPELINE_THINK_TOOLS,
            PIPELINE_TOOLS,
            PIPELINE_WEB_SEARCH,
        )
        assert PIPELINE_DIRECT == "direct"
        assert PIPELINE_TOOLS == "tools"
        assert PIPELINE_CODE_VERIFY == "code_verify"
        assert PIPELINE_THINK == "think"
        assert PIPELINE_WEB_SEARCH == "web_search"
        assert PIPELINE_THINK_TOOLS == "think_tools"
        assert PIPELINE_REASONING == "reasoning"
        assert PIPELINE_CONSENSUS == "consensus"
        assert PIPELINE_SELF_CORRECT == "self_correct"

    def test_singleton_exists(self):
        """Le singleton self_correction_engine existe."""
        from opti_oignon.self_correction import self_correction_engine
        assert self_correction_engine is not None
        assert isinstance(self_correction_engine, SelfCorrectionEngine)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
