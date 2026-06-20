#!/usr/bin/env python3
"""
TESTS REASONING ENGINE - OPTI-OIGNON v1.5.3 (S49)
=====================================================

Tests pour le moteur de raisonnement multi-strategies:
- Decompose-and-Solve
- Tree-of-Thought
- Self-Consistency
- Integration avec AgenticExecutor
- Configuration
- Edge cases

Usage:
    pytest tests/test_reasoning.py -v
    pytest tests/test_reasoning.py -v -k "quick"
"""

import json
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from opti_oignon.agentic_executor import (
    PIPELINE_CODE_VERIFY,
    PIPELINE_DIRECT,
    PIPELINE_REASONING,
    PIPELINE_THINK,
    PIPELINE_THINK_TOOLS,
    PIPELINE_TOOLS,
    PIPELINE_WEB_SEARCH,
    AgenticExecutor,
    _quick_classify,
    _select_pipeline,
)

# ============================================================
# IMPORT DU MODULE SOUS TEST
# ============================================================
from opti_oignon.reasoning import (
    OLLAMA_AVAILABLE,
    YAML_AVAILABLE,
    ReasoningConfig,
    ReasoningEngine,
    ReasoningResult,
    ReasoningStep,
    TreeBranch,
    reasoning_engine,
)

# ============================================================
# FIXTURES
# ============================================================

def _make_reasoning_engine(config=None):
    """Cree un ReasoningEngine avec un mock Ollama."""
    engine = ReasoningEngine(
        config=config or ReasoningConfig(),
        default_model="test-model",
    )
    return engine


def _make_mock_llm_response(content: str):
    """Cree un mock de reponse Ollama."""
    return {"message": {"content": content}}


def _make_routing(model="qwen3:32b", task_type="general", temperature=0.3):
    """Cree un objet RoutingResult minimal pour les tests."""
    return SimpleNamespace(
        model=model,
        task_type=task_type,
        temperature=temperature,
        prompt_variant="standard",
        model_type="general",
        priority_used="primary",
        explanation="test",
        timeout=60,
        routing_reason="test",
    )


# ============================================================
# TESTS: ReasoningConfig
# ============================================================

class TestReasoningConfig:
    """Tests de la configuration du reasoning."""

    def test_quick_default_values(self):
        """Configuration par defaut correcte."""
        config = ReasoningConfig()
        assert config.max_sub_steps == 5
        assert config.tree_branches == 3
        assert config.self_consistency_runs == 3
        assert config.temperature_variance == 0.1
        assert config.base_temperature == 0.3
        assert config.min_complexity_for_reasoning == "complex"
        assert config.decompose_model is None
        assert config.evaluate_model is None
        assert config.timeout_per_step == 60

    def test_quick_from_dict(self):
        """Configuration depuis un dictionnaire."""
        data = {
            "reasoning": {
                "max_sub_steps": 8,
                "tree_branches": 5,
                "self_consistency_runs": 7,
                "temperature_variance": 0.2,
                "base_temperature": 0.5,
                "decompose_model": "my-model",
            }
        }
        config = ReasoningConfig.from_dict(data)
        assert config.max_sub_steps == 8
        assert config.tree_branches == 5
        assert config.self_consistency_runs == 7
        assert config.temperature_variance == 0.2
        assert config.base_temperature == 0.5
        assert config.decompose_model == "my-model"

    def test_quick_from_dict_flat(self):
        """Configuration depuis un dict plat (sans cle 'reasoning')."""
        data = {"max_sub_steps": 3, "tree_branches": 2}
        config = ReasoningConfig.from_dict(data)
        assert config.max_sub_steps == 3
        assert config.tree_branches == 2

    def test_quick_from_dict_empty(self):
        """Configuration depuis un dict vide utilise les valeurs par defaut."""
        config = ReasoningConfig.from_dict({})
        assert config.max_sub_steps == 5
        assert config.tree_branches == 3

    def test_quick_from_yaml_nonexistent(self):
        """from_yaml avec un fichier inexistant retourne les defauts."""
        config = ReasoningConfig.from_yaml("/nonexistent/path.yaml")
        assert config.max_sub_steps == 5

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML not available")
    def test_quick_from_yaml_valid(self, tmp_path):
        """from_yaml avec un fichier valide."""
        yaml_content = """
reasoning:
  max_sub_steps: 10
  tree_branches: 4
  base_temperature: 0.7
"""
        yaml_file = tmp_path / "test_reasoning.yaml"
        yaml_file.write_text(yaml_content)

        config = ReasoningConfig.from_yaml(str(yaml_file))
        assert config.max_sub_steps == 10
        assert config.tree_branches == 4
        assert config.base_temperature == 0.7


# ============================================================
# TESTS: ReasoningEngine - Proprietes
# ============================================================

class TestReasoningEngineProperties:
    """Tests des proprietes du ReasoningEngine."""

    def test_quick_default_model(self):
        """Modele par defaut."""
        engine = ReasoningEngine(default_model="test:latest")
        assert engine._default_model == "test:latest"

    def test_quick_config_access(self):
        """Acces a la configuration."""
        config = ReasoningConfig(max_sub_steps=10)
        engine = ReasoningEngine(config=config)
        assert engine.config.max_sub_steps == 10

    def test_quick_last_result_none_initially(self):
        """last_result est None au debut."""
        engine = _make_reasoning_engine()
        assert engine.last_result is None

    def test_quick_available_depends_on_ollama(self):
        """available depend de la disponibilite d'Ollama."""
        engine = _make_reasoning_engine()
        assert engine.available == OLLAMA_AVAILABLE


# ============================================================
# TESTS: ReasoningEngine - JSON Parsing
# ============================================================

class TestJsonParsing:
    """Tests du parsing JSON des reponses LLM."""

    def test_quick_parse_valid_array(self):
        """Parse un tableau JSON valide."""
        engine = _make_reasoning_engine()
        result = engine._parse_json_response('[{"title": "test", "question": "q?"}]')
        assert isinstance(result, list)
        assert result[0]["title"] == "test"

    def test_quick_parse_valid_object(self):
        """Parse un objet JSON valide."""
        engine = _make_reasoning_engine()
        result = engine._parse_json_response('{"score": 0.8}')
        assert isinstance(result, dict)
        assert result["score"] == 0.8

    def test_quick_parse_markdown_fenced(self):
        """Parse du JSON dans un bloc markdown."""
        engine = _make_reasoning_engine()
        text = '```json\n[{"title": "test"}]\n```'
        result = engine._parse_json_response(text)
        assert isinstance(result, list)

    def test_quick_parse_with_prefix_text(self):
        """Parse du JSON precede de texte."""
        engine = _make_reasoning_engine()
        text = 'Here are the steps:\n[{"title": "test"}]'
        result = engine._parse_json_response(text)
        assert isinstance(result, list)

    def test_quick_parse_invalid_json(self):
        """Retourne None pour du JSON invalide."""
        engine = _make_reasoning_engine()
        result = engine._parse_json_response("this is not json")
        assert result is None

    def test_quick_parse_empty_string(self):
        """Retourne None pour une chaine vide."""
        engine = _make_reasoning_engine()
        result = engine._parse_json_response("")
        assert result is None


# ============================================================
# TESTS: Decompose-and-Solve
# ============================================================

class TestDecomposeAndSolve:
    """Tests de la strategie decompose-and-solve."""

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_basic_decomposition(self, mock_ollama):
        """Decomposition basique en sous-etapes."""
        # Mock: decomposition retourne 2 etapes
        decompose_json = json.dumps([
            {"title": "Understand the problem", "question": "What is asked?"},
            {"title": "Find solution", "question": "How to solve?"},
        ])
        # Les reponses successives: decompose, solve step 1, solve step 2, synthesize
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(decompose_json),
            _make_mock_llm_response("The problem is about X."),
            _make_mock_llm_response("The solution is Y."),
            _make_mock_llm_response("In summary, X leads to Y."),
        ]

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve("Complex question?")

        assert result.strategy == "decompose"
        assert len(result.steps) == 2
        assert result.steps[0].title == "Understand the problem"
        assert result.steps[1].title == "Find solution"
        assert "summary" in result.final_answer.lower()
        assert result.confidence > 0

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_with_callback(self, mock_ollama):
        """Callback appele a chaque etape."""
        decompose_json = json.dumps([
            {"title": "Step A", "question": "Q?"},
        ])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(decompose_json),
            _make_mock_llm_response("Answer A."),
            _make_mock_llm_response("Final."),
        ]

        steps_received = []
        engine = _make_reasoning_engine()
        engine.decompose_and_solve(
            "Question?",
            on_step=lambda s: steps_received.append(s),
        )

        assert len(steps_received) == 1
        assert steps_received[0].title == "Step A"

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_bad_json_fallback(self, mock_ollama):
        """Fallback quand la decomposition retourne du JSON invalide."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("Not valid JSON at all"),
            _make_mock_llm_response("Direct answer."),
            _make_mock_llm_response("Synthesized."),
        ]

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve("Question?")

        # Devrait fallback vers une seule etape directe
        assert len(result.steps) == 1
        assert result.steps[0].title == "Direct answer"

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_max_steps_limit(self, mock_ollama):
        """Respecte la limite max_steps."""
        many_steps = json.dumps([
            {"title": f"Step {i}", "question": f"Q{i}?"} for i in range(10)
        ])
        # decompose + 3 solve + 1 synthesize
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(many_steps),
        ] + [
            _make_mock_llm_response(f"Answer {i}") for i in range(3)
        ] + [
            _make_mock_llm_response("Final."),
        ]

        config = ReasoningConfig(max_sub_steps=3)
        engine = ReasoningEngine(config=config, default_model="test")
        result = engine.decompose_and_solve("Question?")

        assert len(result.steps) <= 3

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_step_error_continues(self, mock_ollama):
        """Continue meme si une etape echoue."""
        decompose_json = json.dumps([
            {"title": "Step 1", "question": "Q1?"},
            {"title": "Step 2", "question": "Q2?"},
        ])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(decompose_json),
            Exception("LLM error"),  # Step 1 fails
            _make_mock_llm_response("Answer 2."),
            _make_mock_llm_response("Final."),
        ]

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve("Question?")

        assert len(result.steps) == 2
        assert "[Error" in result.steps[0].content

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_stores_last_result(self, mock_ollama):
        """Le dernier resultat est stocke."""
        decompose_json = json.dumps([
            {"title": "Step", "question": "Q?"},
        ])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(decompose_json),
            _make_mock_llm_response("Answer."),
            _make_mock_llm_response("Final."),
        ]

        engine = _make_reasoning_engine()
        assert engine.last_result is None
        engine.decompose_and_solve("Q?")
        assert engine.last_result is not None
        assert engine.last_result.strategy == "decompose"


# ============================================================
# TESTS: Tree-of-Thought
# ============================================================

class TestTreeOfThought:
    """Tests de la strategie Tree-of-Thought."""

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_basic_tree(self, mock_ollama):
        """Arbre de pensee basique."""
        branches_json = json.dumps([
            {"approach": "Historical analysis"},
            {"approach": "Statistical analysis"},
        ])
        eval1_json = json.dumps({"score": 0.7, "justification": "Good"})
        eval2_json = json.dumps({"score": 0.9, "justification": "Better"})

        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(branches_json),
            _make_mock_llm_response(eval1_json),
            _make_mock_llm_response(eval2_json),
            _make_mock_llm_response("Statistical analysis reveals..."),
        ]

        engine = _make_reasoning_engine()
        result = engine.tree_of_thought("Compare approaches?")

        assert result.strategy == "tree_of_thought"
        assert len(result.steps) == 3  # generate + evaluate + elaborate
        assert result.confidence > 0
        assert "statistical" in result.final_answer.lower() or result.final_answer

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_tree_selects_best_branch(self, mock_ollama):
        """Selectionne la branche avec le meilleur score."""
        branches_json = json.dumps([
            {"approach": "Approach A"},
            {"approach": "Approach B"},
        ])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(branches_json),
            _make_mock_llm_response('{"score": 0.3}'),
            _make_mock_llm_response('{"score": 0.8}'),
            _make_mock_llm_response("Elaborated B."),
        ]

        engine = _make_reasoning_engine()
        result = engine.tree_of_thought("Question?")

        assert result.metadata["best_branch_score"] == 0.8

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_tree_bad_branch_json_fallback(self, mock_ollama):
        """Fallback quand la generation de branches echoue."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("not json"),
            _make_mock_llm_response('{"score": 0.5}'),
            _make_mock_llm_response("Direct answer."),
        ]

        engine = _make_reasoning_engine()
        result = engine.tree_of_thought("Question?")

        assert len(result.steps) >= 2
        assert result.final_answer

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_tree_eval_failure_uses_default_score(self, mock_ollama):
        """Score par defaut quand l'evaluation echoue."""
        branches_json = json.dumps([{"approach": "Test"}])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(branches_json),
            Exception("Eval failed"),
            _make_mock_llm_response("Elaborated."),
        ]

        engine = _make_reasoning_engine()
        result = engine.tree_of_thought("Question?")

        assert result.metadata["branches"][0]["score"] == 0.5  # default


# ============================================================
# TESTS: Self-Consistency
# ============================================================

class TestSelfConsistency:
    """Tests de la strategie Self-Consistency."""

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_basic_consistency(self, mock_ollama):
        """Self-consistency basique avec 3 runs."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("The answer is 42."),
            _make_mock_llm_response("The answer is 42, clearly."),
            _make_mock_llm_response("The answer is 42 without doubt."),
        ]

        engine = _make_reasoning_engine()
        result = engine.self_consistency("What is the answer?", n_runs=3)

        assert result.strategy == "self_consistency"
        assert len(result.steps) == 3
        assert result.confidence > 0
        assert "42" in result.final_answer

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_consistency_with_failures(self, mock_ollama):
        """Gestion des echecs dans les runs."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("Good answer."),
            Exception("API error"),
            _make_mock_llm_response("Good answer again."),
        ]

        engine = _make_reasoning_engine()
        result = engine.self_consistency("Q?", n_runs=3)

        assert result.metadata["runs_successful"] == 2
        assert result.metadata["runs_failed"] == 1

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_consistency_all_fail(self, mock_ollama):
        """Toutes les runs echouent."""
        mock_ollama.chat.side_effect = [
            Exception("Error 1"),
            Exception("Error 2"),
        ]

        engine = _make_reasoning_engine()
        result = engine.self_consistency("Q?", n_runs=2)

        assert result.confidence == 0.0
        assert "[All runs failed]" in result.final_answer

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_consistency_single_run(self, mock_ollama):
        """Un seul run reussi."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("Only answer."),
        ]

        engine = _make_reasoning_engine()
        result = engine.self_consistency("Q?", n_runs=1)

        assert result.confidence == 0.5  # Confidence limitee avec 1 seul run

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_consistency_callback(self, mock_ollama):
        """Callback appele pour chaque run."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("A1."),
            _make_mock_llm_response("A2."),
        ]

        steps_received = []
        engine = _make_reasoning_engine()
        engine.self_consistency(
            "Q?", n_runs=2,
            on_step=lambda s: steps_received.append(s),
        )

        assert len(steps_received) == 2


# ============================================================
# TESTS: Consistency Selection Algorithm
# ============================================================

class TestConsistencySelection:
    """Tests de l'algorithme de selection de reponse la plus coherente."""

    def test_quick_empty_answers(self):
        """Liste vide retourne chaine vide."""
        engine = _make_reasoning_engine()
        answer, score = engine._select_most_consistent([])
        assert answer == ""
        assert score == 0.0

    def test_quick_single_answer(self):
        """Une seule reponse retourne score 1.0."""
        engine = _make_reasoning_engine()
        answer, score = engine._select_most_consistent(["The only answer."])
        assert answer == "The only answer."
        assert score == 1.0

    def test_quick_identical_answers(self):
        """Reponses identiques ont un score eleve."""
        engine = _make_reasoning_engine()
        answers = ["The answer is clearly yes.", "The answer is clearly yes."]
        answer, score = engine._select_most_consistent(answers)
        assert score == 1.0

    def test_quick_different_answers(self):
        """Reponses differentes ont un score plus faible."""
        engine = _make_reasoning_engine()
        answers = [
            "Cats are better pets than dogs for apartments.",
            "Quantum physics describes wave-particle duality.",
        ]
        answer, score = engine._select_most_consistent(answers)
        assert score < 0.5

    def test_quick_majority_selection(self):
        """Selectionne la reponse qui partage le plus de mots."""
        engine = _make_reasoning_engine()
        answers = [
            "Python is great programming language for data science.",
            "Python is excellent programming language for data analysis.",
            "Java enterprise framework microservices architecture.",
        ]
        answer, score = engine._select_most_consistent(answers)
        assert "python" in answer.lower() or "programming" in answer.lower()


# ============================================================
# TESTS: Streaming Execution
# ============================================================

class TestStreamingExecution:
    """Tests de l'execution en streaming."""

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_execute_reasoning_decompose(self, mock_ollama):
        """execute_reasoning yield les bons types de chunks."""
        decompose_json = json.dumps([
            {"title": "Step 1", "question": "Q?"},
        ])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(decompose_json),
            _make_mock_llm_response("Answer."),
            _make_mock_llm_response("Final answer."),
        ]

        engine = _make_reasoning_engine()
        chunks = list(engine.execute_reasoning("Complex Q?", strategy="decompose"))

        # Verifier qu'on a des reasoning_step, reasoning_done, et du texte
        reasoning_steps = [c for c in chunks if isinstance(c, tuple) and c[0] == "reasoning_step"]
        reasoning_done = [c for c in chunks if isinstance(c, tuple) and c[0] == "reasoning_done"]
        text_chunks = [c for c in chunks if isinstance(c, str)]

        assert len(reasoning_steps) > 0
        assert len(reasoning_done) == 1
        assert len(text_chunks) > 0

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_execute_reasoning_tree(self, mock_ollama):
        """execute_reasoning avec strategie tree_of_thought."""
        branches_json = json.dumps([{"approach": "Test"}])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(branches_json),
            _make_mock_llm_response('{"score": 0.8}'),
            _make_mock_llm_response("Elaborated."),
        ]

        engine = _make_reasoning_engine()
        chunks = list(engine.execute_reasoning("Q?", strategy="tree_of_thought"))

        reasoning_done = [c for c in chunks if isinstance(c, tuple) and c[0] == "reasoning_done"]
        assert len(reasoning_done) == 1
        assert reasoning_done[0][1].strategy == "tree_of_thought"


# ============================================================
# TESTS: should_use_reasoning (Static)
# ============================================================

class TestShouldUseReasoning:
    """Tests de la detection de besoin de raisonnement."""

    def test_quick_explicit_complex(self):
        """complexity='complex' retourne True."""
        assert ReasoningEngine.should_use_reasoning("hello", "complex") is True

    def test_quick_explicit_simple(self):
        """complexity='simple' retourne False."""
        assert ReasoningEngine.should_use_reasoning("hello", "simple") is False

    def test_quick_step_by_step_keyword(self):
        """Detection du keyword 'step by step'."""
        assert ReasoningEngine.should_use_reasoning("explain step by step how") is True

    def test_quick_decompose_keyword(self):
        """Detection du keyword 'break down'."""
        assert ReasoningEngine.should_use_reasoning("break down this problem") is True

    def test_quick_french_keyword(self):
        """Detection des keywords francais."""
        assert ReasoningEngine.should_use_reasoning("analyse en detail ce probleme") is True

    def test_quick_simple_query(self):
        """Requete simple ne declenche pas le raisonnement."""
        assert ReasoningEngine.should_use_reasoning("what is 2+2?") is False

    def test_quick_long_multi_question(self):
        """Longue requete avec multiples questions."""
        long_msg = "word " * 55 + "First question? Second question?"
        assert ReasoningEngine.should_use_reasoning(long_msg) is True


# ============================================================
# TESTS: AgenticExecutor Integration
# ============================================================

class TestAgenticExecutorIntegration:
    """Tests de l'integration avec l'AgenticExecutor."""

    def test_quick_classify_reasoning_keywords(self):
        """_quick_classify detecte les mots-cles de raisonnement."""
        result = _quick_classify("Please break down this problem step by step")
        assert result["needs_reasoning"] is True

    def test_quick_classify_no_reasoning(self):
        """_quick_classify ne detecte pas de raisonnement pour une requete simple."""
        result = _quick_classify("What time is it?")
        assert result["needs_reasoning"] is False

    def test_quick_classify_french_reasoning(self):
        """Detection du raisonnement en francais."""
        result = _quick_classify("Decompose ce probleme en etape par etape")
        assert result["needs_reasoning"] is True

    def test_quick_select_pipeline_reasoning(self):
        """_select_pipeline choisit PIPELINE_REASONING quand disponible."""
        classification = {
            "needs_tools": False,
            "needs_web": False,
            "is_code": False,
            "is_complex": False,
            "needs_reasoning": True,
        }
        pipeline = _select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=True,
        )
        assert pipeline == PIPELINE_REASONING

    def test_quick_select_pipeline_reasoning_not_available(self):
        """Fallback quand reasoning non disponible."""
        classification = {
            "needs_tools": False,
            "needs_web": False,
            "is_code": False,
            "is_complex": True,
            "needs_reasoning": True,
        }
        pipeline = _select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=False,
        )
        # Devrait tomber sur THINK car is_complex=True
        assert pipeline == PIPELINE_THINK

    def test_quick_select_pipeline_web_overrides_reasoning(self):
        """Web search override a priorite sur le reasoning."""
        classification = {
            "needs_tools": False,
            "needs_web": True,
            "is_code": False,
            "is_complex": False,
            "needs_reasoning": True,
        }
        pipeline = _select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=True,
        )
        assert pipeline == PIPELINE_WEB_SEARCH

    def test_quick_select_pipeline_think_override(self):
        """think=True override le reasoning."""
        classification = {
            "needs_tools": False,
            "needs_web": False,
            "is_code": False,
            "is_complex": False,
            "needs_reasoning": True,
        }
        pipeline = _select_pipeline(
            classification=classification,
            think_override=True,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=True,
        )
        assert pipeline == PIPELINE_THINK

    def test_quick_agentic_executor_has_reasoning_property(self):
        """AgenticExecutor a la propriete reasoning_available."""
        ae = AgenticExecutor()
        assert hasattr(ae, 'reasoning_available')
        assert isinstance(ae.reasoning_available, bool)

    def test_quick_agentic_executor_has_last_reasoning_result(self):
        """AgenticExecutor a la propriete last_reasoning_result."""
        ae = AgenticExecutor()
        assert hasattr(ae, 'last_reasoning_result')
        assert ae.last_reasoning_result is None

    def test_quick_agentic_executor_reset_clears_reasoning(self):
        """reset() efface le dernier resultat de raisonnement."""
        ae = AgenticExecutor()
        ae._last_reasoning_result = "test"
        ae.reset()
        assert ae._last_reasoning_result is None


# ============================================================
# TESTS: Dataclasses
# ============================================================

class TestDataclasses:
    """Tests des dataclasses de resultat."""

    def test_quick_reasoning_step(self):
        """Creation d'un ReasoningStep."""
        step = ReasoningStep(
            step_number=1,
            title="Analyze",
            content="The analysis shows...",
            duration_ms=1234,
        )
        assert step.step_number == 1
        assert step.title == "Analyze"
        assert step.duration_ms == 1234

    def test_quick_reasoning_result(self):
        """Creation d'un ReasoningResult."""
        result = ReasoningResult(strategy="decompose")
        assert result.strategy == "decompose"
        assert result.steps == []
        assert result.final_answer == ""
        assert result.confidence == 0.0
        assert result.metadata == {}

    def test_quick_tree_branch(self):
        """Creation d'un TreeBranch."""
        branch = TreeBranch(
            branch_id=0,
            approach="Statistical method",
            score=0.85,
        )
        assert branch.branch_id == 0
        assert branch.score == 0.85


# ============================================================
# TESTS: Edge Cases
# ============================================================

class TestEdgeCases:
    """Tests des cas limites."""

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_empty_question(self, mock_ollama):
        """Question vide."""
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response("[]"),
            _make_mock_llm_response("No question."),
            _make_mock_llm_response("No question."),
        ]

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve("")

        # Devrait fallback vers une seule etape
        assert len(result.steps) >= 1

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_very_long_question(self, mock_ollama):
        """Question tres longue."""
        long_q = "Explain " * 200
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(json.dumps([{"title": "Step", "question": "Q?"}])),
            _make_mock_llm_response("Answer."),
            _make_mock_llm_response("Final."),
        ]

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve(long_q)
        assert result.final_answer

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_decompose_all_calls_fail(self, mock_ollama):
        """Toutes les etapes echouent mais on a quand meme un resultat."""
        mock_ollama.chat.side_effect = Exception("Total failure")

        engine = _make_reasoning_engine()
        result = engine.decompose_and_solve("Q?")

        # Le fallback devrait quand meme creer un resultat
        assert result.strategy == "decompose"
        assert len(result.steps) >= 1

    @patch("opti_oignon.reasoning.ollama")
    def test_quick_tree_zero_branches_config(self, mock_ollama):
        """Config avec 0 branches utilise le minimum."""
        branches_json = json.dumps([{"approach": "Only one"}])
        mock_ollama.chat.side_effect = [
            _make_mock_llm_response(branches_json),
            _make_mock_llm_response('{"score": 0.5}'),
            _make_mock_llm_response("Answer."),
        ]

        config = ReasoningConfig(tree_branches=0)
        engine = ReasoningEngine(config=config, default_model="test")
        result = engine.tree_of_thought("Q?", n_branches=1)

        assert result.strategy == "tree_of_thought"

    def test_quick_select_consistency_with_short_words(self):
        """L'algorithme de consistance filtre les mots courts."""
        engine = _make_reasoning_engine()
        answers = [
            "a b c the is an",  # Que des mots courts
            "x y z it we do",
        ]
        answer, score = engine._select_most_consistent(answers)
        # Les mots courts sont filtres, donc le score est bas
        assert isinstance(score, float)


# ============================================================
# TESTS: Backward Compatibility
# ============================================================

class TestBackwardCompatibility:
    """Tests de retrocompatibilite."""

    def test_quick_pipeline_constants_unchanged(self):
        """Les constantes de pipeline existantes sont inchangees."""
        assert PIPELINE_DIRECT == "direct"
        assert PIPELINE_TOOLS == "tools"
        assert PIPELINE_CODE_VERIFY == "code_verify"
        assert PIPELINE_THINK == "think"
        assert PIPELINE_WEB_SEARCH == "web_search"
        assert PIPELINE_THINK_TOOLS == "think_tools"
        assert PIPELINE_REASONING == "reasoning"

    def test_quick_select_pipeline_backward_compat(self):
        """_select_pipeline fonctionne sans le parametre reasoning_available."""
        # Sans reasoning_available (valeur par defaut False)
        classification = {
            "needs_tools": False,
            "needs_web": False,
            "is_code": False,
            "is_complex": False,
            "needs_reasoning": False,
        }
        pipeline = _select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
        )
        assert pipeline == PIPELINE_DIRECT

    def test_quick_agentic_executor_accepts_no_reasoning(self):
        """AgenticExecutor fonctionne sans reasoning_engine disponible."""
        # Patcher le singleton pour simuler l'absence du module
        with patch("opti_oignon.agentic_executor._default_reasoning_engine", None):
            ae = AgenticExecutor(reasoning_engine=None)
            assert ae.reasoning_available is False

    def test_quick_singleton_exists(self):
        """Le singleton reasoning_engine existe."""
        assert reasoning_engine is not None
        assert isinstance(reasoning_engine, ReasoningEngine)

    def test_quick_classify_returns_needs_reasoning(self):
        """_quick_classify inclut toujours la cle needs_reasoning."""
        result = _quick_classify("hello")
        assert "needs_reasoning" in result


# ============================================================
# TESTS: Module-level
# ============================================================

class TestModuleLevel:
    """Tests du niveau module."""

    def test_quick_imports_available(self):
        """Les imports fonctionnent."""
        from opti_oignon.reasoning import (
            ReasoningConfig,
            ReasoningEngine,
            ReasoningResult,
            ReasoningStep,
            TreeBranch,
            reasoning_engine,
        )
        assert ReasoningEngine is not None
        assert ReasoningConfig is not None
        assert reasoning_engine is not None

    def test_quick_agentic_imports(self):
        """Les imports agentiques incluent PIPELINE_REASONING."""
        from opti_oignon.agentic_executor import PIPELINE_REASONING
        assert PIPELINE_REASONING == "reasoning"
