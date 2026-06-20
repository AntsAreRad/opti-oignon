#!/usr/bin/env python3
"""
TESTS CONSENSUS ENGINE - OPTI-OIGNON v1.5.4 (S50)
=====================================================

Tests pour le moteur de consensus multi-modele:
- Execution parallele
- Comparaison de reponses (Jaccard)
- Strategie Best-of-N
- Strategie Weighted Vote
- Strategie LLM Merge
- Configuration YAML
- Integration avec AgenticExecutor
- Edge cases (single model, all fail, timeout)
- Backward compatible

Usage:
    pytest tests/test_consensus.py -v
    pytest tests/test_consensus.py -v -k "quick"
"""

import json
import os
import tempfile
import time
from types import SimpleNamespace
from typing import List
from unittest.mock import MagicMock, PropertyMock, call, patch

import pytest

from opti_oignon.agentic_executor import (
    PIPELINE_CODE_VERIFY,
    PIPELINE_CONSENSUS,
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
from opti_oignon.consensus import (
    DEFAULT_QUALITY_WEIGHTS,
    OLLAMA_AVAILABLE,
    STRATEGY_BEST_OF_N,
    STRATEGY_LLM_MERGE,
    STRATEGY_WEIGHTED_VOTE,
    VALID_STRATEGIES,
    YAML_AVAILABLE,
    ConsensusComparison,
    ConsensusConfig,
    ConsensusEngine,
    ConsensusResult,
    ModelResponse,
    consensus_engine,
)

# ============================================================
# FIXTURES
# ============================================================

def _make_consensus_engine(config=None):
    """Cree un ConsensusEngine avec config personnalisee."""
    cfg = config or ConsensusConfig(
        default_models=["model_a", "model_b", "model_c"],
        strategy=STRATEGY_BEST_OF_N,
        judge_model="model_a",
        max_models=3,
        timeout_per_model=30,
        min_agreement_threshold=0.3,
    )
    engine = ConsensusEngine.__new__(ConsensusEngine)
    engine._config = cfg
    engine._default_model = "model_a"
    engine._last_result = None
    return engine


def _make_model_response(
    model="model_a",
    content="Test response content",
    duration_ms=500,
    success=True,
    error="",
    quality_tier="medium",
):
    """Cree un ModelResponse pour les tests."""
    return ModelResponse(
        model=model,
        content=content,
        duration_ms=duration_ms,
        success=success,
        error=error,
        quality_tier=quality_tier,
    )


def _make_routing(model="model_a"):
    """Cree un mock RoutingResult."""
    return SimpleNamespace(
        model=model,
        task_type="simple_question",
        temperature=0.5,
        prompt_variant="standard",
        routing_reason="test",
    )


def _make_executor_with_consensus(consensus_engine=None):
    """Cree un AgenticExecutor avec un ConsensusEngine mock."""
    mock_executor = MagicMock()
    mock_executor.execute.return_value = iter(["test response"])
    mock_executor.reset.return_value = None
    mock_executor.cancel.return_value = None

    engine = AgenticExecutor(
        executor=mock_executor,
        tool_executor=None,
        structured_engine=None,
        verification_engine=None,
        reasoning_engine=None,
        consensus_engine=consensus_engine,
    )
    return engine


# ============================================================
# TESTS: CONFIGURATION
# ============================================================

class TestConsensusConfig:
    """Tests pour la configuration du consensus."""

    def test_default_config(self):
        """La config par defaut a des valeurs saines."""
        cfg = ConsensusConfig()
        assert cfg.strategy == STRATEGY_BEST_OF_N
        assert len(cfg.default_models) == 3
        assert cfg.max_models == 3
        assert cfg.timeout_per_model == 60
        assert cfg.min_agreement_threshold == 0.3

    def test_invalid_strategy_fallback(self):
        """Une strategie invalide retombe sur best_of_n."""
        cfg = ConsensusConfig(strategy="invalid_strategy")
        assert cfg.strategy == STRATEGY_BEST_OF_N

    def test_min_max_models(self):
        """max_models ne peut pas etre inferieur a 1."""
        cfg = ConsensusConfig(max_models=0)
        assert cfg.max_models == 1

    def test_min_timeout(self):
        """timeout_per_model ne peut pas etre inferieur a 5."""
        cfg = ConsensusConfig(timeout_per_model=2)
        assert cfg.timeout_per_model == 5

    def test_quality_weights(self):
        """Les poids de qualite sont correctement initialises."""
        cfg = ConsensusConfig()
        assert cfg.quality_weights["high"] == 1.0
        assert cfg.quality_weights["medium"] == 0.7
        assert cfg.quality_weights["low"] == 0.4

    def test_from_yaml_missing_file(self):
        """Retourne la config par defaut si le fichier est absent."""
        cfg = ConsensusConfig.from_yaml("/nonexistent/path.yaml")
        assert cfg.strategy == STRATEGY_BEST_OF_N

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML requis")
    def test_from_yaml_valid_file(self, tmp_path):
        """Charge correctement depuis un fichier YAML valide."""
        yaml_content = """
consensus:
  default_models:
    - "test_model_1"
    - "test_model_2"
  strategy: "weighted_vote"
  judge_model: "test_model_1"
  max_models: 2
  timeout_per_model: 45
  min_agreement_threshold: 0.5
  temperature: 0.4
"""
        yaml_path = tmp_path / "consensus.yaml"
        yaml_path.write_text(yaml_content)
        cfg = ConsensusConfig.from_yaml(str(yaml_path))
        assert cfg.strategy == STRATEGY_WEIGHTED_VOTE
        assert len(cfg.default_models) == 2
        assert cfg.default_models[0] == "test_model_1"
        assert cfg.max_models == 2
        assert cfg.timeout_per_model == 45
        assert cfg.min_agreement_threshold == 0.5

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML requis")
    def test_from_yaml_empty_file(self, tmp_path):
        """Retourne la config par defaut pour un fichier vide."""
        yaml_path = tmp_path / "empty.yaml"
        yaml_path.write_text("")
        cfg = ConsensusConfig.from_yaml(str(yaml_path))
        assert cfg.strategy == STRATEGY_BEST_OF_N

    @pytest.mark.skipif(not YAML_AVAILABLE, reason="PyYAML requis")
    def test_from_yaml_no_consensus_key(self, tmp_path):
        """Retourne la config par defaut si la cle consensus est absente."""
        yaml_path = tmp_path / "other.yaml"
        yaml_path.write_text("something_else: true")
        cfg = ConsensusConfig.from_yaml(str(yaml_path))
        assert cfg.strategy == STRATEGY_BEST_OF_N


# ============================================================
# TESTS: TOKENIZATION ET SIMILARITE
# ============================================================

class TestTokenizationAndSimilarity:
    """Tests pour les fonctions de comparaison."""

    def test_tokenize_basic(self):
        """Tokenise correctement un texte simple."""
        result = ConsensusEngine._tokenize("The quick brown fox jumps")
        assert "quick" in result
        assert "brown" in result
        assert "jumps" in result
        # Mots courts filtres
        assert "the" not in result
        assert "fox" not in result

    def test_tokenize_empty(self):
        """Retourne un ensemble vide pour un texte vide."""
        result = ConsensusEngine._tokenize("")
        assert result == set()

    def test_jaccard_identical(self):
        """Deux ensembles identiques ont un score de 1.0."""
        s = {"hello", "world", "test"}
        assert ConsensusEngine._jaccard_similarity(s, s) == 1.0

    def test_jaccard_disjoint(self):
        """Deux ensembles disjoints ont un score de 0.0."""
        s1 = {"hello", "world"}
        s2 = {"foo", "bar"}
        assert ConsensusEngine._jaccard_similarity(s1, s2) == 0.0

    def test_jaccard_partial_overlap(self):
        """Score correct pour un recouvrement partiel."""
        s1 = {"hello", "world", "test"}
        s2 = {"hello", "world", "other"}
        # Intersection: 2, Union: 4
        assert ConsensusEngine._jaccard_similarity(s1, s2) == pytest.approx(0.5)

    def test_jaccard_empty_sets(self):
        """Retourne 0.0 pour des ensembles vides."""
        assert ConsensusEngine._jaccard_similarity(set(), set()) == 0.0
        assert ConsensusEngine._jaccard_similarity({"a"}, set()) == 0.0


# ============================================================
# TESTS: COMPARAISON DE REPONSES
# ============================================================

class TestCompareResponses:
    """Tests pour la comparaison inter-modeles."""

    def test_compare_two_similar_responses(self):
        """Compare correctement deux reponses similaires."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Python is a programming language used widely"),
            _make_model_response(model="m2", content="Python is a popular programming language used everywhere"),
        ]
        comp = engine.compare_responses(responses)
        assert comp.average_agreement > 0.0
        assert "m1" in comp.agreement_matrix
        assert "m2" in comp.agreement_matrix

    def test_compare_dissimilar_responses(self):
        """Deux reponses tres differentes ont un faible accord."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Quantum physics explains particle behavior"),
            _make_model_response(model="m2", content="Renaissance paintings showcase beautiful artwork"),
        ]
        comp = engine.compare_responses(responses)
        assert comp.average_agreement < 0.5

    def test_compare_single_response(self):
        """Une seule reponse donne un accord parfait."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Single response here"),
        ]
        comp = engine.compare_responses(responses)
        assert comp.average_agreement == 1.0

    def test_compare_no_responses(self):
        """Aucune reponse donne un accord de 0.0."""
        engine = _make_consensus_engine()
        comp = engine.compare_responses([])
        assert comp.average_agreement == 0.0

    def test_compare_with_failures(self):
        """Les reponses echouees sont ignorees."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Valid response"),
            _make_model_response(model="m2", content="", success=False, error="timeout"),
        ]
        comp = engine.compare_responses(responses)
        assert comp.average_agreement == 1.0  # Seule une reponse valide

    def test_compare_three_models(self):
        """Matrice de comparaison correcte pour 3 modeles."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Machine learning algorithms process large datasets"),
            _make_model_response(model="m2", content="Machine learning models analyze large datasets efficiently"),
            _make_model_response(model="m3", content="Deep learning neural networks train on datasets"),
        ]
        comp = engine.compare_responses(responses)
        assert len(comp.agreement_matrix) == 3
        # Chaque modele a des scores avec les deux autres
        for model_scores in comp.agreement_matrix.values():
            assert len(model_scores) == 3

    def test_areas_of_agreement(self):
        """Identifie correctement les mots communs."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Python language programming efficient"),
            _make_model_response(model="m2", content="Python language programming powerful"),
        ]
        comp = engine.compare_responses(responses)
        # "python", "language", "programming" devraient etre dans les zones d'accord
        agreement_lower = [w.lower() for w in comp.areas_of_agreement]
        assert "python" in agreement_lower
        assert "language" in agreement_lower
        assert "programming" in agreement_lower


# ============================================================
# TESTS: STRATEGIE BEST-OF-N
# ============================================================

class TestBestOfN:
    """Tests pour la strategie Best-of-N."""

    def test_best_of_n_selects_most_central(self):
        """Selectionne la reponse la plus similaire aux autres."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Python programming language widely used"),
            _make_model_response(model="m2", content="Python programming language very popular"),
            _make_model_response(model="m3", content="JavaScript framework browser rendering"),
        ]
        comp = engine.compare_responses(responses)
        content, model, confidence = engine._best_of_n(responses, comp)
        # m1 ou m2 devraient etre selectionnes (plus proches entre eux)
        assert model in ("m1", "m2")
        assert confidence > 0.0

    def test_best_of_n_single_response(self):
        """Une seule reponse est retournee avec confiance 0.5."""
        engine = _make_consensus_engine()
        responses = [_make_model_response(model="m1", content="Solo response")]
        comp = engine.compare_responses(responses)
        content, model, confidence = engine._best_of_n(responses, comp)
        assert model == "m1"
        assert confidence == 0.5

    def test_best_of_n_no_responses(self):
        """Aucune reponse retourne des valeurs vides."""
        engine = _make_consensus_engine()
        comp = ConsensusComparison()
        content, model, confidence = engine._best_of_n([], comp)
        assert content == ""
        assert model == ""
        assert confidence == 0.0


# ============================================================
# TESTS: STRATEGIE WEIGHTED VOTE
# ============================================================

class TestWeightedVote:
    """Tests pour la strategie Weighted Vote."""

    def test_weighted_vote_prefers_high_quality(self):
        """Favorise les modeles de haute qualite."""
        engine = _make_consensus_engine()
        responses = [
            _make_model_response(model="m1", content="Python programming language widely used", quality_tier="high"),
            _make_model_response(model="m2", content="Python programming language widely used", quality_tier="low"),
        ]
        comp = engine.compare_responses(responses)
        content, model, confidence = engine._weighted_vote(responses, comp)
        assert model == "m1"  # Le modele high devrait gagner

    def test_weighted_vote_single_response(self):
        """Une seule reponse retourne confiance 0.5."""
        engine = _make_consensus_engine()
        responses = [_make_model_response(model="m1", content="Solo")]
        comp = engine.compare_responses(responses)
        content, model, confidence = engine._weighted_vote(responses, comp)
        assert confidence == 0.5

    def test_weighted_vote_no_responses(self):
        """Aucune reponse retourne des valeurs vides."""
        engine = _make_consensus_engine()
        comp = ConsensusComparison()
        content, model, confidence = engine._weighted_vote([], comp)
        assert content == ""


# ============================================================
# TESTS: STRATEGIE LLM MERGE
# ============================================================

class TestLLMMerge:
    """Tests pour la strategie LLM Merge."""

    def test_llm_merge_calls_judge(self):
        """Appelle le modele juge avec le bon prompt."""
        engine = _make_consensus_engine()
        engine._call_llm = MagicMock(return_value="Merged answer combining best parts")

        responses = [
            _make_model_response(model="m1", content="Response from m1"),
            _make_model_response(model="m2", content="Response from m2"),
        ]
        content, model, confidence = engine._llm_merge(responses, "test query")

        assert engine._call_llm.called
        assert content == "Merged answer combining best parts"
        assert "[merge:" in model
        assert confidence > 0.5

    def test_llm_merge_fallback_on_error(self):
        """Retombe sur la premiere reponse si le merge echoue."""
        engine = _make_consensus_engine()
        engine._call_llm = MagicMock(side_effect=RuntimeError("LLM error"))

        responses = [
            _make_model_response(model="m1", content="First response"),
            _make_model_response(model="m2", content="Second response"),
        ]
        content, model, confidence = engine._llm_merge(responses, "query")

        assert content == "First response"
        assert model == "m1"
        assert confidence == 0.3

    def test_llm_merge_single_response(self):
        """Retourne directement une reponse unique."""
        engine = _make_consensus_engine()
        responses = [_make_model_response(model="m1", content="Solo")]
        content, model, confidence = engine._llm_merge(responses, "query")
        assert content == "Solo"
        assert model == "m1"


# ============================================================
# TESTS: EXECUTION PARALLELE
# ============================================================

class TestParallelExecution:
    """Tests pour l'execution parallele des modeles."""

    def test_query_models_parallel_mock(self):
        """Interroge les modeles en parallele et retourne les reponses."""
        engine = _make_consensus_engine()

        call_count = 0
        def mock_query_model(model, messages, temp):
            nonlocal call_count
            call_count += 1
            return _make_model_response(
                model=model,
                content=f"Response from {model}",
            )

        engine._query_model = mock_query_model

        responses = engine.query_models_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=["m1", "m2", "m3"],
        )
        assert len(responses) == 3
        assert call_count == 3

    def test_query_models_respects_max(self):
        """Respecte la limite max_models."""
        cfg = ConsensusConfig(max_models=2, default_models=["m1", "m2", "m3", "m4"])
        engine = _make_consensus_engine(config=cfg)
        engine._query_model = lambda m, msgs, t: _make_model_response(model=m)

        responses = engine.query_models_parallel(
            messages=[{"role": "user", "content": "test"}],
        )
        assert len(responses) == 2

    def test_query_models_callback(self):
        """Appelle le callback on_model_done pour chaque reponse."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(model=m)

        callback_results = []
        engine.query_models_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=["m1", "m2"],
            on_model_done=lambda resp: callback_results.append(resp.model),
        )
        assert len(callback_results) == 2

    def test_query_models_empty_list(self):
        """Retourne les resultats des modeles par defaut pour une liste vide (falsy)."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(model=m)

        # models=[] est falsy, donc les modeles par defaut sont utilises
        responses = engine.query_models_parallel(
            messages=[{"role": "user", "content": "test"}],
            models=[],
        )
        # Les modeles par defaut de la config sont utilises
        assert len(responses) == len(engine.config.default_models)


# ============================================================
# TESTS: RUN_CONSENSUS (INTEGRATION)
# ============================================================

class TestRunConsensus:
    """Tests pour le point d'entree principal."""

    def test_run_consensus_best_of_n(self):
        """Execute un consensus complet avec best_of_n."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(
            model=m,
            content=f"Response about Python from {m}",
        )

        result = engine.run_consensus("What is Python?", strategy=STRATEGY_BEST_OF_N)
        assert isinstance(result, ConsensusResult)
        assert result.strategy == STRATEGY_BEST_OF_N
        assert result.selected_response != ""
        assert result.confidence > 0.0
        assert len(result.individual_responses) == 3
        assert result.total_duration_ms >= 0

    def test_run_consensus_weighted_vote(self):
        """Execute un consensus avec weighted_vote."""
        engine = _make_consensus_engine()
        call_idx = [0]
        tiers = ["high", "medium", "low"]

        def mock_query(model, msgs, temp):
            idx = call_idx[0]
            call_idx[0] += 1
            return _make_model_response(
                model=model,
                content="Similar response about Python programming",
                quality_tier=tiers[idx % 3],
            )

        engine._query_model = mock_query

        result = engine.run_consensus("What is Python?", strategy=STRATEGY_WEIGHTED_VOTE)
        assert result.strategy == STRATEGY_WEIGHTED_VOTE
        assert result.selected_response != ""

    def test_run_consensus_llm_merge(self):
        """Execute un consensus avec llm_merge."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(
            model=m,
            content=f"Unique perspective from {m}",
        )
        engine._call_llm = MagicMock(return_value="Merged comprehensive answer")

        result = engine.run_consensus("Complex question", strategy=STRATEGY_LLM_MERGE)
        assert result.strategy == STRATEGY_LLM_MERGE
        assert "Merged" in result.selected_response

    def test_run_consensus_metadata(self):
        """Le resultat contient les metadonnees correctes."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(model=m)

        result = engine.run_consensus("test")
        assert "models_queried" in result.metadata
        assert "models_succeeded" in result.metadata
        assert "strategy_used" in result.metadata
        assert len(result.metadata["models_queried"]) == 3

    def test_run_consensus_with_system_prompt(self):
        """Le system prompt est inclus dans les messages."""
        engine = _make_consensus_engine()
        captured_messages = []

        def mock_query(model, messages, temp):
            captured_messages.append(messages)
            return _make_model_response(model=model)

        engine._query_model = mock_query

        result = engine.run_consensus("test", system_prompt="You are helpful.")
        # Verifier que les messages incluent le system prompt
        assert any(
            any(m.get("role") == "system" for m in msgs)
            for msgs in captured_messages
        )


# ============================================================
# TESTS: STREAMING (execute_consensus)
# ============================================================

class TestConsensusStreaming:
    """Tests pour l'execution en mode streaming."""

    def test_execute_consensus_yields_chunks(self):
        """Le generateur yield les chunks attendus."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(
            model=m,
            content=f"Response from {m}",
        )

        chunks = list(engine.execute_consensus("test"))
        # On doit avoir des tuples consensus_model_done, consensus_done, et du texte
        model_done_chunks = [c for c in chunks if isinstance(c, tuple) and c[0] == "consensus_model_done"]
        done_chunks = [c for c in chunks if isinstance(c, tuple) and c[0] == "consensus_done"]
        text_chunks = [c for c in chunks if isinstance(c, str)]

        assert len(model_done_chunks) == 3  # 3 modeles
        assert len(done_chunks) == 1
        assert len(text_chunks) > 0


# ============================================================
# TESTS: SERIALISATION
# ============================================================

class TestSerialization:
    """Tests pour la serialisation des resultats."""

    def test_result_to_dict(self):
        """Convertit correctement un ConsensusResult en dict."""
        result = ConsensusResult(
            strategy=STRATEGY_BEST_OF_N,
            selected_response="Best answer",
            selected_model="m1",
            confidence=0.85,
            individual_responses=[
                _make_model_response(model="m1", content="Answer 1"),
                _make_model_response(model="m2", content="Answer 2"),
            ],
            comparison=ConsensusComparison(
                agreement_matrix={"m1": {"m2": 0.7}, "m2": {"m1": 0.7}},
                average_agreement=0.7,
            ),
            total_duration_ms=1500,
            metadata={"test": True},
        )

        d = ConsensusEngine.result_to_dict(result)
        assert d["strategy"] == STRATEGY_BEST_OF_N
        assert d["selected_response"] == "Best answer"
        assert d["confidence"] == 0.85
        assert len(d["individual_responses"]) == 2
        assert d["comparison"]["average_agreement"] == 0.7
        assert d["total_duration_ms"] == 1500

    def test_result_to_dict_json_serializable(self):
        """Le dict est JSON-serialisable."""
        result = ConsensusResult(
            strategy=STRATEGY_BEST_OF_N,
            individual_responses=[_make_model_response()],
            comparison=ConsensusComparison(),
        )
        d = ConsensusEngine.result_to_dict(result)
        json_str = json.dumps(d)
        assert isinstance(json_str, str)


# ============================================================
# TESTS: INTEGRATION AGENTIC EXECUTOR
# ============================================================

class TestAgenticExecutorIntegration:
    """Tests d'integration avec l'AgenticExecutor."""

    def test_pipeline_consensus_constant_exists(self):
        """La constante PIPELINE_CONSENSUS existe."""
        assert PIPELINE_CONSENSUS == "consensus"

    def test_executor_has_consensus_available_property(self):
        """L'executor expose consensus_available."""
        mock_consensus = MagicMock()
        mock_consensus.available = True
        executor = _make_executor_with_consensus(mock_consensus)
        assert executor.consensus_available is True

    def test_executor_consensus_unavailable(self):
        """consensus_available retourne False si le moteur n'est pas fonctionnel."""
        mock_consensus = MagicMock()
        mock_consensus.available = False
        executor = _make_executor_with_consensus(mock_consensus)
        assert executor.consensus_available is False

    def test_executor_consensus_none_engine(self):
        """consensus_available avec _consensus_engine = None si globale patchee."""
        mock_executor = MagicMock()
        mock_executor.execute.return_value = iter(["test"])
        mock_executor.reset.return_value = None
        mock_executor.cancel.return_value = None

        # Patcher le singleton global pour forcer None
        with patch("opti_oignon.agentic_executor._default_consensus_engine", None):
            engine = AgenticExecutor(
                executor=mock_executor,
                consensus_engine=None,
            )
            assert engine.consensus_available is False

    def test_executor_consensus_explicit_dispatch(self):
        """Le consensus explicite utilise le pipeline consensus."""
        mock_consensus = MagicMock()
        mock_consensus.available = True
        mock_consensus.execute_consensus.return_value = iter([
            ("consensus_done", ConsensusResult(strategy="best_of_n")),
            "Consensus answer",
        ])

        executor = _make_executor_with_consensus(mock_consensus)
        routing = _make_routing()

        chunks = list(executor.execute(
            message="test question",
            routing=routing,
            consensus=True,
        ))
        assert executor._last_pipeline == PIPELINE_CONSENSUS

    def test_executor_consensus_fallback_when_unavailable(self):
        """Si consensus=True mais module non fonctionnel, fallback direct."""
        mock_consensus = MagicMock()
        mock_consensus.available = False
        executor = _make_executor_with_consensus(mock_consensus)
        routing = _make_routing()

        # consensus_available est False, donc le consensus est ignore
        # et le pipeline standard est utilise
        chunks = list(executor.execute(
            message="test question",
            routing=routing,
            consensus=True,
        ))
        # Le pipeline ne devrait pas etre consensus
        assert executor._last_pipeline != PIPELINE_CONSENSUS

    def test_executor_last_consensus_result(self):
        """L'executor stocke le dernier resultat de consensus."""
        mock_consensus = MagicMock()
        mock_consensus.available = True
        expected_result = ConsensusResult(strategy="best_of_n")
        mock_consensus.execute_consensus.return_value = iter([
            ("consensus_done", expected_result),
            "answer text",
        ])

        executor = _make_executor_with_consensus(mock_consensus)
        routing = _make_routing()
        list(executor.execute(message="test", routing=routing, consensus=True))

        assert executor.last_consensus_result == expected_result


# ============================================================
# TESTS: EDGE CASES
# ============================================================

class TestEdgeCases:
    """Tests pour les cas limites."""

    def test_all_models_fail(self):
        """Resultat correct quand tous les modeles echouent."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(
            model=m, content="", success=False, error="timeout",
        )

        result = engine.run_consensus("test")
        assert result.selected_response == ""
        assert result.confidence == 0.0

    def test_single_model_consensus(self):
        """Consensus avec un seul modele fonctionne."""
        cfg = ConsensusConfig(default_models=["solo"], max_models=1)
        engine = _make_consensus_engine(config=cfg)
        engine._query_model = lambda m, msgs, t: _make_model_response(
            model=m, content="Solo answer",
        )

        result = engine.run_consensus("test")
        assert result.selected_response == "Solo answer"
        assert result.selected_model == "solo"
        assert result.confidence == 0.5

    def test_mixed_success_failure(self):
        """Gere correctement un mix de reussites et echecs."""
        engine = _make_consensus_engine()
        call_idx = [0]
        def mock_query(model, msgs, temp):
            idx = call_idx[0]
            call_idx[0] += 1
            if idx == 1:
                return _make_model_response(model=model, content="", success=False)
            return _make_model_response(model=model, content=f"Valid response {idx}")

        engine._query_model = mock_query

        result = engine.run_consensus("test")
        assert result.selected_response != ""
        assert len(result.metadata["models_failed"]) >= 1

    def test_invalid_strategy_fallback_runtime(self):
        """Strategie invalide au runtime retombe sur best_of_n."""
        engine = _make_consensus_engine()
        engine._query_model = lambda m, msgs, t: _make_model_response(model=m)

        result = engine.run_consensus("test", strategy="nonexistent")
        assert result.strategy == STRATEGY_BEST_OF_N

    def test_singleton_exists(self):
        """Le singleton consensus_engine est instancie."""
        assert consensus_engine is not None
        assert isinstance(consensus_engine, ConsensusEngine)

    def test_engine_available_property(self):
        """La propriete available depend d'Ollama."""
        engine = _make_consensus_engine()
        assert engine.available == OLLAMA_AVAILABLE

    def test_last_result_initially_none(self):
        """Le dernier resultat est None apres initialisation."""
        engine = _make_consensus_engine()
        assert engine.last_result is None

    def test_config_property(self):
        """La propriete config retourne la configuration."""
        engine = _make_consensus_engine()
        assert engine.config is not None
        assert isinstance(engine.config, ConsensusConfig)


# ============================================================
# TESTS: BACKWARD COMPATIBILITY
# ============================================================

class TestBackwardCompatibility:
    """Verifie que les fonctionnalites existantes ne sont pas cassees."""

    def test_existing_pipeline_constants(self):
        """Les constantes de pipeline existantes sont preservees."""
        assert PIPELINE_DIRECT == "direct"
        assert PIPELINE_TOOLS == "tools"
        assert PIPELINE_CODE_VERIFY == "code_verify"
        assert PIPELINE_THINK == "think"
        assert PIPELINE_WEB_SEARCH == "web_search"
        assert PIPELINE_THINK_TOOLS == "think_tools"
        assert PIPELINE_REASONING == "reasoning"
        assert PIPELINE_CONSENSUS == "consensus"

    def test_executor_without_consensus_works(self):
        """L'executor fonctionne sans consensus engine."""
        executor = _make_executor_with_consensus(None)
        routing = _make_routing()

        # L'execution standard doit fonctionner
        chunks = list(executor.execute(
            message="simple question",
            routing=routing,
        ))
        assert executor._last_pipeline != PIPELINE_CONSENSUS

    def test_select_pipeline_unchanged(self):
        """La fonction _select_pipeline n'est pas modifiee."""
        classification = _quick_classify("Hello world")
        pipeline = _select_pipeline(
            classification=classification,
            think_override=None,
            web_search_override=None,
            tool_executor_available=False,
            verification_available=False,
            reasoning_available=False,
        )
        assert pipeline in (
            PIPELINE_DIRECT, PIPELINE_THINK, PIPELINE_WEB_SEARCH,
            PIPELINE_TOOLS, PIPELINE_CODE_VERIFY, PIPELINE_THINK_TOOLS,
            PIPELINE_REASONING,
        )

    def test_valid_strategies_set(self):
        """L'ensemble des strategies valides est correct."""
        assert {
            STRATEGY_BEST_OF_N,
            STRATEGY_WEIGHTED_VOTE,
            STRATEGY_LLM_MERGE,
        } == VALID_STRATEGIES
