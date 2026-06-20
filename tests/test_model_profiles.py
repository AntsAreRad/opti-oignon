#!/usr/bin/env python3
"""Tests pour le systeme de profils de modeles (S46 -- Smart Auto-Routing)."""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from opti_oignon.model_profiles import (
    VALID_QUALITY_TIERS,
    VALID_SPEED_TIERS,
    ModelProfile,
    ModelProfileManager,
    RoutingReason,
    find_best_for_task,
    get_profile,
    profile_manager,
)

# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_yaml_content():
    """Contenu YAML minimal pour les tests."""
    return {
        "profiles": {
            "qwen3-coder:30b": {
                "display_name": "Qwen3 Coder 30B",
                "capabilities": ["code", "reasoning"],
                "strengths": ["python", "debugging"],
                "weaknesses": ["creative_writing"],
                "context_window": 262144,
                "speed_tier": "medium",
                "quality_tier": "high",
                "recommended_for": ["code_python", "debug", "refactor"],
                "not_recommended_for": ["creative_writing"],
            },
            "nemotron-3-nano:30b": {
                "display_name": "Nemotron 3 Nano 30B",
                "capabilities": ["general", "fast"],
                "strengths": ["speed"],
                "weaknesses": ["complex_reasoning"],
                "context_window": 1048576,
                "speed_tier": "fast",
                "quality_tier": "medium",
                "recommended_for": ["quick_answer", "chat", "simple_question"],
                "not_recommended_for": ["complex_code"],
            },
            "deepseek-r1:32b": {
                "display_name": "DeepSeek R1 32B",
                "capabilities": ["reasoning", "code"],
                "strengths": ["chain_of_thought"],
                "weaknesses": ["speed"],
                "context_window": 131072,
                "speed_tier": "slow",
                "quality_tier": "high",
                "recommended_for": ["planning_deep", "mathematical", "reasoning"],
                "not_recommended_for": ["quick_answer", "chat"],
            },
        }
    }


@pytest.fixture
def yaml_file(sample_yaml_content, tmp_path):
    """Fichier YAML temporaire avec profils."""
    filepath = tmp_path / "model_profiles.yaml"
    with open(filepath, "w") as f:
        yaml.dump(sample_yaml_content, f)
    return filepath


@pytest.fixture
def manager(yaml_file):
    """ModelProfileManager charge avec les profils de test."""
    mgr = ModelProfileManager(profiles_path=yaml_file)
    mgr.load()
    return mgr


# =============================================================================
# TESTS: ModelProfile
# =============================================================================

class TestModelProfile:
    """Tests de la classe ModelProfile."""

    def test_creation_basic(self):
        """Test creation d'un profil basique."""
        profile = ModelProfile(name="test:7b")
        assert profile.name == "test:7b"
        assert profile.display_name == "test:7b"
        assert profile.capabilities == []
        assert profile.speed_tier == "medium"
        assert profile.quality_tier == "medium"

    def test_creation_with_fields(self):
        """Test creation avec tous les champs."""
        profile = ModelProfile(
            name="qwen3:32b",
            display_name="Qwen3 32B",
            capabilities=["code", "reasoning"],
            strengths=["python"],
            weaknesses=["slow"],
            context_window=262144,
            speed_tier="medium",
            quality_tier="high",
            recommended_for=["code_python"],
            not_recommended_for=["creative_writing"],
        )
        assert profile.display_name == "Qwen3 32B"
        assert "code" in profile.capabilities
        assert profile.context_window == 262144
        assert profile.quality_tier == "high"

    def test_invalid_speed_tier_fallback(self):
        """Test que les tiers invalides sont corriges."""
        profile = ModelProfile(name="test:7b", speed_tier="turbo")
        assert profile.speed_tier == "medium"

    def test_invalid_quality_tier_fallback(self):
        """Test que les tiers invalides sont corriges."""
        profile = ModelProfile(name="test:7b", quality_tier="ultra")
        assert profile.quality_tier == "medium"

    def test_matches_task_direct(self):
        """Test correspondance directe de tache."""
        profile = ModelProfile(
            name="coder:30b",
            recommended_for=["code_python", "debug"],
        )
        assert profile.matches_task("code_python") is True
        assert profile.matches_task("debug") is True
        assert profile.matches_task("creative_writing") is False

    def test_matches_task_prefix(self):
        """Test correspondance par prefixe."""
        profile = ModelProfile(
            name="coder:30b",
            recommended_for=["code"],
        )
        # "code" prefix match "code_python"
        assert profile.matches_task("code_python") is True
        assert profile.matches_task("code_r") is True

    def test_matches_task_exclusion(self):
        """Test exclusion explicite."""
        profile = ModelProfile(
            name="coder:30b",
            recommended_for=["code_python"],
            not_recommended_for=["creative_writing"],
        )
        assert profile.matches_task("creative_writing") is False

    def test_has_capability(self):
        """Test verification de capacite."""
        profile = ModelProfile(
            name="test:7b",
            capabilities=["code", "reasoning"],
        )
        assert profile.has_capability("code") is True
        assert profile.has_capability("vision") is False

    def test_score_for_task_recommended(self):
        """Test score pour tache recommandee."""
        profile = ModelProfile(
            name="coder:30b",
            quality_tier="high",
            recommended_for=["code_python"],
        )
        score = profile.score_for_task("code_python")
        assert score > 0.5

    def test_score_for_task_excluded(self):
        """Test score nul pour tache exclue."""
        profile = ModelProfile(
            name="coder:30b",
            not_recommended_for=["creative_writing"],
        )
        score = profile.score_for_task("creative_writing")
        assert score == 0.0

    def test_score_with_requirements(self):
        """Test score avec capacites requises."""
        profile = ModelProfile(
            name="test:7b",
            capabilities=["code", "reasoning"],
            recommended_for=["code_python"],
            quality_tier="high",
        )
        score_with = profile.score_for_task("code_python", requirements=["code"])
        score_without = profile.score_for_task("code_python")
        assert score_with >= score_without

    def test_score_unmatched_requirements(self):
        """Test score avec capacites non correspondantes."""
        profile = ModelProfile(
            name="test:7b",
            capabilities=["general"],
            recommended_for=["chat"],
            quality_tier="medium",
        )
        score = profile.score_for_task("chat", requirements=["vision"])
        # Score plus bas car vision non presente
        score_all = profile.score_for_task("chat", requirements=["general"])
        assert score_all >= score

    def test_to_dict(self):
        """Test serialisation en dictionnaire."""
        profile = ModelProfile(
            name="test:7b",
            display_name="Test 7B",
            capabilities=["code"],
            quality_tier="high",
        )
        d = profile.to_dict()
        assert d["name"] == "test:7b"
        assert d["display_name"] == "Test 7B"
        assert d["capabilities"] == ["code"]
        assert d["quality_tier"] == "high"


# =============================================================================
# TESTS: RoutingReason
# =============================================================================

class TestRoutingReason:
    """Tests de la classe RoutingReason."""

    def test_creation_default(self):
        """Test creation avec valeurs par defaut."""
        reason = RoutingReason(model="test:7b")
        assert reason.model == "test:7b"
        assert reason.profile_used is False
        assert reason.alternatives == []

    def test_to_dict(self):
        """Test serialisation."""
        reason = RoutingReason(
            model="qwen3:32b",
            display_name="Qwen3 32B",
            task_type="code_python",
            reason="Recommended for code_python",
            score=0.85,
            alternatives=["coder:30b"],
            profile_used=True,
        )
        d = reason.to_dict()
        assert d["model"] == "qwen3:32b"
        assert d["score"] == 0.85
        assert d["profile_used"] is True
        assert len(d["alternatives"]) == 1


# =============================================================================
# TESTS: ModelProfileManager
# =============================================================================

class TestModelProfileManager:
    """Tests du gestionnaire de profils."""

    def test_load_from_yaml(self, manager):
        """Test chargement depuis YAML."""
        assert manager.loaded is True
        assert manager.count == 3

    def test_load_nonexistent_file(self, tmp_path):
        """Test chargement fichier inexistant."""
        mgr = ModelProfileManager(profiles_path=tmp_path / "nonexistent.yaml")
        count = mgr.load()
        assert count == 0
        assert mgr.loaded is True

    def test_load_invalid_yaml(self, tmp_path):
        """Test chargement YAML invalide."""
        filepath = tmp_path / "bad.yaml"
        filepath.write_text(": invalid: yaml: [\n")
        mgr = ModelProfileManager(profiles_path=filepath)
        count = mgr.load()
        assert count == 0
        assert mgr.loaded is True

    def test_load_empty_profiles(self, tmp_path):
        """Test chargement avec section profiles vide."""
        filepath = tmp_path / "empty.yaml"
        with open(filepath, "w") as f:
            yaml.dump({"profiles": {}}, f)
        mgr = ModelProfileManager(profiles_path=filepath)
        count = mgr.load()
        assert count == 0

    def test_load_idempotent(self, yaml_file):
        """Test que le chargement est idempotent sans force_reload."""
        mgr = ModelProfileManager(profiles_path=yaml_file)
        count1 = mgr.load()
        count2 = mgr.load()
        assert count1 == count2

    def test_force_reload(self, yaml_file):
        """Test rechargement force."""
        mgr = ModelProfileManager(profiles_path=yaml_file)
        mgr.load()
        assert mgr.count == 3
        # Force reload
        count = mgr.load(force_reload=True)
        assert count == 3

    def test_get_profile_exists(self, manager):
        """Test recuperation d'un profil existant."""
        profile = manager.get_profile("qwen3-coder:30b")
        assert profile is not None
        assert profile.name == "qwen3-coder:30b"
        assert profile.display_name == "Qwen3 Coder 30B"

    def test_get_profile_not_found(self, manager):
        """Test recuperation d'un profil inexistant."""
        profile = manager.get_profile("nonexistent:7b")
        assert profile is None

    def test_list_profiles(self, manager):
        """Test liste de tous les profils."""
        profiles = manager.list_profiles()
        assert len(profiles) == 3
        names = [p.name for p in profiles]
        assert "qwen3-coder:30b" in names

    def test_list_profile_names(self, manager):
        """Test liste des noms de profils."""
        names = manager.list_profile_names()
        assert len(names) == 3
        assert "nemotron-3-nano:30b" in names

    def test_find_best_for_task_code(self, manager):
        """Test recherche pour tache code."""
        results = manager.find_best_for_task("code_python")
        assert len(results) > 0
        # Le coder devrait etre en tete (score le plus haut)
        assert results[0].name == "qwen3-coder:30b"

    def test_find_best_for_task_quick(self, manager):
        """Test recherche pour reponse rapide."""
        results = manager.find_best_for_task("quick_answer")
        assert len(results) > 0
        # Nemotron devrait matcher
        names = [p.name for p in results]
        assert "nemotron-3-nano:30b" in names

    def test_find_best_speed_filter(self, manager):
        """Test filtre par tier de vitesse."""
        results = manager.find_best_for_task("quick_answer", speed_tier="fast")
        # Seul nemotron est fast
        for p in results:
            assert p.speed_tier == "fast"

    def test_find_best_quality_filter(self, manager):
        """Test filtre par tier de qualite."""
        results = manager.find_best_for_task("code_python", quality_tier="high")
        for p in results:
            assert p.quality_tier == "high"

    def test_find_best_with_requirements(self, manager):
        """Test recherche avec capacites requises."""
        results = manager.find_best_for_task(
            "code_python",
            requirements=["code", "reasoning"],
        )
        assert len(results) > 0

    def test_find_best_limit(self, manager):
        """Test limitation du nombre de resultats."""
        results = manager.find_best_for_task("code_python", limit=1)
        assert len(results) <= 1

    def test_find_by_capability(self, manager):
        """Test recherche par capacite."""
        results = manager.find_by_capability("reasoning")
        names = [p.name for p in results]
        assert "deepseek-r1:32b" in names
        assert "qwen3-coder:30b" in names

    def test_find_by_capability_none(self, manager):
        """Test recherche par capacite inexistante."""
        results = manager.find_by_capability("vision")
        assert len(results) == 0

    def test_build_routing_reason_with_profile(self, manager):
        """Test construction raison de routage avec profil."""
        reason = manager.build_routing_reason(
            selected_model="qwen3-coder:30b",
            task_type="code_python",
            pipeline="tools",
        )
        assert reason.model == "qwen3-coder:30b"
        assert reason.display_name == "Qwen3 Coder 30B"
        assert reason.profile_used is True
        assert "code_python" in reason.reason.lower() or "recommend" in reason.reason.lower()

    def test_build_routing_reason_without_profile(self, manager):
        """Test construction raison sans profil (modele inconnu)."""
        reason = manager.build_routing_reason(
            selected_model="unknown:7b",
            task_type="chat",
        )
        assert reason.model == "unknown:7b"
        assert reason.profile_used is False
        assert "no profile" in reason.reason.lower()

    def test_build_routing_reason_with_alternatives(self, manager):
        """Test raison avec alternatives."""
        reason = manager.build_routing_reason(
            selected_model="qwen3-coder:30b",
            task_type="code_python",
            alternatives=["deepseek-r1:32b"],
        )
        assert "deepseek-r1:32b" in reason.alternatives

    def test_to_dict(self, manager):
        """Test export dictionnaire."""
        d = manager.to_dict()
        assert "profiles" in d
        assert "count" in d
        assert d["count"] == 3
        assert "qwen3-coder:30b" in d["profiles"]

    def test_auto_load_on_access(self, yaml_file):
        """Test chargement automatique au premier acces."""
        mgr = ModelProfileManager(profiles_path=yaml_file)
        assert mgr.loaded is False
        # L'acces declenche le chargement
        profile = mgr.get_profile("qwen3-coder:30b")
        assert mgr.loaded is True
        assert profile is not None


# =============================================================================
# TESTS: Singleton et fonctions de commodite
# =============================================================================

class TestSingleton:
    """Tests du singleton et des fonctions raccourcis."""

    def test_singleton_exists(self):
        """Test que le singleton est cree."""
        assert profile_manager is not None
        assert isinstance(profile_manager, ModelProfileManager)

    def test_convenience_get_profile(self):
        """Test fonction raccourci get_profile."""
        # Peut retourner None si profils pas charges, mais ne plante pas
        result = get_profile("nonexistent:7b")
        assert result is None

    def test_convenience_find_best(self):
        """Test fonction raccourci find_best_for_task."""
        results = find_best_for_task("code_python")
        assert isinstance(results, list)


# =============================================================================
# TESTS: Integration avec le Router
# =============================================================================

class TestRouterIntegration:
    """Tests d'integration avec le Router existant."""

    def test_router_import(self):
        """Test que le router importe correctement les profils."""
        from opti_oignon.router import MODEL_PROFILES_AVAILABLE
        # Doit etre True car model_profiles.py existe
        assert MODEL_PROFILES_AVAILABLE is True

    def test_routing_result_has_reason(self):
        """Test que RoutingResult a le champ routing_reason."""
        from opti_oignon.router import RoutingResult
        result = RoutingResult(
            model="test:7b",
            temperature=0.5,
            task_type="code_python",
            prompt_variant="standard",
            model_type="code",
            priority_used="primary",
            explanation="test",
            timeout=300,
            routing_reason={"model": "test:7b", "reason": "test"},
        )
        assert result.routing_reason is not None
        d = result.to_dict()
        assert "routing_reason" in d

    def test_routing_result_without_reason(self):
        """Test que RoutingResult fonctionne sans routing_reason (backward compat)."""
        from opti_oignon.router import RoutingResult
        result = RoutingResult(
            model="test:7b",
            temperature=0.5,
            task_type="code_python",
            prompt_variant="standard",
            model_type="code",
            priority_used="primary",
            explanation="test",
            timeout=300,
        )
        assert result.routing_reason is None
        d = result.to_dict()
        assert "routing_reason" not in d

    @patch("opti_oignon.router.ollama")
    def test_router_route_with_profiles(self, mock_ollama, yaml_file):
        """Test routage complet avec profils."""
        from opti_oignon.analyzer import AnalysisResult, Complexity, Language, TaskType
        from opti_oignon.router import ModelRouter

        # Simuler les modeles disponibles
        mock_model = MagicMock()
        mock_model.model = "qwen3-coder:30b"
        mock_response = MagicMock()
        mock_response.models = [mock_model]
        mock_ollama.list.return_value = mock_response

        # Configurer le profile manager avec notre YAML de test
        from opti_oignon import model_profiles
        original_manager = model_profiles.profile_manager
        test_manager = ModelProfileManager(profiles_path=yaml_file)
        test_manager.load()

        with patch.object(model_profiles, "profile_manager", test_manager):
            router = ModelRouter()
            analysis = AnalysisResult(
                task_type=TaskType.CODE_PYTHON,
                complexity=Complexity.MEDIUM,
                confidence=0.9,
                language=Language.PYTHON,
                keywords=["python"],
                is_debug=False,
                is_code=True,
                suggested_model_type="code",
                explanation="test",
            )
            result = router.route(analysis)

            assert result.model == "qwen3-coder:30b"
            assert result.routing_reason is not None

    def test_router_backward_compatible(self):
        """Test que le router fonctionne sans profils (backward compat)."""
        from opti_oignon.router import ModelRouter
        # Le router ne plante pas meme si les profils ne sont pas charges
        router = ModelRouter()
        assert router is not None


# =============================================================================
# TESTS: Chargement YAML du vrai fichier config
# =============================================================================

class TestRealYAMLConfig:
    """Tests sur le vrai fichier model_profiles.yaml."""

    def test_real_yaml_loads(self):
        """Test que le vrai fichier YAML se charge sans erreur."""
        from opti_oignon.model_profiles import _DEFAULT_PROFILES_PATH
        if not _DEFAULT_PROFILES_PATH.exists():
            pytest.skip("model_profiles.yaml not found")

        mgr = ModelProfileManager()
        count = mgr.load()
        assert count > 0

    def test_real_yaml_profiles_valid(self):
        """Test que tous les profils du vrai YAML sont valides."""
        from opti_oignon.model_profiles import _DEFAULT_PROFILES_PATH
        if not _DEFAULT_PROFILES_PATH.exists():
            pytest.skip("model_profiles.yaml not found")

        mgr = ModelProfileManager()
        mgr.load()
        for profile in mgr.list_profiles():
            assert profile.name
            assert profile.speed_tier in VALID_SPEED_TIERS
            assert profile.quality_tier in VALID_QUALITY_TIERS
            assert profile.context_window > 0

    def test_real_yaml_has_key_models(self):
        """Test que les modeles cles de Leon sont presents."""
        from opti_oignon.model_profiles import _DEFAULT_PROFILES_PATH
        if not _DEFAULT_PROFILES_PATH.exists():
            pytest.skip("model_profiles.yaml not found")

        mgr = ModelProfileManager()
        mgr.load()
        names = mgr.list_profile_names()
        # Modeles cles attendus
        assert "qwen3-coder:30b" in names
        assert "qwen3:32b" in names
        assert "nemotron-3-nano:30b" in names
        assert "deepseek-r1:32b" in names


# =============================================================================
# TESTS: Edge cases
# =============================================================================

class TestEdgeCases:
    """Tests des cas limites."""

    def test_empty_task_type(self, manager):
        """Test recherche avec type de tache vide."""
        results = manager.find_best_for_task("")
        # Ne devrait pas planter
        assert isinstance(results, list)

    def test_profile_with_no_recommendations(self):
        """Test profil sans recommandations."""
        profile = ModelProfile(name="bare:7b")
        assert profile.matches_task("anything") is False
        assert profile.score_for_task("anything") > 0  # Bonus qualite minimal

    def test_score_capped_at_one(self):
        """Test que le score ne depasse pas 1.0."""
        profile = ModelProfile(
            name="super:7b",
            capabilities=["code", "reasoning", "general"],
            quality_tier="high",
            recommended_for=["code"],
        )
        score = profile.score_for_task("code", requirements=["code", "reasoning", "general"])
        assert score <= 1.0

    def test_yaml_with_missing_fields(self, tmp_path):
        """Test chargement YAML avec champs manquants."""
        content = {
            "profiles": {
                "minimal:7b": {
                    "display_name": "Minimal",
                    # Pas de capabilities, strengths, etc.
                }
            }
        }
        filepath = tmp_path / "minimal.yaml"
        with open(filepath, "w") as f:
            yaml.dump(content, f)

        mgr = ModelProfileManager(profiles_path=filepath)
        count = mgr.load()
        assert count == 1
        profile = mgr.get_profile("minimal:7b")
        assert profile is not None
        assert profile.capabilities == []
        assert profile.speed_tier == "medium"

    def test_yaml_with_invalid_profile_entry(self, tmp_path):
        """Test que les entrees invalides sont ignorees."""
        content = {
            "profiles": {
                "valid:7b": {"display_name": "Valid"},
                "invalid": "not_a_dict",
            }
        }
        filepath = tmp_path / "mixed.yaml"
        with open(filepath, "w") as f:
            yaml.dump(content, f)

        mgr = ModelProfileManager(profiles_path=filepath)
        count = mgr.load()
        assert count == 1
