#!/usr/bin/env python3
"""
Tests S53 -- Pipeline Editor: ExecutionPipeline, PipelineStore, API
===================================================================

Tests unitaires couvrant:
- ExecutionStep: creation, validation, serialisation
- ExecutionPipeline: CRUD, validation, serialisation
- PipelineStore: chargement builtin, CRUD custom, YAML persistence
- API endpoints: list, get, create, update, delete, duplicate, step-types
- Pipeline Runner: condition evaluation, model override

Mode quick: pytest tests/test_pipelines.py --quick
Mode complet: pytest tests/test_pipelines.py -v
"""

import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

# Ajouter le repertoire parent au path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Mock des modules optionnels non disponibles en CI
for _mod_name in ["ollama", "chromadb", "chromadb.config", "chromadb.utils",
                   "chromadb.utils.embedding_functions"]:
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = type(sys)("mock_" + _mod_name)


# =====================================================================
# FIXTURES
# =====================================================================

@pytest.fixture
def temp_dirs():
    """Cree des repertoires temporaires pour config et data."""
    config_dir = Path(tempfile.mkdtemp())
    data_dir = Path(tempfile.mkdtemp())
    yield config_dir, data_dir
    shutil.rmtree(config_dir, ignore_errors=True)
    shutil.rmtree(data_dir, ignore_errors=True)


@pytest.fixture
def sample_step_dict():
    """Dictionnaire d'un step de test."""
    return {
        "step_type": "think",
        "label": "Planning Phase",
        "model_override": "qwen3:32b",
        "parameters": {"temperature": 0.7},
        "condition": "always",
        "pass_previous_output": True,
    }


@pytest.fixture
def sample_pipeline_dict():
    """Dictionnaire d'un pipeline de test."""
    return {
        "name": "Test Pipeline",
        "description": "Pipeline de test",
        "steps": [
            {"step_type": "think", "label": "Reflexion"},
            {"step_type": "code_verify", "label": "Verification"},
            {"step_type": "self_correct", "label": "Correction"},
        ],
    }


@pytest.fixture
def store_with_builtin(temp_dirs):
    """PipelineStore avec un pipeline builtin charge."""
    config_dir, data_dir = temp_dirs
    # Creer un fichier builtin
    builtin_data = {
        "id": "test-builtin",
        "name": "Test Builtin",
        "description": "Un pipeline builtin de test",
        "steps": [
            {"step_type": "direct", "label": "Response"},
        ],
    }
    with open(config_dir / "builtin_test.yaml", "w") as f:
        yaml.dump(builtin_data, f)

    from opti_oignon.pipelines import PipelineStore
    return PipelineStore(config_dir=config_dir, data_dir=data_dir)


# =====================================================================
# TESTS: ExecutionStep
# =====================================================================

class TestExecutionStep:
    """Tests pour ExecutionStep."""

    def test_creation_default(self):
        """Step avec valeurs par defaut."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="direct")
        assert step.step_type == "direct"
        assert step.label == "Direct"
        assert step.model_override is None
        assert step.parameters == {}
        assert step.condition is None
        assert step.pass_previous_output is True

    def test_creation_with_params(self, sample_step_dict):
        """Step avec tous les parametres."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(**sample_step_dict)
        assert step.step_type == "think"
        assert step.label == "Planning Phase"
        assert step.model_override == "qwen3:32b"
        assert step.parameters == {"temperature": 0.7}

    def test_from_dict(self, sample_step_dict):
        """Creation depuis un dictionnaire."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep.from_dict(sample_step_dict)
        assert step.step_type == "think"
        assert step.label == "Planning Phase"

    def test_to_dict(self):
        """Serialisation en dictionnaire."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(
            step_type="tools",
            label="Tool Execution",
            model_override="qwen3-coder:30b",
        )
        d = step.to_dict()
        assert d["step_type"] == "tools"
        assert d["label"] == "Tool Execution"
        assert d["model_override"] == "qwen3-coder:30b"
        # Champs vides non inclus
        assert "parameters" not in d
        assert "condition" not in d

    def test_to_dict_minimal(self):
        """Serialisation minimale sans champs optionnels."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="direct", label="Go")
        d = step.to_dict()
        assert set(d.keys()) == {"step_type", "label"}

    def test_to_dict_pass_previous_false(self):
        """pass_previous_output=False est inclus dans le dict."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="direct", pass_previous_output=False)
        d = step.to_dict()
        assert d["pass_previous_output"] is False

    def test_validate_valid(self):
        """Validation d'un step valide."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="think", label="Think")
        errors = step.validate()
        assert errors == []

    def test_validate_invalid_type(self):
        """Validation avec type invalide."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="invalid_type", label="Bad")
        errors = step.validate()
        assert len(errors) == 1
        assert "invalide" in errors[0]

    def test_validate_empty_label(self):
        """Validation avec label vide."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="direct", label="   ")
        errors = step.validate()
        assert len(errors) == 1
        assert "Label" in errors[0]

    def test_auto_label(self):
        """Label automatique depuis le type."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep(step_type="code_verify")
        assert step.label == "Code Verify"

    def test_roundtrip(self, sample_step_dict):
        """Aller-retour dict -> Step -> dict."""
        from opti_oignon.pipelines import ExecutionStep
        step = ExecutionStep.from_dict(sample_step_dict)
        d = step.to_dict()
        step2 = ExecutionStep.from_dict(d)
        assert step2.step_type == step.step_type
        assert step2.label == step.label
        assert step2.model_override == step.model_override

    def test_all_valid_types(self):
        """Tous les types valides passent la validation."""
        from opti_oignon.pipelines import VALID_STEP_TYPES, ExecutionStep
        for st in VALID_STEP_TYPES:
            step = ExecutionStep(step_type=st)
            assert step.validate() == [], f"Type {st} should be valid"


# =====================================================================
# TESTS: ExecutionPipeline
# =====================================================================

class TestExecutionPipeline:
    """Tests pour ExecutionPipeline."""

    def test_creation_minimal(self):
        """Pipeline avec le minimum requis."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test",
            name="Test",
            steps=[ExecutionStep(step_type="direct")],
        )
        assert pipeline.id == "test"
        assert pipeline.name == "Test"
        assert pipeline.step_count == 1
        assert pipeline.created_at != ""

    def test_from_dict(self, sample_pipeline_dict):
        """Creation depuis un dictionnaire."""
        from opti_oignon.pipelines import ExecutionPipeline
        pipeline = ExecutionPipeline.from_dict("test-id", sample_pipeline_dict)
        assert pipeline.id == "test-id"
        assert pipeline.name == "Test Pipeline"
        assert pipeline.step_count == 3

    def test_to_dict(self):
        """Serialisation en dictionnaire."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test",
            name="Test",
            description="A test",
            steps=[
                ExecutionStep(step_type="think", label="Think"),
                ExecutionStep(step_type="direct", label="Answer"),
            ],
        )
        d = pipeline.to_dict()
        assert d["name"] == "Test"
        assert d["description"] == "A test"
        assert len(d["steps"]) == 2
        assert d["steps"][0]["step_type"] == "think"

    def test_step_types_summary(self):
        """Resume des types d'etapes."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test",
            name="Test",
            steps=[
                ExecutionStep(step_type="think"),
                ExecutionStep(step_type="code_verify"),
                ExecutionStep(step_type="self_correct"),
            ],
        )
        assert pipeline.step_types_summary == "think -> code_verify -> self_correct"

    def test_validate_valid(self):
        """Validation d'un pipeline valide."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="valid_pipeline",
            name="Valid",
            steps=[ExecutionStep(step_type="direct")],
        )
        assert pipeline.validate() == []

    def test_validate_no_id(self):
        """Validation sans ID."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="",
            name="Test",
            steps=[ExecutionStep(step_type="direct")],
        )
        errors = pipeline.validate()
        assert any("ID" in e for e in errors)

    def test_validate_bad_id(self):
        """Validation avec ID invalide."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="123bad",
            name="Test",
            steps=[ExecutionStep(step_type="direct")],
        )
        errors = pipeline.validate()
        assert any("ID" in e for e in errors)

    def test_validate_no_name(self):
        """Validation sans nom."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test",
            name="  ",
            steps=[ExecutionStep(step_type="direct")],
        )
        errors = pipeline.validate()
        assert any("Nom" in e for e in errors)

    def test_validate_no_steps(self):
        """Validation sans etapes."""
        from opti_oignon.pipelines import ExecutionPipeline
        pipeline = ExecutionPipeline(id="test", name="Test", steps=[])
        errors = pipeline.validate()
        assert any("etape" in e for e in errors)

    def test_validate_invalid_step(self):
        """Validation avec etape invalide."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test",
            name="Test",
            steps=[ExecutionStep(step_type="nonexistent", label="Bad")],
        )
        errors = pipeline.validate()
        assert any("invalide" in e for e in errors)

    def test_from_dict_with_raw_dicts(self):
        """Creation avec des dicts bruts en guise de steps."""
        from opti_oignon.pipelines import ExecutionPipeline
        pipeline = ExecutionPipeline(
            id="test",
            name="Test",
            steps=[{"step_type": "direct", "label": "Go"}],
        )
        assert pipeline.step_count == 1
        assert pipeline.steps[0].step_type == "direct"

    def test_is_builtin_default_false(self):
        """is_builtin est False par defaut."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test", name="Test",
            steps=[ExecutionStep(step_type="direct")],
        )
        assert pipeline.is_builtin is False


# =====================================================================
# TESTS: PipelineStore
# =====================================================================

class TestPipelineStore:
    """Tests pour PipelineStore."""

    def test_empty_store(self, temp_dirs):
        """Store vide sans fichiers."""
        config_dir, data_dir = temp_dirs
        from opti_oignon.pipelines import PipelineStore
        store = PipelineStore(config_dir=config_dir, data_dir=data_dir)
        assert store.list_all() == []

    def test_load_builtin(self, store_with_builtin):
        """Chargement d'un pipeline builtin."""
        store = store_with_builtin
        all_pipelines = store.list_all()
        assert len(all_pipelines) == 1
        assert all_pipelines[0].id == "test-builtin"
        assert all_pipelines[0].is_builtin is True

    def test_get_existing(self, store_with_builtin):
        """Recuperation par ID."""
        store = store_with_builtin
        p = store.get("test-builtin")
        assert p is not None
        assert p.name == "Test Builtin"

    def test_get_nonexistent(self, store_with_builtin):
        """Recuperation d'un ID inexistant."""
        store = store_with_builtin
        assert store.get("nonexistent") is None

    def test_create_custom(self, store_with_builtin):
        """Creation d'un pipeline custom."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="custom-test",
            name="Custom Test",
            steps=[ExecutionStep(step_type="think")],
        )
        assert store.create(pipeline) is True
        assert store.get("custom-test") is not None
        assert len(store.list_custom()) == 1

    def test_create_duplicate_id(self, store_with_builtin):
        """Impossible de creer avec un ID existant."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="test-builtin",
            name="Duplicate",
            steps=[ExecutionStep(step_type="direct")],
        )
        assert store.create(pipeline) is False

    def test_create_invalid(self, store_with_builtin):
        """Impossible de creer un pipeline invalide."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline
        pipeline = ExecutionPipeline(id="", name="Bad", steps=[])
        assert store.create(pipeline) is False

    def test_update_custom(self, store_with_builtin):
        """Mise a jour d'un pipeline custom."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        # Creer d'abord
        pipeline = ExecutionPipeline(
            id="updatable",
            name="Before",
            steps=[ExecutionStep(step_type="direct")],
        )
        store.create(pipeline)
        # Modifier
        updated = ExecutionPipeline(
            id="updatable",
            name="After",
            steps=[
                ExecutionStep(step_type="think"),
                ExecutionStep(step_type="direct"),
            ],
        )
        assert store.update("updatable", updated) is True
        result = store.get("updatable")
        assert result.name == "After"
        assert result.step_count == 2

    def test_update_builtin_fails(self, store_with_builtin):
        """Impossible de modifier un builtin."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        updated = ExecutionPipeline(
            id="test-builtin",
            name="Modified",
            steps=[ExecutionStep(step_type="direct")],
        )
        assert store.update("test-builtin", updated) is False

    def test_delete_custom(self, store_with_builtin):
        """Suppression d'un pipeline custom."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        pipeline = ExecutionPipeline(
            id="deletable",
            name="Deletable",
            steps=[ExecutionStep(step_type="direct")],
        )
        store.create(pipeline)
        assert store.delete("deletable") is True
        assert store.get("deletable") is None

    def test_delete_builtin_fails(self, store_with_builtin):
        """Impossible de supprimer un builtin."""
        store = store_with_builtin
        assert store.delete("test-builtin") is False

    def test_delete_nonexistent(self, store_with_builtin):
        """Suppression d'un ID inexistant retourne False."""
        store = store_with_builtin
        assert store.delete("nonexistent") is False

    def test_duplicate_builtin(self, store_with_builtin):
        """Duplication d'un builtin."""
        store = store_with_builtin
        dup = store.duplicate("test-builtin", "my-copy")
        assert dup is not None
        assert dup.id == "my-copy"
        assert dup.is_builtin is False
        assert dup.step_count == store.get("test-builtin").step_count

    def test_duplicate_nonexistent(self, store_with_builtin):
        """Duplication d'un pipeline inexistant."""
        store = store_with_builtin
        assert store.duplicate("nonexistent", "copy") is None

    def test_duplicate_existing_target(self, store_with_builtin):
        """Duplication vers un ID existant echoue."""
        store = store_with_builtin
        assert store.duplicate("test-builtin", "test-builtin") is None

    def test_list_builtin_only(self, store_with_builtin):
        """Filtrage builtin uniquement."""
        store = store_with_builtin
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep
        store.create(ExecutionPipeline(
            id="custom1", name="C1",
            steps=[ExecutionStep(step_type="direct")],
        ))
        assert len(store.list_builtin()) == 1
        assert len(store.list_custom()) == 1
        assert len(store.list_all()) == 2

    def test_persistence_yaml(self, temp_dirs):
        """Les custom sont persistes en YAML et recharges."""
        config_dir, data_dir = temp_dirs
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep, PipelineStore

        store1 = PipelineStore(config_dir=config_dir, data_dir=data_dir)
        store1.create(ExecutionPipeline(
            id="persisted",
            name="Persisted",
            steps=[ExecutionStep(step_type="reasoning")],
        ))

        # Recharger depuis les fichiers
        store2 = PipelineStore(config_dir=config_dir, data_dir=data_dir)
        p = store2.get("persisted")
        assert p is not None
        assert p.name == "Persisted"
        assert p.steps[0].step_type == "reasoning"

    def test_get_step_types(self, store_with_builtin):
        """Liste des types de step."""
        store = store_with_builtin
        types = store.get_step_types()
        assert len(types) == 9
        type_ids = [t["type"] for t in types]
        assert "direct" in type_ids
        assert "reasoning" in type_ids
        assert "self_correct" in type_ids

    def test_load_multi_format_yaml(self, temp_dirs):
        """Chargement du format multi-pipeline dans un fichier."""
        config_dir, data_dir = temp_dirs
        multi_data = {
            "pipeline-a": {
                "name": "Pipeline A",
                "steps": [{"step_type": "direct", "label": "Go"}],
            },
            "pipeline-b": {
                "name": "Pipeline B",
                "steps": [{"step_type": "think", "label": "Think"}],
            },
        }
        with open(config_dir / "multi.yaml", "w") as f:
            yaml.dump(multi_data, f)

        from opti_oignon.pipelines import PipelineStore
        store = PipelineStore(config_dir=config_dir, data_dir=data_dir)
        assert len(store.list_all()) == 2
        assert store.get("pipeline-a") is not None
        assert store.get("pipeline-b") is not None


# =====================================================================
# TESTS: PipelineRunner
# =====================================================================

class TestPipelineRunner:
    """Tests pour PipelineRunner."""

    def test_evaluate_condition_always(self):
        """Condition 'always' retourne True."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        assert runner._evaluate_condition("always", "test", "") is True

    def test_evaluate_condition_empty(self):
        """Condition vide retourne True."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        assert runner._evaluate_condition("", "test", "") is True

    def test_evaluate_condition_code_detected(self):
        """Condition 'if_code_detected' avec du code."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        assert runner._evaluate_condition(
            "if_code_detected", "def hello():\n    pass", ""
        ) is True

    def test_evaluate_condition_code_not_detected(self):
        """Condition 'if_code_detected' sans code."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        assert runner._evaluate_condition(
            "if_code_detected", "Hello world", ""
        ) is False

    def test_evaluate_condition_long_input(self):
        """Condition 'if_long_input' avec texte long."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        long_text = "a" * 600
        assert runner._evaluate_condition("if_long_input", long_text, "") is True
        assert runner._evaluate_condition("if_long_input", "short", "") is False

    def test_evaluate_condition_unknown(self):
        """Condition inconnue retourne True par defaut."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        assert runner._evaluate_condition("unknown_condition", "x", "") is True

    def test_override_model(self):
        """Override de modele sur le routing."""
        from opti_oignon.pipelines import PipelineRunner
        runner = PipelineRunner()
        routing = MagicMock()
        routing.model = "original"
        new_routing = runner._override_model(routing, "override_model")
        assert new_routing.model == "override_model"

    def test_execute_no_executor(self):
        """Execution sans executor disponible."""
        from opti_oignon.pipelines import ExecutionPipeline, ExecutionStep, PipelineRunner
        runner = PipelineRunner(agentic_executor=None)
        # Mock pour empecher le lazy import
        runner._get_executor = lambda: None
        pipeline = ExecutionPipeline(
            id="test", name="Test",
            steps=[ExecutionStep(step_type="direct")],
        )
        chunks = list(runner.execute(pipeline, "hello", MagicMock()))
        assert any("[ERR]" in str(c) for c in chunks)

    def test_execute_empty_pipeline(self):
        """Execution d'un pipeline vide."""
        from opti_oignon.pipelines import ExecutionPipeline, PipelineRunner
        runner = PipelineRunner(agentic_executor=MagicMock())
        pipeline = ExecutionPipeline(id="empty", name="Empty", steps=[])
        chunks = list(runner.execute(pipeline, "hello", MagicMock()))
        assert any("[ERR]" in str(c) for c in chunks)


# =====================================================================
# TESTS: API Endpoints
# =====================================================================

class TestExecPipelineAPI:
    """Tests pour les endpoints API des execution pipelines."""

    @pytest.fixture(autouse=True)
    def setup_client(self, temp_dirs):
        """Configure le client de test FastAPI."""
        config_dir, data_dir = temp_dirs

        # Creer un builtin pour les tests
        builtin = {
            "id": "api-builtin",
            "name": "API Builtin",
            "steps": [{"step_type": "direct", "label": "Go"}],
        }
        with open(config_dir / "test.yaml", "w") as f:
            yaml.dump(builtin, f)

        # Patcher le store singleton
        from opti_oignon.pipelines import PipelineStore
        test_store = PipelineStore(config_dir=config_dir, data_dir=data_dir)

        with patch(
            "opti_oignon.api.routes_exec_pipelines.get_pipeline_store",
            return_value=test_store,
        ), patch(
            "opti_oignon.api.routes_exec_pipelines.EXEC_PIPELINES_AVAILABLE",
            True,
        ):
            from fastapi import FastAPI
            from fastapi.testclient import TestClient

            from opti_oignon.api.routes_exec_pipelines import router

            app = FastAPI()
            app.include_router(router)
            self.client = TestClient(app)
            self.store = test_store
            yield

    def test_list_pipelines(self):
        """GET /api/execution-pipelines retourne la liste."""
        resp = self.client.get("/api/execution-pipelines")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_list_builtin_only(self):
        """Filtrage builtin uniquement."""
        resp = self.client.get("/api/execution-pipelines?builtin_only=true")
        assert resp.status_code == 200
        data = resp.json()
        assert all(p["is_builtin"] for p in data)

    def test_list_step_types(self):
        """GET /api/execution-pipelines/step-types."""
        resp = self.client.get("/api/execution-pipelines/step-types")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 9
        types = [t["type"] for t in data]
        assert "direct" in types
        assert "reasoning" in types

    def test_get_pipeline(self):
        """GET /api/execution-pipelines/{id}."""
        resp = self.client.get("/api/execution-pipelines/api-builtin")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "api-builtin"
        assert data["name"] == "API Builtin"

    def test_get_not_found(self):
        """GET pipeline inexistant retourne 404."""
        resp = self.client.get("/api/execution-pipelines/nonexistent")
        assert resp.status_code == 404

    def test_create_pipeline(self):
        """POST /api/execution-pipelines."""
        resp = self.client.post("/api/execution-pipelines", json={
            "id": "new-pipeline",
            "name": "New Pipeline",
            "description": "A test pipeline",
            "steps": [
                {"step_type": "think", "label": "Think"},
                {"step_type": "direct", "label": "Answer"},
            ],
        })
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "new-pipeline"
        assert data["step_count"] == 2

    def test_create_duplicate_id(self):
        """Creation avec ID existant retourne 409."""
        resp = self.client.post("/api/execution-pipelines", json={
            "id": "api-builtin",
            "name": "Dup",
            "steps": [{"step_type": "direct", "label": "Go"}],
        })
        assert resp.status_code == 409

    def test_create_empty_id(self):
        """Creation avec ID vide retourne 422."""
        resp = self.client.post("/api/execution-pipelines", json={
            "id": "  ",
            "name": "Bad",
            "steps": [{"step_type": "direct", "label": "Go"}],
        })
        assert resp.status_code == 422

    def test_create_invalid_step_type(self):
        """Creation avec type de step invalide retourne 422."""
        resp = self.client.post("/api/execution-pipelines", json={
            "id": "bad-steps",
            "name": "Bad Steps",
            "steps": [{"step_type": "nonexistent", "label": "Bad"}],
        })
        assert resp.status_code == 422

    def test_update_pipeline(self):
        """PUT /api/execution-pipelines/{id}."""
        # Creer d'abord
        self.client.post("/api/execution-pipelines", json={
            "id": "updatable",
            "name": "Before",
            "steps": [{"step_type": "direct", "label": "Go"}],
        })
        # Modifier
        resp = self.client.put("/api/execution-pipelines/updatable", json={
            "name": "After",
            "steps": [
                {"step_type": "think", "label": "Think"},
                {"step_type": "direct", "label": "Answer"},
            ],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["name"] == "After"
        assert data["step_count"] == 2

    def test_update_builtin_fails(self):
        """Modification d'un builtin retourne 403."""
        resp = self.client.put("/api/execution-pipelines/api-builtin", json={
            "name": "Modified",
        })
        assert resp.status_code == 403

    def test_update_not_found(self):
        """Modification d'un pipeline inexistant retourne 404."""
        resp = self.client.put("/api/execution-pipelines/nonexistent", json={
            "name": "X",
        })
        assert resp.status_code == 404

    def test_delete_pipeline(self):
        """DELETE /api/execution-pipelines/{id}."""
        self.client.post("/api/execution-pipelines", json={
            "id": "deletable",
            "name": "Delete Me",
            "steps": [{"step_type": "direct", "label": "Go"}],
        })
        resp = self.client.delete("/api/execution-pipelines/deletable")
        assert resp.status_code == 200
        assert resp.json()["deleted"] is True

    def test_delete_builtin_fails(self):
        """Suppression d'un builtin retourne 403."""
        resp = self.client.delete("/api/execution-pipelines/api-builtin")
        assert resp.status_code == 403

    def test_delete_not_found(self):
        """Suppression d'un pipeline inexistant retourne 404."""
        resp = self.client.delete("/api/execution-pipelines/nonexistent")
        assert resp.status_code == 404

    def test_duplicate_pipeline(self):
        """POST /api/execution-pipelines/{id}/duplicate."""
        resp = self.client.post(
            "/api/execution-pipelines/api-builtin/duplicate",
            json={"new_id": "my-copy"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["id"] == "my-copy"
        assert data["is_builtin"] is False

    def test_duplicate_not_found(self):
        """Duplication d'un pipeline inexistant retourne 404."""
        resp = self.client.post(
            "/api/execution-pipelines/nonexistent/duplicate",
            json={"new_id": "copy"},
        )
        assert resp.status_code == 404

    def test_duplicate_existing_target(self):
        """Duplication vers un ID existant retourne 409."""
        resp = self.client.post(
            "/api/execution-pipelines/api-builtin/duplicate",
            json={"new_id": "api-builtin"},
        )
        assert resp.status_code == 409

    def test_pipeline_info_fields(self):
        """Verification de tous les champs de PipelineInfo."""
        self.client.post("/api/execution-pipelines", json={
            "id": "full-info",
            "name": "Full Info",
            "description": "Test all fields",
            "steps": [
                {"step_type": "think", "label": "A"},
                {"step_type": "code_verify", "label": "B"},
            ],
        })
        resp = self.client.get("/api/execution-pipelines/full-info")
        data = resp.json()
        assert data["step_count"] == 2
        assert data["step_types_summary"] == "think -> code_verify"
        assert data["is_builtin"] is False
        assert "created_at" in data
        assert "updated_at" in data


# =====================================================================
# TESTS: Builtin Pipeline Loading
# =====================================================================

class TestBuiltinPipelines:
    """Tests pour le chargement des pipelines builtin du projet."""

    def test_builtin_config_dir_exists(self):
        """Le repertoire config/pipelines/ existe."""
        config_dir = Path(__file__).resolve().parent.parent / "opti_oignon" / "config" / "pipelines"
        assert config_dir.exists(), f"Missing: {config_dir}"

    def test_builtin_yaml_files_exist(self):
        """Au moins 3 fichiers YAML builtin."""
        config_dir = Path(__file__).resolve().parent.parent / "opti_oignon" / "config" / "pipelines"
        yamls = list(config_dir.glob("*.yaml"))
        assert len(yamls) >= 3, f"Found only {len(yamls)} YAML files"

    def test_builtin_yaml_parseable(self):
        """Tous les YAML builtin sont valides."""
        config_dir = Path(__file__).resolve().parent.parent / "opti_oignon" / "config" / "pipelines"
        for yaml_file in config_dir.glob("*.yaml"):
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
            assert data is not None, f"Empty YAML: {yaml_file}"
            assert "steps" in data or any(
                isinstance(v, dict) and "steps" in v
                for v in data.values()
                if isinstance(v, dict)
            ), f"No steps in {yaml_file}"

    def test_load_real_builtins(self):
        """Chargement des vrais builtins du projet."""
        config_dir = Path(__file__).resolve().parent.parent / "opti_oignon" / "config" / "pipelines"
        data_dir = Path(tempfile.mkdtemp())
        try:
            from opti_oignon.pipelines import PipelineStore
            store = PipelineStore(config_dir=config_dir, data_dir=data_dir)
            builtins = store.list_builtin()
            assert len(builtins) >= 3
            # Verifier quelques pipelines specifiques
            ids = [p.id for p in builtins]
            assert "code-expert" in ids
            assert "research-assistant" in ids
            assert "creative-writer" in ids
        finally:
            shutil.rmtree(data_dir, ignore_errors=True)


# =====================================================================
# TESTS: Constants and Module-level
# =====================================================================

class TestModuleConstants:
    """Tests pour les constantes du module."""

    def test_valid_step_types_count(self):
        """9 types de step valides."""
        from opti_oignon.pipelines import VALID_STEP_TYPES
        assert len(VALID_STEP_TYPES) == 9

    def test_step_type_descriptions(self):
        """Chaque type a une description."""
        from opti_oignon.pipelines import STEP_TYPE_DESCRIPTIONS, VALID_STEP_TYPES
        for st in VALID_STEP_TYPES:
            assert st in STEP_TYPE_DESCRIPTIONS
            assert len(STEP_TYPE_DESCRIPTIONS[st]) > 0

    def test_singleton_store(self):
        """get_pipeline_store retourne un singleton."""
        from opti_oignon.pipelines import get_pipeline_store
        s1 = get_pipeline_store()
        s2 = get_pipeline_store()
        assert s1 is s2

    def test_singleton_runner(self):
        """get_pipeline_runner retourne un singleton."""
        from opti_oignon.pipelines import get_pipeline_runner
        r1 = get_pipeline_runner()
        r2 = get_pipeline_runner()
        assert r1 is r2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
