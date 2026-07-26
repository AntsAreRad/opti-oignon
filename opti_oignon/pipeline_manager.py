#!/usr/bin/env python3
"""
PIPELINE MANAGER - OPTI-OIGNON 1.2
==================================

Management of custom multi-agent pipelines.

Features:
- View all pipelines (builtin + custom)
- Create new custom pipelines
- Modify/delete existing pipelines
- Auto-generate system prompts via LLM
- Import/export pipelines

Builtin pipelines (in agents/config.yaml) are read-only.
Custom pipelines are stored in data/pipelines_custom.yaml.

Author: Léon
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .config import DATA_DIR, load_yaml, save_yaml

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PipelineStep:
    """
    Represents a step in a pipeline.

    Attributes:
        name: Step name (e.g. "Analysis", "Code Generation")
        agent: Agent ID to use (e.g. "coder", "reviewer")
        prompt_template: Template ID or None if custom
        description: Step description
        system_prompt: Custom system prompt (if prompt_template=None)
        model: Specific model to use (overrides agent default)
    """
    name: str
    agent: str
    prompt_template: str | None = None
    description: str = ""
    system_prompt: str | None = None
    model: str | None = None  # Override model for this step

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        d = {
            "name": self.name,
            "agent": self.agent,
            "description": self.description,
        }
        if self.prompt_template:
            d["prompt_template"] = self.prompt_template
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.model:
            d["model"] = self.model
        return d

    @classmethod
    def from_dict(cls, data: dict) -> 'PipelineStep':
        """Create a PipelineStep from a dictionary."""
        return cls(
            name=data.get("name", "Step"),
            agent=data.get("agent", "coder"),
            prompt_template=data.get("prompt_template"),
            description=data.get("description", ""),
            system_prompt=data.get("system_prompt"),
            model=data.get("model"),
        )


@dataclass
class Pipeline:
    """
    Represents a complete multi-agent pipeline.

    Attributes:
        id: Unique pipeline identifier
        name: Display name
        description: Pipeline description
        pattern: Orchestration pattern (chain, verifier, decomposition, iterative)
        emoji: Emoji for display
        steps: List of pipeline steps
        keywords: Keywords for auto-detection
        detection_weight: Weight for scoring (0.0-1.0)
        created_at: Creation date
        is_builtin: True if defined in config.yaml (read-only)
    """
    id: str
    name: str
    description: str = ""
    pattern: str = "chain"
    emoji: str = "🔧"
    steps: list[PipelineStep] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    detection_weight: float = 0.5
    created_at: str | None = None
    is_builtin: bool = False

    def __post_init__(self):
        # Ensure steps are PipelineStep objects
        if self.steps and isinstance(self.steps[0], dict):
            self.steps = [PipelineStep.from_dict(s) for s in self.steps]

        # Clamp detection_weight between 0 and 1
        self.detection_weight = max(0.0, min(1.0, self.detection_weight))

        # Creation date if not set
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict:
        """Convert to dictionary for YAML export."""
        return {
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "emoji": self.emoji,
            "steps": [s.to_dict() for s in self.steps],
            "auto_detect": {
                "keywords": self.keywords,
            } if self.keywords else None,
            "detection_weight": self.detection_weight,
            "created_at": self.created_at,
            "is_builtin": self.is_builtin,
        }

    def to_config_dict(self) -> dict:
        """
        Convert to config.yaml format (compatible with the orchestrator).
        """
        d = {
            "name": self.name,
            "description": self.description,
            "pattern": self.pattern,
            "steps": [s.to_dict() for s in self.steps],
        }
        if self.keywords:
            d["auto_detect"] = {"keywords": self.keywords}
        return d

    @classmethod
    def from_dict(cls, pipeline_id: str, data: dict, is_builtin: bool = False) -> 'Pipeline':
        """Create a Pipeline from a dictionary."""
        # Extract keywords from auto_detect if present
        auto_detect = data.get("auto_detect", {}) or {}
        keywords = auto_detect.get("keywords", []) if isinstance(auto_detect, dict) else []

        # Parser les steps
        steps_data = data.get("steps", [])
        steps = [PipelineStep.from_dict(s) for s in steps_data]

        return cls(
            id=pipeline_id,
            name=data.get("name", pipeline_id),
            description=data.get("description", ""),
            pattern=data.get("pattern", "chain"),
            emoji=data.get("emoji", "🔧"),
            steps=steps,
            keywords=keywords,
            detection_weight=data.get("detection_weight", 0.5),
            created_at=data.get("created_at"),
            is_builtin=is_builtin,
        )

    def matches_keywords(self, text: str) -> int:
        """
        Count how many keywords match in the text.

        Args:
            text: Text to analyze

        Returns:
            Number of keywords found
        """
        if not self.keywords:
            return 0

        text_lower = text.lower()
        matches = 0

        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                matches += 1

        return matches

    def get_weighted_score(self, text: str) -> float:
        """Compute the weighted score based on keyword matches."""
        matches = self.matches_keywords(text)
        return matches * self.detection_weight

    @property
    def step_count(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    def validate(self, available_agents: list[str], available_templates: list[str]) -> list[str]:
        """
        Validate the pipeline.

        Args:
            available_agents: List of available agent IDs
            available_templates: List of available template IDs

        Returns:
            List of errors (empty if valid)
        """
        errors = []

        if not self.id:
            errors.append("Pipeline ID is required")
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', self.id):
            errors.append("ID can only contain letters, digits, _ and -")

        if not self.name:
            errors.append("Pipeline name is required")

        if not self.steps:
            errors.append("At least one step is required")

        for i, step in enumerate(self.steps):
            if not step.name:
                errors.append(f"Step {i+1}: name required")

            if step.agent and step.agent != "auto" and step.agent not in available_agents:
                errors.append(f"Step {i+1}: agent '{step.agent}' unknown")

            if step.prompt_template and step.prompt_template not in available_templates:
                # Not a blocking error, system_prompt can be used
                if not step.system_prompt:
                    errors.append(f"Step {i+1}: template '{step.prompt_template}' unknown and no system_prompt")

        return errors


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager:
    """
    Multi-agent pipeline manager.

    Manages builtin pipelines (config.yaml, read-only) and
    custom pipelines (pipelines_custom.yaml, editable).

    Usage:
        manager = PipelineManager()
        pipelines = manager.list_all()
        manager.create(pipeline)
        manager.update(pipeline_id, pipeline)
        manager.delete(pipeline_id)
    """

    def __init__(self):
        """Initialize the manager."""
        self._pipelines: dict[str, Pipeline] = {}

        # File paths
        self._config_file = Path(__file__).parent / "agents" / "config.yaml"
        self._custom_file = DATA_DIR / "pipelines_custom.yaml"

        # Cache of available agents and templates
        self._available_agents: list[str] = []
        self._available_templates: list[str] = []

        self._load_all()

    def _load_all(self) -> None:
        """Load all pipelines (builtin + custom)."""
        self._pipelines = {}

        # 1. Load builtin pipelines from config.yaml
        self._load_builtin()

        # 2. Load custom pipelines
        self._load_custom()

        # 3. Update template list (includes those from custom pipelines)
        self._update_templates_list()

        logger.info(f"{len(self._pipelines)} pipelines loaded")

    def _update_templates_list(self) -> None:
        """Update the list of available templates from all pipelines."""
        # Collect templates used across all pipelines
        used_templates = set(self._available_templates)  # Keep already defined ones
        for pipeline in self._pipelines.values():
            for step in pipeline.steps:
                if step.prompt_template:
                    used_templates.add(step.prompt_template)
        self._available_templates = sorted(used_templates)

    def _load_builtin(self) -> None:
        """Load pipelines from agents/config.yaml."""
        if not self._config_file.exists():
            logger.warning(f"Config file not found: {self._config_file}")
            return

        try:
            with open(self._config_file, encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}

            # Load pipelines
            pipelines_data = config.get("pipelines", {})
            for pipe_id, pipe_data in pipelines_data.items():
                self._pipelines[pipe_id] = Pipeline.from_dict(
                    pipe_id, pipe_data, is_builtin=True
                )

            # Load list of available agents
            self._available_agents = list(config.get("agents", {}).keys())

            # Load available templates (defined in prompt_templates)
            defined_templates = set(config.get("prompt_templates", {}).keys())

            # Also collect templates used in pipelines
            used_templates = set()
            for pipeline in self._pipelines.values():
                for step in pipeline.steps:
                    if step.prompt_template:
                        used_templates.add(step.prompt_template)

            # Merge: defined templates + used templates
            self._available_templates = sorted(defined_templates | used_templates)

            logger.debug(f"Loaded {len(pipelines_data)} builtin pipelines")

        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")

    def _load_custom(self) -> None:
        """Load custom pipelines from data/pipelines_custom.yaml."""
        if not self._custom_file.exists():
            return

        try:
            data = load_yaml(self._custom_file)
            pipelines_data = data.get("pipelines", {})

            for pipe_id, pipe_data in pipelines_data.items():
                # Custom entries must not overwrite builtins
                # (mirrors import_from_yaml and the execution-pipeline store).
                if pipe_id in self._pipelines and self._pipelines[pipe_id].is_builtin:
                    logger.warning(
                        f"Skipping custom pipeline '{pipe_id}': "
                        f"ID already used by a builtin"
                    )
                    continue
                self._pipelines[pipe_id] = Pipeline.from_dict(
                    pipe_id, pipe_data, is_builtin=False
                )

            logger.debug(f"Loaded {len(pipelines_data)} custom pipelines")

        except Exception as e:
            logger.error(f"Error loading custom pipelines: {e}")

    def _save_custom(self) -> bool:
        """Save custom pipelines."""
        try:
            custom_pipelines = {}

            for pipe_id, pipeline in self._pipelines.items():
                if not pipeline.is_builtin:
                    custom_pipelines[pipe_id] = pipeline.to_dict()

            # Ensure directory exists
            self._custom_file.parent.mkdir(parents=True, exist_ok=True)

            return save_yaml(self._custom_file, {"pipelines": custom_pipelines})

        except Exception as e:
            logger.error(f"Error saving custom pipelines: {e}")
            return False

    def reload(self) -> None:
        """Reload all pipelines."""
        self._load_all()

    # -------------------------------------------------------------------------
    # Pipeline access
    # -------------------------------------------------------------------------

    def get(self, pipeline_id: str) -> Pipeline | None:
        """
        Retrieve a pipeline by its ID.

        Args:
            pipeline_id: Pipeline identifier

        Returns:
            Pipeline or None if not found
        """
        return self._pipelines.get(pipeline_id)

    def list_all(self) -> list[Pipeline]:
        """
        List all pipelines sorted by type (builtin first) then by name.

        Returns:
            List of pipelines
        """
        pipelines = list(self._pipelines.values())
        # Sort: builtin first, then by name
        pipelines.sort(key=lambda p: (not p.is_builtin, p.name.lower()))
        return pipelines

    def list_builtin(self) -> list[Pipeline]:
        """List only builtin pipelines."""
        return [p for p in self._pipelines.values() if p.is_builtin]

    def list_custom(self) -> list[Pipeline]:
        """List only custom pipelines."""
        return [p for p in self._pipelines.values() if not p.is_builtin]

    def get_all(self) -> dict[str, Pipeline]:
        """Return all pipelines as a dictionary."""
        return self._pipelines.copy()

    def get_available_agents(self) -> list[str]:
        """Return the list of available agents."""
        return self._available_agents.copy()

    def get_available_templates(self) -> list[str]:
        """Return the list of available templates."""
        return self._available_templates.copy()

    def find_by_keywords(self, text: str, min_matches: int = 1) -> Pipeline | None:
        """
        Find the best pipeline based on keywords.

        Args:
            text: Text to analyze
            min_matches: Minimum number of required matches

        Returns:
            Best Pipeline or None
        """
        best_pipeline = None
        best_score = 0.0

        for pipeline in self._pipelines.values():
            matches = pipeline.matches_keywords(text)
            if matches >= min_matches:
                score = pipeline.get_weighted_score(text)
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline

        return best_pipeline

    def find_by_keywords_with_scores(
        self,
        text: str,
        min_matches: int = 1
    ) -> list[tuple[Pipeline, float, int]]:
        """
        Find all matching pipelines with their scores.

        Returns:
            List of (Pipeline, score, matches) sorted by score desc
        """
        results = []

        for pipeline in self._pipelines.values():
            matches = pipeline.matches_keywords(text)
            if matches >= min_matches:
                score = pipeline.get_weighted_score(text)
                results.append((pipeline, score, matches))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # -------------------------------------------------------------------------
    # Create/Modify/Delete
    # -------------------------------------------------------------------------

    def validate_for_write(self, pipeline: Pipeline) -> list[str]:
        """
        Validate a pipeline before create/update.

        Uses the loaded agent registry when available. When config.yaml did
        not load (empty registry), the agent-existence check is skipped so a
        degraded install can still create custom pipelines; the ID format,
        name and step checks always apply.

        Returns:
            List of errors (empty if valid)
        """
        available_agents = self._available_agents
        if not available_agents:
            available_agents = [s.agent for s in pipeline.steps if s.agent]
        return pipeline.validate(available_agents, self._available_templates)

    def create(self, pipeline: Pipeline) -> bool:
        """
        Create a new custom pipeline.

        Args:
            pipeline: Pipeline to create

        Returns:
            True if created successfully
        """
        if pipeline.id in self._pipelines:
            logger.warning(f"Pipeline '{pipeline.id}' already exists")
            return False

        # Validate before persisting (id format, name, steps,
        # agent/template existence). Mirrors the execution-pipeline store.
        errors = self.validate_for_write(pipeline)
        if errors:
            logger.warning(f"Pipeline '{pipeline.id}' rejected: {errors}")
            return False

        # Force is_builtin to False
        pipeline.is_builtin = False

        # Ensure creation date is set
        if not pipeline.created_at:
            pipeline.created_at = datetime.now().isoformat()

        self._pipelines[pipeline.id] = pipeline
        success = self._save_custom()

        if success:
            logger.info(f"Pipeline created: {pipeline.id}")

        return success

    def update(self, pipeline_id: str, pipeline: Pipeline) -> bool:
        """
        Update an existing pipeline.

        Args:
            pipeline_id: ID of pipeline to modify
            pipeline: New data

        Returns:
            True if updated successfully
        """
        existing = self._pipelines.get(pipeline_id)
        if not existing:
            logger.warning(f"Pipeline not found: {pipeline_id}")
            return False

        if existing.is_builtin:
            logger.warning(f"Cannot modify builtin pipeline: {pipeline_id}")
            return False

        # Validate before persisting, same as create. The id
        # is normalized to the target pipeline_id first (callers may pass a
        # payload without it).
        pipeline.id = pipeline_id
        errors = self.validate_for_write(pipeline)
        if errors:
            logger.warning(f"Pipeline '{pipeline_id}' update rejected: {errors}")
            return False

        # Keep is_builtin and created_at from existing
        pipeline.is_builtin = False
        pipeline.created_at = existing.created_at
        pipeline.id = pipeline_id

        self._pipelines[pipeline_id] = pipeline
        success = self._save_custom()

        if success:
            logger.info(f"Pipeline updated: {pipeline_id}")

        return success

    def delete(self, pipeline_id: str) -> bool:
        """
        Delete a custom pipeline.

        Args:
            pipeline_id: ID of pipeline to delete

        Returns:
            True if deleted successfully
        """
        pipeline = self._pipelines.get(pipeline_id)
        if not pipeline:
            return False

        if pipeline.is_builtin:
            logger.warning(f"Cannot delete builtin pipeline: {pipeline_id}")
            return False

        del self._pipelines[pipeline_id]
        success = self._save_custom()

        if success:
            logger.info(f"Pipeline deleted: {pipeline_id}")

        return success

    def duplicate(self, pipeline_id: str, new_id: str) -> Pipeline | None:
        """
        Duplicate a pipeline (builtin or custom).

        Args:
            pipeline_id: Source pipeline ID
            new_id: New pipeline ID

        Returns:
            New Pipeline or None on failure
        """
        source = self._pipelines.get(pipeline_id)
        if not source:
            return None

        if new_id in self._pipelines:
            logger.warning(f"Pipeline ID '{new_id}' already exists")
            return None

        # Create a deep copy
        new_pipeline = Pipeline(
            id=new_id,
            name=f"{source.name} (Copy)",
            description=f"Copy of {source.name}",
            pattern=source.pattern,
            emoji=source.emoji,
            steps=[
                PipelineStep(
                    name=s.name,
                    agent=s.agent,
                    prompt_template=s.prompt_template,
                    description=s.description,
                    system_prompt=s.system_prompt,
                    model=s.model,
                )
                for s in source.steps
            ],
            keywords=source.keywords.copy() if source.keywords else [],
            detection_weight=source.detection_weight,
            is_builtin=False,
        )

        if self.create(new_pipeline):
            return new_pipeline

        return None

    # -------------------------------------------------------------------------
    # Import/Export
    # -------------------------------------------------------------------------

    def export_all(self) -> str:
        """
        Export all pipelines (builtin + custom) as YAML.

        Returns:
            YAML content
        """
        all_pipelines = {}
        for pipe_id, pipeline in self._pipelines.items():
            all_pipelines[pipe_id] = pipeline.to_dict()

        return yaml.dump(
            {"pipelines": all_pipelines},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    def export_custom(self) -> str:
        """
        Export only custom pipelines as YAML.

        Returns:
            YAML content
        """
        custom_pipelines = {}
        for pipe_id, pipeline in self._pipelines.items():
            if not pipeline.is_builtin:
                custom_pipelines[pipe_id] = pipeline.to_dict()

        return yaml.dump(
            {"pipelines": custom_pipelines},
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )

    def export_to_file(self, filepath: Path) -> bool:
        """
        Export all pipelines to a file.

        Args:
            filepath: Destination file path

        Returns:
            True on success
        """
        try:
            filepath = Path(filepath)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(self.export_all())

            return True
        except Exception as e:
            logger.error(f"Export error: {e}")
            return False

    def import_from_yaml(self, yaml_content: str) -> list[str]:
        """
        Import pipelines from YAML content.

        Args:
            yaml_content: YAML content

        Returns:
            List of imported IDs
        """
        try:
            data = yaml.safe_load(yaml_content)
            pipelines_data = data.get("pipelines", data)

            imported = []

            for pipe_id, pipe_data in pipelines_data.items():
                if not isinstance(pipe_data, dict):
                    continue

                # Don't overwrite builtins
                if pipe_id in self._pipelines and self._pipelines[pipe_id].is_builtin:
                    logger.warning(f"Skipping builtin pipeline: {pipe_id}")
                    continue

                pipeline = Pipeline.from_dict(pipe_id, pipe_data, is_builtin=False)
                self._pipelines[pipe_id] = pipeline
                imported.append(pipe_id)

            if imported:
                self._save_custom()

            return imported

        except Exception as e:
            logger.error(f"Import error: {e}")
            return []

    def import_from_file(self, filepath: Path) -> list[str]:
        """
        Import pipelines from a YAML file.

        Args:
            filepath: File path

        Returns:
            List of imported IDs
        """
        try:
            with open(filepath, encoding='utf-8') as f:
                content = f.read()
            return self.import_from_yaml(content)
        except Exception as e:
            logger.error(f"Import file error: {e}")
            return []

    # -------------------------------------------------------------------------
    # Utilitaires
    # -------------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Return pipeline statistics."""
        builtin_count = sum(1 for p in self._pipelines.values() if p.is_builtin)
        custom_count = len(self._pipelines) - builtin_count

        total_steps = sum(len(p.steps) for p in self._pipelines.values())
        total_keywords = sum(len(p.keywords) for p in self._pipelines.values())

        patterns = {}
        for p in self._pipelines.values():
            patterns[p.pattern] = patterns.get(p.pattern, 0) + 1

        return {
            "total": len(self._pipelines),
            "builtin": builtin_count,
            "custom": custom_count,
            "total_steps": total_steps,
            "total_keywords": total_keywords,
            "by_pattern": patterns,
            "available_agents": len(self._available_agents),
            "available_templates": len(self._available_templates),
        }

    def validate_pipeline_id(self, pipeline_id: str) -> bool:
        """Check if a pipeline ID is valid."""
        if not pipeline_id or not pipeline_id.strip():
            return False
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', pipeline_id))

    def get_pipelines_for_orchestrator(self) -> dict[str, dict]:
        """
        Return pipelines in the format expected by the orchestrator.

        Returns:
            Dict compatible with config["pipelines"]
        """
        result = {}
        for pipe_id, pipeline in self._pipelines.items():
            result[pipe_id] = pipeline.to_config_dict()
        return result


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

_pipeline_manager: PipelineManager | None = None


def get_pipeline_manager() -> PipelineManager:
    """Retrieve the global pipeline manager instance."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


def list_pipelines() -> list[Pipeline]:
    """Convenience function to list pipelines."""
    return get_pipeline_manager().list_all()


def get_pipeline(pipeline_id: str) -> Pipeline | None:
    """Convenience function to retrieve a pipeline."""
    return get_pipeline_manager().get(pipeline_id)


# =============================================================================
# CLI POUR TESTS
# =============================================================================

if __name__ == "__main__":
    print("=== Pipeline Manager Test ===\n")

    manager = PipelineManager()

    # Afficher les stats
    stats = manager.get_stats()
    print(f"Total pipelines: {stats['total']}")
    print(f"  - Builtin: {stats['builtin']}")
    print(f"  - Custom: {stats['custom']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Available agents: {stats['available_agents']}")
    print(f"Available templates: {stats['available_templates']}")

    print("\nAvailable pipelines:")
    for pipeline in manager.list_all():
        status = "📌" if pipeline.is_builtin else "🧅"
        print(f"  {pipeline.emoji} {status} {pipeline.id}: {pipeline.name} ({pipeline.step_count} steps)")

    print("\nAvailable agents:")
    for agent in manager.get_available_agents():
        print(f"  - {agent}")

    print("\nAvailable templates:")
    for template in manager.get_available_templates()[:10]:
        print(f"  - {template}")
    if len(manager.get_available_templates()) > 10:
        print(f"  ... et {len(manager.get_available_templates()) - 10} autres")

    # Creation test
    print("\nTesting custom pipeline creation...")
    test_pipeline = Pipeline(
        id="test_custom",
        name="Test Custom Pipeline",
        description="Test pipeline",
        pattern="chain",
        emoji="🧪",
        steps=[
            PipelineStep(
                name="Analyse",
                agent="reviewer",
                prompt_template="error_analysis",
                description="Analyze the problem",
            ),
            PipelineStep(
                name="Solution",
                agent="coder",
                system_prompt="You are an expert problem solver. Fix the issue.",
                description="Propose a solution",
            ),
        ],
        keywords=["test", "debug"],
        detection_weight=0.7,
    )

    if manager.create(test_pipeline):
        print(f"[OK] Pipeline '{test_pipeline.id}' created")

        # Verify
        loaded = manager.get("test_custom")
        if loaded:
            print(f"   Loaded: {loaded.name} with {loaded.step_count} steps")

        # Supprimer
        if manager.delete("test_custom"):
            print("[OK] Pipeline 'test_custom' deleted")
    else:
        print("[ERR] Creation failed")

    print("\n[OK] Pipeline Manager fonctionnel")
