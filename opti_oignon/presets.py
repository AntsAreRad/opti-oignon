#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PRESETS - OPTI-OIGNON 1.0
=========================

Preset management (task + model + prompt + parameters combinations).

Presets allow quick saving and loading of configurations
for specific use cases.

ENHANCED: 
- Keywords support for automatic routing based on question content
- Detection weight for scoring
- Auto-suggestion of keywords based on task/name/description

Author: Léon
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
import yaml
import logging
import re

from .config import config, DATA_DIR, save_yaml, load_yaml

logger = logging.getLogger(__name__)

# =============================================================================
# KEYWORD SUGGESTION DATABASE
# =============================================================================

# Keywords by task type (used for auto-suggestion)
TASK_KEYWORDS = {
    "code_r": [
        # Core R
        "ggplot", "dplyr", "tidyverse", "vegan", "mutate", "filter", "pipe", 
        "%>%", "|>", "dataframe", "tibble", "library(", "data.frame", "matrix",
        "function(", "lapply", "sapply", "purrr", "readr", "tidyr", "stringr",
        # Ecology/Bioinformatics
        "phyloseq", "deseq2", "edger", "bioconductor", "shannon", "simpson",
        "rarefaction", "pcoa", "nmds", "ordination", "abundance", "diversity",
        "ecology", "biodiversity", "bray-curtis", "permanova", "adonis",
        "metabarcoding", "ASV", "OTU",
        # Stats
        "lm", "glm", "anova", "t.test", "cor.test", "wilcox", "kruskal",
    ],
    "debug_r": [
        "error", "erreur", "bug", "traceback", "warning", "fix", "corriger",
        "doesn't work", "ne marche pas", "failed", "échoue", "problem",
        "issue", "undefined", "NULL", "NA", "NaN", "object not found",
    ],
    "code_python": [
        # Core Python
        "import", "def", "class", "pandas", "numpy", "matplotlib", "seaborn",
        "scipy", "sklearn", "scikit-learn", "pytorch", "tensorflow", "keras",
        "requests", "json", "csv", "argparse", "pathlib", "os", "sys",
        # Bioinformatics
        "biopython", "Bio.", "fasta", "fastq", "alignment", "phylogeny",
        "blast", "sequence", "protein", "DNA", "RNA",
        # Data
        "dataframe", "df.", "pd.", "np.", "plt.",
    ],
    "debug_python": [
        "error", "exception", "traceback", "TypeError", "ValueError",
        "AttributeError", "ImportError", "SyntaxError", "KeyError",
        "IndexError", "NameError", "ModuleNotFoundError", "FileNotFoundError",
        "bug", "fix", "doesn't work", "failed", "crash",
    ],
    "scientific_writing": [
        "abstract", "introduction", "methods", "results", "discussion",
        "paper", "article", "manuscript", "thesis", "report", "scientific",
        "académique", "academic", "publication", "review", "reviewer",
        "figure", "table", "reference", "citation", "hypothesis",
    ],
    "planning": [
        "plan", "organize", "steps", "todo", "list", "brainstorm", "ideas",
        "structure", "strategy", "timeline", "milestone", "task", "project",
        "workflow", "process", "roadmap", "goal", "objective",
    ],
    "planning_deep": [
        "analyze", "think deeply", "consider", "evaluate", "pros cons",
        "trade-offs", "strategy", "compare", "alternatives", "decision",
        "in-depth", "detailed analysis", "comprehensive", "thorough",
    ],
    "linux": [
        "bash", "shell", "terminal", "command", "apt", "systemctl", "chmod",
        "grep", "sed", "awk", "pipe", "sudo", "kubuntu", "ubuntu", "linux",
        "ssh", "scp", "rsync", "cron", "systemd", "docker", "git",
    ],
    "simple_question": [
        "what is", "c'est quoi", "explain", "explique", "how does", "comment",
        "why", "pourquoi", "definition", "meaning", "difference between",
    ],
    "bioinformatics": [
        "metabarcoding", "bioacoustic", "ASV", "OTU", "DADA2", "qiime",
        "blast", "fasta", "fastq", "alignment", "phylogeny", "taxonomy",
        "species", "abundance", "alpha diversity", "beta diversity",
        "rarefaction", "primer", "amplicon", "16S", "18S", "ITS", "COI",
    ],
}

# Keywords by common name patterns
NAME_KEYWORDS = {
    "bioinformatics": ["bioconductor", "phyloseq", "deseq2", "vegan", "ggplot", "tidyverse"],
    "ecology": ["ecology", "biodiversity", "diversity", "species", "abundance", "vegan"],
    "statistics": ["lm", "glm", "anova", "regression", "correlation", "p-value"],
    "visualization": ["ggplot", "matplotlib", "seaborn", "plot", "chart", "graph"],
    "data": ["pandas", "dplyr", "tidyverse", "dataframe", "csv", "xlsx"],
    "machine learning": ["sklearn", "tensorflow", "pytorch", "keras", "model", "train"],
    "web": ["flask", "django", "fastapi", "requests", "api", "http", "json"],
    "automation": ["script", "automate", "cron", "schedule", "batch", "pipeline"],
}


def suggest_keywords(
    task: str,
    name: str = "",
    description: str = "",
    existing_keywords: List[str] = None,
    max_suggestions: int = 15
) -> List[str]:
    """
    Suggest keywords based on task type, name, and description.
    
    Args:
        task: Task type (e.g., "code_r", "debug_python")
        name: Preset name
        description: Preset description
        existing_keywords: Already defined keywords (to avoid duplicates)
        max_suggestions: Maximum number of suggestions
        
    Returns:
        List of suggested keywords
    """
    suggestions = []
    existing = set(k.lower() for k in (existing_keywords or []))
    
    # 1. Add keywords from task type
    task_kw = TASK_KEYWORDS.get(task, [])
    for kw in task_kw:
        if kw.lower() not in existing:
            suggestions.append(kw)
    
    # 2. Add keywords based on name patterns
    name_lower = name.lower()
    desc_lower = description.lower()
    combined = f"{name_lower} {desc_lower}"
    
    for pattern, keywords in NAME_KEYWORDS.items():
        if pattern in combined:
            for kw in keywords:
                if kw.lower() not in existing and kw not in suggestions:
                    suggestions.append(kw)
    
    # 3. Extract potential keywords from name and description
    # Look for technical terms (words with special chars or capital letters)
    words = re.findall(r'\b[A-Za-z][a-zA-Z0-9_.-]+\b', f"{name} {description}")
    for word in words:
        if len(word) > 2 and word.lower() not in existing and word not in suggestions:
            # Check if it looks like a technical term
            if any(c.isupper() for c in word[1:]) or '_' in word or '.' in word:
                suggestions.append(word)
    
    # Deduplicate and limit
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s.lower() not in seen:
            seen.add(s.lower())
            unique_suggestions.append(s)
    
    return unique_suggestions[:max_suggestions]


# =============================================================================
# PRESET STRUCTURE
# =============================================================================

@dataclass
class Preset:
    """Representation of a preset."""
    id: str
    name: str
    description: str
    task: str
    model: str
    temperature: float
    prompt_variant: str
    icon: str = "⚙️"
    tags: List[str] = None
    keywords: List[str] = None  # Keywords for auto-routing
    detection_weight: float = 0.5  # NEW: Weight for keyword scoring (0.0-1.0)
    custom_prompt: Optional[str] = None  # Custom prompt (override)
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.keywords is None:
            self.keywords = []
        # Clamp detection_weight between 0 and 1
        self.detection_weight = max(0.0, min(1.0, self.detection_weight))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, preset_id: str, data: Dict) -> 'Preset':
        """Create a Preset from a dictionary."""
        return cls(
            id=preset_id,
            name=data.get("name", preset_id),
            description=data.get("description", ""),
            task=data.get("task", "simple_question"),
            model=data.get("model", "qwen3-coder:30b"),
            temperature=data.get("temperature", 0.5),
            prompt_variant=data.get("prompt_variant", "standard"),
            icon=data.get("icon", "⚙️"),
            tags=data.get("tags", []),
            keywords=data.get("keywords", []),
            detection_weight=data.get("detection_weight", 0.5),
            custom_prompt=data.get("custom_prompt"),
        )
    
    def matches_keywords(self, text: str) -> int:
        """
        Check how many keywords match the given text.
        
        Args:
            text: Text to check against keywords
            
        Returns:
            Number of matching keywords (0 = no match)
        """
        if not self.keywords:
            return 0
        
        text_lower = text.lower()
        matches = 0
        
        for keyword in self.keywords:
            # Use word boundary matching for better accuracy
            pattern = re.compile(r'\b' + re.escape(keyword.lower()) + r'\b', re.IGNORECASE)
            if pattern.search(text_lower):
                matches += 1
        
        return matches
    
    def get_weighted_score(self, text: str) -> float:
        """
        Calculate weighted score based on keyword matches.
        
        Args:
            text: Text to check against keywords
            
        Returns:
            Weighted score (matches * detection_weight)
        """
        matches = self.matches_keywords(text)
        return matches * self.detection_weight
    
    def suggest_keywords(self, max_suggestions: int = 15) -> List[str]:
        """
        Suggest keywords for this preset based on its configuration.
        
        Returns:
            List of suggested keywords
        """
        return suggest_keywords(
            task=self.task,
            name=self.name,
            description=self.description,
            existing_keywords=self.keywords,
            max_suggestions=max_suggestions,
        )


# =============================================================================
# PRESET MANAGER
# =============================================================================

class PresetManager:
    """
    Preset manager.
    
    Handles:
    - Loading presets from config
    - Creating/modifying/deleting user presets
    - Searching and filtering presets
    - Keyword-based auto-routing with weighted scoring
    
    Usage:
        manager = PresetManager()
        preset = manager.get("r_fast")
        manager.create("my_preset", {...})
        best_preset = manager.find_by_keywords("How to use ggplot2?")
    """
    
    def __init__(self):
        """Initialize the manager."""
        self._presets: Dict[str, Preset] = {}
        self._user_presets_file = DATA_DIR / "user_presets.yaml"
        self._load_presets()
    
    def _load_presets(self) -> None:
        """Load all presets (config + user)."""
        self._presets = {}
        
        # Load presets from global config
        config_presets = config.get_all_presets()
        for preset_id, preset_data in config_presets.items():
            self._presets[preset_id] = Preset.from_dict(preset_id, preset_data)
        
        # Load user presets (override config)
        if self._user_presets_file.exists():
            user_presets = load_yaml(self._user_presets_file)
            for preset_id, preset_data in user_presets.get("presets", {}).items():
                self._presets[preset_id] = Preset.from_dict(preset_id, preset_data)
        
        logger.info(f"{len(self._presets)} presets loaded")
    
    def reload(self) -> None:
        """Reload presets from files."""
        config.reload()
        self._load_presets()
    
    # -------------------------------------------------------------------------
    # Accessing presets
    # -------------------------------------------------------------------------
    
    def get(self, preset_id: str) -> Optional[Preset]:
        """
        Get a preset by its ID.
        
        Args:
            preset_id: Preset identifier
            
        Returns:
            The Preset or None if not found
        """
        return self._presets.get(preset_id)
    
    def get_all(self) -> Dict[str, Preset]:
        """Return all presets."""
        return self._presets.copy()
    
    def get_ordered(self) -> List[Preset]:
        """Return presets in the configured display order."""
        display_order = config.get_preset_display_order()
        ordered = []
        
        # First, presets in the configured order
        for preset_id in display_order:
            if preset_id in self._presets:
                ordered.append(self._presets[preset_id])
        
        # Then, other unlisted presets
        for preset_id, preset in self._presets.items():
            if preset_id not in display_order:
                ordered.append(preset)
        
        return ordered
    
    def get_for_task(self, task: str) -> Optional[Preset]:
        """
        Get the auto-selected preset for a task.
        
        Args:
            task: Task type
            
        Returns:
            The corresponding Preset or None
        """
        preset_id = config.get_auto_select_preset(task)
        if preset_id:
            return self.get(preset_id)
        return None
    
    def find_by_keywords(self, text: str, min_matches: int = 1) -> Optional[Preset]:
        """
        Find the best matching preset based on keywords in text.
        Uses weighted scoring for better accuracy.
        
        Args:
            text: Text to analyze (usually the user's question)
            min_matches: Minimum number of keyword matches required
            
        Returns:
            Best matching Preset or None if no match
        """
        best_preset = None
        best_score = 0.0
        
        for preset in self._presets.values():
            matches = preset.matches_keywords(text)
            if matches >= min_matches:
                score = preset.get_weighted_score(text)
                if score > best_score:
                    best_score = score
                    best_preset = preset
        
        if best_preset:
            logger.debug(f"Keyword match: '{best_preset.id}' with score {best_score:.2f}")
        
        return best_preset
    
    def find_by_keywords_with_scores(self, text: str, min_matches: int = 1) -> List[Tuple[Preset, float, int]]:
        """
        Find all matching presets with their scores.
        
        Args:
            text: Text to analyze
            min_matches: Minimum number of keyword matches required
            
        Returns:
            List of (Preset, weighted_score, match_count) tuples, sorted by score desc
        """
        results = []
        
        for preset in self._presets.values():
            matches = preset.matches_keywords(text)
            if matches >= min_matches:
                score = preset.get_weighted_score(text)
                results.append((preset, score, matches))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def search(self, query: str) -> List[Preset]:
        """
        Search presets by name, description, tags, or keywords.
        
        Args:
            query: Search term
            
        Returns:
            List of matching presets
        """
        query = query.lower()
        results = []
        
        for preset in self._presets.values():
            if (query in preset.name.lower() or 
                query in preset.description.lower() or
                any(query in tag.lower() for tag in preset.tags) or
                any(query in kw.lower() for kw in preset.keywords)):
                results.append(preset)
        
        return results
    
    def filter_by_tag(self, tag: str) -> List[Preset]:
        """Filter presets by tag."""
        return [p for p in self._presets.values() if tag in p.tags]
    
    def filter_by_task(self, task: str) -> List[Preset]:
        """Filter presets by task type."""
        return [p for p in self._presets.values() if p.task == task]
    
    # -------------------------------------------------------------------------
    # Creating/modifying presets
    # -------------------------------------------------------------------------
    
    def create(
        self,
        preset_id: str,
        name: str,
        task: str,
        model: str,
        temperature: float = 0.5,
        prompt_variant: str = "standard",
        description: str = "",
        icon: str = "⚙️",
        tags: List[str] = None,
        keywords: List[str] = None,
        detection_weight: float = 0.5,
        custom_prompt: Optional[str] = None,
    ) -> Preset:
        """
        Create a new user preset.
        
        Args:
            preset_id: Unique identifier
            name: Display name
            task: Task type
            model: Ollama model
            temperature: Temperature
            prompt_variant: Prompt variant
            description: Optional description
            icon: Emoji/icon
            tags: Tags for search
            keywords: Keywords for auto-routing
            detection_weight: Weight for keyword scoring (0.0-1.0)
            custom_prompt: Custom prompt (override)
            
        Returns:
            The created Preset
        """
        preset = Preset(
            id=preset_id,
            name=name,
            description=description,
            task=task,
            model=model,
            temperature=temperature,
            prompt_variant=prompt_variant,
            icon=icon,
            tags=tags or [],
            keywords=keywords or [],
            detection_weight=detection_weight,
            custom_prompt=custom_prompt,
        )
        
        self._presets[preset_id] = preset
        self._save_user_presets()
        
        logger.info(f"Preset created: {preset_id}")
        return preset
    
    def update(self, preset_id: str, **kwargs) -> Optional[Preset]:
        """
        Update an existing preset.
        
        Args:
            preset_id: ID of the preset to modify
            **kwargs: Fields to update
            
        Returns:
            The modified Preset or None if not found
        """
        preset = self._presets.get(preset_id)
        if not preset:
            logger.warning(f"Preset not found: {preset_id}")
            return None
        
        # Update fields
        for key, value in kwargs.items():
            if hasattr(preset, key):
                setattr(preset, key, value)
        
        # Ensure detection_weight is clamped
        if 'detection_weight' in kwargs:
            preset.detection_weight = max(0.0, min(1.0, preset.detection_weight))
        
        self._save_user_presets()
        logger.info(f"Preset updated: {preset_id}")
        return preset
    
    def delete(self, preset_id: str) -> bool:
        """
        Delete a user preset.
        
        Args:
            preset_id: ID of the preset to delete
            
        Returns:
            True if deleted, False otherwise
        """
        if preset_id not in self._presets:
            return False
        
        # Don't allow deleting the default preset
        if preset_id == "default":
            logger.warning("Cannot delete the default preset")
            return False
        
        del self._presets[preset_id]
        self._save_user_presets()
        
        logger.info(f"Preset deleted: {preset_id}")
        return True
    
    def duplicate(self, preset_id: str, new_id: str, new_name: str) -> Optional[Preset]:
        """
        Duplicate an existing preset.
        
        Args:
            preset_id: ID of the preset to duplicate
            new_id: ID of the new preset
            new_name: Name of the new preset
            
        Returns:
            The new Preset or None if source not found
        """
        source = self._presets.get(preset_id)
        if not source:
            return None
        
        return self.create(
            preset_id=new_id,
            name=new_name,
            task=source.task,
            model=source.model,
            temperature=source.temperature,
            prompt_variant=source.prompt_variant,
            description=f"Copy of {source.name}",
            icon=source.icon,
            tags=source.tags.copy(),
            keywords=source.keywords.copy() if source.keywords else [],
            detection_weight=source.detection_weight,
            custom_prompt=source.custom_prompt,
        )
    
    # -------------------------------------------------------------------------
    # Saving
    # -------------------------------------------------------------------------
    
    def _save_user_presets(self) -> bool:
        """Save user presets."""
        # Only save presets that are not in global config
        config_presets = config.get_all_presets()
        
        user_presets = {}
        for preset_id, preset in self._presets.items():
            if preset_id not in config_presets:
                user_presets[preset_id] = preset.to_dict()
            else:
                # Check if preset has been modified
                config_preset = config_presets[preset_id]
                preset_dict = preset.to_dict()
                if preset_dict != Preset.from_dict(preset_id, config_preset).to_dict():
                    user_presets[preset_id] = preset_dict
        
        return save_yaml(self._user_presets_file, {"presets": user_presets})
    
    def export_preset(self, preset_id: str, filepath: Path) -> bool:
        """
        Export a preset to a file.
        
        Args:
            preset_id: ID of the preset to export
            filepath: Destination path
            
        Returns:
            True if successful
        """
        preset = self._presets.get(preset_id)
        if not preset:
            return False
        
        return save_yaml(filepath, {preset_id: preset.to_dict()})
    
    def export_all_presets(self, filepath: Path) -> bool:
        """
        Export all presets to a file.
        
        Args:
            filepath: Destination path
            
        Returns:
            True if successful
        """
        all_presets = {}
        for preset_id, preset in self._presets.items():
            all_presets[preset_id] = preset.to_dict()
        
        return save_yaml(filepath, {"presets": all_presets})
    
    def import_preset(self, filepath: Path) -> List[str]:
        """
        Import presets from a file.
        
        Args:
            filepath: Path of file to import
            
        Returns:
            List of imported IDs
        """
        data = load_yaml(filepath)
        imported = []
        
        # Handle both formats: direct presets or wrapped in "presets" key
        presets_data = data.get("presets", data)
        
        for preset_id, preset_data in presets_data.items():
            if isinstance(preset_data, dict) and "name" in preset_data:
                self._presets[preset_id] = Preset.from_dict(preset_id, preset_data)
                imported.append(preset_id)
        
        self._save_user_presets()
        return imported
    
    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about presets."""
        tasks = {}
        models = {}
        keywords_count = 0
        total_weight = 0.0
        
        for preset in self._presets.values():
            tasks[preset.task] = tasks.get(preset.task, 0) + 1
            models[preset.model] = models.get(preset.model, 0) + 1
            keywords_count += len(preset.keywords)
            total_weight += preset.detection_weight
        
        avg_weight = total_weight / len(self._presets) if self._presets else 0.0
        
        return {
            "total": len(self._presets),
            "by_task": tasks,
            "by_model": models,
            "total_keywords": keywords_count,
            "avg_detection_weight": avg_weight,
        }
    
    def get_all_keywords(self) -> List[str]:
        """Return all unique keywords across all presets."""
        keywords = set()
        for preset in self._presets.values():
            keywords.update(preset.keywords)
        return sorted(keywords)
    
    def validate_preset_id(self, preset_id: str) -> bool:
        """Check if a preset ID is valid (no special chars, not empty)."""
        if not preset_id or not preset_id.strip():
            return False
        # Only allow alphanumeric, underscore, hyphen
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', preset_id))
    
    def suggest_keywords_for_preset(self, preset_id: str, max_suggestions: int = 15) -> List[str]:
        """
        Suggest keywords for an existing preset.
        
        Args:
            preset_id: ID of the preset
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested keywords
        """
        preset = self.get(preset_id)
        if not preset:
            return []
        return preset.suggest_keywords(max_suggestions)


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

preset_manager = PresetManager()


def get_preset(preset_id: str) -> Optional[Preset]:
    """Convenience function to get a preset."""
    return preset_manager.get(preset_id)


def get_all_presets() -> Dict[str, Preset]:
    """Convenience function to get all presets."""
    return preset_manager.get_all()


def get_ordered_presets() -> List[Preset]:
    """Convenience function to get ordered presets."""
    return preset_manager.get_ordered()


def find_preset_by_keywords(text: str) -> Optional[Preset]:
    """Convenience function to find preset by keywords."""
    return preset_manager.find_by_keywords(text)


# =============================================================================
# CLI FOR TESTS
# =============================================================================

if __name__ == "__main__":
    print("=== Presets Test ===\n")
    
    manager = PresetManager()
    
    # Display all presets
    print("Available presets:")
    for preset in manager.get_ordered():
        kw_count = len(preset.keywords)
        print(f"  {preset.icon} {preset.id}: {preset.name} ({kw_count} keywords, weight={preset.detection_weight})")
        print(f"      Task: {preset.task}, Model: {preset.model}")
    
    print()
    
    # Stats
    stats = manager.get_stats()
    print(f"Total: {stats['total']} presets")
    print(f"Total keywords: {stats['total_keywords']}")
    print(f"Avg detection weight: {stats['avg_detection_weight']:.2f}")
    print(f"By task: {stats['by_task']}")
    
    print()
    
    # Test keyword matching
    test_questions = [
        "How to create a ggplot2 bar chart?",
        "Error: object not found in my R script",
        "Write a bash script to backup files",
        "What is biodiversity?",
    ]
    
    print("Keyword matching tests (with weighted scoring):")
    for q in test_questions:
        results = manager.find_by_keywords_with_scores(q)
        if results:
            best = results[0]
            print(f"  '{q[:40]}...' -> {best[0].icon} {best[0].id} (score={best[1]:.2f}, matches={best[2]})")
        else:
            print(f"  '{q[:40]}...' -> No match")
    
    print()
    
    # Test keyword suggestions
    print("Keyword suggestions test:")
    suggestions = suggest_keywords("code_r", "Ecology R Analysis", "Analyze biodiversity data with vegan")
    print(f"  Task: code_r, Name: 'Ecology R Analysis'")
    print(f"  Suggestions: {suggestions[:10]}")
    
    print()
    
    # Test creation
    print("Creating a test preset...")
    test_preset = manager.create(
        preset_id="test_preset",
        name="Test Preset",
        task="code_r",
        model="qwen3-coder:30b",
        temperature=0.4,
        description="Test preset",
        tags=["test"],
        keywords=["test_keyword"],
        detection_weight=0.7,
    )
    print(f"Created: {test_preset.id} with {len(test_preset.keywords)} keywords, weight={test_preset.detection_weight}")
    
    # Deletion
    manager.delete("test_preset")
    print("Deleted: test_preset")
