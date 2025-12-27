#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PIPELINE MANAGER - OPTI-OIGNON 1.2
==================================

Gestion des pipelines multi-agents personnalisés.

Permet de :
- Visualiser tous les pipelines (builtin + custom)
- Créer de nouveaux pipelines personnalisés
- Modifier/supprimer des pipelines existants
- Générer automatiquement les prompts système via LLM
- Importer/exporter des pipelines

Les pipelines builtin (dans agents/config.yaml) sont en lecture seule.
Les pipelines custom sont stockés dans data/pipelines_custom.yaml.

Author: Léon
"""

from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict, field
from pathlib import Path
from datetime import datetime
import yaml
import logging
import re
import copy

from .config import DATA_DIR, save_yaml, load_yaml

logger = logging.getLogger(__name__)


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PipelineStep:
    """
    Représente une étape dans un pipeline.
    
    Attributes:
        name: Nom de l'étape (ex: "Analysis", "Code Generation")
        agent: ID de l'agent à utiliser (ex: "coder", "reviewer")
        prompt_template: ID du template OU None si custom
        description: Description de l'étape
        system_prompt: Prompt système custom (si prompt_template=None)
        model: Modèle spécifique à utiliser (override celui de l'agent)
    """
    name: str
    agent: str
    prompt_template: Optional[str] = None
    description: str = ""
    system_prompt: Optional[str] = None
    model: Optional[str] = None  # Override model for this step
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
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
    def from_dict(cls, data: Dict) -> 'PipelineStep':
        """Crée un PipelineStep depuis un dictionnaire."""
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
    Représente un pipeline multi-agent complet.
    
    Attributes:
        id: Identifiant unique du pipeline
        name: Nom d'affichage
        description: Description du pipeline
        pattern: Pattern d'orchestration (chain, verifier, decomposition, iterative)
        emoji: Emoji pour l'affichage
        steps: Liste des étapes du pipeline
        keywords: Mots-clés pour l'auto-détection
        detection_weight: Poids pour le scoring (0.0-1.0)
        created_at: Date de création
        is_builtin: True si défini dans config.yaml (non modifiable)
    """
    id: str
    name: str
    description: str = ""
    pattern: str = "chain"
    emoji: str = "🔧"
    steps: List[PipelineStep] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    detection_weight: float = 0.5
    created_at: Optional[str] = None
    is_builtin: bool = False
    
    def __post_init__(self):
        # Assurer que les steps sont des objets PipelineStep
        if self.steps and isinstance(self.steps[0], dict):
            self.steps = [PipelineStep.from_dict(s) for s in self.steps]
        
        # Clamper detection_weight entre 0 et 1
        self.detection_weight = max(0.0, min(1.0, self.detection_weight))
        
        # Date de création si non définie
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour export YAML."""
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
    
    def to_config_dict(self) -> Dict:
        """
        Convertit dans le format config.yaml (compatible avec l'orchestrateur).
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
    def from_dict(cls, pipeline_id: str, data: Dict, is_builtin: bool = False) -> 'Pipeline':
        """Crée un Pipeline depuis un dictionnaire."""
        # Extraire les keywords depuis auto_detect si présent
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
        Compte combien de keywords matchent dans le texte.
        
        Args:
            text: Texte à analyser
            
        Returns:
            Nombre de keywords trouvés
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
        """Calcule le score pondéré basé sur les matches de keywords."""
        matches = self.matches_keywords(text)
        return matches * self.detection_weight
    
    @property
    def step_count(self) -> int:
        """Retourne le nombre d'étapes."""
        return len(self.steps)
    
    def validate(self, available_agents: List[str], available_templates: List[str]) -> List[str]:
        """
        Valide le pipeline.
        
        Args:
            available_agents: Liste des IDs d'agents disponibles
            available_templates: Liste des IDs de templates disponibles
            
        Returns:
            Liste des erreurs (vide si valide)
        """
        errors = []
        
        if not self.id:
            errors.append("L'ID du pipeline est requis")
        elif not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', self.id):
            errors.append("L'ID ne peut contenir que des lettres, chiffres, _ et -")
        
        if not self.name:
            errors.append("Le nom du pipeline est requis")
        
        if not self.steps:
            errors.append("Au moins une étape est requise")
        
        for i, step in enumerate(self.steps):
            if not step.name:
                errors.append(f"Étape {i+1}: nom requis")
            
            if step.agent and step.agent != "auto" and step.agent not in available_agents:
                errors.append(f"Étape {i+1}: agent '{step.agent}' inconnu")
            
            if step.prompt_template and step.prompt_template not in available_templates:
                # Pas une erreur bloquante, on peut utiliser system_prompt
                if not step.system_prompt:
                    errors.append(f"Étape {i+1}: template '{step.prompt_template}' inconnu et pas de system_prompt")
        
        return errors


# =============================================================================
# PIPELINE MANAGER
# =============================================================================

class PipelineManager:
    """
    Gestionnaire de pipelines multi-agents.
    
    Gère les pipelines builtin (config.yaml, lecture seule) et
    les pipelines custom (pipelines_custom.yaml, modifiables).
    
    Usage:
        manager = PipelineManager()
        pipelines = manager.list_all()
        manager.create(pipeline)
        manager.update(pipeline_id, pipeline)
        manager.delete(pipeline_id)
    """
    
    def __init__(self):
        """Initialise le gestionnaire."""
        self._pipelines: Dict[str, Pipeline] = {}
        
        # Chemins des fichiers
        self._config_file = Path(__file__).parent / "agents" / "config.yaml"
        self._custom_file = DATA_DIR / "pipelines_custom.yaml"
        
        # Cache des agents et templates disponibles
        self._available_agents: List[str] = []
        self._available_templates: List[str] = []
        
        self._load_all()
    
    def _load_all(self) -> None:
        """Charge tous les pipelines (builtin + custom)."""
        self._pipelines = {}
        
        # 1. Charger les pipelines builtin depuis config.yaml
        self._load_builtin()
        
        # 2. Charger les pipelines custom
        self._load_custom()
        
        # 3. Mettre à jour la liste des templates (inclut ceux des custom pipelines)
        self._update_templates_list()
        
        logger.info(f"{len(self._pipelines)} pipelines chargés")
    
    def _update_templates_list(self) -> None:
        """Met à jour la liste des templates disponibles depuis tous les pipelines."""
        # Collecter les templates utilisés dans tous les pipelines
        used_templates = set(self._available_templates)  # Garder ceux déjà définis
        for pipeline in self._pipelines.values():
            for step in pipeline.steps:
                if step.prompt_template:
                    used_templates.add(step.prompt_template)
        self._available_templates = sorted(used_templates)
    
    def _load_builtin(self) -> None:
        """Charge les pipelines depuis agents/config.yaml."""
        if not self._config_file.exists():
            logger.warning(f"Config file not found: {self._config_file}")
            return
        
        try:
            with open(self._config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f) or {}
            
            # Charger les pipelines
            pipelines_data = config.get("pipelines", {})
            for pipe_id, pipe_data in pipelines_data.items():
                self._pipelines[pipe_id] = Pipeline.from_dict(
                    pipe_id, pipe_data, is_builtin=True
                )
            
            # Charger la liste des agents disponibles
            self._available_agents = list(config.get("agents", {}).keys())
            
            # Charger la liste des templates disponibles (définis dans prompt_templates)
            defined_templates = set(config.get("prompt_templates", {}).keys())
            
            # Aussi collecter les templates utilisés dans les pipelines
            used_templates = set()
            for pipeline in self._pipelines.values():
                for step in pipeline.steps:
                    if step.prompt_template:
                        used_templates.add(step.prompt_template)
            
            # Fusionner: templates définis + templates utilisés
            self._available_templates = sorted(defined_templates | used_templates)
            
            logger.debug(f"Loaded {len(pipelines_data)} builtin pipelines")
            
        except Exception as e:
            logger.error(f"Error loading config.yaml: {e}")
    
    def _load_custom(self) -> None:
        """Charge les pipelines custom depuis data/pipelines_custom.yaml."""
        if not self._custom_file.exists():
            return
        
        try:
            data = load_yaml(self._custom_file)
            pipelines_data = data.get("pipelines", {})
            
            for pipe_id, pipe_data in pipelines_data.items():
                self._pipelines[pipe_id] = Pipeline.from_dict(
                    pipe_id, pipe_data, is_builtin=False
                )
            
            logger.debug(f"Loaded {len(pipelines_data)} custom pipelines")
            
        except Exception as e:
            logger.error(f"Error loading custom pipelines: {e}")
    
    def _save_custom(self) -> bool:
        """Sauvegarde les pipelines custom."""
        try:
            custom_pipelines = {}
            
            for pipe_id, pipeline in self._pipelines.items():
                if not pipeline.is_builtin:
                    custom_pipelines[pipe_id] = pipeline.to_dict()
            
            # S'assurer que le dossier existe
            self._custom_file.parent.mkdir(parents=True, exist_ok=True)
            
            return save_yaml(self._custom_file, {"pipelines": custom_pipelines})
            
        except Exception as e:
            logger.error(f"Error saving custom pipelines: {e}")
            return False
    
    def reload(self) -> None:
        """Recharge tous les pipelines."""
        self._load_all()
    
    # -------------------------------------------------------------------------
    # Accès aux pipelines
    # -------------------------------------------------------------------------
    
    def get(self, pipeline_id: str) -> Optional[Pipeline]:
        """
        Récupère un pipeline par son ID.
        
        Args:
            pipeline_id: Identifiant du pipeline
            
        Returns:
            Pipeline ou None si non trouvé
        """
        return self._pipelines.get(pipeline_id)
    
    def list_all(self) -> List[Pipeline]:
        """
        Liste tous les pipelines triés par type (builtin d'abord) puis par nom.
        
        Returns:
            Liste des pipelines
        """
        pipelines = list(self._pipelines.values())
        # Trier: builtin en premier, puis par nom
        pipelines.sort(key=lambda p: (not p.is_builtin, p.name.lower()))
        return pipelines
    
    def list_builtin(self) -> List[Pipeline]:
        """Liste uniquement les pipelines builtin."""
        return [p for p in self._pipelines.values() if p.is_builtin]
    
    def list_custom(self) -> List[Pipeline]:
        """Liste uniquement les pipelines custom."""
        return [p for p in self._pipelines.values() if not p.is_builtin]
    
    def get_all(self) -> Dict[str, Pipeline]:
        """Retourne tous les pipelines en dictionnaire."""
        return self._pipelines.copy()
    
    def get_available_agents(self) -> List[str]:
        """Retourne la liste des agents disponibles."""
        return self._available_agents.copy()
    
    def get_available_templates(self) -> List[str]:
        """Retourne la liste des templates disponibles."""
        return self._available_templates.copy()
    
    def find_by_keywords(self, text: str, min_matches: int = 1) -> Optional[Pipeline]:
        """
        Trouve le meilleur pipeline basé sur les keywords.
        
        Args:
            text: Texte à analyser
            min_matches: Nombre minimum de matches requis
            
        Returns:
            Meilleur Pipeline ou None
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
    ) -> List[Tuple[Pipeline, float, int]]:
        """
        Trouve tous les pipelines qui matchent avec leurs scores.
        
        Returns:
            Liste de (Pipeline, score, matches) triée par score desc
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
    # Création/Modification/Suppression
    # -------------------------------------------------------------------------
    
    def create(self, pipeline: Pipeline) -> bool:
        """
        Crée un nouveau pipeline custom.
        
        Args:
            pipeline: Pipeline à créer
            
        Returns:
            True si créé avec succès
        """
        if pipeline.id in self._pipelines:
            logger.warning(f"Pipeline '{pipeline.id}' already exists")
            return False
        
        # Forcer is_builtin à False
        pipeline.is_builtin = False
        
        # S'assurer d'avoir une date de création
        if not pipeline.created_at:
            pipeline.created_at = datetime.now().isoformat()
        
        self._pipelines[pipeline.id] = pipeline
        success = self._save_custom()
        
        if success:
            logger.info(f"Pipeline created: {pipeline.id}")
        
        return success
    
    def update(self, pipeline_id: str, pipeline: Pipeline) -> bool:
        """
        Met à jour un pipeline existant.
        
        Args:
            pipeline_id: ID du pipeline à modifier
            pipeline: Nouvelles données
            
        Returns:
            True si mis à jour avec succès
        """
        existing = self._pipelines.get(pipeline_id)
        if not existing:
            logger.warning(f"Pipeline not found: {pipeline_id}")
            return False
        
        if existing.is_builtin:
            logger.warning(f"Cannot modify builtin pipeline: {pipeline_id}")
            return False
        
        # Conserver is_builtin et created_at de l'existant
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
        Supprime un pipeline custom.
        
        Args:
            pipeline_id: ID du pipeline à supprimer
            
        Returns:
            True si supprimé avec succès
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
    
    def duplicate(self, pipeline_id: str, new_id: str) -> Optional[Pipeline]:
        """
        Duplique un pipeline (builtin ou custom).
        
        Args:
            pipeline_id: ID du pipeline source
            new_id: ID du nouveau pipeline
            
        Returns:
            Nouveau Pipeline ou None si échec
        """
        source = self._pipelines.get(pipeline_id)
        if not source:
            return None
        
        if new_id in self._pipelines:
            logger.warning(f"Pipeline ID '{new_id}' already exists")
            return None
        
        # Créer une copie profonde
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
        Exporte tous les pipelines (builtin + custom) en YAML.
        
        Returns:
            Contenu YAML
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
        Exporte uniquement les pipelines custom en YAML.
        
        Returns:
            Contenu YAML
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
        Exporte tous les pipelines dans un fichier.
        
        Args:
            filepath: Chemin du fichier de destination
            
        Returns:
            True si succès
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
    
    def import_from_yaml(self, yaml_content: str) -> List[str]:
        """
        Importe des pipelines depuis du contenu YAML.
        
        Args:
            yaml_content: Contenu YAML
            
        Returns:
            Liste des IDs importés
        """
        try:
            data = yaml.safe_load(yaml_content)
            pipelines_data = data.get("pipelines", data)
            
            imported = []
            
            for pipe_id, pipe_data in pipelines_data.items():
                if not isinstance(pipe_data, dict):
                    continue
                
                # Ne pas écraser les builtins
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
    
    def import_from_file(self, filepath: Path) -> List[str]:
        """
        Importe des pipelines depuis un fichier YAML.
        
        Args:
            filepath: Chemin du fichier
            
        Returns:
            Liste des IDs importés
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            return self.import_from_yaml(content)
        except Exception as e:
            logger.error(f"Import file error: {e}")
            return []
    
    # -------------------------------------------------------------------------
    # Génération de prompts via LLM
    # -------------------------------------------------------------------------
    
    # Modèle rapide par défaut pour la génération de prompts
    FAST_MODEL = "nemotron-mini"  # Rapide et efficace pour les tâches simples
    FALLBACK_MODEL = "qwen3:32b"  # Fallback si modèle rapide indisponible
    
    def _get_prompt_model(self) -> str:
        """
        Retourne le modèle à utiliser pour la génération de prompts.
        Vérifie si le modèle rapide est disponible, sinon utilise le fallback.
        Retourne le nom COMPLET du modèle (avec tag).
        """
        try:
            import requests
            response = requests.get("http://localhost:11434/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                
                if not models:
                    logger.warning("No models found in Ollama")
                    return self.FALLBACK_MODEL
                
                # Liste des noms complets disponibles
                model_names_full = [m.get("name", "") for m in models]
                logger.debug(f"Available models: {model_names_full}")
                
                # Chercher le modèle rapide ou ses variantes
                fast_candidates = [
                    "nemotron-mini", "nemotron-3-nano", "gemma3:4b", 
                    "phi4-mini", "phi3:mini", "llama3.2:3b", "qwen2.5:7b"
                ]
                
                for candidate in fast_candidates:
                    for full_name in model_names_full:
                        # Match si le candidat est dans le nom complet
                        if candidate in full_name:
                            logger.info(f"Selected fast model: {full_name}")
                            return full_name
                
                # Vérifier si le fallback existe
                for full_name in model_names_full:
                    if self.FALLBACK_MODEL.split(":")[0] in full_name:
                        logger.info(f"Using fallback model: {full_name}")
                        return full_name
                
                # Dernier recours : premier modèle disponible
                first_model = model_names_full[0]
                logger.info(f"Using first available model: {first_model}")
                return first_model
            else:
                logger.error(f"Ollama API returned status {response.status_code}")
                    
        except requests.exceptions.ConnectionError:
            logger.error("Cannot connect to Ollama at localhost:11434 - is it running?")
        except Exception as e:
            logger.error(f"Error checking available models: {e}")
        
        return self.FALLBACK_MODEL
    
    def generate_step_prompt(
        self,
        step_name: str,
        step_description: str,
        pipeline_context: str,
        agent_type: str,
        model: Optional[str] = None,
    ) -> Optional[str]:
        """
        Génère un prompt système optimisé pour une étape via LLM.
        
        Args:
            step_name: Nom de l'étape
            step_description: Description de ce que l'étape doit faire
            pipeline_context: Contexte du pipeline (objectif global)
            agent_type: Type d'agent (coder, reviewer, etc.)
            model: Modèle Ollama à utiliser (auto-sélectionné si None)
            
        Returns:
            Prompt système généré ou None si erreur
        """
        if model is None:
            model = self._get_prompt_model()
            
        try:
            import requests
            
            meta_prompt = f"""Tu es un expert en prompt engineering pour LLMs.
Génère un prompt système concis et efficace pour un agent '{agent_type}'.

Contexte du pipeline: {pipeline_context}
Nom de l'étape: {step_name}
Description de l'étape: {step_description}

Règles:
- Le prompt doit être direct et professionnel
- En anglais
- Pas plus de 200 mots
- Inclure les attentes de format de sortie si pertinent
- Adapter au type d'agent ({agent_type})

Retourne UNIQUEMENT le prompt, sans explication ni balises."""

            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model,
                    "prompt": meta_prompt,
                    "stream": False,
                    "options": {"temperature": 0.3}
                },
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            else:
                logger.error(f"LLM request failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating prompt: {e}")
            return None
    
    def generate_step_prompt_with_keepalive(
        self,
        step_name: str,
        step_description: str,
        pipeline_context: str,
        agent_type: str,
    ):
        """
        Génère un prompt avec keepalive pour éviter timeout Gradio.
        
        Yields des messages de statut toutes les 2 secondes pendant la génération,
        puis le prompt final ou un message d'erreur.
        
        Args:
            step_name: Nom de l'étape
            step_description: Description de ce que l'étape doit faire
            pipeline_context: Contexte du pipeline (objectif global)
            agent_type: Type d'agent (coder, reviewer, etc.)
            
        Yields:
            Messages de statut puis le prompt final
        """
        import threading
        import time as _time
        
        # Vérification des inputs
        if not step_description:
            yield "[ERR] Error: Step description is required"
            return
        
        # Sélectionner le modèle
        model = self._get_prompt_model()
        yield f"Initializing with model: {model}..."
        
        # Container pour le résultat du thread
        result = {"prompt": None, "error": None, "done": False}
        
        def generate_in_thread():
            try:
                prompt = self.generate_step_prompt(
                    step_name=step_name or "Step",
                    step_description=step_description,
                    pipeline_context=pipeline_context or "General pipeline",
                    agent_type=agent_type or "coder",
                    model=model,
                )
                result["prompt"] = prompt
            except Exception as e:
                result["error"] = str(e)
            finally:
                result["done"] = True
        
        # Lancer la génération dans un thread séparé
        thread = threading.Thread(target=generate_in_thread, daemon=True)
        thread.start()
        
        # Keepalive: yield un message toutes les 2 secondes
        dots = 0
        elapsed = 0
        while not result["done"]:
            dots = (dots % 3) + 1
            elapsed += 2
            yield f"Generating prompt{'.' * dots} ({elapsed}s)"
            _time.sleep(2)
        
        # Résultat final
        if result["error"]:
            yield f"[ERR] Error: {result['error']}"
        elif result["prompt"]:
            yield result["prompt"]
        else:
            # Vérifier pourquoi ça a échoué
            try:
                import requests
                response = requests.get("http://localhost:11434/api/tags", timeout=3)
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    if not models:
                        yield "[ERR] Ollama is running but no models are installed. Run: ollama pull qwen3:32b"
                    else:
                        yield f"[ERR] Generation failed. Model '{model}' may not exist. Available: {[m['name'] for m in models[:3]]}"
                else:
                    yield f"[ERR] Ollama API error (status {response.status_code})"
            except requests.exceptions.ConnectionError:
                yield "[ERR] Cannot connect to Ollama at localhost:11434. Is Ollama running?"
            except Exception as e:
                yield f"[ERR] Unknown error: {e}"
    
    # -------------------------------------------------------------------------
    # Utilitaires
    # -------------------------------------------------------------------------
    
    def get_stats(self) -> Dict[str, Any]:
        """Retourne des statistiques sur les pipelines."""
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
        """Vérifie si un ID de pipeline est valide."""
        if not pipeline_id or not pipeline_id.strip():
            return False
        return bool(re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', pipeline_id))
    
    def get_pipelines_for_orchestrator(self) -> Dict[str, Dict]:
        """
        Retourne les pipelines dans le format attendu par l'orchestrateur.
        
        Returns:
            Dict compatible avec config["pipelines"]
        """
        result = {}
        for pipe_id, pipeline in self._pipelines.items():
            result[pipe_id] = pipeline.to_config_dict()
        return result


# =============================================================================
# INSTANCE GLOBALE
# =============================================================================

_pipeline_manager: Optional[PipelineManager] = None


def get_pipeline_manager() -> PipelineManager:
    """Récupère l'instance globale du gestionnaire de pipelines."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager


def list_pipelines() -> List[Pipeline]:
    """Fonction de commodité pour lister les pipelines."""
    return get_pipeline_manager().list_all()


def get_pipeline(pipeline_id: str) -> Optional[Pipeline]:
    """Fonction de commodité pour récupérer un pipeline."""
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
    
    print("\nPipelines disponibles:")
    for pipeline in manager.list_all():
        status = "📌" if pipeline.is_builtin else "🧅"
        print(f"  {pipeline.emoji} {status} {pipeline.id}: {pipeline.name} ({pipeline.step_count} steps)")
    
    print("\nAgents disponibles:")
    for agent in manager.get_available_agents():
        print(f"  - {agent}")
    
    print("\nTemplates disponibles:")
    for template in manager.get_available_templates()[:10]:
        print(f"  - {template}")
    if len(manager.get_available_templates()) > 10:
        print(f"  ... et {len(manager.get_available_templates()) - 10} autres")
    
    # Test de création
    print("\nTest de création d'un pipeline custom...")
    test_pipeline = Pipeline(
        id="test_custom",
        name="Test Custom Pipeline",
        description="Pipeline de test",
        pattern="chain",
        emoji="🧪",
        steps=[
            PipelineStep(
                name="Analyse",
                agent="reviewer",
                prompt_template="error_analysis",
                description="Analyser le problème",
            ),
            PipelineStep(
                name="Solution",
                agent="coder",
                system_prompt="You are an expert problem solver. Fix the issue.",
                description="Proposer une solution",
            ),
        ],
        keywords=["test", "debug"],
        detection_weight=0.7,
    )
    
    if manager.create(test_pipeline):
        print(f"[OK] Pipeline '{test_pipeline.id}' créé")
        
        # Vérifier
        loaded = manager.get("test_custom")
        if loaded:
            print(f"   Loaded: {loaded.name} with {loaded.step_count} steps")
        
        # Supprimer
        if manager.delete("test_custom"):
            print(f"[OK] Pipeline 'test_custom' supprimé")
    else:
        print("[ERR] Échec de création")
    
    print("\n[OK] Pipeline Manager fonctionnel")
