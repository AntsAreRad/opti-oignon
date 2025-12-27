#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DYNAMIC PIPELINE - OPTI-OIGNON 1.1
==================================

Module de planification dynamique de pipeline multi-modèles.

Ce module analyse le prompt utilisateur et génère automatiquement
un pipeline optimal en orchestrant plusieurs modèles spécialisés.

Fonctionnalités:
- Analyse de complexité et type de tâche via LLM
- Planification automatique des étapes
- Génération de plan JSON structuré
- Validation et parsing robuste du plan
- Estimation du temps d'exécution
- Mode preview avec confirmation utilisateur

Author: Léon
Version: 1.1 (Phase A5)
"""

import json
import time
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Generator
from enum import Enum
from datetime import datetime

# Import conditionnel d'ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    ollama = None

logger = logging.getLogger("DynamicPipeline")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class PlanComplexity(Enum):
    """Niveau de complexité du plan."""
    SIMPLE = "simple"      # 1 étape, modèle unique suffit
    MEDIUM = "medium"      # 2-3 étapes, orchestration légère
    COMPLEX = "complex"    # 4+ étapes, orchestration complète


@dataclass
class PipelineStep:
    """
    Une étape du pipeline planifié dynamiquement.
    
    Attributes:
        step_number: Numéro d'ordre de l'étape (1-based)
        agent_type: Type d'agent ("planner", "coder", "reviewer", "explainer")
        model: Modèle Ollama spécifique à utiliser
        task_description: Description de la tâche pour cette étape
        expected_output: Type de sortie attendue
        depends_on: Liste des étapes dont celle-ci dépend
        timeout: Timeout en secondes pour cette étape
    """
    step_number: int
    agent_type: str
    model: str
    task_description: str
    expected_output: str
    depends_on: List[int] = field(default_factory=list)
    timeout: int = 120
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire."""
        return {
            "step": self.step_number,
            "agent": self.agent_type,
            "model": self.model,
            "task": self.task_description,
            "output": self.expected_output,
            "depends_on": self.depends_on,
            "timeout": self.timeout,
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'PipelineStep':
        """Crée une instance depuis un dictionnaire."""
        return cls(
            step_number=data.get("step", 1),
            agent_type=data.get("agent", "coder"),
            model=data.get("model", "qwen3-coder:30b"),
            task_description=data.get("task", ""),
            expected_output=data.get("output", ""),
            depends_on=data.get("depends_on", []),
            timeout=data.get("timeout", 120),
        )


@dataclass
class DynamicPipelinePlan:
    """
    Plan de pipeline généré dynamiquement par le LLM planificateur.
    
    Attributes:
        analysis: Analyse textuelle du problème
        complexity: Niveau de complexité détecté
        reasoning: Justification du plan choisi
        recommended_pipeline: Liste des étapes planifiées
        single_model_sufficient: Si True, pas besoin de multi-agent
        estimated_time: Estimation du temps total
        raw_response: Réponse brute du LLM (debug)
        generation_time: Temps de génération du plan
        planner_model: Modèle utilisé pour la planification
    """
    analysis: str
    complexity: PlanComplexity
    reasoning: str
    recommended_pipeline: List[PipelineStep]
    single_model_sufficient: bool
    estimated_time: str
    raw_response: str = ""
    generation_time: float = 0.0
    planner_model: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict:
        """Convertit en dictionnaire pour sérialisation."""
        return {
            "analysis": self.analysis,
            "complexity": self.complexity.value,
            "reasoning": self.reasoning,
            "recommended_pipeline": [s.to_dict() for s in self.recommended_pipeline],
            "single_model_sufficient": self.single_model_sufficient,
            "estimated_time": self.estimated_time,
            "generation_time": self.generation_time,
            "planner_model": self.planner_model,
            "timestamp": self.timestamp,
        }
    
    @property
    def step_count(self) -> int:
        """Nombre d'étapes dans le pipeline."""
        return len(self.recommended_pipeline)
    
    def format_preview(self, show_models: bool = True) -> str:
        """
        Formate le plan pour affichage utilisateur.
        
        Returns:
            Chaîne formatée pour affichage Markdown
        """
        lines = [
            "═" * 60,
            "DYNAMIC PIPELINE PLAN",
            "═" * 60,
            "",
            f"**Analysis:** {self.analysis}",
            "",
            f"**Complexity:** {self.complexity.value}",
            f"**Estimated time:** {self.estimated_time}",
            "",
            "**Planned Steps:**",
            "",
        ]
        
        for step in self.recommended_pipeline:
            agent_emoji = {
                "planner": "🧠",
                "coder": "💻",
                "reviewer": "🔍",
                "explainer": "📚",
                "writer": "✍️",
            }.get(step.agent_type, "⚙️")
            
            model_info = f" ({step.model})" if show_models else ""
            lines.append(f"  **Step {step.step_number}:** {agent_emoji} {step.agent_type.upper()}{model_info}")
            lines.append(f"  └─ Task: {step.task_description}")
            lines.append("")
        
        lines.extend([
            "═" * 60,
            "",
            f"**Reasoning:** {self.reasoning}",
        ])
        
        return "\n".join(lines)


# =============================================================================
# DYNAMIC PIPELINE PLANNER
# =============================================================================

class DynamicPipelinePlanner:
    """
    Planificateur de pipeline dynamique utilisant un LLM pour analyser
    les requêtes et générer des plans d'exécution optimaux.
    
    Le planificateur utilise un modèle de reasoning (par défaut deepseek-r1:32b)
    pour analyser la complexité de la tâche et déterminer la meilleure
    séquence d'agents spécialisés.
    
    Example:
        planner = DynamicPipelinePlanner(config)
        plan = planner.plan("Écris une fonction R avec tests")
        if not plan.single_model_sufficient:
            # Execute multi-agent pipeline
            pass
    """
    
    # Prompt de planification optimisé pour générer un JSON structuré
    PLANNING_PROMPT = '''You are a task orchestration expert for a local LLM system.
Analyze the user request and determine if a multi-model pipeline would be beneficial.

## Available Specialized Agents

| Agent | Model | Best For | Speed |
|-------|-------|----------|-------|
| **planner** | deepseek-r1:32b | Complex reasoning, task decomposition, strategy | ~60s |
| **coder** | qwen3-coder:30b | Code generation (R/Python), fast and accurate | ~30s |
| **reviewer** | qwen3-coder:30b | Code review, bug detection, quality checks | ~30s |
| **explainer** | qwen3:32b | Clear explanations, documentation, teaching | ~45s |
| **writer** | qwen3:32b | Scientific writing, reports, documentation | ~45s |

## User Request
{user_prompt}

## Additional Context (if any)
{context}

## Your Analysis Task

Respond ONLY with a valid JSON object (no markdown, no explanation outside JSON):

```json
{{
  "analysis": "Brief analysis of what this task requires (1-2 sentences)",
  "complexity": "simple|medium|complex",
  "single_model_sufficient": true|false,
  "reasoning": "Why you chose this approach (1-2 sentences)",
  "recommended_pipeline": [
    {{
      "step": 1,
      "agent": "planner|coder|reviewer|explainer|writer",
      "model": "model-name:size",
      "task": "What this step accomplishes",
      "output": "Expected output type (code/analysis/review/explanation)"
    }}
  ],
  "estimated_time": "~X minutes"
}}
```

## Decision Rules

1. **Simple tasks** (single_model_sufficient: true):
   - Direct questions, simple explanations
   - Basic code snippets (< 50 lines)
   - Pipeline: just 1 step with the appropriate agent

2. **Medium tasks** (2-3 steps):
   - Code with testing/review needed
   - Multi-part questions
   - Pattern: planner → coder → reviewer

3. **Complex tasks** (3-4 steps):
   - Full implementations with tests
   - Research + implementation
   - Debugging + fixing + validation
   - Pattern varies by task type

## Common Patterns

- **Code generation**: coder (or planner → coder for complex)
- **Code + tests**: planner → coder → reviewer
- **Debugging**: reviewer → coder → reviewer
- **Explanation**: explainer alone
- **Scientific writing**: planner → writer → reviewer
- **Complex analysis**: planner → coder → explainer

## Important
- Maximum 4 steps unless truly necessary
- Prefer fewer steps when possible (efficiency)
- Always use qwen3-coder:30b for code tasks
- Always use deepseek-r1:32b for complex reasoning
- Output ONLY valid JSON, nothing else'''

    # Prompt simplifié pour modèle rapide (fallback)
    FAST_PLANNING_PROMPT = '''Analyze this request briefly. Respond with JSON only.

Request: {user_prompt}

JSON format:
{{"analysis": "...", "complexity": "simple|medium|complex", "single_model_sufficient": true|false, "reasoning": "...", "recommended_pipeline": [{{"step": 1, "agent": "coder", "model": "qwen3-coder:30b", "task": "...", "output": "..."}}], "estimated_time": "~1 min"}}'''

    # Configuration par défaut des modèles pour chaque type d'agent
    DEFAULT_AGENT_MODELS = {
        "planner": "deepseek-r1:32b",
        "coder": "qwen3-coder:30b",
        "reviewer": "qwen3-coder:30b",
        "explainer": "qwen3:32b",
        "writer": "qwen3:32b",
    }
    
    def __init__(
        self, 
        config: Optional[Dict] = None,
        planning_model: Optional[str] = None,
        fast_mode: bool = False,
    ):
        """
        Initialise le planificateur.
        
        Args:
            config: Configuration globale (optionnel)
            planning_model: Modèle pour la planification (défaut: deepseek-r1:32b)
            fast_mode: Si True, utilise un prompt simplifié et modèle rapide
        """
        self.config = config or {}
        self.fast_mode = fast_mode
        
        # Modèle de planification
        if planning_model:
            self.planning_model = planning_model
        elif fast_mode:
            self.planning_model = self.config.get("fast_planning_model", "qwen3:32b")
        else:
            self.planning_model = self.config.get("planning_model", "deepseek-r1:32b")
        
        # Modèles par agent (peut être override via config)
        self.agent_models = self.DEFAULT_AGENT_MODELS.copy()
        if "agent_models" in self.config:
            self.agent_models.update(self.config["agent_models"])
        
        # Paramètres
        self.temperature = self.config.get("planning_temperature", 0.3)
        self.timeout = self.config.get("planning_timeout", 180)
        self.max_retries = self.config.get("planning_retries", 2)
        
        logger.info(f"DynamicPipelinePlanner initialized with model: {self.planning_model}")
    
    def plan(
        self, 
        user_prompt: str, 
        context: str = "",
        force_complexity: Optional[str] = None,
    ) -> DynamicPipelinePlan:
        """
        Génère un plan de pipeline dynamique pour la requête.
        
        Args:
            user_prompt: Prompt/question de l'utilisateur
            context: Contexte additionnel (document, code, etc.)
            force_complexity: Force un niveau de complexité
            
        Returns:
            DynamicPipelinePlan avec les étapes recommandées
        """
        start_time = time.time()
        
        # Vérifier disponibilité d'ollama
        if not OLLAMA_AVAILABLE:
            logger.warning("Ollama not available, using fallback plan")
            return self._create_fallback_plan(user_prompt, "Ollama not available", time.time() - start_time)
        
        # Construction du prompt
        template = self.FAST_PLANNING_PROMPT if self.fast_mode else self.PLANNING_PROMPT
        prompt = template.format(
            user_prompt=user_prompt[:2000],  # Limite pour éviter context overflow
            context=context[:1000] if context else "None provided",
        )
        
        # Appel au LLM avec retries
        raw_response = ""
        last_error = None
        
        for attempt in range(self.max_retries + 1):
            try:
                response = ollama.chat(
                    model=self.planning_model,
                    messages=[
                        {"role": "system", "content": "You are a pipeline planning assistant. Output only valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": self.temperature},
                )
                raw_response = response.get("message", {}).get("content", "")
                
                # Parse le JSON
                plan_data = self._parse_plan_json(raw_response)
                
                if plan_data:
                    generation_time = time.time() - start_time
                    plan = self._build_plan(plan_data, raw_response, generation_time)
                    
                    # Override complexity si demandé
                    if force_complexity:
                        try:
                            plan.complexity = PlanComplexity(force_complexity)
                        except ValueError:
                            pass
                    
                    logger.info(f"Plan generated in {generation_time:.1f}s: {plan.step_count} steps, {plan.complexity.value}")
                    return plan
                    
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Planning attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries:
                    time.sleep(1)  # Petit délai avant retry
        
        # Fallback: plan par défaut si échec
        logger.error(f"Planning failed after {self.max_retries + 1} attempts: {last_error}")
        return self._create_fallback_plan(user_prompt, raw_response, time.time() - start_time)
    
    def _parse_plan_json(self, raw_response: str) -> Optional[Dict]:
        """
        Parse le JSON depuis la réponse LLM.
        
        Gère plusieurs formats de sortie possibles:
        - JSON pur
        - JSON dans bloc ```json```
        - JSON avec texte avant/après
        """
        if not raw_response:
            return None
        
        # Tentative 1: JSON direct
        try:
            return json.loads(raw_response.strip())
        except json.JSONDecodeError:
            pass
        
        # Tentative 2: Extraction depuis bloc de code
        json_patterns = [
            r'```json\s*([\s\S]*?)\s*```',
            r'```\s*([\s\S]*?)\s*```',
            r'\{[\s\S]*\}',
        ]
        
        for pattern in json_patterns:
            match = re.search(pattern, raw_response)
            if match:
                try:
                    json_str = match.group(1) if '```' in pattern else match.group(0)
                    return json.loads(json_str.strip())
                except json.JSONDecodeError:
                    continue
        
        # Tentative 3: Nettoyage agressif
        try:
            # Supprime le contenu avant le premier { et après le dernier }
            start = raw_response.find('{')
            end = raw_response.rfind('}')
            if start != -1 and end != -1 and end > start:
                cleaned = raw_response[start:end + 1]
                return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        
        logger.warning("Failed to parse plan JSON from response")
        return None
    
    def _build_plan(
        self, 
        plan_data: Dict, 
        raw_response: str, 
        generation_time: float
    ) -> DynamicPipelinePlan:
        """Construit un DynamicPipelinePlan depuis les données parsées."""
        
        # Parse complexity
        complexity_str = plan_data.get("complexity", "medium").lower()
        try:
            complexity = PlanComplexity(complexity_str)
        except ValueError:
            complexity = PlanComplexity.MEDIUM
        
        # Parse pipeline steps
        steps = []
        raw_steps = plan_data.get("recommended_pipeline", [])
        
        for i, step_data in enumerate(raw_steps):
            # Validation et normalisation
            agent = step_data.get("agent", "coder").lower()
            if agent not in self.agent_models:
                agent = "coder"  # Fallback
            
            # Utilise le modèle configuré pour l'agent si non spécifié
            model = step_data.get("model")
            if not model or model == "auto":
                model = self.agent_models.get(agent, "qwen3-coder:30b")
            
            step = PipelineStep(
                step_number=step_data.get("step", i + 1),
                agent_type=agent,
                model=model,
                task_description=step_data.get("task", f"Step {i + 1}"),
                expected_output=step_data.get("output", "result"),
                depends_on=step_data.get("depends_on", []),
            )
            steps.append(step)
        
        # Si pas d'étapes, en crée une par défaut
        if not steps:
            steps = [PipelineStep(
                step_number=1,
                agent_type="coder",
                model=self.agent_models["coder"],
                task_description="Process the user request",
                expected_output="result",
            )]
        
        return DynamicPipelinePlan(
            analysis=plan_data.get("analysis", "Task analysis"),
            complexity=complexity,
            reasoning=plan_data.get("reasoning", "Standard processing"),
            recommended_pipeline=steps,
            single_model_sufficient=plan_data.get("single_model_sufficient", len(steps) == 1),
            estimated_time=plan_data.get("estimated_time", f"~{len(steps)} min"),
            raw_response=raw_response,
            generation_time=generation_time,
            planner_model=self.planning_model,
        )
    
    def _create_fallback_plan(
        self, 
        user_prompt: str, 
        raw_response: str, 
        generation_time: float
    ) -> DynamicPipelinePlan:
        """
        Crée un plan de fallback en cas d'échec du parsing.
        
        Utilise des heuristiques simples pour déterminer le type de tâche.
        """
        prompt_lower = user_prompt.lower()
        
        # Détection simple du type de tâche
        is_code = any(kw in prompt_lower for kw in [
            "code", "function", "script", "implement", "write",
            "fonction", "implémenter", "écrire", "créer"
        ])
        is_debug = any(kw in prompt_lower for kw in [
            "error", "bug", "fix", "debug", "crash",
            "erreur", "réparer", "corriger"
        ])
        is_explain = any(kw in prompt_lower for kw in [
            "explain", "what is", "how does", "why",
            "expliquer", "qu'est-ce", "comment", "pourquoi"
        ])
        
        # Choix de l'agent et complexité
        if is_debug:
            agent = "reviewer"
            complexity = PlanComplexity.MEDIUM
            task = "Debug and fix the issue"
        elif is_code:
            agent = "coder"
            complexity = PlanComplexity.SIMPLE
            task = "Generate the requested code"
        elif is_explain:
            agent = "explainer"
            complexity = PlanComplexity.SIMPLE
            task = "Provide a clear explanation"
        else:
            agent = "coder"
            complexity = PlanComplexity.SIMPLE
            task = "Process the request"
        
        return DynamicPipelinePlan(
            analysis="Fallback analysis - could not parse LLM response",
            complexity=complexity,
            reasoning="Using fallback heuristics due to planning failure",
            recommended_pipeline=[PipelineStep(
                step_number=1,
                agent_type=agent,
                model=self.agent_models[agent],
                task_description=task,
                expected_output="result",
            )],
            single_model_sufficient=True,
            estimated_time="~1 min",
            raw_response=raw_response,
            generation_time=generation_time,
            planner_model=self.planning_model,
        )
    
    def should_use_pipeline(self, plan: DynamicPipelinePlan) -> bool:
        """
        Détermine si le pipeline multi-agent vaut le coup.
        
        Args:
            plan: Plan généré
            
        Returns:
            True si multi-agent recommandé, False sinon
        """
        # Si explicitement marqué comme suffisant avec un seul modèle
        if plan.single_model_sufficient:
            return False
        
        # Si complexité simple, généralement pas besoin
        if plan.complexity == PlanComplexity.SIMPLE:
            return False
        
        # Si plusieurs étapes, utiliser le pipeline
        if plan.step_count >= 2:
            return True
        
        return False
    
    def quick_analyze(self, user_prompt: str) -> Tuple[str, bool]:
        """
        Analyse rapide sans appel LLM (heuristiques uniquement).
        
        Utile pour pré-filtrer avant de lancer la planification complète.
        
        Returns:
            (complexity_hint, might_need_pipeline)
        """
        prompt_lower = user_prompt.lower()
        word_count = len(user_prompt.split())
        
        # Indicateurs de complexité
        complexity_markers = {
            "complex": [
                "implement", "create", "build", "develop", "test",
                "with", "including", "and also", "multi", "several",
                "implémente", "crée", "développe", "teste", "avec",
            ],
            "simple": [
                "what is", "how to", "explain", "show me",
                "qu'est-ce", "comment", "explique", "montre",
            ]
        }
        
        complex_count = sum(1 for m in complexity_markers["complex"] if m in prompt_lower)
        simple_count = sum(1 for m in complexity_markers["simple"] if m in prompt_lower)
        
        # Décision
        if simple_count > complex_count or word_count < 15:
            return "simple", False
        elif complex_count >= 2 or word_count > 50:
            return "complex", True
        else:
            return "medium", complex_count > 0


# =============================================================================
# DYNAMIC PIPELINE EXECUTOR
# =============================================================================

class DynamicPipelineExecutor:
    """
    Exécute un plan de pipeline dynamique.
    
    Coordonne l'exécution séquentielle des étapes avec:
    - Passage du contexte entre étapes
    - Callbacks pour suivi de progression
    - Gestion des erreurs et rollback
    - Streaming optionnel
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialise l'exécuteur.
        
        Args:
            config: Configuration globale
        """
        self.config = config or {}
        self._cancel_event = None
        
        # Import des agents (lazy)
        self._orchestrator = None
        
    def _get_orchestrator(self):
        """Récupère l'orchestrateur de manière lazy."""
        if self._orchestrator is None:
            try:
                from .orchestrator import get_orchestrator
                self._orchestrator = get_orchestrator()
            except ImportError:
                logger.warning("Orchestrator not available")
        return self._orchestrator
    
    def execute(
        self,
        plan: DynamicPipelinePlan,
        user_input: str,
        context: Optional[Dict] = None,
        stream: bool = True,
        on_step_start: Optional[callable] = None,
        on_step_complete: Optional[callable] = None,
        on_token: Optional[callable] = None,
    ) -> Generator[str, None, Dict]:
        """
        Exécute le plan de pipeline avec streaming.
        
        Args:
            plan: Plan à exécuter
            user_input: Input utilisateur original
            context: Contexte additionnel
            stream: Active le streaming
            on_step_start: Callback au début de chaque étape
            on_step_complete: Callback à la fin de chaque étape
            on_token: Callback pour chaque token (si streaming)
            
        Yields:
            Tokens de la réponse (si streaming)
            
        Returns:
            Dict avec résultat final et métadonnées
        """
        context = context or {}
        context["original_input"] = user_input
        context["plan"] = plan.to_dict()
        
        results = []
        total_start = time.time()
        
        for step in plan.recommended_pipeline:
            step_start = time.time()
            
            # Callback début d'étape
            if on_step_start:
                on_step_start(step.step_number, step.agent_type, step.model)
            
            yield f"\n\n---\n**Step {step.step_number}: {step.agent_type.upper()}** ({step.model})\n"
            yield f"*{step.task_description}*\n\n"
            
            # Construit le prompt pour cette étape
            step_prompt = self._build_step_prompt(step, user_input, results, context)
            
            # Exécute l'étape
            try:
                step_result = ""
                
                for token in self._execute_step(step, step_prompt, stream):
                    step_result += token
                    if on_token:
                        on_token(token)
                    yield token
                
                step_time = time.time() - step_start
                
                results.append({
                    "step": step.step_number,
                    "agent": step.agent_type,
                    "model": step.model,
                    "output": step_result,
                    "time": step_time,
                    "status": "completed",
                })
                
                # Met à jour le contexte
                context["previous_output"] = step_result
                context[f"step_{step.step_number}_output"] = step_result
                
                # Callback fin d'étape
                if on_step_complete:
                    on_step_complete(step.step_number, step_result, step_time)
                    
            except Exception as e:
                logger.error(f"Step {step.step_number} failed: {e}")
                results.append({
                    "step": step.step_number,
                    "agent": step.agent_type,
                    "status": "failed",
                    "error": str(e),
                })
                yield f"\n\n[ERROR] Step {step.step_number} failed: {e}"
                break
        
        total_time = time.time() - total_start
        
        # Résultat final
        final_output = results[-1].get("output", "") if results else ""
        
        return {
            "success": all(r.get("status") == "completed" for r in results),
            "final_output": final_output,
            "steps": results,
            "total_time": total_time,
            "plan": plan.to_dict(),
        }
    
    def _build_step_prompt(
        self,
        step: PipelineStep,
        original_input: str,
        previous_results: List[Dict],
        context: Dict,
    ) -> str:
        """Construit le prompt pour une étape."""
        
        # Prompt de base
        prompt_parts = [f"Task: {step.task_description}\n"]
        
        # Ajoute l'input original
        prompt_parts.append(f"Original request:\n{original_input}\n")
        
        # Ajoute les résultats précédents si pertinents
        if previous_results:
            prompt_parts.append("\n--- Previous steps output ---\n")
            for prev in previous_results[-2:]:  # Les 2 derniers max
                if prev.get("output"):
                    prompt_parts.append(f"[Step {prev['step']} - {prev['agent']}]:\n")
                    # Tronque si trop long
                    output = prev["output"]
                    if len(output) > 2000:
                        output = output[:2000] + "\n... (truncated)"
                    prompt_parts.append(f"{output}\n")
        
        # Ajoute le contexte document si présent
        if context.get("document"):
            prompt_parts.append(f"\n--- Document context ---\n{context['document'][:1500]}\n")
        
        return "\n".join(prompt_parts)
    
    def _execute_step(
        self, 
        step: PipelineStep, 
        prompt: str, 
        stream: bool
    ) -> Generator[str, None, None]:
        """Exécute une étape individuelle."""
        
        # Vérifier disponibilité d'ollama
        if not OLLAMA_AVAILABLE:
            yield f"[ERROR] Ollama not available for step execution"
            return
        
        # System prompt basé sur le type d'agent
        system_prompts = {
            "planner": "You are an expert planner. Analyze the task and create a clear action plan. Be structured and thorough.",
            "coder": "You are an expert programmer. Write clean, well-documented code. Use best practices and include comments.",
            "reviewer": "You are a code reviewer. Analyze the code carefully, identify issues, and suggest improvements.",
            "explainer": "You are a teacher. Explain concepts clearly with examples. Make complex topics accessible.",
            "writer": "You are a scientific writer. Write clear, well-structured content following academic conventions.",
        }
        
        system = system_prompts.get(step.agent_type, "You are a helpful assistant.")
        
        try:
            if stream:
                response_stream = ollama.chat(
                    model=step.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.4},
                    stream=True,
                )
                
                for chunk in response_stream:
                    if "message" in chunk and "content" in chunk["message"]:
                        yield chunk["message"]["content"]
            else:
                response = ollama.chat(
                    model=step.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt}
                    ],
                    options={"temperature": 0.4},
                )
                yield response.get("message", {}).get("content", "")
                
        except Exception as e:
            logger.error(f"Step execution error: {e}")
            yield f"[Error: {e}]"


# =============================================================================
# GLOBAL INSTANCES AND CONVENIENCE FUNCTIONS
# =============================================================================

_planner: Optional[DynamicPipelinePlanner] = None
_executor: Optional[DynamicPipelineExecutor] = None


def get_planner(config: Optional[Dict] = None) -> DynamicPipelinePlanner:
    """Récupère l'instance globale du planificateur."""
    global _planner
    if _planner is None:
        _planner = DynamicPipelinePlanner(config)
    return _planner


def get_executor(config: Optional[Dict] = None) -> DynamicPipelineExecutor:
    """Récupère l'instance globale de l'exécuteur."""
    global _executor
    if _executor is None:
        _executor = DynamicPipelineExecutor(config)
    return _executor


def plan_pipeline(
    user_prompt: str, 
    context: str = "",
    fast_mode: bool = False,
) -> DynamicPipelinePlan:
    """
    Fonction de commodité pour planifier un pipeline.
    
    Args:
        user_prompt: Requête utilisateur
        context: Contexte additionnel
        fast_mode: Mode rapide (moins précis mais plus rapide)
        
    Returns:
        Plan de pipeline
    """
    planner = get_planner()
    if fast_mode:
        planner.fast_mode = True
    return planner.plan(user_prompt, context)


def execute_dynamic_pipeline(
    plan: DynamicPipelinePlan,
    user_input: str,
    context: Optional[Dict] = None,
    stream: bool = True,
) -> Generator[str, None, Dict]:
    """
    Fonction de commodité pour exécuter un pipeline.
    
    Args:
        plan: Plan à exécuter
        user_input: Input utilisateur
        context: Contexte additionnel
        stream: Active le streaming
        
    Yields:
        Tokens de la réponse
        
    Returns:
        Résultat final
    """
    executor = get_executor()
    return executor.execute(plan, user_input, context, stream)


# =============================================================================
# CLI POUR TESTS
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧅 DYNAMIC PIPELINE PLANNER - Test")
    print("=" * 60)
    
    # Test prompts
    test_prompts = [
        "What is the Shannon diversity index?",
        "Write a Python function to calculate mean with error handling and tests",
        "I have an error in my R code: Error in sum(x): 'x' must be numeric",
        "Create a complete R package for biodiversity analysis with vegan",
    ]
    
    if len(sys.argv) > 1:
        test_prompts = [" ".join(sys.argv[1:])]
    
    planner = DynamicPipelinePlanner()
    
    for prompt in test_prompts:
        print(f"\n{'─' * 60}")
        print(f"Prompt: {prompt[:60]}...")
        print("─" * 60)
        
        # Quick analyze d'abord
        hint, might_need = planner.quick_analyze(prompt)
        print(f"Quick analysis: {hint}, needs pipeline: {might_need}")
        
        # Plan complet
        print("\nGenerating plan...")
        plan = planner.plan(prompt)
        
        print(f"\nPlan generated in {plan.generation_time:.1f}s")
        print(f"   Complexity: {plan.complexity.value}")
        print(f"   Single model sufficient: {plan.single_model_sufficient}")
        print(f"   Steps: {plan.step_count}")
        print(f"   Estimated time: {plan.estimated_time}")
        
        print("\nPipeline steps:")
        for step in plan.recommended_pipeline:
            print(f"   {step.step_number}. [{step.agent_type}] {step.model}")
            print(f"      → {step.task_description}")
        
        print(f"\nReasoning: {plan.reasoning}")
        
        should_use = planner.should_use_pipeline(plan)
        print(f"\n[OK] Should use multi-agent pipeline: {should_use}")
    
    print("\n" + "=" * 60)
    print("[OK] Test completed")
