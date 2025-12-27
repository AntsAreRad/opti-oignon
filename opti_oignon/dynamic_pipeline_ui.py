#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DYNAMIC PIPELINE UI - OPTI-OIGNON 1.1
=====================================

Module d'intégration UI pour le Dynamic Pipeline.

Ce module fait le pont entre l'interface Gradio et le système
de planification dynamique multi-modèles.

Fonctionnalités:
- Décision si dynamic pipeline nécessaire
- Formatage du plan pour affichage
- Streaming de l'exécution
- Gestion des callbacks UI
- Enregistrement dans l'historique

Author: Léon
Version: 1.1.3 (Phase A5 - Keepalives + History)
"""

import logging
import time
from typing import Generator, Tuple, Optional, Dict, Any, List

logger = logging.getLogger("DynamicPipelineUI")

# Import de l'historique
try:
    from .history import history
    HISTORY_AVAILABLE = True
except ImportError:
    HISTORY_AVAILABLE = False
    history = None

# Import conditionnel du dynamic pipeline (dans agents/)
try:
    from .agents.dynamic_pipeline import (
        DynamicPipelinePlanner,
        DynamicPipelineExecutor,
        DynamicPipelinePlan,
        PipelineStep,
        PlanComplexity,
        get_planner,
        get_executor,
        plan_pipeline,
    )
    DYNAMIC_PIPELINE_AVAILABLE = True
    logger.info("Dynamic pipeline module loaded successfully")
except ImportError as e:
    logger.warning(f"Dynamic pipeline module not available: {e}")
    DYNAMIC_PIPELINE_AVAILABLE = False
    DynamicPipelinePlanner = None
    DynamicPipelinePlan = None

# Import orchestrator pour l'exécution (dans agents/)
try:
    from .agents.orchestrator import get_orchestrator
    from .agents import is_multi_agent_enabled as is_dynamic_pipeline_available
    ORCHESTRATOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Orchestrator not available: {e}")
    ORCHESTRATOR_AVAILABLE = False
    get_orchestrator = None
    is_dynamic_pipeline_available = lambda: False


# =============================================================================
# CSS STYLES FOR DYNAMIC PIPELINE
# =============================================================================

DYNAMIC_PIPELINE_CSS = """
/* Dynamic Pipeline Plan Display */
.dynamic-plan {
    background: #1e2030;
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 16px;
    margin: 8px 0;
}

.dynamic-plan-header {
    color: #a78bfa;
    font-weight: bold;
    font-size: 1.1em;
    margin-bottom: 12px;
}

.dynamic-step {
    background: #252840;
    border-left: 3px solid #60a5fa;
    padding: 8px 12px;
    margin: 6px 0;
    border-radius: 4px;
}

.dynamic-step.running {
    border-left-color: #fbbf24;
    animation: pulse 2s infinite;
}

.dynamic-step.completed {
    border-left-color: #4ade80;
}

.dynamic-step.failed {
    border-left-color: #f87171;
}

.step-header {
    display: flex;
    align-items: center;
    gap: 8px;
}

.step-number {
    background: #60a5fa;
    color: #1a1a2e;
    width: 24px;
    height: 24px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: bold;
    font-size: 0.8em;
}

.step-agent {
    font-weight: bold;
    text-transform: uppercase;
    font-size: 0.9em;
}

.step-model {
    color: #6b7280;
    font-size: 0.8em;
}

.step-task {
    margin-top: 4px;
    color: #a0a0a0;
    font-size: 0.9em;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.6; }
}
"""


# =============================================================================
# DECISION FUNCTIONS
# =============================================================================

def should_use_dynamic_pipeline(
    question: str,
    enable_dynamic: bool = True,
    document: str = "",
    min_complexity: str = "medium",
) -> bool:
    """
    Détermine si le dynamic pipeline devrait être utilisé.
    
    Cette fonction fait une analyse rapide (heuristique) pour décider
    si la planification dynamique vaut le coup.
    
    Args:
        question: Question de l'utilisateur
        enable_dynamic: Si le dynamic est activé dans l'UI
        document: Document/code associé (optionnel)
        min_complexity: Complexité minimum pour activer ("simple", "medium", "complex")
        
    Returns:
        True si dynamic pipeline recommandé
    """
    # Vérifications de base
    if not enable_dynamic:
        logger.debug("Dynamic pipeline disabled by user")
        return False
    
    if not DYNAMIC_PIPELINE_AVAILABLE:
        logger.warning("Dynamic pipeline module not available")
        return False
    
    if not question or len(question.strip()) < 5:
        logger.debug("Question too short for dynamic pipeline")
        return False
    
    # Analyse rapide via le planner
    try:
        planner = get_planner()
        hint, might_need = planner.quick_analyze(question)
        
        logger.info(f"Quick analysis: complexity={hint}, might_need_pipeline={might_need}")
        
        # Si document fourni, augmenter les chances d'utiliser le pipeline
        if document and len(document) > 100:
            # Avec un document, on est plus susceptible d'avoir besoin d'analyse
            if hint in ("medium", "complex"):
                logger.info("Document present + medium/complex complexity → use dynamic")
                return True
        
        # Décision basée sur la complexité
        complexity_levels = {"simple": 1, "medium": 2, "complex": 3}
        min_level = complexity_levels.get(min_complexity, 2)
        hint_level = complexity_levels.get(hint, 1)
        
        should_use = might_need or hint_level >= min_level
        logger.info(f"Decision: should_use_dynamic={should_use}")
        return should_use
        
    except Exception as e:
        logger.error(f"Error in quick analysis: {e}")
        return False


def get_complexity_from_question(question: str, document: str = "") -> str:
    """
    Estime la complexité d'une question.
    
    Args:
        question: Question utilisateur
        document: Document optionnel
        
    Returns:
        "simple", "medium", ou "complex"
    """
    if not DYNAMIC_PIPELINE_AVAILABLE:
        return "simple"
    
    try:
        planner = get_planner()
        hint, _ = planner.quick_analyze(question)
        return hint
    except Exception:
        return "medium"


# =============================================================================
# FORMATTING FUNCTIONS
# =============================================================================

def format_dynamic_plan_markdown(plan: 'DynamicPipelinePlan') -> str:
    """
    Formate un plan dynamique en Markdown pour l'affichage UI.
    
    Args:
        plan: Plan généré par le DynamicPipelinePlanner
        
    Returns:
        String Markdown formatée
    """
    if plan is None:
        return "_No plan generated_"
    
    # Emojis par type d'agent
    agent_emojis = {
        "planner": "🧠",
        "coder": "💻",
        "reviewer": "🔍",
        "explainer": "📚",
        "writer": "✍️",
    }
    
    # Couleurs par complexité
    complexity_colors = {
        "simple": "🟢",
        "medium": "🟡",
        "complex": "🔴",
    }
    
    lines = [
        "### Dynamic Pipeline Plan",
        "",
        f"**Analysis:** {plan.analysis}",
        "",
        f"**Complexity:** {complexity_colors.get(plan.complexity.value, '⚪')} {plan.complexity.value.upper()}",
        f"**Estimated time:** {plan.estimated_time}",
        f"**Planner:** {plan.planner_model}",
        "",
        "---",
        "",
        "**Planned Steps:**",
        "",
    ]
    
    for step in plan.recommended_pipeline:
        emoji = agent_emojis.get(step.agent_type, "⚙️")
        lines.append(f"**Step {step.step_number}:** {emoji} `{step.agent_type.upper()}` ({step.model})")
        lines.append(f"> {step.task_description}")
        lines.append("")
    
    lines.extend([
        "---",
        "",
        f"**Reasoning:** {plan.reasoning}",
    ])
    
    if plan.single_model_sufficient:
        lines.extend([
            "",
            "[i] _Single model sufficient - simplified execution_",
        ])
    
    return "\n".join(lines)


def format_step_progress(
    step: 'PipelineStep',
    status: str = "pending",
    output_preview: str = "",
) -> str:
    """
    Formate une étape pour l'affichage de progression.
    
    Args:
        step: Étape du pipeline
        status: "pending", "running", "completed", "failed"
        output_preview: Aperçu de la sortie (si complétée)
        
    Returns:
        Ligne formatée
    """
    status_icons = {
        "pending": "⏳",
        "running": "🔄",
        "completed": "✅",
        "failed": "❌",
    }
    
    agent_emojis = {
        "planner": "🧠",
        "coder": "💻",
        "reviewer": "🔍",
        "explainer": "📚",
        "writer": "✍️",
    }
    
    icon = status_icons.get(status, "❓")
    agent = agent_emojis.get(step.agent_type, "⚙️")
    
    line = f"{icon} **Step {step.step_number}:** {agent} {step.agent_type.upper()}"
    
    if status == "running":
        line += " _(in progress...)_"
    elif status == "completed" and output_preview:
        preview = output_preview[:100] + "..." if len(output_preview) > 100 else output_preview
        line += f"\n> _{preview}_"
    elif status == "failed":
        line += " _(failed)_"
    
    return line


# =============================================================================
# PROCESSING FUNCTION
# =============================================================================

def process_with_dynamic_pipeline(
    question: str,
    document: str = "",
    enable_dynamic: bool = True,
    auto_execute: bool = True,
    fast_mode: bool = False,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Traite une requête avec le Dynamic Pipeline.
    
    Cette fonction:
    1. Génère un plan dynamique via le LLM de planification
    2. Affiche le plan à l'utilisateur
    3. Exécute le pipeline si auto_execute=True
    4. Streame les résultats
    
    Args:
        question: Question utilisateur
        document: Document/code associé
        enable_dynamic: Si dynamic pipeline activé
        auto_execute: Exécuter automatiquement après planification
        fast_mode: Mode rapide (planification moins précise mais plus rapide)
        
    Yields:
        Tuple (status, plan_markdown, response)
    """
    import threading
    import queue
    
    if not DYNAMIC_PIPELINE_AVAILABLE:
        yield "[ERR] Dynamic pipeline not available", "", "Module not installed"
        return
    
    if not enable_dynamic:
        yield "[INFO] Dynamic pipeline disabled", "", ""
        return
    
    start_time = time.time()
    
    try:
        # ===== PHASE 1: PLANIFICATION (avec keepalives) =====
        yield "[>] Dynamic: Analyzing and planning...", "", ""
        
        planner = get_planner()
        planner.fast_mode = fast_mode
        
        # Utiliser un thread pour la planification avec keepalives
        result_queue = queue.Queue()
        error_queue = queue.Queue()
        
        def plan_thread():
            try:
                plan = planner.plan(question, document)
                result_queue.put(plan)
            except Exception as e:
                error_queue.put(e)
        
        thread = threading.Thread(target=plan_thread, daemon=True)
        thread.start()
        
        # Envoyer des keepalives pendant la planification
        dots = 0
        while thread.is_alive():
            elapsed = time.time() - start_time
            dots = (dots % 3) + 1
            yield f"[>] Planning{'.' * dots} ({elapsed:.0f}s)", "", ""
            thread.join(timeout=2.0)  # Keepalive toutes les 2 secondes
        
        # Vérifier les erreurs
        if not error_queue.empty():
            error = error_queue.get()
            yield f"[ERR] Planning failed: {error}", "", str(error)
            return
        
        # Récupérer le plan
        if result_queue.empty():
            yield "[ERR] No plan generated", "", "Planning returned empty"
            return
        
        plan = result_queue.get()
        plan_md = format_dynamic_plan_markdown(plan)
        
        planning_time = time.time() - start_time
        yield f"[OK] Plan generated in {planning_time:.1f}s", plan_md, ""
        
        # Vérifie si multi-agent nécessaire
        should_use_multi = planner.should_use_pipeline(plan)
        
        if not should_use_multi:
            yield "[INFO] Single model sufficient", plan_md, ""
            
            # Exécution simple avec un seul modèle (avec keepalives via thread)
            if auto_execute:
                step = plan.recommended_pipeline[0]
                yield f"[>] Executing with {step.model}...", plan_md, ""
                
                executor = get_executor()
                full_response = ""
                token_queue = queue.Queue()
                exec_error = [None]
                exec_done = [False]
                
                def single_exec_thread():
                    try:
                        for token in executor._execute_step(
                            step,
                            f"{question}\n\n{document}" if document else question,
                            stream=True
                        ):
                            token_queue.put(("token", token))
                        token_queue.put(("done", None))
                    except Exception as e:
                        exec_error[0] = e
                        token_queue.put(("error", str(e)))
                    finally:
                        exec_done[0] = True
                
                thread = threading.Thread(target=single_exec_thread, daemon=True)
                thread.start()
                
                # Boucle avec keepalives
                while not exec_done[0] or not token_queue.empty():
                    try:
                        msg_type, msg_data = token_queue.get(timeout=2.0)
                        if msg_type == "token":
                            full_response += msg_data
                            elapsed = time.time() - start_time
                            yield f"[>] Generating... ({len(full_response)} chars, {elapsed:.0f}s)", plan_md, full_response
                        elif msg_type == "done":
                            break
                        elif msg_type == "error":
                            yield f"[ERR] Execution failed: {msg_data}", plan_md, full_response
                            return
                    except queue.Empty:
                        # Keepalive
                        elapsed = time.time() - start_time
                        dots = int(elapsed) % 3 + 1
                        yield f"[>] Generating{'.' * dots} ({elapsed:.0f}s)", plan_md, full_response
                
                thread.join(timeout=5.0)
                
                total_time = time.time() - start_time
                
                # Enregistrer dans l'historique
                _save_dynamic_history(
                    question=question,
                    document=document,
                    response=full_response,
                    plan=plan,
                    step_results=[{"step": 1, "agent": step.agent_type, "model": step.model, "status": "completed"}],
                    total_time=total_time,
                    mode="single"
                )
                
                yield f"[OK] Completed in {total_time:.1f}s (1 step)", plan_md, full_response
            return
        
        # ===== PHASE 2: EXÉCUTION MULTI-AGENT =====
        if not auto_execute:
            yield "[WAIT] Plan ready - waiting for confirmation", plan_md, ""
            return
        
        yield f"[>] Executing {plan.step_count}-step pipeline...", plan_md, ""
        
        # Exécution du pipeline
        executor = get_executor()
        context = {"document": document}
        
        step_results = []
        full_response = ""
        
        for i, step in enumerate(plan.recommended_pipeline):
            step_status = f"[>] Step {step.step_number}/{plan.step_count}: {step.agent_type.upper()}..."
            yield step_status, plan_md, full_response
            
            # Construit le prompt avec contexte des étapes précédentes
            step_prompt = _build_step_prompt(step, question, step_results, document)
            
            # Exécute l'étape avec keepalives via thread + queue
            step_output = ""
            token_queue = queue.Queue()
            step_error = [None]  # Mutable pour capturer l'erreur
            step_done = [False]
            
            def execute_step_thread():
                try:
                    for token in executor._execute_step(step, step_prompt, stream=True):
                        token_queue.put(("token", token))
                    token_queue.put(("done", None))
                except Exception as e:
                    step_error[0] = e
                    token_queue.put(("error", str(e)))
                finally:
                    step_done[0] = True
            
            exec_thread = threading.Thread(target=execute_step_thread, daemon=True)
            exec_thread.start()
            
            last_yield_time = time.time()
            
            # Boucle avec keepalives
            while not step_done[0] or not token_queue.empty():
                try:
                    msg_type, msg_data = token_queue.get(timeout=2.0)
                    if msg_type == "token":
                        step_output += msg_data
                        # Yield le contenu
                        elapsed = time.time() - start_time
                        yield f"[>] Step {step.step_number}... ({elapsed:.0f}s)", plan_md, step_output
                        last_yield_time = time.time()
                    elif msg_type == "done":
                        break
                    elif msg_type == "error":
                        raise Exception(msg_data)
                except queue.Empty:
                    # Timeout - envoyer keepalive
                    elapsed = time.time() - start_time
                    dots = int(elapsed) % 3 + 1
                    yield f"[>] Step {step.step_number}{'.' * dots} ({elapsed:.0f}s)", plan_md, step_output
                    last_yield_time = time.time()
            
            # Attendre la fin du thread
            exec_thread.join(timeout=5.0)
            
            if step_error[0]:
                logger.error(f"Step {step.step_number} failed: {step_error[0]}")
                step_results.append({
                    "step": step.step_number,
                    "agent": step.agent_type,
                    "error": str(step_error[0]),
                    "status": "failed",
                })
                yield f"[ERR] Step {step.step_number} failed: {step_error[0]}", plan_md, full_response
                break
            else:
                step_results.append({
                    "step": step.step_number,
                    "agent": step.agent_type,
                    "output": step_output,
                    "status": "completed",
                })
                full_response = step_output  # La dernière étape est la réponse finale
        
        # ===== PHASE 3: FINALISATION =====
        total_time = time.time() - start_time
        completed_steps = sum(1 for r in step_results if r.get("status") == "completed")
        
        # Enregistrer dans l'historique
        _save_dynamic_history(
            question=question,
            document=document,
            response=full_response,
            plan=plan,
            step_results=step_results,
            total_time=total_time,
            mode="multi"
        )
        
        if completed_steps == plan.step_count:
            yield f"[OK] Completed in {total_time:.1f}s ({plan.step_count} steps)", plan_md, full_response
        else:
            yield f"[!] Partial completion: {completed_steps}/{plan.step_count} steps ({total_time:.1f}s)", plan_md, full_response
        
    except Exception as e:
        logger.exception("Dynamic pipeline error")
        yield f"[ERR] Dynamic pipeline error: {e}", "", f"Error: {str(e)}"


def _save_dynamic_history(
    question: str,
    document: str,
    response: str,
    plan: 'DynamicPipelinePlan',
    step_results: List[Dict],
    total_time: float,
    mode: str = "dynamic"
) -> None:
    """
    Enregistre une exécution Dynamic Pipeline dans l'historique.
    
    Args:
        question: Question originale
        document: Document/code fourni
        response: Réponse finale
        plan: Plan généré
        step_results: Résultats des étapes
        total_time: Temps total d'exécution
        mode: "single" ou "multi"
    """
    if not HISTORY_AVAILABLE or history is None:
        logger.warning("History not available, skipping save")
        return
    
    try:
        # Déterminer le modèle principal utilisé
        if step_results:
            last_step = step_results[-1]
            model_used = last_step.get("model", "dynamic-pipeline")
        else:
            model_used = plan.planner_model if plan else "dynamic-pipeline"
        
        # Construire les métadonnées
        metadata = {
            "dynamic_pipeline": True,
            "mode": mode,
            "step_count": len(step_results),
            "complexity": plan.complexity.value if plan and hasattr(plan.complexity, 'value') else "unknown",
            "planner_model": plan.planner_model if plan else None,
            "analysis": plan.analysis if plan else None,
        }
        
        # Ajouter le résumé des étapes
        if step_results:
            metadata["steps"] = [
                {
                    "step": sr.get("step", i+1),
                    "agent": sr.get("agent", "unknown"),
                    "model": sr.get("model"),
                    "status": sr.get("status", "unknown"),
                }
                for i, sr in enumerate(step_results)
            ]
        
        # Enregistrer
        history.add(
            question=question,
            refined_question=f"[Dynamic Pipeline: {mode}] {question}",
            response=response or "No response",
            model=model_used,
            task_type=f"dynamic_pipeline_{mode}",
            duration_seconds=total_time,
            document=document if document else None,
            metadata=metadata,
        )
        logger.info(f"Dynamic pipeline history saved: {mode}, {len(step_results)} steps")
        
    except Exception as e:
        logger.error(f"Failed to save dynamic pipeline history: {e}")


def _build_step_prompt(
    step: 'PipelineStep',
    original_question: str,
    previous_results: list,
    document: str,
) -> str:
    """
    Construit le prompt pour une étape du pipeline.
    
    Args:
        step: Étape courante
        original_question: Question originale
        previous_results: Résultats des étapes précédentes
        document: Document original
        
    Returns:
        Prompt formaté
    """
    parts = [f"**Task:** {step.task_description}\n"]
    
    # Question originale
    parts.append(f"**Original Request:**\n{original_question}\n")
    
    # Document si présent
    if document:
        doc_preview = document[:50000] + "..." if len(document) > 50000 else document
        parts.append(f"\n**Document/Code:**\n```\n{doc_preview}\n```\n")
    
    # Résultats précédents (les 2 derniers max)
    if previous_results:
        parts.append("\n**Previous Steps Output:**\n")
        for result in previous_results[-3:]:
            if result.get("output"):
                output = result["output"]
                if len(output) > 10000:
                    output = output[:10000] + "\n... (truncated)"
                parts.append(f"\n[Step {result['step']} - {result['agent'].upper()}]:\n{output}\n")
    
    return "\n".join(parts)


# =============================================================================
# UI COMPONENTS (GRADIO)
# =============================================================================

def create_dynamic_pipeline_section():
    """
    Crée la section UI pour le Dynamic Pipeline.
    
    À utiliser dans l'interface Gradio.
    
    Returns:
        Tuple de composants Gradio
    """
    import gradio as gr
    
    with gr.Group():
        gr.Markdown("### Dynamic Pipeline")
        
        use_dynamic = gr.Checkbox(
            label="Enable Dynamic Pipeline (auto-planning)",
            value=False,
            info="LLM analyzes your request and creates optimal multi-step plan"
        )
        
        auto_execute = gr.Checkbox(
            label="Auto-execute plan",
            value=True,
            visible=True,
        )
        
        fast_mode = gr.Checkbox(
            label="Fast mode (less accurate planning)",
            value=False,
            visible=True,
        )
        
        plan_preview = gr.Markdown(
            value="",
            visible=True,
        )
    
    return use_dynamic, auto_execute, fast_mode, plan_preview


def create_dynamic_options_compact():
    """
    Crée une version compacte des options Dynamic Pipeline.
    
    Returns:
        Tuple de composants Gradio
    """
    import gradio as gr
    
    use_dynamic = gr.Checkbox(
        label="Dynamic Pipeline (auto-planning)",
        value=False,
        info="LLM analyzes your request and creates optimal multi-step plan"
    )
    
    auto_execute = gr.Checkbox(
        label="Auto-execute plan",
        value=True,
        visible=False,
    )
    
    return use_dynamic, auto_execute


# =============================================================================
# STATUS AND UTILITY FUNCTIONS
# =============================================================================

def get_dynamic_pipeline_status() -> Dict[str, Any]:
    """
    Récupère le statut du système Dynamic Pipeline.
    
    Returns:
        Dict avec les informations de statut
    """
    status = {
        "available": DYNAMIC_PIPELINE_AVAILABLE,
        "orchestrator_available": ORCHESTRATOR_AVAILABLE,
    }
    
    if DYNAMIC_PIPELINE_AVAILABLE:
        try:
            planner = get_planner()
            status["planner_model"] = planner.planning_model
            status["fast_mode"] = planner.fast_mode
            status["agent_models"] = planner.agent_models
        except Exception as e:
            status["error"] = str(e)
    
    return status


def format_status_for_ui() -> str:
    """
    Formate le statut pour affichage dans l'UI.
    
    Returns:
        String formatée
    """
    status = get_dynamic_pipeline_status()
    
    if not status["available"]:
        return "[x] Dynamic Pipeline: Not available"
    
    lines = ["[OK] Dynamic Pipeline: Ready"]
    
    if "planner_model" in status:
        lines.append(f"   Planner: {status['planner_model']}")
    
    if "error" in status:
        lines.append(f"   ⚠️ Warning: {status['error']}")
    
    return "\n".join(lines)


# =============================================================================
# CLI FOR TESTING
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("DYNAMIC PIPELINE UI - Test")
    print("=" * 60)
    
    # Test status
    print("\nStatus:")
    print(format_status_for_ui())
    
    # Test decision function
    test_questions = [
        "What is the Shannon index?",
        "Write a Python function to calculate mean with error handling and tests",
        "What does this python script do?",
        "Fix the bug in this code",
    ]
    
    print("\nDecision Tests:")
    for q in test_questions:
        should_use = should_use_dynamic_pipeline(q, True)
        complexity = get_complexity_from_question(q)
        print(f"  [{complexity:7}] {should_use!s:5} | {q[:50]}...")
    
    # Interactive test if argument provided
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
        print(f"\nProcessing: {question[:60]}...")
        print("-" * 60)
        
        for status, plan_md, response in process_with_dynamic_pipeline(question):
            print(f"\nStatus: {status}")
            if plan_md:
                print(f"\nPlan:\n{plan_md[:500]}...")
            if response:
                print(f"\nResponse:\n{response[:500]}...")
    
    print("\n" + "=" * 60)
    print("Test completed")
