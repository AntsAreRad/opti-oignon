#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI - OPTI-OIGNON 1.0
========================================

Unified Gradio interface with all systems integrated.

Features:
- Dark mode
- Automatic task detection
- Intelligent model selection
- One-click presets
- Browsable history
- Markdown export
- Keyboard shortcuts
- Detailed progress indicator
- RAG: Contextual enrichment from personal documents
- MULTI-AGENT: Multi-model orchestration for complex tasks

MULTILINGUAL RESPONSE:
- Interface is in English
- BUT responses match user's language
- If user asks in French → response in French
- If user asks in English → response in English

Author: Léon
Version: 1.0 (with Multi-Agent)
"""

import gradio as gr
from pathlib import Path
from typing import Generator, Optional, Tuple, List, Dict, Any
import time
import logging
import threading

from .config import config, DATA_DIR
from .analyzer import analyzer, AnalysisResult
from .router import router, RoutingResult
from .executor import executor
from .presets import preset_manager, Preset, suggest_keywords
from .history import history

# Pipeline Manager Import (for custom pipeline management)
try:
    from .pipeline_manager import (
        get_pipeline_manager,
        Pipeline,
        PipelineStep,
    )
    PIPELINE_MANAGER_AVAILABLE = True
except ImportError:
    PIPELINE_MANAGER_AVAILABLE = False
    get_pipeline_manager = None

# Context Manager Import (for context limit handling)
try:
    from .context_manager import (
        check_context,
        estimate_tokens,
        get_model_limits,
        format_context_indicator,
        get_quick_context_status,
        ContextCheck,
    )
    CONTEXT_MANAGER_AVAILABLE = True
except ImportError:
    CONTEXT_MANAGER_AVAILABLE = False
    check_context = None
    estimate_tokens = None

# RAG Import (optional - graceful fallback if not installed)
try:
    from .rag import ContexteurRAGIntegration, check_ollama_status
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False
    ContexteurRAGIntegration = None
    check_ollama_status = None

# Multi-Agent Import (optional - graceful fallback)
try:
    from .agents import (
        is_multi_agent_enabled,
        set_multi_agent_enabled,
        run_auto,
        run_pipeline,
        get_orchestrator,
        status as multi_agent_status,
        Orchestrator,
        PipelineResult,
        PipelineStatus,
        StepResult,  # Added for callback signature
    )
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    MULTI_AGENT_AVAILABLE = False
    is_multi_agent_enabled = lambda: False
    set_multi_agent_enabled = lambda x: False

# Dynamic Pipeline UI (UPDATED)
try:
    from .dynamic_pipeline_ui import (
        create_dynamic_pipeline_section,
        create_dynamic_options_compact,
        process_with_dynamic_pipeline,
        should_use_dynamic_pipeline,  # Now accepts document parameter
        format_dynamic_plan_markdown,
        DYNAMIC_PIPELINE_CSS,
    )
    DYNAMIC_UI_AVAILABLE = True
except ImportError:
    DYNAMIC_UI_AVAILABLE = False

logger = logging.getLogger(__name__)

# =============================================================================
# CUSTOM THEME (DARK MODE)
# =============================================================================

CUSTOM_CSS = """
/* Enhanced dark mode */
.dark {
    --background-fill-primary: #1a1a2e;
    --background-fill-secondary: #16213e;
    --border-color-primary: #0f3460;
    --text-color-primary: #e8e8e8;
}

/* Colored status */
.status-success { color: #4ade80; }
.status-error { color: #f87171; }
.status-working { color: #fbbf24; }

/* Presets */
.preset-btn {
    min-width: 120px;
    margin: 2px;
}

/* Formatted response */
.response-area {
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
}

/* Code blocks */
.response-area pre {
    background: #1e1e2e;
    padding: 12px;
    border-radius: 8px;
    overflow-x: auto;
}

/* Hide Gradio footer */
footer { display: none !important; }

/* Improve label contrast */
.label-wrap { font-weight: 500; }

/* Loading animation */
.generating {
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}

/* RAG Sources */
.rag-sources {
    background: #1e2030;
    border-left: 3px solid #4ade80;
    padding: 8px 12px;
    margin-top: 8px;
    border-radius: 4px;
    font-size: 0.9em;
}

.rag-badge {
    background: #4ade80;
    color: #1a1a2e;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}

/* Multi-Agent Steps */
.agent-steps {
    background: #1e2030;
    border-left: 3px solid #60a5fa;
    padding: 12px;
    margin-top: 8px;
    border-radius: 4px;
    font-size: 0.9em;
}

.agent-step {
    margin: 8px 0;
    padding: 8px;
    background: #252840;
    border-radius: 4px;
}

.agent-step-header {
    font-weight: bold;
    color: #60a5fa;
}

.agent-step-content {
    margin-top: 4px;
    padding-left: 16px;
    color: #a0a0a0;
}

/* Multi-Agent Badge */
.multi-agent-badge {
    background: #60a5fa;
    color: #1a1a2e;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: bold;
}

/* Pipeline selector */
.pipeline-card {
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 12px;
    margin: 4px;
    cursor: pointer;
    transition: all 0.2s;
}

.pipeline-card:hover {
    border-color: #60a5fa;
    background: #252840;
}

.pipeline-card.selected {
    border-color: #4ade80;
    background: #1e3a2e;
}

/* Context Indicator */
.context-indicator {
    background: #1e2030;
    border: 1px solid #3b4261;
    border-radius: 8px;
    padding: 12px;
    margin: 8px 0;
    font-size: 0.9em;
}

.context-indicator.warning {
    border-color: #fbbf24;
    background: #2a2520;
}

.context-indicator.danger {
    border-color: #f87171;
    background: #2a2020;
}

.context-bar {
    height: 8px;
    background: #3b4261;
    border-radius: 4px;
    overflow: hidden;
    margin: 8px 0;
}

.context-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.3s ease;
}

.context-bar-fill.safe {
    background: linear-gradient(90deg, #4ade80, #22c55e);
}

.context-bar-fill.warning {
    background: linear-gradient(90deg, #fbbf24, #f59e0b);
}

.context-bar-fill.danger {
    background: linear-gradient(90deg, #f87171, #ef4444);
}
"""

# Ajouter le CSS du Dynamic Pipeline si disponible
if DYNAMIC_UI_AVAILABLE:
    CUSTOM_CSS = CUSTOM_CSS + DYNAMIC_PIPELINE_CSS


# =============================================================================
# RAG INITIALIZATION
# =============================================================================

_rag_instance: Optional['ContexteurRAGIntegration'] = None

def get_rag() -> Optional['ContexteurRAGIntegration']:
    """Get RAG instance (singleton)."""
    global _rag_instance
    if not RAG_AVAILABLE:
        return None
    if _rag_instance is None:
        try:
            _rag_instance = ContexteurRAGIntegration()
        except Exception as e:
            logger.error(f"RAG initialization error: {e}")
            return None
    return _rag_instance


def get_rag_status() -> str:
    """Return RAG status for UI."""
    if not RAG_AVAILABLE:
        return "[ERR] RAG not installed (pip install -e . in rag_project/)"
    
    rag = get_rag()
    if rag is None:
        return "[ERR] RAG initialization error"
    
    try:
        stats = rag.get_stats()
        if stats.get("total_chunks", 0) == 0:
            return "[!] Empty index - use `rag index <folder>` to index"
        return f"[OK] RAG active: {stats.get('total_chunks', 0)} chunks, {stats.get('total_files', 0)} files"
    except Exception as e:
        return f"[!] RAG: {str(e)}"


# =============================================================================
# MULTI-AGENT INITIALIZATION
# =============================================================================

def get_multi_agent_status() -> str:
    """Return Multi-Agent status for UI."""
    if not MULTI_AGENT_AVAILABLE:
        return "[ERR] Multi-Agent not installed"
    
    try:
        status = multi_agent_status()
        enabled = status.get("multi_agent_enabled", False)
        agents = len(status.get("available_agents", []))
        pipelines = len(status.get("available_pipelines", []))
        
        if enabled:
            return f"[OK] Multi-Agent active: {agents} agents, {pipelines} pipelines"
        else:
            return f"[OFF] Multi-Agent disabled ({agents} agents available)"
    except Exception as e:
        return f"[!] Multi-Agent: {str(e)}"


def get_pipeline_choices() -> List[Tuple[str, str]]:
    """Return pipeline choices for dropdown, including Dynamic Pipeline."""
    choices = [
        ("Disabled (single model)", "disabled"),
        ("Auto-detection", "auto"),
    ]
    
    # Add Dynamic Pipeline option if available
    if DYNAMIC_UI_AVAILABLE:
        choices.append(("Dynamic Pipeline (LLM planning)", "dynamic"))
    
    if not MULTI_AGENT_AVAILABLE:
        return choices
    
    # Add regular pipelines
    try:
        orch = get_orchestrator()
        for p in orch.list_pipelines():
            emoji = p.get("emoji", "[>]")
            name = p.get("name", p.get("id"))
            choices.append((f"{emoji} {name}", p["id"]))
    except Exception:
        pass
    
    return choices


def get_pipeline_dropdown_update():
    """Return a Gradio update for the pipeline dropdown with fresh choices."""
    return gr.update(choices=get_pipeline_choices())


def get_preset_choices() -> List[Tuple[str, str]]:
    """Return preset choices for dropdown."""
    try:
        presets = preset_manager.get_ordered()
        choices = []
        for p in presets:
            choices.append((f"{p.icon} {p.name}", p.id))
        if not choices:
            choices = [("⚙️ Default", "default")]
        return choices
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        return [("⚙️ Default", "default")]


def get_preset_dropdown_update():
    """Return a Gradio update for the preset dropdown with fresh choices."""
    return gr.update(choices=get_preset_choices())


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_read_file(filepath: str) -> Tuple[str, Optional[str]]:
    """Read a file safely."""
    encodings = ["utf-8", "latin-1", "cp1252"]
    
    for enc in encodings:
        try:
            content = Path(filepath).read_text(encoding=enc, errors='replace')
            return content, None
        except Exception as e:
            continue
    
    return "", "Unable to read file"


def format_analysis(analysis: AnalysisResult) -> str:
    """Format analysis result for display."""
    lines = [
        f"[>] **Detected task:** {analysis.task_type.value}",
        f"Confidence: {analysis.confidence:.0%}",
        f"**Language:** {analysis.language.value}",
        f"Complexity: {analysis.complexity.value}",
    ]
    
    if analysis.keywords:
        lines.append(f"**Keywords:** {', '.join(analysis.keywords)}")
    
    return "\n".join(lines)


def format_routing(routing: RoutingResult) -> str:
    """Format routing result for display."""
    lines = [
        f"Model: {routing.model}",
        f"🌡️ **Temperature:** {routing.temperature}",
        f"Prompt variant: {routing.prompt_variant}",
        f"⏱️ **Timeout:** {routing.timeout}s",
    ]
    return "\n".join(lines)


def format_rag_sources(sources: List[dict]) -> str:
    """Format RAG sources for display."""
    if not sources:
        return ""
    
    lines = ["### RAG Sources Used\n"]
    for i, src in enumerate(sources[:5], 1):
        score = src.get('score', 0)
        location = src.get('location', src.get('source_file', 'unknown'))
        lines.append(f"{i}. **{location}** (score: {score:.2f})")
    
    return "\n".join(lines)


def format_agent_steps(steps: List[Dict], elapsed_time: float = 0) -> str:
    """Format multi-agent steps for display with better visibility."""
    if not steps:
        return "_Waiting for steps..._"
    
    lines = ["### Pipeline Progress\n"]
    lines.append(f"**Time elapsed:** {elapsed_time:.0f}s\n")
    lines.append("---\n")
    
    for i, step in enumerate(steps):
        status = step.get("status", "pending")
        status_display = {
            "completed": "OK",
            "failed": "ERR", 
            "skipped": "SKIP",
            "running": ">>",
            "pending": "..",
        }.get(status, "?")
        
        name = step.get("step_name", step.get("name", f"Step {i+1}"))
        
        # Format line based on status
        if status == "running":
            lines.append(f"**[{status_display}] {name}** _(in progress...)_")
        elif status == "completed":
            lines.append(f"[{status_display}] ~~{name}~~")
            if step.get("output"):
                output = step["output"]
                if hasattr(output, 'content'):
                    output = output.content
                if isinstance(output, str) and len(output) > 150:
                    output = output[:150] + "..."
                # Show preview of output
                lines.append(f"> _{output}_\n")
        else:
            lines.append(f"[{status_display}] {name}")
    
    return "\n".join(lines)


# =============================================================================
# CONTEXT MANAGEMENT FUNCTIONS
# =============================================================================

def get_context_indicator_html(model: str, question: str = "", document: str = "", system_prompt: str = "") -> str:
    """
    Generate context indicator display for UI.
    
    Args:
        model: Model name
        question: User's question
        document: Document content
        system_prompt: System prompt
        
    Returns:
        Markdown formatted context indicator
    """
    if not CONTEXT_MANAGER_AVAILABLE or not model:
        return ""
    
    try:
        check = check_context(
            prompt=question,
            document=document,
            system_prompt=system_prompt,
            model=model
        )
        
        # Create progress bar
        bar_length = 20
        filled = int(bar_length * min(check.usage_percent, 100) / 100)
        bar = "█" * filled + "░" * (bar_length - filled)
        
        # Build display
        lines = [
            f"**{check.status_emoji} Context Usage** • `{model}`",
            "",
            f"Input: ~{check.total_tokens:,} / {check.available_for_input:,} tokens",
            f"`[{bar}]` {check.usage_percent:.0f}%",
        ]
        
        if check.document_tokens > 100:
            lines.extend([
                "",
                f"*Prompt: ~{check.prompt_tokens:,} • "
                f"Doc: ~{check.document_tokens:,} • "
                f"System: ~{check.system_tokens:,}*"
            ])
        
        if check.warning_message:
            lines.extend(["", f"⚠️ {check.warning_message}"])
        
        return "\n".join(lines)
        
    except Exception as e:
        logger.debug(f"Context indicator error: {e}")
        return ""


def format_context_check(check) -> str:
    """
    Format a ContextCheck for display.
    
    Args:
        check: ContextCheck object
        
    Returns:
        Formatted string
    """
    if check is None:
        return ""
    
    lines = [
        f"{check.status_emoji} **Context:** {check.usage_percent:.0f}% used",
        f"• Total: ~{check.total_tokens:,} tokens",
        f"• Limit: {check.available_for_input:,} tokens",
    ]
    
    if check.warning_message:
        lines.append(f"• ⚠️ {check.warning_message}")
    
    return "\n".join(lines)


def update_context_indicator(model: str, question: str, document: str) -> str:
    """
    Update context indicator when inputs change.
    
    Args:
        model: Selected model
        question: Current question
        document: Current document
        
    Returns:
        Updated indicator markdown
    """
    if not model:
        return "*Select a model to see context usage*"
    
    return get_context_indicator_html(model, question, document, "")


def get_effective_model(
    question: str,
    force_model: str,
    use_presets: bool,
    selected_preset: str,
    priority: str = "balanced"
) -> Tuple[str, str]:
    """
    Determine the effective model based on settings.
    
    Priority:
    1. force_model if set
    2. selected_preset.model if use_presets and preset selected
    3. Auto-detected preset via keywords
    4. Router auto-selection based on question analysis
    
    Args:
        question: User's question
        force_model: Manually forced model
        use_presets: Whether presets mode is enabled
        selected_preset: Selected preset ID
        priority: Routing priority
        
    Returns:
        Tuple of (model_name, source) where source is one of:
        "forced", "preset", "auto_preset", "auto_router"
    """
    # 1. Force model takes priority
    if force_model:
        return force_model, "forced"
    
    # 2. Manual preset selection
    if use_presets and selected_preset:
        preset = preset_manager.get(selected_preset)
        if preset and preset.model:
            return preset.model, "preset"
    
    # 3. Auto-detect preset via keywords
    if question.strip():
        detected_preset = preset_manager.find_by_keywords(question)
        if detected_preset and detected_preset.model:
            return detected_preset.model, "auto_preset"
    
    # 4. Auto-routing based on question analysis
    if question.strip():
        try:
            analysis = analyzer.analyze(question)
            routing = router.route(analysis, priority=priority)
            return routing.model, "auto_router"
        except Exception as e:
            logger.debug(f"Auto-routing failed: {e}")
    
    # Fallback: no model determined
    return "", "none"


def update_context_full(
    question: str,
    document: str,
    file_upload,
    force_model: str,
    use_presets: bool,
    selected_preset: str,
    priority: str = "balanced"
) -> str:
    """
    Update context indicator with full context awareness.
    
    Takes into account:
    - Question text
    - Document text area
    - Uploaded file content
    - Forced model OR preset model OR auto-detected model
    
    Args:
        question: User's question
        document: Document text area content
        file_upload: Uploaded file object
        force_model: Manually selected model
        use_presets: Whether preset mode is enabled
        selected_preset: Selected preset ID
        priority: Routing priority
        
    Returns:
        Formatted context indicator markdown
    """
    # Determine effective model
    model, source = get_effective_model(
        question, force_model, use_presets, selected_preset, priority
    )
    
    if not model:
        return "*Enter a question to auto-detect model and see context usage*"
    
    # Combine document with file content
    combined_document = document or ""
    
    if file_upload is not None:
        try:
            content, error = safe_read_file(file_upload.name)
            if not error and content:
                if combined_document:
                    combined_document = f"{combined_document}\n\n--- Uploaded File ---\n{content}"
                else:
                    combined_document = content
        except Exception as e:
            logger.debug(f"Error reading file for context indicator: {e}")
    
    # Generate context indicator
    indicator = get_context_indicator_html(model, question, combined_document, "")
    
    # Add model source info
    source_labels = {
        "forced": "Manual selection",
        "preset": f"Preset: {selected_preset}",
        "auto_preset": "Auto-detected preset",
        "auto_router": "Auto-routed",
    }
    source_label = source_labels.get(source, "")
    
    if indicator and source_label:
        indicator = f"*Model: `{model}` ({source_label})*\n\n{indicator}"
    elif not indicator and model:
        indicator = f"*Model: `{model}` ({source_label})*"
    
    return indicator


# =============================================================================
# MULTI-AGENT PROCESSING FUNCTION
# =============================================================================

def process_with_multi_agent(
    question: str,
    document: str,
    pipeline_id: str,
    show_steps: bool,
) -> Generator[Tuple[str, str, str], None, None]:
    """
    Process a request with the multi-agent system.
    
    Uses a separate thread to run the pipeline while yielding updates
    to keep the Gradio connection alive.
    
    Yields:
        (status, steps_display, response)
    """
    import queue
    
    if not MULTI_AGENT_AVAILABLE:
        yield "[ERR] Multi-Agent not available", "", "The multi-agent system is not installed."
        return
    
    if not is_multi_agent_enabled():
        yield "[!] Multi-Agent disabled", "", "Enable multi-agent to use this feature."
        return
    
    start_time = time.time()
    steps_md = ""
    current_steps = []
    
    # Queue for thread communication
    update_queue = queue.Queue()
    result_container = {"result": None, "error": None}
    
    try:
        orch = get_orchestrator()
        
        # Callbacks for tracking - now push to queue
        def on_step_start(step_name: str, step_index: int):
            update_queue.put(("step_start", step_name, step_index))
        
        def on_step_complete(step_result: StepResult):
            output_content = step_result.output.content if step_result.output else ""
            update_queue.put(("step_complete", step_result.step_index, output_content))
        
        orch.set_callbacks(
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
        )
        
        # Determine pipeline
        if pipeline_id == "auto":
            yield "[...] Auto-detecting pipeline...", "", ""
            detected = orch.detect_pipeline(question)
            if detected:
                pipeline_id = detected
                yield f"[>] Detected pipeline: {pipeline_id}", "", ""
            else:
                pipeline_id = "quick"
                yield "[FAST] Quick mode (no complex pipeline detected)", "", ""
        
        # Build input
        full_input = question
        if document:
            full_input = f"{question}\n\n--- Document ---\n{document}"
        
        yield f"[RUN] Running pipeline: {pipeline_id}", "### Pipeline Starting...\n_Initializing agents..._", ""
        
        # Run pipeline in a separate thread
        def run_pipeline_thread():
            try:
                result_container["result"] = orch.run_pipeline(pipeline_id, full_input)
            except Exception as e:
                result_container["error"] = str(e)
            finally:
                update_queue.put(("done", None, None))
        
        pipeline_thread = threading.Thread(target=run_pipeline_thread, daemon=True)
        pipeline_thread.start()
        
        # Process updates from the queue while pipeline runs
        last_yield_time = time.time()
        while True:
            try:
                # Wait for update with timeout (keeps connection alive)
                update = update_queue.get(timeout=2.0)
                
                event_type, arg1, arg2 = update
                
                if event_type == "done":
                    break
                elif event_type == "step_start":
                    step_name, step_index = arg1, arg2
                    current_steps.append({
                        "step_name": step_name,
                        "step_index": step_index,
                        "status": "running",
                    })
                    elapsed = time.time() - start_time
                    if show_steps:
                        steps_md = format_agent_steps(current_steps, elapsed)
                    yield f"[RUN] Step {step_index + 1}: {step_name}... ({elapsed:.0f}s)", steps_md, ""
                    
                elif event_type == "step_complete":
                    step_index, output_content = arg1, arg2
                    for step in current_steps:
                        if step.get("step_index") == step_index:
                            step["status"] = "completed"
                            step["output"] = output_content
                            break
                    elapsed = time.time() - start_time
                    if show_steps:
                        steps_md = format_agent_steps(current_steps, elapsed)
                    yield f"[RUN] Step {step_index + 1} completed ({elapsed:.0f}s)", steps_md, ""
                    
            except queue.Empty:
                # No update, but yield to keep connection alive
                elapsed = time.time() - start_time
                current_step_name = current_steps[-1]["step_name"] if current_steps else "Initializing"
                if show_steps and current_steps:
                    steps_md = format_agent_steps(current_steps, elapsed)
                yield f"[RUN] {current_step_name}... ({elapsed:.0f}s)", steps_md, ""
        
        # Wait for thread to finish
        pipeline_thread.join(timeout=5.0)
        
        # Check for errors
        if result_container["error"]:
            yield f"[ERR] Error: {result_container['error']}", steps_md, f"An error occurred: {result_container['error']}"
            return
        
        result = result_container["result"]
        if result is None:
            yield "[ERR] No result from pipeline", steps_md, "Pipeline did not return a result"
            return
        
        elapsed = time.time() - start_time
        
        # Format final steps
        if show_steps and result.steps:
            final_steps = []
            for sr in result.steps:
                final_steps.append({
                    "step_name": sr.step_name,
                    "step_index": sr.step_index,
                    "status": sr.status.value if hasattr(sr.status, 'value') else str(sr.status),
                    "output": sr.output.content if sr.output else "",
                })
            steps_md = format_agent_steps(final_steps, elapsed)
        
        # Final status
        if result.status == PipelineStatus.COMPLETED:
            status = f"[OK] Completed in {elapsed:.1f}s ({len(result.steps)} steps)"
        else:
            status = f"[!] Pipeline finished with status: {result.status.value}"
        
        # Save to history (FIX: multi-agent history was not being saved!)
        try:
            # Determine model used (use the last step's model or first available)
            model_used = "multi-agent"
            if result.steps:
                # Try to get model from the last step's metadata
                for step in reversed(result.steps):
                    if hasattr(step, 'model') and step.model:
                        model_used = step.model
                        break
                    elif hasattr(step, 'agent') and step.agent:
                        model_used = f"multi-agent:{step.agent}"
                        break
            
            # Build metadata with pipeline info
            pipeline_metadata = {
                "pipeline_id": pipeline_id,
                "steps_count": len(result.steps),
                "multi_agent": True,
            }
            
            # Add step summaries to metadata
            if result.steps:
                pipeline_metadata["steps"] = [
                    {
                        "name": sr.step_name,
                        "status": sr.status.value if hasattr(sr.status, 'value') else str(sr.status),
                    }
                    for sr in result.steps
                ]
            
            history.add(
                question=question,
                refined_question=f"[Multi-Agent Pipeline: {pipeline_id}] {question}",
                response=result.final_output or "No result",
                model=model_used,
                task_type=f"multi_agent_{pipeline_id}",
                duration_seconds=elapsed,
                document=document if document else None,
                metadata=pipeline_metadata,
            )
            logger.debug(f"Multi-agent history saved: {pipeline_id}")
        except Exception as e:
            logger.error(f"Failed to save multi-agent history: {e}")
        
        yield status, steps_md, result.final_output or "No result"
        
    except Exception as e:
        logger.exception("Multi-agent error")
        yield f"[ERR] Error: {str(e)}", steps_md, f"An error occurred: {str(e)}"


# =============================================================================
# MAIN INTERFACE FUNCTIONS
# =============================================================================

def analyze_question(question: str, document: str = "") -> Tuple[str, str]:
    """Analyze a question and return detection info."""
    if not question.strip():
        return "Enter a question...", ""
    
    analysis = analyzer.analyze(question, document if document else None)
    routing = router.route(analysis)
    
    return format_analysis(analysis), format_routing(routing)


def load_preset(preset_id: str) -> Tuple[str, str, float, str]:
    """Load a preset and return its values."""
    preset = preset_manager.get(preset_id)
    if not preset:
        return "", "", 0.5, "standard"
    
    return preset.task, preset.model, preset.temperature, preset.prompt_variant


def process_with_streaming(
    question: str,
    document: str,
    file_upload,
    use_presets: bool,
    selected_preset: str,
    force_task: str,
    force_model: str,
    force_temp: float,
    force_variant: str,
    do_refine: bool,
    priority: str,
    use_rag: bool,
    rag_n_chunks: int,
    use_multi_agent: bool,
    multi_agent_pipeline: str,
    show_agent_steps: bool,
    use_dynamic: bool = False,
    auto_execute_dynamic: bool = True,
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    """
    Process a question with all options including dynamic pipeline support.
    
    Pipeline selection logic:
    - "disabled": Single model mode (standard)
    - "dynamic": Dynamic Pipeline (LLM-planned multi-step)
    - "auto": Auto-detect best pipeline
    - Other: Specific multi-agent pipeline
    
    Yields:
        (status, analysis, routing, agent_steps/rag_sources, response)
    """
    if not question.strip():
        yield "Enter a question...", "", "", "", ""
        return

    # =========================================================================
    # FIX: Handle file upload FIRST (before any pipeline decisions)
    # =========================================================================
    if file_upload is not None:
        try:
            content, error = safe_read_file(file_upload.name)
            if error:
                yield f"[ERR] File error: {error}", "", "", "", ""
                return
            document = content
            logger.info(f"File loaded: {len(document)} chars")
        except Exception as e:
            yield f"[ERR] File error: {str(e)}", "", "", "", ""
            return

    # =========================================================================
    # Dynamic Pipeline Mode (explicit selection from dropdown)
    # =========================================================================
    if multi_agent_pipeline == "dynamic" and DYNAMIC_UI_AVAILABLE:
        logger.info(f"Dynamic pipeline selected explicitly")
        logger.info(f"Document length: {len(document) if document else 0}")
        
        yield "[>] Dynamic pipeline mode...", "", "", "", ""
        
        for status, plan_md, response in process_with_dynamic_pipeline(
            question=question,
            document=document,
            enable_dynamic=True,
            auto_execute=auto_execute_dynamic,
        ):
            logger.debug(f"Dynamic status: {status}")
            yield status, "", "", plan_md, response
        return
    
    # =========================================================================
    # Legacy Dynamic Pipeline (checkbox - kept for backward compatibility)
    # =========================================================================
    if use_dynamic and DYNAMIC_UI_AVAILABLE and multi_agent_pipeline not in ["disabled", "dynamic"]:
        logger.info(f"Dynamic pipeline check via checkbox: use_dynamic={use_dynamic}")
        
        if should_use_dynamic_pipeline(question, use_dynamic, document):
            yield "[>] Dynamic pipeline mode...", "", "", "", ""
            
            for status, plan_md, response in process_with_dynamic_pipeline(
                question=question,
                document=document,
                enable_dynamic=True,
                auto_execute=auto_execute_dynamic,
            ):
                yield status, "", "", plan_md, response
            return
        else:
            logger.info("Dynamic pipeline: should_use returned False, using standard mode")
    
    # =========================================================================
    # Disabled mode (single model) - skip multi-agent entirely
    # =========================================================================
    if multi_agent_pipeline == "disabled":
        logger.info("Multi-agent disabled, using single model mode")
        # Fall through to standard mode below
        use_multi_agent = False
    
    # =========================================================================
    # Multi-agent mode (auto or specific pipeline)
    # =========================================================================
    if use_multi_agent and MULTI_AGENT_AVAILABLE and is_multi_agent_enabled() and multi_agent_pipeline != "disabled":
        for status, steps_md, response in process_with_multi_agent(
            question, document, multi_agent_pipeline, show_agent_steps
        ):
            yield status, "", "", steps_md, response
        return
    
    # Standard mode
    yield "[>] Analyzing question...", "", "", "", ""
    
    # RAG enrichment
    enriched_context = ""
    rag_sources_md = ""
    if use_rag and RAG_AVAILABLE:
        rag = get_rag()
        if rag:
            try:
                yield "[>] RAG: Searching relevant context...", "", "", "", ""
                enriched = rag.enrich_query(question, n_results=rag_n_chunks)
                if enriched.get("sources"):
                    enriched_context = enriched.get("enriched_prompt", "")
                    rag_sources_md = format_rag_sources(enriched.get("sources", []))
                    yield f"[>] RAG: Found {len(enriched.get('sources', []))} sources", "", "", rag_sources_md, ""
            except Exception as e:
                logger.error(f"RAG error: {e}")
                yield f"[!] RAG error: {str(e)}", "", "", "", ""
    
    # Analysis
    combined_doc = document
    if enriched_context:
        combined_doc = f"{document}\n\n--- RAG Context ---\n{enriched_context}" if document else enriched_context
    
    analysis = analyzer.analyze(question, combined_doc if combined_doc else None)
    analysis_md = format_analysis(analysis)
    
    # Routing
    routing = router.route(
        analysis,
        priority=priority,
        force_model=force_model if force_model else None,
        force_variant=force_variant if force_variant else None,
    )
    routing_md = format_routing(routing)
    
    yield "[>] Generating...", analysis_md, routing_md, rag_sources_md, ""
    
    # Context validation (if available)
    context_warning = ""
    if CONTEXT_MANAGER_AVAILABLE:
        try:
            ctx_check = check_context(
                prompt=question,
                document=combined_doc,
                system_prompt="",  # Will be set by executor
                model=routing.model
            )
            if ctx_check.exceeds_limit:
                context_warning = f"⚠️ Context exceeds limit ({ctx_check.usage_percent:.0f}%). Response may be truncated."
                yield f"[!] {context_warning}", analysis_md, routing_md, rag_sources_md, ""
            elif ctx_check.exceeds_recommended:
                context_warning = f"Context is large ({ctx_check.usage_percent:.0f}%). Quality may vary."
        except Exception as e:
            logger.debug(f"Context check error: {e}")
    
    # Generation
    response_parts = []
    try:
        for chunk in executor.execute(
            question=question,
            routing=routing,
            document=combined_doc,
            refine=do_refine,
        ):
            response_parts.append(chunk)
            yield "[>] Generating...", analysis_md, routing_md, rag_sources_md, "".join(response_parts)
        
        full_response = "".join(response_parts)
        # FIX: Retrieve refined question from executor instance
        refined_q = executor.last_refined_question or question
        history.add(
            question=question,
            refined_question=refined_q,  # Now correctly retrieved!
            response=full_response,
            model=routing.model,
            task_type=routing.task_type,
        )
        
        yield "[OK] Complete", analysis_md, routing_md, rag_sources_md, full_response
        
    except Exception as e:
        logger.exception("Generation error")
        yield f"[ERR] Error: {str(e)}", analysis_md, routing_md, rag_sources_md, f"Error: {str(e)}"


def cancel_generation():
    """Cancel ongoing generation."""
    executor.cancel()
    return "[X] Cancelled"


def clear_inputs():
    """Clear all inputs."""
    return "", "", None, ""


# =============================================================================
# RAG FUNCTIONS
# =============================================================================

def index_rag_folder(folder: str, recursive: bool, force: bool) -> str:
    """Index a folder with RAG."""
    if not RAG_AVAILABLE:
        return "[ERR] RAG not installed"
    
    rag = get_rag()
    if rag is None:
        return "[ERR] RAG not available"
    
    try:
        folder_path = Path(folder).expanduser()
        if not folder_path.exists():
            return f"[ERR] Folder not found: {folder}"
        
        result = rag.index_folder(str(folder_path), recursive=recursive, force=force)
        return f"[OK] Indexed: {result.get('files_processed', 0)} files, {result.get('chunks_created', 0)} chunks"
    except Exception as e:
        return f"[ERR] Indexing error: {str(e)}"


def refresh_rag_stats() -> str:
    """Refresh RAG stats."""
    return get_rag_status()


def preview_rag_search(query: str, n_results: int) -> str:
    """Preview RAG search results."""
    if not RAG_AVAILABLE:
        return "RAG not available"
    
    if not query.strip():
        return "Enter a search query"
    
    rag = get_rag()
    if rag is None:
        return "RAG not initialized"
    
    try:
        results = rag.search(query, n_results=int(n_results))
        if not results:
            return "No results found"
        
        lines = []
        for i, r in enumerate(results, 1):
            score = r.get('score', 0)
            source = r.get('source_file', 'unknown')
            content = r.get('content', '')[:200]
            lines.append(f"### {i}. {source} (score: {score:.2f})\n```\n{content}...\n```\n")
        
        return "\n".join(lines)
    except Exception as e:
        return f"Search error: {str(e)}"


# =============================================================================
# MULTI-AGENT FUNCTIONS
# =============================================================================

def toggle_multi_agent(enabled: bool) -> str:
    """Toggle multi-agent mode."""
    if not MULTI_AGENT_AVAILABLE:
        return "[ERR] Multi-Agent not installed"
    
    try:
        set_multi_agent_enabled(enabled)
        return get_multi_agent_status()
    except Exception as e:
        return f"[ERR] {str(e)}"


def refresh_multi_agent_stats() -> Tuple[str, Any]:
    """Refresh multi-agent stats and pipeline dropdown. Returns (status, dropdown_update)."""
    return get_multi_agent_status(), get_pipeline_dropdown_update()


# =============================================================================
# HISTORY FUNCTIONS
# =============================================================================

def get_history_choices() -> List[Tuple[str, str]]:
    """Get history choices for dropdown."""
    try:
        entries = history.get_recent(20)  # Fixed: was list_recent
        choices = []
        for entry in entries:
            # entry is a HistoryEntry dataclass, not a dict
            label = entry.question[:50] if entry.question else ""
            if len(entry.question) > 50:
                label += "..."
            # Add timestamp for better identification
            timestamp_short = entry.timestamp[:10] if entry.timestamp else ""
            display_label = f"[{timestamp_short}] {label}"
            choices.append((display_label, entry.id))
        return choices
    except Exception as e:
        logger.error(f"Error getting history choices: {e}")
        return []


def load_history_entry(entry_id: str) -> Tuple[str, str, str]:
    """Load a history entry."""
    if not entry_id:
        return "", "", ""
    
    try:
        entry = history.get(entry_id)
        if entry:
            # entry is a HistoryEntry dataclass, access properties directly
            return (
                entry.question or "",
                entry.refined_question or "",
                entry.response or "",
            )
    except Exception as e:
        logger.error(f"Error loading history entry: {e}")
    
    return "", "", ""


def export_history_markdown(entry_id: str) -> str:
    """Export history entry to Markdown."""
    if not entry_id:
        return "Select an entry first"
    
    try:
        entry = history.get(entry_id)
        if not entry:
            return "Entry not found"
        
        export_dir = DATA_DIR / "exports"
        export_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"export_{int(time.time())}.md"
        filepath = export_dir / filename
        
        # entry is a HistoryEntry dataclass, access properties directly
        content = f"""# Opti-Oignon Export

## Question
{entry.question or ""}

## Refined Question
{entry.refined_question or ""}

## Response
{entry.response or ""}

---
Model: {entry.model or ""}
Task: {entry.task_type or ""}
Date: {entry.timestamp or ""}
"""
        filepath.write_text(content, encoding="utf-8")
        return str(filepath)
    except Exception as e:
        logger.error(f"Export error: {e}")
        return f"Export error: {str(e)}"


# =============================================================================
# PRESET MANAGEMENT FUNCTIONS
# =============================================================================

def get_preset_table_data() -> List[List[str]]:
    """Get preset data for display in table (enhanced with weight)."""
    data = []
    try:
        for p in preset_manager.get_ordered():
            keywords_str = ", ".join(p.keywords[:3]) if p.keywords else "-"
            if len(p.keywords) > 3:
                keywords_str += f" (+{len(p.keywords)-3})"
            # Get detection_weight with fallback for old presets
            weight = getattr(p, 'detection_weight', 0.5)
            data.append([
                p.icon,
                p.id,
                p.name,
                p.task,
                p.model.split(":")[0],  # Short model name
                str(p.temperature),
                f"{weight:.1f}",  # Detection weight
                keywords_str,
            ])
    except Exception as e:
        logger.error(f"Error getting preset table data: {e}")
    return data


def refresh_preset_table():
    """Refresh the preset table display."""
    return get_preset_table_data()


def get_preset_details(preset_id: str) -> Tuple[str, str, str, str, float, str, str, str, str, float]:
    """Get all details of a preset for editing (with detection_weight)."""
    if not preset_id:
        return "", "", "", "", 0.5, "standard", "", "", "", 0.5
    
    try:
        preset = preset_manager.get(preset_id)
        if preset:
            weight = getattr(preset, 'detection_weight', 0.5)
            return (
                preset.id,
                preset.name,
                preset.description,
                preset.task,
                preset.temperature,
                preset.prompt_variant,
                preset.model,
                ", ".join(preset.tags),
                ", ".join(preset.keywords),
                weight,  # NEW: detection_weight
            )
    except Exception as e:
        logger.error(f"Error getting preset details: {e}")
    
    return "", "", "", "", 0.5, "standard", "", "", "", 0.5


def create_new_preset(
    preset_id: str,
    name: str,
    description: str,
    task: str,
    temperature: float,
    prompt_variant: str,
    model: str,
    tags_str: str,
    keywords_str: str,
    detection_weight: float = 0.5,
) -> Tuple[str, List[List[str]], Any]:
    """Create a new preset. Returns (status, table_data, dropdown_update)."""
    # Validate
    if not preset_id or not preset_id.strip():
        return "[ERR] Preset ID is required", get_preset_table_data(), get_preset_dropdown_update()
    
    if not preset_manager.validate_preset_id(preset_id):
        return "[ERR] Invalid ID (use letters, numbers, _ or -)", get_preset_table_data(), get_preset_dropdown_update()
    
    if preset_manager.get(preset_id):
        return f"[ERR] Preset '{preset_id}' already exists", get_preset_table_data(), get_preset_dropdown_update()
    
    if not name or not name.strip():
        return "[ERR] Name is required", get_preset_table_data(), get_preset_dropdown_update()
    
    # Parse tags and keywords
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
    
    try:
        preset_manager.create(
            preset_id=preset_id.strip(),
            name=name.strip(),
            description=description.strip() if description else "",
            task=task,
            model=model,
            temperature=temperature,
            prompt_variant=prompt_variant,
            tags=tags,
            keywords=keywords,
            detection_weight=detection_weight,
        )
        return f"[OK] Preset '{preset_id}' created ({len(keywords)} keywords, weight={detection_weight})", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Creation failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


def update_existing_preset(
    preset_id: str,
    name: str,
    description: str,
    task: str,
    temperature: float,
    prompt_variant: str,
    model: str,
    tags_str: str,
    keywords_str: str,
    detection_weight: float = 0.5,
) -> Tuple[str, List[List[str]], Any]:
    """Update an existing preset. Returns (status, table_data, dropdown_update)."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data(), get_preset_dropdown_update()
    
    if not preset_manager.get(preset_id):
        return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data(), get_preset_dropdown_update()
    
    # Parse tags and keywords
    tags = [t.strip() for t in tags_str.split(",") if t.strip()] if tags_str else []
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
    
    try:
        preset_manager.update(
            preset_id,
            name=name.strip() if name else preset_id,
            description=description.strip() if description else "",
            task=task,
            model=model,
            temperature=temperature,
            prompt_variant=prompt_variant,
            tags=tags,
            keywords=keywords,
            detection_weight=detection_weight,
        )
        return f"[OK] Preset '{preset_id}' updated", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Update failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


def delete_preset(preset_id: str) -> Tuple[str, List[List[str]], Any]:
    """Delete a preset. Returns (status, table_data, dropdown_update)."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data(), get_preset_dropdown_update()
    
    if preset_id == "default":
        return "[ERR] Cannot delete the default preset", get_preset_table_data(), get_preset_dropdown_update()
    
    try:
        if preset_manager.delete(preset_id):
            return f"[OK] Preset '{preset_id}' deleted", get_preset_table_data(), get_preset_dropdown_update()
        else:
            return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Delete failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


def duplicate_preset(preset_id: str, new_id: str) -> Tuple[str, List[List[str]], Any]:
    """Duplicate a preset. Returns (status, table_data, dropdown_update)."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data(), get_preset_dropdown_update()
    
    if not new_id or not new_id.strip():
        return "[ERR] New ID is required", get_preset_table_data(), get_preset_dropdown_update()
    
    if not preset_manager.validate_preset_id(new_id):
        return "[ERR] Invalid new ID", get_preset_table_data(), get_preset_dropdown_update()
    
    try:
        source = preset_manager.get(preset_id)
        if not source:
            return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data(), get_preset_dropdown_update()
        
        new_preset = preset_manager.duplicate(preset_id, new_id.strip(), f"Copy of {source.name}")
        if new_preset:
            return f"[OK] Preset duplicated as '{new_id}'", get_preset_table_data(), get_preset_dropdown_update()
        else:
            return "[ERR] Duplication failed", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Duplication failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


def test_keyword_matching(test_text: str) -> str:
    """Test which preset matches the given text based on keywords (with weighted scoring)."""
    if not test_text or not test_text.strip():
        return "Enter some text to test keyword matching"
    
    try:
        # Use the enhanced method that returns scores
        results = preset_manager.find_by_keywords_with_scores(test_text)
        if results:
            lines = ["### Matching Presets\n"]
            for preset, score, matches in results[:5]:  # Top 5
                lines.append(f"- **{preset.icon} {preset.name}** (`{preset.id}`)")
                lines.append(f"  - Score: {score:.2f} ({matches} keywords matched)")
                weight = getattr(preset, 'detection_weight', 0.5)
                lines.append(f"  - Weight: {weight}")
                matched_kw = [kw for kw in preset.keywords if kw.lower() in test_text.lower()][:5]
                if matched_kw:
                    lines.append(f"  - Matched: {', '.join(matched_kw)}")
                lines.append("")
            return "\n".join(lines)
        else:
            return "No preset matched. The system will use task detection instead."
    except Exception as e:
        return f"Error: {str(e)}"


def reload_presets() -> Tuple[str, List[List[str]], Any]:
    """Reload presets from config files. Returns (status, table_data, dropdown_update)."""
    try:
        preset_manager.reload()
        stats = preset_manager.get_stats()
        avg_weight = stats.get('avg_detection_weight', 0.5)
        return f"[OK] Reloaded {stats['total']} presets ({stats['total_keywords']} keywords, avg weight={avg_weight:.2f})", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Reload failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


def suggest_keywords_for_preset(
    task: str,
    name: str,
    description: str,
    existing_keywords: str,
) -> str:
    """Auto-suggest keywords based on task, name, and description."""
    try:
        existing = [k.strip() for k in existing_keywords.split(",") if k.strip()] if existing_keywords else []
        
        suggestions = suggest_keywords(
            task=task,
            name=name,
            description=description,
            existing_keywords=existing,
            max_suggestions=15,
        )
        
        if suggestions:
            # Return as comma-separated for easy copy
            return ", ".join(suggestions)
        else:
            return "No suggestions available for this configuration"
    except Exception as e:
        return f"Error: {str(e)}"


def export_all_presets() -> str:
    """Export all presets to a YAML file."""
    try:
        export_path = DATA_DIR / "presets_export.yaml"
        if preset_manager.export_all_presets(export_path):
            return f"[OK] Exported to: {export_path}"
        else:
            return "[ERR] Export failed"
    except Exception as e:
        return f"[ERR] Export failed: {str(e)}"


def import_presets_from_file(file) -> Tuple[str, List[List[str]], Any]:
    """Import presets from an uploaded YAML file. Returns (status, table_data, dropdown_update)."""
    if file is None:
        return "[ERR] No file selected", get_preset_table_data(), get_preset_dropdown_update()
    
    try:
        filepath = Path(file.name) if hasattr(file, 'name') else Path(file)
        imported = preset_manager.import_preset(filepath)
        if imported:
            return f"[OK] Imported {len(imported)} presets: {', '.join(imported)}", get_preset_table_data(), get_preset_dropdown_update()
        else:
            return "[ERR] No valid presets found in file", get_preset_table_data(), get_preset_dropdown_update()
    except Exception as e:
        return f"[ERR] Import failed: {str(e)}", get_preset_table_data(), get_preset_dropdown_update()


# =============================================================================
# PIPELINE MANAGER HELPERS
# =============================================================================

def get_pipeline_table_data() -> List[List[str]]:
    """Get pipeline data for display in table."""
    data = []
    if not PIPELINE_MANAGER_AVAILABLE:
        return data
    
    try:
        pm = get_pipeline_manager()
        for p in pm.list_all():
            status = "📌" if p.is_builtin else "🧅"
            keywords_str = ", ".join(p.keywords[:3]) if p.keywords else "-"
            if p.keywords and len(p.keywords) > 3:
                keywords_str += f" (+{len(p.keywords)-3})"
            
            data.append([
                status,
                p.id,
                p.name,
                str(p.step_count),
                p.pattern or "chain",
                f"{p.detection_weight:.1f}",
                keywords_str,
            ])
    except Exception as e:
        logger.error(f"Error getting pipeline table data: {e}")
    return data


def refresh_pipeline_table():
    """Refresh the pipeline table display."""
    return get_pipeline_table_data()


def get_pipeline_details(pipeline_id: str) -> Tuple:
    """Get all details of a pipeline for editing."""
    default = ("", "", "", "chain", "🔧", "", "", 0.5, "")
    
    if not PIPELINE_MANAGER_AVAILABLE or not pipeline_id:
        return default
    
    try:
        pm = get_pipeline_manager()
        pipeline = pm.get(pipeline_id)
        if pipeline:
            # Format steps for display
            steps_text = ""
            for i, step in enumerate(pipeline.steps):
                prompt_type = "template" if step.prompt_template else "custom"
                prompt_val = step.prompt_template or step.system_prompt or ""
                steps_text += f"Step {i+1}: {step.name}\n"
                steps_text += f"  Agent: {step.agent}\n"
                steps_text += f"  Prompt ({prompt_type}): {prompt_val[:50]}...\n"
                steps_text += f"  Description: {step.description}\n\n"
            
            return (
                pipeline.id,
                pipeline.name,
                pipeline.description,
                pipeline.pattern,
                pipeline.emoji,
                ", ".join(pipeline.keywords) if pipeline.keywords else "",
                steps_text.strip(),
                pipeline.detection_weight,
                "builtin" if pipeline.is_builtin else "custom",
            )
    except Exception as e:
        logger.error(f"Error getting pipeline details: {e}")
    
    return default


def get_pipeline_steps_json(pipeline_id: str) -> str:
    """Get pipeline steps as JSON for editing."""
    if not PIPELINE_MANAGER_AVAILABLE or not pipeline_id:
        return "[]"
    
    try:
        pm = get_pipeline_manager()
        pipeline = pm.get(pipeline_id)
        if pipeline:
            import json
            steps = []
            for step in pipeline.steps:
                steps.append({
                    "name": step.name,
                    "agent": step.agent,
                    "prompt_template": step.prompt_template,
                    "system_prompt": step.system_prompt,
                    "description": step.description,
                })
            return json.dumps(steps, indent=2, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Error getting pipeline steps: {e}")
    
    return "[]"


def create_new_pipeline(
    pipeline_id: str,
    name: str,
    description: str,
    pattern: str,
    emoji: str,
    keywords_str: str,
    steps_json: str,
    detection_weight: float = 0.5,
) -> Tuple[str, List[List[str]], Any]:
    """Create a new custom pipeline. Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    # Validate
    if not pipeline_id or not pipeline_id.strip():
        return "[ERR] Pipeline ID is required", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    pm = get_pipeline_manager()
    
    if not pm.validate_pipeline_id(pipeline_id):
        return "[ERR] Invalid ID (use letters, numbers, _ or -)", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if pm.get(pipeline_id):
        return f"[ERR] Pipeline '{pipeline_id}' already exists", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if not name or not name.strip():
        return "[ERR] Name is required", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    # Parse keywords
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
    
    # Parse steps from JSON
    try:
        import json
        steps_data = json.loads(steps_json) if steps_json else []
        if not steps_data:
            return "[ERR] At least one step is required", get_pipeline_table_data(), get_pipeline_dropdown_update()
        
        steps = []
        for s in steps_data:
            steps.append(PipelineStep(
                name=s.get("name", "Step"),
                agent=s.get("agent", "coder"),
                prompt_template=s.get("prompt_template"),
                system_prompt=s.get("system_prompt"),
                description=s.get("description", ""),
                model=s.get("model"),  # Include model override
            ))
    except json.JSONDecodeError as e:
        return f"[ERR] Invalid steps JSON: {e}", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        pipeline = Pipeline(
            id=pipeline_id.strip(),
            name=name.strip(),
            description=description.strip() if description else "",
            pattern=pattern or "chain",
            emoji=emoji or "🔧",
            steps=steps,
            keywords=keywords,
            detection_weight=detection_weight,
        )
        
        if pm.create(pipeline):
            return f"[OK] Pipeline '{pipeline_id}' created ({len(steps)} steps)", get_pipeline_table_data(), get_pipeline_dropdown_update()
        else:
            return f"[ERR] Creation failed", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Creation failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


def update_existing_pipeline(
    pipeline_id: str,
    name: str,
    description: str,
    pattern: str,
    emoji: str,
    keywords_str: str,
    steps_json: str,
    detection_weight: float = 0.5,
) -> Tuple[str, List[List[str]], Any]:
    """Update an existing custom pipeline. Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if not pipeline_id:
        return "[ERR] No pipeline selected", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    pm = get_pipeline_manager()
    existing = pm.get(pipeline_id)
    
    if not existing:
        return f"[ERR] Pipeline '{pipeline_id}' not found", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if existing.is_builtin:
        return "[ERR] Cannot modify builtin pipelines. Duplicate it first.", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    # Parse keywords
    keywords = [k.strip() for k in keywords_str.split(",") if k.strip()] if keywords_str else []
    
    # Parse steps from JSON
    try:
        import json
        steps_data = json.loads(steps_json) if steps_json else []
        if not steps_data:
            return "[ERR] At least one step is required", get_pipeline_table_data(), get_pipeline_dropdown_update()
        
        steps = []
        for s in steps_data:
            steps.append(PipelineStep(
                name=s.get("name", "Step"),
                agent=s.get("agent", "coder"),
                prompt_template=s.get("prompt_template"),
                system_prompt=s.get("system_prompt"),
                description=s.get("description", ""),
                model=s.get("model"),  # Include model override
            ))
    except json.JSONDecodeError as e:
        return f"[ERR] Invalid steps JSON: {e}", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        pipeline = Pipeline(
            id=pipeline_id,
            name=name.strip() if name else existing.name,
            description=description.strip() if description else "",
            pattern=pattern or existing.pattern,
            emoji=emoji or existing.emoji,
            steps=steps,
            keywords=keywords,
            detection_weight=detection_weight,
        )
        
        if pm.update(pipeline_id, pipeline):
            return f"[OK] Pipeline '{pipeline_id}' updated", get_pipeline_table_data(), get_pipeline_dropdown_update()
        else:
            return f"[ERR] Update failed", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Update failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


def delete_pipeline(pipeline_id: str) -> Tuple[str, List[List[str]], Any]:
    """Delete a custom pipeline. Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if not pipeline_id:
        return "[ERR] No pipeline selected", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    pm = get_pipeline_manager()
    existing = pm.get(pipeline_id)
    
    if not existing:
        return f"[ERR] Pipeline '{pipeline_id}' not found", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if existing.is_builtin:
        return "[ERR] Cannot delete builtin pipelines", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        if pm.delete(pipeline_id):
            return f"[OK] Pipeline '{pipeline_id}' deleted", get_pipeline_table_data(), get_pipeline_dropdown_update()
        else:
            return f"[ERR] Deletion failed", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Deletion failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


def duplicate_pipeline(pipeline_id: str, new_id: str) -> Tuple[str, List[List[str]], Any]:
    """Duplicate a pipeline (builtin or custom). Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if not pipeline_id:
        return "[ERR] No pipeline selected", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if not new_id or not new_id.strip():
        return "[ERR] New ID is required", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    pm = get_pipeline_manager()
    
    if not pm.validate_pipeline_id(new_id):
        return "[ERR] Invalid new ID", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        new_pipeline = pm.duplicate(pipeline_id, new_id.strip())
        if new_pipeline:
            return f"[OK] Pipeline duplicated as '{new_id}'", get_pipeline_table_data(), get_pipeline_dropdown_update()
        else:
            return "[ERR] Duplication failed", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Duplication failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


def reload_pipelines() -> Tuple[str, List[List[str]], Any]:
    """Reload pipelines from files. Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        pm = get_pipeline_manager()
        pm.reload()
        stats = pm.get_stats()
        return f"[OK] Reloaded {stats['total']} pipelines ({stats['builtin']} builtin, {stats['custom']} custom)", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Reload failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


def export_all_pipelines() -> str:
    """Export all pipelines to YAML file."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available"
    
    try:
        pm = get_pipeline_manager()
        filepath = DATA_DIR / "exports" / f"pipelines_export_{time.strftime('%Y%m%d_%H%M%S')}.yaml"
        if pm.export_to_file(filepath):
            return f"[OK] Exported to {filepath}"
        else:
            return "[ERR] Export failed"
    except Exception as e:
        return f"[ERR] Export failed: {str(e)}"


def import_pipelines_from_file(file) -> Tuple[str, List[List[str]], Any]:
    """Import pipelines from an uploaded YAML file. Returns (status, table_data, dropdown_update)."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return "[ERR] Pipeline Manager not available", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    if file is None:
        return "[ERR] No file selected", get_pipeline_table_data(), get_pipeline_dropdown_update()
    
    try:
        filepath = Path(file.name) if hasattr(file, 'name') else Path(file)
        pm = get_pipeline_manager()
        imported = pm.import_from_file(filepath)
        if imported:
            return f"[OK] Imported {len(imported)} pipelines: {', '.join(imported)}", get_pipeline_table_data(), get_pipeline_dropdown_update()
        else:
            return "[ERR] No valid pipelines found in file", get_pipeline_table_data(), get_pipeline_dropdown_update()
    except Exception as e:
        return f"[ERR] Import failed: {str(e)}", get_pipeline_table_data(), get_pipeline_dropdown_update()


# =============================================================================
# VISUAL STEPS UI HELPERS (supports up to 10 steps)
# =============================================================================

MAX_PIPELINE_STEPS = 10
STEP_FIELDS = 8  # enabled, name, agent, model, ptype, template, prompt, desc


def steps_ui_to_json(*step_values) -> str:
    """
    Convert the visual step blocks to JSON for storage.
    Accepts 8 values per step (enabled, name, agent, model, ptype, template, prompt, desc).
    """
    import json
    steps = []
    
    # Process steps in groups of 8 fields
    num_steps = len(step_values) // STEP_FIELDS
    
    for i in range(num_steps):
        base = i * STEP_FIELDS
        enabled = step_values[base]
        name = step_values[base + 1]
        agent = step_values[base + 2]
        model = step_values[base + 3]  # Model override
        ptype = step_values[base + 4]
        template = step_values[base + 5]
        prompt = step_values[base + 6]
        desc = step_values[base + 7]
        
        if enabled and name:
            step = {
                "name": name,
                "agent": agent or "coder",
                "description": desc or "",
            }
            # Include model override if specified
            if model and model.strip():
                step["model"] = model.strip()
            if ptype == "template" and template:
                step["prompt_template"] = template
            elif prompt:
                step["system_prompt"] = prompt
            steps.append(step)
    
    return json.dumps(steps, indent=2)


def json_to_steps_ui(steps_json: str, num_steps: int = MAX_PIPELINE_STEPS) -> Tuple:
    """
    Convert JSON steps to values for the visual step blocks.
    Returns a tuple of (8 * num_steps) values.
    Also returns visibility states for accordions.
    """
    import json
    
    # Default values for empty step (enabled, name, agent, model, ptype, template, prompt, desc)
    empty_step = (False, "", "coder", "", "custom", "", "", "")
    
    try:
        steps = json.loads(steps_json) if steps_json else []
    except:
        steps = []
    
    result = []
    for i in range(num_steps):
        if i < len(steps):
            step = steps[i]
            enabled = True
            name = step.get("name", "")
            agent = step.get("agent", "coder")
            model = step.get("model", "")  # Model override
            
            # Determine prompt type
            if step.get("prompt_template"):
                ptype = "template"
                template = step.get("prompt_template", "")
                prompt = ""
            else:
                ptype = "custom"
                template = ""
                prompt = step.get("system_prompt", "")
            
            desc = step.get("description", "")
            result.extend([enabled, name, agent, model, ptype, template, prompt, desc])
        else:
            result.extend(empty_step)
    
    return tuple(result)


def json_to_steps_ui_with_visibility(steps_json: str, num_steps: int = MAX_PIPELINE_STEPS) -> Tuple:
    """
    Convert JSON steps to values AND accordion visibility states.
    Returns: (*step_values, *accordion_visible_states, visible_count)
    """
    import json
    
    try:
        steps = json.loads(steps_json) if steps_json else []
    except:
        steps = []
    
    # Get step values
    step_values = json_to_steps_ui(steps_json, num_steps)
    
    # Calculate how many accordions should be visible
    # Show at least 2, or the number of steps + 1 (for adding), up to MAX
    visible_count = max(2, min(len(steps) + 1, num_steps))
    
    # Generate visibility states (gr.update objects for accordions)
    visibility_states = []
    for i in range(num_steps):
        visibility_states.append(gr.update(visible=(i < visible_count)))
    
    return step_values + tuple(visibility_states) + (visible_count,)


def load_pipeline_to_visual_steps(pipeline_id: str) -> Tuple:
    """
    Load a pipeline and convert its steps to the visual step UI values.
    Returns: (id, name, desc, pattern, keywords, weight, type, 
              *steps_values, *accordion_visibilities, visible_count)
    """
    empty_base = ("", "", "", "chain", "", 0.5, "🧅 custom")
    empty_steps = json_to_steps_ui_with_visibility("[]")
    
    if not PIPELINE_MANAGER_AVAILABLE:
        return empty_base + empty_steps
    
    try:
        pm = get_pipeline_manager()
        pipeline = pm.get(pipeline_id)
        if not pipeline:
            return empty_base + empty_steps
        
        # Get basic info
        keywords_str = ", ".join(pipeline.keywords) if pipeline.keywords else ""
        pipe_type = "📌 builtin" if pipeline.is_builtin else "🧅 custom"
        
        # Convert steps to JSON then to UI values with visibility
        import json
        steps_list = [s.to_dict() for s in pipeline.steps]
        steps_json = json.dumps(steps_list)
        steps_values = json_to_steps_ui_with_visibility(steps_json)
        
        return (
            pipeline.id,
            pipeline.name,
            pipeline.description,
            pipeline.pattern,
            keywords_str,
            pipeline.detection_weight,
            pipe_type,
        ) + steps_values
        
    except Exception as e:
        logger.error(f"Error loading pipeline: {e}")
        return empty_base + empty_steps


def create_pipeline_from_visual_steps(
    pipeline_id: str,
    name: str,
    description: str,
    pattern: str,
    keywords: str,
    weight: float,
    *step_values,  # All step fields as *args
) -> Tuple[str, List, Any]:
    """Create a new pipeline from the visual step UI. Returns (status, table, dropdown_update)."""
    steps_json = steps_ui_to_json(*step_values)
    return create_new_pipeline(
        pipeline_id, name, description, pattern, "🔧", keywords, steps_json, weight
    )


def update_pipeline_from_visual_steps(
    pipeline_id: str,
    name: str,
    description: str,
    pattern: str,
    keywords: str,
    weight: float,
    *step_values,  # All step fields as *args
) -> Tuple[str, List, Any]:
    """Update an existing pipeline from the visual step UI. Returns (status, table, dropdown_update)."""
    steps_json = steps_ui_to_json(*step_values)
    # Pass empty emoji to keep existing one
    return update_existing_pipeline(
        pipeline_id, name, description, pattern, "", keywords, steps_json, weight
    )


def get_template_content(template_id: str) -> str:
    """
    Get the content of a prompt template by its ID.
    Returns the template content or empty string if not found.
    """
    if not PIPELINE_MANAGER_AVAILABLE or not template_id:
        return ""
    
    try:
        # Templates are stored in agents/config.yaml under prompt_templates
        # They are direct strings, not dicts
        from .agents import get_orchestrator
        orch = get_orchestrator()
        if orch and hasattr(orch, 'config'):
            prompt_templates = orch.config.get('prompt_templates', {})
            if template_id in prompt_templates:
                template_content = prompt_templates[template_id]
                # Templates are strings directly, not dicts
                if isinstance(template_content, str):
                    return template_content
                elif isinstance(template_content, dict):
                    # Fallback if it's a dict with system_prompt key
                    return template_content.get('system_prompt', str(template_content))
        
        return ""
    except Exception as e:
        logger.warning(f"Could not load template {template_id}: {e}")
        return ""


def on_template_select(template_id: str, current_ptype: str):
    """
    When a template is selected, load its content into the prompt field.
    Only loads if ptype is 'template'.
    """
    if current_ptype == "template" and template_id:
        content = get_template_content(template_id)
        return content
    return gr.update()  # No change


def on_prompt_edit(new_prompt: str, current_ptype: str, original_prompt: str):
    """
    When the prompt is edited, switch to 'custom' mode if it was 'template'.
    Returns (new_ptype,) - only switches if content actually changed.
    """
    # If we're in template mode and the prompt was modified, switch to custom
    if current_ptype == "template" and new_prompt != original_prompt:
        return "custom"
    return gr.update()


def swap_steps(
    # Step A values
    a_enabled, a_name, a_agent, a_ptype, a_template, a_prompt, a_desc,
    # Step B values  
    b_enabled, b_name, b_agent, b_ptype, b_template, b_prompt, b_desc,
):
    """
    Swap the content of two steps.
    Returns all values for both steps, swapped.
    """
    return (
        # New values for step A (from B)
        b_enabled, b_name, b_agent, b_ptype, b_template, b_prompt, b_desc,
        # New values for step B (from A)
        a_enabled, a_name, a_agent, a_ptype, a_template, a_prompt, a_desc,
    )


def generate_step_prompt_llm(
    step_name: str,
    step_description: str,
    pipeline_context: str,
    agent_type: str,
):
    """
    Generate a system prompt for a step using LLM with keepalive.
    
    This is a generator function that yields progress updates to prevent
    Gradio timeout during long LLM calls.
    """
    if not PIPELINE_MANAGER_AVAILABLE:
        yield "[ERR] Pipeline Manager not available"
        return
    
    if not step_description:
        yield "[ERR] Step description is required"
        return
    
    try:
        pm = get_pipeline_manager()
        # Use the new keepalive generator
        for update in pm.generate_step_prompt_with_keepalive(
            step_name=step_name or "Step",
            step_description=step_description,
            pipeline_context=pipeline_context or "General pipeline",
            agent_type=agent_type or "coder",
        ):
            yield update
    except Exception as e:
        yield f"[ERR] Generation failed: {str(e)}"


def get_available_agents_list() -> List[str]:
    """Get list of available agent IDs."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return ["coder", "reviewer", "explainer", "planner", "writer"]
    
    try:
        pm = get_pipeline_manager()
        return pm.get_available_agents()
    except:
        return ["coder", "reviewer", "explainer", "planner", "writer"]


def get_available_templates_list() -> List[str]:
    """Get list of available prompt template IDs."""
    if not PIPELINE_MANAGER_AVAILABLE:
        return []
    
    try:
        pm = get_pipeline_manager()
        return pm.get_available_templates()
    except:
        return []


def get_available_models_list() -> List[Tuple[str, str]]:
    """Get list of available Ollama models for step model override dropdown."""
    models = [("(Use agent default)", "")]  # Empty = use agent's default model
    
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, 'models'):
            for m in response.models:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
                if name:
                    models.append((name, name))
        elif isinstance(response, dict):
            for m in response.get("models", []):
                name = m.get("model") or m.get("name", "")
                if name:
                    models.append((name, name))
    except Exception:
        # Fallback models
        for m in ["qwen3-coder:30b", "deepseek-r1:32b", "qwen3:32b", "nemotron-3-nano:30b"]:
            models.append((m, m))
    
    return models


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def create_app() -> gr.Blocks:
    """Create the Gradio application."""
    
    # Get available models
    available_models = []
    try:
        import ollama
        response = ollama.list()
        if hasattr(response, 'models'):
            for m in response.models:
                name = getattr(m, 'model', None) or getattr(m, 'name', None)
                if name:
                    available_models.append(name)
        elif isinstance(response, dict):
            for m in response.get("models", []):
                available_models.append(m.get("model") or m.get("name", ""))
    except Exception:
        available_models = ["qwen3-coder:30b", "deepseek-r1:32b", "qwen3:32b"]
    
    # Preset choices with improved error handling
    preset_choices = []
    try:
        presets = preset_manager.get_ordered()
        for p in presets:
            preset_choices.append((f"{p.icon} {p.name}", p.id))
        if not preset_choices:
            preset_choices = [("⚙️ Default", "default")]
    except Exception as e:
        logger.error(f"Error loading presets: {e}")
        preset_choices = [("⚙️ Default", "default")]
    
    # Task choices
    task_choices = [
        ("Auto", ""),
        ("R Code", "code_r"),
        ("Python Code", "code_python"),
        ("R Debug", "debug_r"),
        ("Python Debug", "debug_python"),
        ("Scientific Writing", "scientific_writing"),
        ("Planning", "planning"),
        ("Simple Question", "simple_question"),
    ]
    
    variant_choices = [
        ("Standard", "standard"),
        ("Reasoning", "reasoning"),
        ("Fast", "fast"),
    ]
    
    priority_choices = [
        ("Balanced", "balanced"),
        ("Fast", "fast"),
        ("Quality", "quality"),
    ]
    
    pipeline_choices = get_pipeline_choices()
    
    with gr.Blocks(
        title="Opti-Oignon LLM",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="slate",
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
        ),
        css=CUSTOM_CSS,
    ) as app:
        
        # Header
        gr.Markdown("""
        # 🧅 LLMs Opti-Oignon
        
        Transform your questions into optimized prompts, enriched with your personal documents, 
        with multi-model orchestration for complex tasks.
        
        """)
        
        with gr.Tabs():
            # =================================================================
            # TAB 1: MAIN (CHAT)
            # =================================================================
            with gr.Tab("Chat", id="chat"):
                with gr.Row():
                    # Left column: Inputs
                    with gr.Column(scale=1):
                        
                        # Question
                        question_input = gr.Textbox(
                            label="Your question",
                            placeholder="Ex: How to calculate Shannon index in R?",
                            lines=3,
                            autofocus=True,
                        )
                        
                        # Document/Code
                        with gr.Accordion("📄 Document / Code", open=False):
                            document_input = gr.Textbox(
                                label="Paste your code here",
                                placeholder="R code, Python code, error message...",
                                lines=10,
                            )
                            file_input = gr.File(
                                label="Or upload a file",
                                file_types=[".r", ".R", ".py", ".sh", ".md", ".txt", ".json", ".yaml"],
                            )
                        
                        # === MULTI-AGENT / PIPELINE SECTION ===
                        with gr.Accordion("Pipeline Mode", open=True):
                            multi_agent_status_display = gr.Textbox(
                                label="Status",
                                value=get_multi_agent_status(),
                                interactive=False,
                                lines=1,
                            )
                            multi_agent_pipeline = gr.Dropdown(
                                label="Select Mode",
                                choices=pipeline_choices,
                                value="disabled",  # Single model by default
                                info="disabled=single model, dynamic=LLM planning, auto=auto-detect",
                            )
                            # Hidden but kept for backward compatibility
                            use_multi_agent = gr.Checkbox(
                                label="Enable Multi-Agent",
                                value=True,  # Always true, dropdown controls mode
                                visible=False,
                            )
                            show_agent_steps = gr.Checkbox(
                                label="Show reasoning steps",
                                value=True,
                                visible=MULTI_AGENT_AVAILABLE,
                            )
                            multi_agent_refresh = gr.Button(
                                "🔄 Refresh", 
                                size="sm", 
                                visible=MULTI_AGENT_AVAILABLE
                            )

                            # Hidden checkboxes for backward compatibility
                            use_dynamic_pipeline = gr.Checkbox(
                                value=False,
                                visible=False,
                            )
                            dynamic_auto_execute = gr.Checkbox(
                                value=True,
                                visible=False,
                            )
                        
                        # === RAG SECTION ===
                        with gr.Accordion("RAG - Document Context", open=False):
                            rag_status = gr.Textbox(
                                label="Status",
                                value=get_rag_status(),
                                interactive=False,
                                lines=1,
                            )
                            use_rag = gr.Checkbox(
                                label="Enable RAG (enrich with my documents)",
                                value=False,  # Off by default
                                interactive=RAG_AVAILABLE,
                            )
                            rag_n_chunks = gr.Slider(
                                label="Number of context chunks",
                                minimum=1,
                                maximum=10,
                                step=1,
                                value=3,
                                visible=RAG_AVAILABLE,
                            )
                            rag_refresh_btn = gr.Button("🔄 Refresh status", size="sm", visible=RAG_AVAILABLE)
                        
                        # Presets (quick mode)
                        with gr.Accordion("Quick Presets", open=False):
                            use_presets = gr.Checkbox(
                                label="Use a preset",
                                value=False,
                            )
                            preset_dropdown = gr.Dropdown(
                                label="Preset",
                                choices=preset_choices,
                                interactive=True,
                            )
                        
                        # Advanced options
                        with gr.Accordion("⚙️ Advanced Options", open=False):
                            force_task = gr.Dropdown(
                                label="Force task type",
                                choices=task_choices,
                                value="",
                            )
                            force_model = gr.Dropdown(
                                label="Force model",
                                choices=[("Auto", "")] + [(m, m) for m in available_models],
                                value="",
                            )
                            force_temp = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.5,
                            )
                            force_variant = gr.Dropdown(
                                label="Prompt variant",
                                choices=variant_choices,
                                value="standard",
                            )
                            do_refine = gr.Checkbox(
                                label="Refine question",
                                value=True,
                            )
                            priority = gr.Dropdown(
                                label="Priority",
                                choices=priority_choices,
                                value="balanced",
                            )
                        
                        # Buttons
                        with gr.Row():
                            submit_btn = gr.Button("Zou!", variant="primary", scale=2)
                            cancel_btn = gr.Button("[x] Stop", variant="stop", scale=1)
                            clear_btn = gr.Button("Clear", scale=1)
                    
                    # Right column: Outputs
                    with gr.Column(scale=2):
                        
                        # Status
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                        )
                        
                        # Detection info (collapsed)
                        with gr.Accordion("Detection & Routing", open=False):
                            with gr.Row():
                                analysis_output = gr.Markdown(label="Analysis")
                                routing_output = gr.Markdown(label="Routing")
                            
                            # Context indicator (if available)
                            if CONTEXT_MANAGER_AVAILABLE:
                                context_indicator = gr.Markdown(
                                    label="Context Usage",
                                    value="*Enter a question and select a model to see context usage*",
                                    elem_classes=["context-indicator"],
                                )
                            else:
                                context_indicator = None
                        
                        # Multi-Agent steps (visible if multi-agent used)
                        agent_steps_output = gr.Markdown(
                            label="Multi-Agent Steps",
                            visible=MULTI_AGENT_AVAILABLE,
                            elem_classes=["agent-steps"],
                        )
                        
                        # RAG sources (visible if RAG used)
                        rag_sources_output = gr.Markdown(
                            label="RAG Sources",
                            visible=RAG_AVAILABLE,
                            elem_classes=["rag-sources"],
                        )
                        
                        # Response
                        response_output = gr.Markdown(
                            label="Response",
                            elem_classes=["response-area"],
                        )
                
                # Event handlers
                submit_btn.click(
                    fn=process_with_streaming,
                    inputs=[
                        question_input, document_input, file_input,
                        use_presets, preset_dropdown,
                        force_task, force_model, force_temp, force_variant,
                        do_refine, priority,
                        use_rag, rag_n_chunks,
                        use_multi_agent, multi_agent_pipeline, show_agent_steps,
                        use_dynamic_pipeline,
                        dynamic_auto_execute,
                    ],
                    outputs=[status_output, analysis_output, routing_output, agent_steps_output, response_output],
                )
                
                cancel_btn.click(
                    fn=cancel_generation,
                    outputs=[status_output],
                )
                
                clear_btn.click(
                    fn=clear_inputs,
                    outputs=[question_input, document_input, file_input, response_output],
                )
                
                rag_refresh_btn.click(
                    fn=refresh_rag_stats,
                    outputs=[rag_status],
                )
                
                multi_agent_refresh.click(
                    fn=refresh_multi_agent_stats,
                    outputs=[multi_agent_status_display, multi_agent_pipeline],
                )
                
                # Live analysis
                question_input.change(
                    fn=analyze_question,
                    inputs=[question_input, document_input],
                    outputs=[analysis_output, routing_output],
                )
                
                # Context indicator updates (if available)
                if CONTEXT_MANAGER_AVAILABLE:
                    # Common inputs for all context updates
                    context_inputs = [
                        question_input, document_input, file_input,
                        force_model, use_presets, preset_dropdown, priority
                    ]
                    
                    # Update on question change
                    question_input.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on document change  
                    document_input.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on file upload
                    file_input.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on model change
                    force_model.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on preset selection change
                    preset_dropdown.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on use_presets toggle
                    use_presets.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
                    
                    # Update on priority change
                    priority.change(
                        fn=update_context_full,
                        inputs=context_inputs,
                        outputs=[context_indicator],
                    )
            
            # =================================================================
            # TAB 2: PRESETS MANAGEMENT (ENHANCED)
            # =================================================================
            with gr.Tab("Presets", id="presets"):
                gr.Markdown("""
                ### Preset Management
                
                Create and manage presets for quick configuration.
                **Keywords** enable automatic preset selection based on your question content.
                **Detection Weight** (0.0-1.0) determines keyword match priority.
                """)
                
                with gr.Row():
                    # Left column: Preset list and testing
                    with gr.Column(scale=1):
                        gr.Markdown("#### Available Presets")
                        
                        preset_table = gr.Dataframe(
                            headers=["Icon", "ID", "Name", "Task", "Model", "Temp", "Weight", "Keywords"],
                            value=get_preset_table_data(),
                            interactive=False,
                            wrap=True,
                        )
                        
                        with gr.Row():
                            preset_refresh_btn = gr.Button("Refresh", size="sm")
                            preset_reload_btn = gr.Button("Reload Config", size="sm")
                        
                        preset_status = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=1,
                        )
                        
                        # Export/Import section
                        with gr.Accordion("Export / Import", open=False):
                            with gr.Row():
                                export_presets_btn = gr.Button("Export All", size="sm")
                                export_result = gr.Textbox(
                                    label="Export Result",
                                    interactive=False,
                                    lines=1,
                                )
                            with gr.Row():
                                import_file = gr.File(
                                    label="Import YAML file",
                                    file_types=[".yaml", ".yml"],
                                )
                                import_btn = gr.Button("Import", size="sm")
                        
                        # Keyword test section
                        gr.Markdown("#### Test Keyword Matching")
                        keyword_test_input = gr.Textbox(
                            label="Test question",
                            placeholder="Enter a question to see which presets match...",
                            lines=2,
                        )
                        keyword_test_btn = gr.Button("Test Keywords")
                        keyword_test_result = gr.Markdown()
                    
                    # Right column: Edit/Create form
                    with gr.Column(scale=1):
                        gr.Markdown("#### Create / Edit Preset")
                        
                        preset_edit_id = gr.Textbox(
                            label="Preset ID (unique, no spaces)",
                            placeholder="my_preset",
                        )
                        preset_edit_name = gr.Textbox(
                            label="Display Name",
                            placeholder="My Custom Preset",
                        )
                        preset_edit_desc = gr.Textbox(
                            label="Description",
                            placeholder="What this preset is for...",
                            lines=2,
                        )
                        
                        with gr.Row():
                            preset_edit_task = gr.Dropdown(
                                label="Task Type",
                                choices=task_choices,
                                value="simple_question",
                            )
                            preset_edit_model = gr.Dropdown(
                                label="Model",
                                choices=[(m, m) for m in available_models],
                                value=available_models[0] if available_models else "qwen3-coder:30b",
                            )
                        
                        with gr.Row():
                            preset_edit_temp = gr.Slider(
                                label="Temperature",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.5,
                            )
                            preset_edit_variant = gr.Dropdown(
                                label="Prompt Variant",
                                choices=variant_choices,
                                value="standard",
                            )
                        
                        preset_edit_tags = gr.Textbox(
                            label="Tags (comma-separated)",
                            placeholder="code, r, fast",
                        )
                        
                        # Keyword section with auto-suggest
                        gr.Markdown("#### Detection Keywords")
                        preset_edit_keywords = gr.Textbox(
                            label="Keywords for Auto-Routing (comma-separated)",
                            placeholder="ggplot, dplyr, tidyverse, vegan",
                            lines=3,
                        )
                        
                        with gr.Row():
                            preset_edit_weight = gr.Slider(
                                label="Detection Weight",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=0.5,
                                info="Higher = higher priority when multiple presets match",
                            )
                            auto_suggest_btn = gr.Button("Auto-Suggest", variant="secondary", size="sm")
                        
                        keyword_suggestions = gr.Textbox(
                            label="Suggested Keywords (copy to add)",
                            interactive=False,
                            lines=2,
                        )
                        
                        gr.Markdown("*Keywords trigger automatic preset selection when found in your question*")
                        
                        # Action buttons
                        with gr.Row():
                            preset_create_btn = gr.Button("Create New", variant="primary")
                            preset_update_btn = gr.Button("Update Selected")
                            preset_delete_btn = gr.Button("Delete", variant="stop")
                        
                        with gr.Row():
                            preset_dup_id = gr.Textbox(
                                label="New ID for duplicate",
                                placeholder="my_preset_copy",
                                scale=2,
                            )
                            preset_dup_btn = gr.Button("Duplicate", scale=1)
                
                # Preset events
                def on_preset_row_select(evt: gr.SelectData, table_data):
                    """When a row is clicked, load its details."""
                    if evt.index and len(evt.index) > 0:
                        row_idx = evt.index[0]
                        if row_idx < len(table_data):
                            preset_id = table_data.iloc[row_idx, 1]  # ID is in column 1
                            return get_preset_details(preset_id)
                    return "", "", "", "", 0.5, "standard", "", "", "", 0.5
                
                preset_table.select(
                    fn=on_preset_row_select,
                    inputs=[preset_table],
                    outputs=[
                        preset_edit_id,
                        preset_edit_name,
                        preset_edit_desc,
                        preset_edit_task,
                        preset_edit_temp,
                        preset_edit_variant,
                        preset_edit_model,
                        preset_edit_tags,
                        preset_edit_keywords,
                        preset_edit_weight,
                    ],
                )
                
                preset_refresh_btn.click(
                    fn=refresh_preset_table,
                    outputs=[preset_table],
                )
                
                preset_reload_btn.click(
                    fn=reload_presets,
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
                
                preset_create_btn.click(
                    fn=create_new_preset,
                    inputs=[
                        preset_edit_id,
                        preset_edit_name,
                        preset_edit_desc,
                        preset_edit_task,
                        preset_edit_temp,
                        preset_edit_variant,
                        preset_edit_model,
                        preset_edit_tags,
                        preset_edit_keywords,
                        preset_edit_weight,
                    ],
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
                
                preset_update_btn.click(
                    fn=update_existing_preset,
                    inputs=[
                        preset_edit_id,
                        preset_edit_name,
                        preset_edit_desc,
                        preset_edit_task,
                        preset_edit_temp,
                        preset_edit_variant,
                        preset_edit_model,
                        preset_edit_tags,
                        preset_edit_keywords,
                        preset_edit_weight,
                    ],
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
                
                preset_delete_btn.click(
                    fn=delete_preset,
                    inputs=[preset_edit_id],
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
                
                preset_dup_btn.click(
                    fn=duplicate_preset,
                    inputs=[preset_edit_id, preset_dup_id],
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
                
                keyword_test_btn.click(
                    fn=test_keyword_matching,
                    inputs=[keyword_test_input],
                    outputs=[keyword_test_result],
                )
                
                auto_suggest_btn.click(
                    fn=suggest_keywords_for_preset,
                    inputs=[
                        preset_edit_task,
                        preset_edit_name,
                        preset_edit_desc,
                        preset_edit_keywords,
                    ],
                    outputs=[keyword_suggestions],
                )
                
                export_presets_btn.click(
                    fn=export_all_presets,
                    outputs=[export_result],
                )
                
                import_btn.click(
                    fn=import_presets_from_file,
                    inputs=[import_file],
                    outputs=[preset_status, preset_table, preset_dropdown],
                )
            
            # =================================================================
            # TAB: PIPELINES (Multi-Agent)
            # =================================================================
            with gr.Tab("Pipelines", id="pipelines"):
                gr.Markdown("""
                ### Pipeline Manager (Multi-Agent)
                
                Create and manage multi-agent pipelines for complex tasks.
                **📌 Builtin** pipelines are read-only. **🧅 Custom** pipelines can be edited.
                """)
                
                with gr.Row():
                    # Left column: Pipeline list (more space)
                    with gr.Column(scale=2):
                        gr.Markdown("#### Available Pipelines")
                        
                        pipeline_table = gr.Dataframe(
                            headers=["Type", "ID", "Name", "Steps", "Pattern", "Weight", "Keywords"],
                            value=get_pipeline_table_data(),
                            interactive=False,
                            wrap=True,
                        )
                        
                        with gr.Row():
                            pipeline_refresh_btn = gr.Button("🔄 Refresh", size="sm")
                            pipeline_reload_btn = gr.Button("Reload Config", size="sm")
                        
                        pipeline_status = gr.Textbox(
                            label="Pipeline Status",
                            interactive=False,
                            lines=1,
                        )
                        
                        # Export/Import section
                        with gr.Accordion("Export / Import", open=False):
                            with gr.Row():
                                pipeline_export_btn = gr.Button("Export All", size="sm")
                                pipeline_export_result = gr.Textbox(
                                    label="Export Result",
                                    interactive=False,
                                    lines=1,
                                )
                            with gr.Row():
                                pipeline_import_file = gr.File(
                                    label="Import YAML file",
                                    file_types=[".yaml", ".yml"],
                                )
                                pipeline_import_btn = gr.Button("Import", size="sm")
                    
                    # Right column: Edit/Create form (slightly narrower)
                    with gr.Column(scale=3):
                        gr.Markdown("#### Create / Edit Pipeline")
                        
                        # Basic info
                        with gr.Row():
                            pipeline_edit_id = gr.Textbox(
                                label="Pipeline ID (unique, no spaces)",
                                placeholder="my_pipeline",
                            )
                        
                        pipeline_edit_name = gr.Textbox(
                            label="Display Name",
                            placeholder="My Custom Pipeline",
                        )
                        pipeline_edit_desc = gr.Textbox(
                            label="Description",
                            placeholder="What this pipeline does...",
                            lines=2,
                        )
                        
                        with gr.Row():
                            pipeline_edit_pattern = gr.Dropdown(
                                label="Pattern",
                                choices=[
                                    ("Chain (sequential)", "chain"),
                                    ("Verifier (with validation)", "verifier"),
                                    ("Decomposition (sub-tasks)", "decomposition"),
                                    ("Iterative (refinement)", "iterative"),
                                ],
                                value="chain",
                            )
                            pipeline_edit_type = gr.Textbox(
                                label="Type",
                                value="🧅 custom",
                                interactive=False,
                            )
                        
                        # =========================================
                        # VISUAL STEPS EDITOR (10 blocks max, with move up/down)
                        # =========================================
                        gr.Markdown("#### Pipeline Steps")
                        
                        # Get available agents and templates for dropdowns
                        agents_list = get_available_agents_list()
                        templates_list = get_available_templates_list()
                        models_list = get_available_models_list()  # For model override dropdown
                        
                        # State to track number of visible steps
                        visible_steps_count = gr.State(value=2)
                        
                        # Add step button at the top
                        with gr.Row():
                            gr.Markdown("*Use ↑↓ to reorder steps. Template prompts are editable (editing switches to Custom).*")
                            add_step_btn = gr.Button("➕ Add Step", size="sm", scale=0)
                        
                        # Step 1
                        with gr.Accordion("Step 1", open=True, visible=True) as step1_accordion:
                            with gr.Row():
                                step1_enabled = gr.Checkbox(label="✓ Enable", value=True, scale=1)
                                step1_name = gr.Textbox(label="Name", placeholder="Analysis", scale=2)
                                step1_agent = gr.Dropdown(label="Agent", choices=agents_list, value="reviewer", scale=2)
                                step1_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step1_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step1_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step1_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step1_prompt = gr.Textbox(label="System Prompt", placeholder="You are an expert...", lines=2)
                            with gr.Row():
                                step1_desc = gr.Textbox(label="Description", placeholder="What this step does", scale=3)
                                step1_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 2
                        with gr.Accordion("Step 2", open=True, visible=True) as step2_accordion:
                            with gr.Row():
                                step2_enabled = gr.Checkbox(label="✓ Enable", value=True, scale=1)
                                step2_name = gr.Textbox(label="Name", placeholder="Solution", scale=2)
                                step2_agent = gr.Dropdown(label="Agent", choices=agents_list, value="coder", scale=2)
                                step2_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step2_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step2_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step2_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step2_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step2_prompt = gr.Textbox(label="System Prompt", placeholder="You are an expert...", lines=2)
                            with gr.Row():
                                step2_desc = gr.Textbox(label="Description", placeholder="What this step does", scale=3)
                                step2_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 3
                        with gr.Accordion("Step 3", open=False, visible=False) as step3_accordion:
                            with gr.Row():
                                step3_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step3_name = gr.Textbox(label="Name", placeholder="Verification", scale=2)
                                step3_agent = gr.Dropdown(label="Agent", choices=agents_list, value="reviewer", scale=2)
                                step3_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step3_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step3_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step3_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step3_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step3_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step3_desc = gr.Textbox(label="Description", scale=3)
                                step3_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 4
                        with gr.Accordion("Step 4", open=False, visible=False) as step4_accordion:
                            with gr.Row():
                                step4_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step4_name = gr.Textbox(label="Name", placeholder="Refinement", scale=2)
                                step4_agent = gr.Dropdown(label="Agent", choices=agents_list, value="coder", scale=2)
                                step4_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step4_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step4_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step4_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step4_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step4_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step4_desc = gr.Textbox(label="Description", scale=3)
                                step4_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 5
                        with gr.Accordion("Step 5", open=False, visible=False) as step5_accordion:
                            with gr.Row():
                                step5_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step5_name = gr.Textbox(label="Name", placeholder="Review", scale=2)
                                step5_agent = gr.Dropdown(label="Agent", choices=agents_list, value="reviewer", scale=2)
                                step5_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step5_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step5_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step5_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step5_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step5_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step5_desc = gr.Textbox(label="Description", scale=3)
                                step5_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 6
                        with gr.Accordion("Step 6", open=False, visible=False) as step6_accordion:
                            with gr.Row():
                                step6_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step6_name = gr.Textbox(label="Name", scale=2)
                                step6_agent = gr.Dropdown(label="Agent", choices=agents_list, value="coder", scale=2)
                                step6_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step6_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step6_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step6_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step6_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step6_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step6_desc = gr.Textbox(label="Description", scale=3)
                                step6_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 7
                        with gr.Accordion("Step 7", open=False, visible=False) as step7_accordion:
                            with gr.Row():
                                step7_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step7_name = gr.Textbox(label="Name", scale=2)
                                step7_agent = gr.Dropdown(label="Agent", choices=agents_list, value="reviewer", scale=2)
                                step7_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step7_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step7_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step7_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step7_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step7_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step7_desc = gr.Textbox(label="Description", scale=3)
                                step7_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 8
                        with gr.Accordion("Step 8", open=False, visible=False) as step8_accordion:
                            with gr.Row():
                                step8_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step8_name = gr.Textbox(label="Name", scale=2)
                                step8_agent = gr.Dropdown(label="Agent", choices=agents_list, value="coder", scale=2)
                                step8_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step8_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step8_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step8_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step8_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step8_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step8_desc = gr.Textbox(label="Description", scale=3)
                                step8_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 9
                        with gr.Accordion("Step 9", open=False, visible=False) as step9_accordion:
                            with gr.Row():
                                step9_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step9_name = gr.Textbox(label="Name", scale=2)
                                step9_agent = gr.Dropdown(label="Agent", choices=agents_list, value="reviewer", scale=2)
                                step9_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step9_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                                step9_down = gr.Button("↓", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step9_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step9_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step9_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step9_desc = gr.Textbox(label="Description", scale=3)
                                step9_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Step 10
                        with gr.Accordion("Step 10", open=False, visible=False) as step10_accordion:
                            with gr.Row():
                                step10_enabled = gr.Checkbox(label="✓ Enable", value=False, scale=1)
                                step10_name = gr.Textbox(label="Name", scale=2)
                                step10_agent = gr.Dropdown(label="Agent", choices=agents_list, value="coder", scale=2)
                                step10_model = gr.Dropdown(label="Model", choices=models_list, value="", scale=2, info="Override agent's default")
                                step10_up = gr.Button("↑", size="sm", scale=0, min_width=40)
                            with gr.Row():
                                step10_ptype = gr.Radio(
                                    choices=[("Template", "template"), ("Custom", "custom")],
                                    value="custom", label="Prompt Type", scale=1,
                                )
                                step10_template = gr.Dropdown(
                                    label="Template", choices=templates_list, allow_custom_value=True,
                                    value=templates_list[0] if templates_list else None,
                                    scale=2,
                                )
                            step10_prompt = gr.Textbox(label="System Prompt", lines=2)
                            with gr.Row():
                                step10_desc = gr.Textbox(label="Description", scale=3)
                                step10_gen_btn = gr.Button("Generate", size="sm", scale=1)
                        
                        # Collect all accordion references for visibility control
                        all_step_accordions = [
                            step1_accordion, step2_accordion, step3_accordion, step4_accordion, step5_accordion,
                            step6_accordion, step7_accordion, step8_accordion, step9_accordion, step10_accordion,
                        ]
                        

                        
                        # Keywords and weight
                        gr.Markdown("#### Auto-Detection")
                        pipeline_edit_keywords = gr.Textbox(
                            label="Detection Keywords (comma-separated)",
                            placeholder="debug, error, fix, code",
                            lines=1,
                        )
                        
                        pipeline_edit_weight = gr.Slider(
                            label="Detection Weight",
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            value=0.5,
                            info="Higher = higher priority when multiple pipelines match",
                        )
                        
                        # Action buttons
                        with gr.Row():
                            pipeline_create_btn = gr.Button("Create New ✨", variant="primary")
                            pipeline_update_btn = gr.Button("Update Selected")
                            pipeline_delete_btn = gr.Button("Delete", variant="stop")
                        
                        with gr.Row():
                            pipeline_dup_id = gr.Textbox(
                                label="New ID for duplicate",
                                placeholder="my_pipeline_copy",
                                scale=2,
                            )
                            pipeline_dup_btn = gr.Button("Duplicate", scale=1)
                
                # =========================================
                # EVENTS
                # =========================================
                
                # Add Step button handler
                def add_step(current_count):
                    """Show the next hidden step accordion."""
                    new_count = min(current_count + 1, MAX_PIPELINE_STEPS)
                    visibility_updates = []
                    for i in range(MAX_PIPELINE_STEPS):
                        visibility_updates.append(gr.update(visible=(i < new_count)))
                    return visibility_updates + [new_count]
                
                add_step_btn.click(
                    fn=add_step,
                    inputs=[visible_steps_count],
                    outputs=all_step_accordions + [visible_steps_count],
                )
                
                # =============================================
                # TEMPLATE SELECTION - Load content into prompt
                # =============================================
                # When template is selected and ptype is "template", load the template content
                step1_template.change(fn=on_template_select, inputs=[step1_template, step1_ptype], outputs=[step1_prompt])
                step2_template.change(fn=on_template_select, inputs=[step2_template, step2_ptype], outputs=[step2_prompt])
                step3_template.change(fn=on_template_select, inputs=[step3_template, step3_ptype], outputs=[step3_prompt])
                step4_template.change(fn=on_template_select, inputs=[step4_template, step4_ptype], outputs=[step4_prompt])
                step5_template.change(fn=on_template_select, inputs=[step5_template, step5_ptype], outputs=[step5_prompt])
                step6_template.change(fn=on_template_select, inputs=[step6_template, step6_ptype], outputs=[step6_prompt])
                step7_template.change(fn=on_template_select, inputs=[step7_template, step7_ptype], outputs=[step7_prompt])
                step8_template.change(fn=on_template_select, inputs=[step8_template, step8_ptype], outputs=[step8_prompt])
                step9_template.change(fn=on_template_select, inputs=[step9_template, step9_ptype], outputs=[step9_prompt])
                step10_template.change(fn=on_template_select, inputs=[step10_template, step10_ptype], outputs=[step10_prompt])
                
                # When ptype changes to "template", load the current template content
                def on_ptype_change(ptype, template_id):
                    """When switching to template mode, load the template content."""
                    if ptype == "template" and template_id:
                        return get_template_content(template_id)
                    return gr.update()
                
                step1_ptype.change(fn=on_ptype_change, inputs=[step1_ptype, step1_template], outputs=[step1_prompt])
                step2_ptype.change(fn=on_ptype_change, inputs=[step2_ptype, step2_template], outputs=[step2_prompt])
                step3_ptype.change(fn=on_ptype_change, inputs=[step3_ptype, step3_template], outputs=[step3_prompt])
                step4_ptype.change(fn=on_ptype_change, inputs=[step4_ptype, step4_template], outputs=[step4_prompt])
                step5_ptype.change(fn=on_ptype_change, inputs=[step5_ptype, step5_template], outputs=[step5_prompt])
                step6_ptype.change(fn=on_ptype_change, inputs=[step6_ptype, step6_template], outputs=[step6_prompt])
                step7_ptype.change(fn=on_ptype_change, inputs=[step7_ptype, step7_template], outputs=[step7_prompt])
                step8_ptype.change(fn=on_ptype_change, inputs=[step8_ptype, step8_template], outputs=[step8_prompt])
                step9_ptype.change(fn=on_ptype_change, inputs=[step9_ptype, step9_template], outputs=[step9_prompt])
                step10_ptype.change(fn=on_ptype_change, inputs=[step10_ptype, step10_template], outputs=[step10_prompt])
                
                # =============================================
                # MOVE UP/DOWN - Swap step contents
                # =============================================
                # Step 1 can only go down
                step1_down.click(
                    fn=swap_steps,
                    inputs=[
                        step1_enabled, step1_name, step1_agent, step1_ptype, step1_template, step1_prompt, step1_desc,
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                    ],
                    outputs=[
                        step1_enabled, step1_name, step1_agent, step1_ptype, step1_template, step1_prompt, step1_desc,
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                    ],
                )
                
                # Step 2
                step2_up.click(
                    fn=swap_steps,
                    inputs=[
                        step1_enabled, step1_name, step1_agent, step1_ptype, step1_template, step1_prompt, step1_desc,
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                    ],
                    outputs=[
                        step1_enabled, step1_name, step1_agent, step1_ptype, step1_template, step1_prompt, step1_desc,
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                    ],
                )
                step2_down.click(
                    fn=swap_steps,
                    inputs=[
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                    ],
                    outputs=[
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                    ],
                )
                
                # Step 3
                step3_up.click(
                    fn=swap_steps,
                    inputs=[
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                    ],
                    outputs=[
                        step2_enabled, step2_name, step2_agent, step2_ptype, step2_template, step2_prompt, step2_desc,
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                    ],
                )
                step3_down.click(
                    fn=swap_steps,
                    inputs=[
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                    ],
                    outputs=[
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                    ],
                )
                
                # Step 4
                step4_up.click(
                    fn=swap_steps,
                    inputs=[
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                    ],
                    outputs=[
                        step3_enabled, step3_name, step3_agent, step3_ptype, step3_template, step3_prompt, step3_desc,
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                    ],
                )
                step4_down.click(
                    fn=swap_steps,
                    inputs=[
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                    ],
                    outputs=[
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                    ],
                )
                
                # Step 5
                step5_up.click(
                    fn=swap_steps,
                    inputs=[
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                    ],
                    outputs=[
                        step4_enabled, step4_name, step4_agent, step4_ptype, step4_template, step4_prompt, step4_desc,
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                    ],
                )
                step5_down.click(
                    fn=swap_steps,
                    inputs=[
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                    ],
                    outputs=[
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                    ],
                )
                
                # Step 6
                step6_up.click(
                    fn=swap_steps,
                    inputs=[
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                    ],
                    outputs=[
                        step5_enabled, step5_name, step5_agent, step5_ptype, step5_template, step5_prompt, step5_desc,
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                    ],
                )
                step6_down.click(
                    fn=swap_steps,
                    inputs=[
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                    ],
                    outputs=[
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                    ],
                )
                
                # Step 7
                step7_up.click(
                    fn=swap_steps,
                    inputs=[
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                    ],
                    outputs=[
                        step6_enabled, step6_name, step6_agent, step6_ptype, step6_template, step6_prompt, step6_desc,
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                    ],
                )
                step7_down.click(
                    fn=swap_steps,
                    inputs=[
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                    ],
                    outputs=[
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                    ],
                )
                
                # Step 8
                step8_up.click(
                    fn=swap_steps,
                    inputs=[
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                    ],
                    outputs=[
                        step7_enabled, step7_name, step7_agent, step7_ptype, step7_template, step7_prompt, step7_desc,
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                    ],
                )
                step8_down.click(
                    fn=swap_steps,
                    inputs=[
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                    ],
                    outputs=[
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                    ],
                )
                
                # Step 9
                step9_up.click(
                    fn=swap_steps,
                    inputs=[
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                    ],
                    outputs=[
                        step8_enabled, step8_name, step8_agent, step8_ptype, step8_template, step8_prompt, step8_desc,
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                    ],
                )
                step9_down.click(
                    fn=swap_steps,
                    inputs=[
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                        step10_enabled, step10_name, step10_agent, step10_ptype, step10_template, step10_prompt, step10_desc,
                    ],
                    outputs=[
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                        step10_enabled, step10_name, step10_agent, step10_ptype, step10_template, step10_prompt, step10_desc,
                    ],
                )
                
                # Step 10 can only go up
                step10_up.click(
                    fn=swap_steps,
                    inputs=[
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                        step10_enabled, step10_name, step10_agent, step10_ptype, step10_template, step10_prompt, step10_desc,
                    ],
                    outputs=[
                        step9_enabled, step9_name, step9_agent, step9_ptype, step9_template, step9_prompt, step9_desc,
                        step10_enabled, step10_name, step10_agent, step10_ptype, step10_template, step10_prompt, step10_desc,
                    ],
                )
                
                # =============================================
                # AUTO-SWITCH TO CUSTOM ON PROMPT EDIT
                # =============================================
                # When user edits the prompt while in template mode, switch to custom
                def switch_to_custom_on_edit(ptype):
                    """Switch to custom mode when prompt is edited in template mode."""
                    if ptype == "template":
                        return "custom"
                    return gr.update()
                
                # Use .input() to detect user keyboard input (not programmatic changes)
                step1_prompt.input(fn=switch_to_custom_on_edit, inputs=[step1_ptype], outputs=[step1_ptype])
                step2_prompt.input(fn=switch_to_custom_on_edit, inputs=[step2_ptype], outputs=[step2_ptype])
                step3_prompt.input(fn=switch_to_custom_on_edit, inputs=[step3_ptype], outputs=[step3_ptype])
                step4_prompt.input(fn=switch_to_custom_on_edit, inputs=[step4_ptype], outputs=[step4_ptype])
                step5_prompt.input(fn=switch_to_custom_on_edit, inputs=[step5_ptype], outputs=[step5_ptype])
                step6_prompt.input(fn=switch_to_custom_on_edit, inputs=[step6_ptype], outputs=[step6_ptype])
                step7_prompt.input(fn=switch_to_custom_on_edit, inputs=[step7_ptype], outputs=[step7_ptype])
                step8_prompt.input(fn=switch_to_custom_on_edit, inputs=[step8_ptype], outputs=[step8_ptype])
                step9_prompt.input(fn=switch_to_custom_on_edit, inputs=[step9_ptype], outputs=[step9_ptype])
                step10_prompt.input(fn=switch_to_custom_on_edit, inputs=[step10_ptype], outputs=[step10_ptype])
                
                # =============================================
                # PIPELINE TABLE SELECTION
                # =============================================
                # All step components for outputs (10 steps x 7 fields = 70 values)
                all_step_outputs = [
                    step1_enabled, step1_name, step1_agent, step1_model, step1_ptype, step1_template, step1_prompt, step1_desc,
                    step2_enabled, step2_name, step2_agent, step2_model, step2_ptype, step2_template, step2_prompt, step2_desc,
                    step3_enabled, step3_name, step3_agent, step3_model, step3_ptype, step3_template, step3_prompt, step3_desc,
                    step4_enabled, step4_name, step4_agent, step4_model, step4_ptype, step4_template, step4_prompt, step4_desc,
                    step5_enabled, step5_name, step5_agent, step5_model, step5_ptype, step5_template, step5_prompt, step5_desc,
                    step6_enabled, step6_name, step6_agent, step6_model, step6_ptype, step6_template, step6_prompt, step6_desc,
                    step7_enabled, step7_name, step7_agent, step7_model, step7_ptype, step7_template, step7_prompt, step7_desc,
                    step8_enabled, step8_name, step8_agent, step8_model, step8_ptype, step8_template, step8_prompt, step8_desc,
                    step9_enabled, step9_name, step9_agent, step9_model, step9_ptype, step9_template, step9_prompt, step9_desc,
                    step10_enabled, step10_name, step10_agent, step10_model, step10_ptype, step10_template, step10_prompt, step10_desc,
                ]
                
                def on_pipeline_row_select(evt: gr.SelectData, table_data):
                    """When a row is clicked, load its details into the visual editor."""
                    if evt.index and len(evt.index) > 0:
                        row_idx = evt.index[0]
                        if row_idx < len(table_data):
                            pipeline_id = table_data.iloc[row_idx, 1]  # ID is in column 1
                            return load_pipeline_to_visual_steps(pipeline_id)
                    # Return empty defaults
                    return load_pipeline_to_visual_steps("")
                
                pipeline_table.select(
                    fn=on_pipeline_row_select,
                    inputs=[pipeline_table],
                    outputs=[
                        pipeline_edit_id,
                        pipeline_edit_name,
                        pipeline_edit_desc,
                        pipeline_edit_pattern,
                        pipeline_edit_keywords,
                        pipeline_edit_weight,
                        pipeline_edit_type,
                    ] + all_step_outputs + all_step_accordions + [visible_steps_count],
                )
                
                pipeline_refresh_btn.click(
                    fn=refresh_pipeline_table,
                    outputs=[pipeline_table],
                )
                
                pipeline_reload_btn.click(
                    fn=reload_pipelines,
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                # All step inputs for create/update (10 steps) - 8 fields per step
                all_step_inputs = [
                    step1_enabled, step1_name, step1_agent, step1_model, step1_ptype, step1_template, step1_prompt, step1_desc,
                    step2_enabled, step2_name, step2_agent, step2_model, step2_ptype, step2_template, step2_prompt, step2_desc,
                    step3_enabled, step3_name, step3_agent, step3_model, step3_ptype, step3_template, step3_prompt, step3_desc,
                    step4_enabled, step4_name, step4_agent, step4_model, step4_ptype, step4_template, step4_prompt, step4_desc,
                    step5_enabled, step5_name, step5_agent, step5_model, step5_ptype, step5_template, step5_prompt, step5_desc,
                    step6_enabled, step6_name, step6_agent, step6_model, step6_ptype, step6_template, step6_prompt, step6_desc,
                    step7_enabled, step7_name, step7_agent, step7_model, step7_ptype, step7_template, step7_prompt, step7_desc,
                    step8_enabled, step8_name, step8_agent, step8_model, step8_ptype, step8_template, step8_prompt, step8_desc,
                    step9_enabled, step9_name, step9_agent, step9_model, step9_ptype, step9_template, step9_prompt, step9_desc,
                    step10_enabled, step10_name, step10_agent, step10_model, step10_ptype, step10_template, step10_prompt, step10_desc,
                ]
                
                pipeline_create_btn.click(
                    fn=create_pipeline_from_visual_steps,
                    inputs=[
                        pipeline_edit_id,
                        pipeline_edit_name,
                        pipeline_edit_desc,
                        pipeline_edit_pattern,
                        pipeline_edit_keywords,
                        pipeline_edit_weight,
                    ] + all_step_inputs,
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                pipeline_update_btn.click(
                    fn=update_pipeline_from_visual_steps,
                    inputs=[
                        pipeline_edit_id,
                        pipeline_edit_name,
                        pipeline_edit_desc,
                        pipeline_edit_pattern,
                        pipeline_edit_keywords,
                        pipeline_edit_weight,
                    ] + all_step_inputs,
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                pipeline_delete_btn.click(
                    fn=delete_pipeline,
                    inputs=[pipeline_edit_id],
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                pipeline_dup_btn.click(
                    fn=duplicate_pipeline,
                    inputs=[pipeline_edit_id, pipeline_dup_id],
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                pipeline_export_btn.click(
                    fn=export_all_pipelines,
                    outputs=[pipeline_export_result],
                )
                
                pipeline_import_btn.click(
                    fn=import_pipelines_from_file,
                    inputs=[pipeline_import_file],
                    outputs=[pipeline_status, pipeline_table, multi_agent_pipeline],
                )
                
                # Generate prompt buttons (with keepalive) for all 10 steps
                step1_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step1_name, step1_desc, pipeline_edit_desc, step1_agent], outputs=[step1_prompt])
                step2_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step2_name, step2_desc, pipeline_edit_desc, step2_agent], outputs=[step2_prompt])
                step3_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step3_name, step3_desc, pipeline_edit_desc, step3_agent], outputs=[step3_prompt])
                step4_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step4_name, step4_desc, pipeline_edit_desc, step4_agent], outputs=[step4_prompt])
                step5_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step5_name, step5_desc, pipeline_edit_desc, step5_agent], outputs=[step5_prompt])
                step6_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step6_name, step6_desc, pipeline_edit_desc, step6_agent], outputs=[step6_prompt])
                step7_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step7_name, step7_desc, pipeline_edit_desc, step7_agent], outputs=[step7_prompt])
                step8_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step8_name, step8_desc, pipeline_edit_desc, step8_agent], outputs=[step8_prompt])
                step9_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step9_name, step9_desc, pipeline_edit_desc, step9_agent], outputs=[step9_prompt])
                step10_gen_btn.click(fn=generate_step_prompt_llm, inputs=[step10_name, step10_desc, pipeline_edit_desc, step10_agent], outputs=[step10_prompt])
            
            
            # =================================================================
            # TAB 3: RAG
            # =================================================================
            with gr.Tab("RAG", id="rag"):
                gr.Markdown("""
                ### RAG Management (Retrieval-Augmented Generation)
                
                Index your documents to automatically enrich your questions with relevant context.
                """)
                
                with gr.Row():
                    # Left column: Indexing
                    with gr.Column():
                        gr.Markdown("#### Indexing")
                        rag_folder_input = gr.Textbox(
                            label="Folder to index",
                            placeholder="~/Documents/code",
                        )
                        rag_recursive = gr.Checkbox(
                            label="Recursive",
                            value=True,
                        )
                        rag_force = gr.Checkbox(
                            label="Force re-indexing",
                            value=False,
                        )
                        rag_index_btn = gr.Button("📥 Index", variant="primary")
                        rag_index_result = gr.Textbox(
                            label="Result",
                            interactive=False,
                        )
                        
                        gr.Markdown("#### Statistics")
                        rag_stats_display = gr.Textbox(
                            label="RAG Status",
                            value=get_rag_status(),
                            interactive=False,
                            lines=2,
                        )
                        rag_stats_refresh = gr.Button("🔄 Refresh")
                    
                    # Right column: Search
                    with gr.Column():
                        gr.Markdown("#### Search Test")
                        rag_search_input = gr.Textbox(
                            label="Test query",
                            placeholder="Ex: diversity index",
                        )
                        rag_search_n = gr.Slider(
                            label="Number of results",
                            minimum=1,
                            maximum=10,
                            step=1,
                            value=5,
                        )
                        rag_search_btn = gr.Button("Search")
                        rag_search_results = gr.Markdown(
                            label="Results",
                        )
                
                # RAG event handlers
                rag_index_btn.click(
                    fn=index_rag_folder,
                    inputs=[rag_folder_input, rag_recursive, rag_force],
                    outputs=[rag_index_result],
                )
                
                rag_stats_refresh.click(
                    fn=refresh_rag_stats,
                    outputs=[rag_stats_display],
                )
                
                rag_search_btn.click(
                    fn=preview_rag_search,
                    inputs=[rag_search_input, rag_search_n],
                    outputs=[rag_search_results],
                )
            
            # =================================================================
            # TAB 4: HISTORY
            # =================================================================
            with gr.Tab("History", id="history"):
                # Pre-load history choices
                initial_history_choices = get_history_choices()
                
                with gr.Row():
                    with gr.Column(scale=1):
                        history_refresh = gr.Button("🔄 Refresh")
                        history_dropdown = gr.Dropdown(
                            label="Recent conversations",
                            choices=initial_history_choices,  # Initialize with existing history
                            interactive=True,
                        )
                        history_search = gr.Textbox(
                            label="Search",
                            placeholder="Keyword...",
                        )
                        history_search_btn = gr.Button("Search")
                        export_btn = gr.Button("📤 Export to Markdown")
                        export_path = gr.Textbox(
                            label="Export path",
                            interactive=False,
                        )
                    
                    with gr.Column(scale=2):
                        history_question = gr.Textbox(
                            label="Original question",
                            lines=3,
                            interactive=False,
                        )
                        history_refined = gr.Textbox(
                            label="Refined question",
                            lines=3,
                            interactive=False,
                        )
                        history_response = gr.Markdown(
                            label="Response",
                        )
                
                # History event handlers
                def refresh_history():
                    choices = get_history_choices()
                    return gr.Dropdown(choices=choices, value=None)
                
                def search_history_entries(query: str):
                    """Search history by keyword."""
                    if not query or not query.strip():
                        # If empty query, return all recent entries
                        choices = get_history_choices()
                        return gr.Dropdown(choices=choices, value=None)
                    
                    try:
                        # Use history.search() method
                        entries = history.search(query.strip())
                        choices = []
                        for entry in entries[:20]:  # Limit results
                            label = entry.question[:50] if entry.question else ""
                            if len(entry.question) > 50:
                                label += "..."
                            timestamp_short = entry.timestamp[:10] if entry.timestamp else ""
                            display_label = f"[{timestamp_short}] {label}"
                            choices.append((display_label, entry.id))
                        
                        if not choices:
                            return gr.Dropdown(choices=[("No results", "")], value=None)
                        return gr.Dropdown(choices=choices, value=None)
                    except Exception as e:
                        logger.error(f"Search error: {e}")
                        return gr.Dropdown(choices=[("Search error", "")], value=None)
                
                history_refresh.click(
                    fn=refresh_history,
                    outputs=[history_dropdown],
                )
                
                history_search_btn.click(
                    fn=search_history_entries,
                    inputs=[history_search],
                    outputs=[history_dropdown],
                )
                
                history_dropdown.change(
                    fn=load_history_entry,
                    inputs=[history_dropdown],
                    outputs=[history_question, history_refined, history_response],
                )
                
                export_btn.click(
                    fn=export_history_markdown,
                    inputs=[history_dropdown],
                    outputs=[export_path],
                )
            
            # =================================================================
            # TAB 5: INFO
            # =================================================================
            with gr.Tab("[i] Info", id="info"):
                gr.Markdown(f"""
                ### 🧅 Opti-Oignon - Local LLM Optimization
                
                **Version:** 1.0
                
                **Available Models:** {len(available_models)}
                
                **Features:**
                - Automatic task detection
                - Intelligent model routing
                - RAG document enrichment
                - Multi-agent orchestration
                - Quick presets
                - Conversation history
                
                **System Status:**
                - RAG: {'[OK] Active' if RAG_AVAILABLE else '[x] Not installed'}
                - Multi-Agent: {'[OK] Active' if MULTI_AGENT_AVAILABLE else '[x] Not installed'}
                
                **Author:** Léon Brouillé
                
                ---
                
                ### Keyboard Shortcuts
                - `Ctrl+Enter`: Send question
                - `Escape`: Cancel generation
                
                ### Tips
                - The system automatically detects your task type (R code, Python, debug, writing...)
                - Responses match your language: ask in French → answer in French
                - Use RAG to enrich your questions with your personal documents
                - Multi-agent mode orchestrates multiple models for complex tasks
                """)
    
    return app


# =============================================================================
# ENTRY POINT
# =============================================================================

def launch(port: int = 7860, share: bool = False, debug: bool = False):
    """
    Launch the Gradio interface.
    
    Args:
        port: Server port (default: 7860)
        share: Create public link (default: False)
        debug: Enable debug mode (default: False)
    """
    app = create_app()
    
    # Display startup message with benchmark tips
    print()
    print("=" * 60)
    print("  OPTI-OIGNON - Interface launched")
    print("=" * 60)
    print()
    print(f"  Local:   http://localhost:{port}/")
    print(f"  Network: http://0.0.0.0:{port}/")
    if share:
        print("  Public:  (Gradio link generating...)")
    print()
    print("-" * 60)
    print("  [TIP] Benchmark your models for optimal routing:")
    print("        opti-oignon benchmark --interactive --confirm")
    print()
    print("  [TIP] Estimate benchmark time:")
    print("        opti-oignon benchmark --estimate")
    print("-" * 60)
    print()
    print("  Ctrl+C to stop")
    print()
    print("=" * 60)
    print()
    
    app.launch(
        server_port=port,
        share=share,
        show_error=debug,
        quiet=not debug,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Opti-Oignon UI")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    args = parser.parse_args()
    launch(args.port, args.share, args.debug)
