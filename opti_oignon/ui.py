#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UI - OPTI-OIGNON 1.0 + RAG + MULTI-AGENT
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
"""

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
    """Return pipeline choices for dropdown."""
    choices = [("Auto-detection", "auto")]
    
    if not MULTI_AGENT_AVAILABLE:
        return choices
    
    try:
        orch = get_orchestrator()
        for p in orch.list_pipelines():
            emoji = p.get("emoji", "[>]")
            name = p.get("name", p.get("id"))
            choices.append((f"{emoji} {name}", p["id"]))
    except Exception:
        pass
    
    return choices


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
        f"💻 **Language:** {analysis.language.value}",
        f"Complexity: {analysis.complexity.value}",
    ]
    
    if analysis.keywords:
        lines.append(f"🔑 **Keywords:** {', '.join(analysis.keywords)}")
    
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
) -> Generator[Tuple[str, str, str, str, str], None, None]:
    """
    Process a question with all options.
    
    Yields:
        (status, analysis, routing, agent_steps/rag_sources, response)
    """
    if not question.strip():
        yield "Enter a question...", "", "", "", ""
        return
    
    # Handle file upload
    if file_upload is not None:
        try:
            content, error = safe_read_file(file_upload.name)
            if error:
                yield f"[ERR] File error: {error}", "", "", "", ""
                return
            document = content
        except Exception as e:
            yield f"[ERR] File error: {str(e)}", "", "", "", ""
            return
    
    # Multi-agent mode
    if use_multi_agent and MULTI_AGENT_AVAILABLE and is_multi_agent_enabled():
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


def refresh_multi_agent_stats() -> str:
    """Refresh multi-agent stats."""
    return get_multi_agent_status()


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
) -> Tuple[str, List[List[str]]]:
    """Create a new preset (with detection_weight)."""
    # Validate
    if not preset_id or not preset_id.strip():
        return "[ERR] Preset ID is required", get_preset_table_data()
    
    if not preset_manager.validate_preset_id(preset_id):
        return "[ERR] Invalid ID (use letters, numbers, _ or -)", get_preset_table_data()
    
    if preset_manager.get(preset_id):
        return f"[ERR] Preset '{preset_id}' already exists", get_preset_table_data()
    
    if not name or not name.strip():
        return "[ERR] Name is required", get_preset_table_data()
    
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
        return f"[OK] Preset '{preset_id}' created ({len(keywords)} keywords, weight={detection_weight})", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Creation failed: {str(e)}", get_preset_table_data()


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
) -> Tuple[str, List[List[str]]]:
    """Update an existing preset (with detection_weight)."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data()
    
    if not preset_manager.get(preset_id):
        return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data()
    
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
        return f"[OK] Preset '{preset_id}' updated", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Update failed: {str(e)}", get_preset_table_data()


def delete_preset(preset_id: str) -> Tuple[str, List[List[str]]]:
    """Delete a preset."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data()
    
    if preset_id == "default":
        return "[ERR] Cannot delete the default preset", get_preset_table_data()
    
    try:
        if preset_manager.delete(preset_id):
            return f"[OK] Preset '{preset_id}' deleted", get_preset_table_data()
        else:
            return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Delete failed: {str(e)}", get_preset_table_data()


def duplicate_preset(preset_id: str, new_id: str) -> Tuple[str, List[List[str]]]:
    """Duplicate a preset."""
    if not preset_id:
        return "[ERR] No preset selected", get_preset_table_data()
    
    if not new_id or not new_id.strip():
        return "[ERR] New ID is required", get_preset_table_data()
    
    if not preset_manager.validate_preset_id(new_id):
        return "[ERR] Invalid new ID", get_preset_table_data()
    
    try:
        source = preset_manager.get(preset_id)
        if not source:
            return f"[ERR] Preset '{preset_id}' not found", get_preset_table_data()
        
        new_preset = preset_manager.duplicate(preset_id, new_id.strip(), f"Copy of {source.name}")
        if new_preset:
            return f"[OK] Preset duplicated as '{new_id}'", get_preset_table_data()
        else:
            return "[ERR] Duplication failed", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Duplication failed: {str(e)}", get_preset_table_data()


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


def reload_presets() -> Tuple[str, List[List[str]]]:
    """Reload presets from config files."""
    try:
        preset_manager.reload()
        stats = preset_manager.get_stats()
        avg_weight = stats.get('avg_detection_weight', 0.5)
        return f"[OK] Reloaded {stats['total']} presets ({stats['total_keywords']} keywords, avg weight={avg_weight:.2f})", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Reload failed: {str(e)}", get_preset_table_data()


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


def import_presets_from_file(file) -> Tuple[str, List[List[str]]]:
    """Import presets from an uploaded YAML file."""
    if file is None:
        return "[ERR] No file selected", get_preset_table_data()
    
    try:
        filepath = Path(file.name) if hasattr(file, 'name') else Path(file)
        imported = preset_manager.import_preset(filepath)
        if imported:
            return f"[OK] Imported {len(imported)} presets: {', '.join(imported)}", get_preset_table_data()
        else:
            return "[ERR] No valid presets found in file", get_preset_table_data()
    except Exception as e:
        return f"[ERR] Import failed: {str(e)}", get_preset_table_data()


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
                        
                        # === MULTI-AGENT SECTION ===
                        with gr.Accordion("Multi-Agent", open=True):
                            multi_agent_status_display = gr.Textbox(
                                label="Status",
                                value=get_multi_agent_status(),
                                interactive=False,
                                lines=1,
                            )
                            use_multi_agent = gr.Checkbox(
                                label="Enable Multi-Agent (multi-model orchestration)",
                                value=MULTI_AGENT_AVAILABLE and is_multi_agent_enabled(),
                                interactive=MULTI_AGENT_AVAILABLE,
                            )
                            multi_agent_pipeline = gr.Dropdown(
                                label="Pipeline",
                                choices=pipeline_choices,
                                value="auto",
                                visible=MULTI_AGENT_AVAILABLE,
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
                                value=RAG_AVAILABLE,
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
                    outputs=[multi_agent_status_display],
                )
                
                # Live analysis
                question_input.change(
                    fn=analyze_question,
                    inputs=[question_input, document_input],
                    outputs=[analysis_output, routing_output],
                )
            
            # =================================================================
            # TAB 2: MULTI-AGENT
            # =================================================================
            with gr.Tab("Multi-Agent", id="multi_agent"):
                gr.Markdown("""
                ### Multi-Agent Orchestration
                
                Use multiple specialized models in sequence for complex tasks.
                """)
                
                with gr.Row():
                    # Left column: Configuration
                    with gr.Column():
                        gr.Markdown("#### Configuration")
                        ma_toggle = gr.Checkbox(
                            label="Enable Multi-Agent",
                            value=MULTI_AGENT_AVAILABLE and is_multi_agent_enabled(),
                            interactive=MULTI_AGENT_AVAILABLE,
                        )
                        ma_status = gr.Textbox(
                            label="Status",
                            value=get_multi_agent_status(),
                            interactive=False,
                        )
                        ma_refresh = gr.Button("🔄 Refresh")
                        
                        gr.Markdown("#### Available Pipelines")
                        
                        if MULTI_AGENT_AVAILABLE:
                            try:
                                status = multi_agent_status()
                                pipelines_data = []
                                for p in status.get("available_pipelines", []):
                                    pipelines_data.append([
                                        p.get("emoji", ""),
                                        p.get("id", ""),
                                        p.get("name", ""),
                                        p.get("description", "")[:50],
                                    ])
                            except:
                                pipelines_data = []
                        else:
                            pipelines_data = []
                        
                        pipelines_table = gr.Dataframe(
                            headers=["", "ID", "Name", "Description"],
                            value=pipelines_data,
                            interactive=False,
                        )
                    
                    # Right column: Agents
                    with gr.Column():
                        gr.Markdown("#### Available Agents")
                        
                        if MULTI_AGENT_AVAILABLE:
                            try:
                                status = multi_agent_status()
                                agents_data = []
                                for agent_id in status.get("available_agents", []):
                                    agents_data.append([agent_id])
                            except:
                                agents_data = []
                        else:
                            agents_data = []
                        
                        agents_table = gr.Dataframe(
                            headers=["Agent"],
                            value=agents_data,
                            interactive=False,
                        )
                        
                        gr.Markdown("""
                        #### How it works
                        
                        The system orchestrates multiple specialized models in sequence:
                        
                        ---
                        
                        **Pipeline: Data Analysis** (`data_analysis`)
                        1. `deepseek-r1:32b` - Understands data and plans analysis
                        2. `qwen3-coder:30b` - Writes R/Python analysis code
                        3. `deepseek-r1:32b` - Verifies code and logic
                        4. `qwen3:32b` - Interprets results
                        
                        ---
                        
                        **Pipeline: Complex Debug** (`debug`)
                        1. `deepseek-r1:32b` - Analyzes error in detail
                        2. `deepseek-r1:32b` - Diagnoses possible causes  
                        3. `qwen3-coder:30b` - Proposes fixes
                        4. `deepseek-r1:32b` - Evaluates each solution
                        5. `qwen3-coder:30b` - Implements best solution
                        
                        ---
                        
                        **Pipeline: Scientific Writing** (`scientific_writing`)
                        1. `deepseek-r1:32b` - Defines document structure
                        2. `qwen3:32b` - Writes each section
                        3. `deepseek-r1:32b` - Verifies coherence
                        4. `qwen3:32b` - Improves style
                        
                        ---
                        
                        **Pipeline: Code + Tests** (`code_with_tests`)
                        1. `deepseek-r1:32b` - Extracts specifications
                        2. `qwen3-coder:30b` - Generates code
                        3. `qwen3-coder:30b` - Creates unit tests
                        4. `deepseek-r1:32b` - Verifies code and tests
                        
                        ---
                        
                        **Pipeline: Quick Response** (`quick`)
                        - Single auto-selected agent
                        - No multi-model orchestration
                        """)
                
                # Events
                ma_toggle.change(
                    fn=toggle_multi_agent,
                    inputs=[ma_toggle],
                    outputs=[ma_status],
                )
                
                ma_refresh.click(
                    fn=refresh_multi_agent_stats,
                    outputs=[ma_status],
                )
            
            # =================================================================
            # TAB: PRESETS MANAGEMENT (ENHANCED)
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
                    outputs=[preset_status, preset_table],
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
                    outputs=[preset_status, preset_table],
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
                    outputs=[preset_status, preset_table],
                )
                
                preset_delete_btn.click(
                    fn=delete_preset,
                    inputs=[preset_edit_id],
                    outputs=[preset_status, preset_table],
                )
                
                preset_dup_btn.click(
                    fn=duplicate_preset,
                    inputs=[preset_edit_id, preset_dup_id],
                    outputs=[preset_status, preset_table],
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
                    outputs=[preset_status, preset_table],
                )
            
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
                        history_search_btn = gr.Button("🔍 Search")
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
            with gr.Tab("ℹ️ Info", id="info"):
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
                
                **Author:** Léon
                
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
