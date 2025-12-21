#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opti-Oignon 1.0 - CLI Entry Point
=================================

Complete command-line interface for the Opti-Oignon local LLM optimization suite.

Usage:
    opti-oignon ui [--port PORT] [--share] [--debug]
    opti-oignon benchmark [--quick] [--interactive] [--confirm]
    opti-oignon rag index <folder> [--recursive] [--force]
    opti-oignon rag search <query> [--n-results N]
    opti-oignon config show
    opti-oignon config set <key> <value>
    opti-oignon ask "question" [--file FILE]
    opti-oignon info models|stats|config
    opti-oignon --version

Author: Léon
"""

import argparse
import sys
from pathlib import Path

__version__ = "1.0.0"


# =============================================================================
# UI COMMAND
# =============================================================================

def cmd_ui(args):
    """Launch the Gradio interface."""
    from .ui import launch
    launch(
        port=args.port,
        share=args.share,
        debug=args.debug
    )


# =============================================================================
# BENCHMARK COMMAND
# =============================================================================

def cmd_benchmark(args):
    """Run model benchmarks."""
    try:
        from .routing.benchmark import ModelBenchmark, DEFAULT_MODELS, QUICK_MODELS, BENCHMARK_TASKS
    except ImportError:
        print("[ERR] Benchmark module not found.")
        print("      Make sure opti_oignon.routing.benchmark exists.")
        sys.exit(1)
    
    # Determine models to test
    models = None
    if args.models:
        models = [m.strip() for m in args.models.split(",")]
    elif args.quick:
        models = QUICK_MODELS
    
    # Create benchmark instance
    benchmark = ModelBenchmark(
        output_dir=args.output_dir,
        timeout=args.timeout,
        temperature=args.temperature,
        interactive=args.interactive,
    )
    
    # Get available models for estimation
    available = benchmark._get_available_models()
    models_to_test = models if models else [m for m in DEFAULT_MODELS if m in available]
    models_to_test = [m for m in models_to_test if m in available]
    
    tasks_to_run = args.tasks.split(",") if args.tasks else list(BENCHMARK_TASKS.keys())
    total_tests = len(models_to_test) * len(tasks_to_run)
    estimated_time_min = (total_tests * 90) / 60
    estimated_time_quick = (total_tests * 45) / 60
    
    # Estimate mode
    if args.estimate:
        print("\n" + "=" * 60)
        print("[>] BENCHMARK ESTIMATION")
        print("=" * 60)
        print(f"\n[>] Models to test: {len(models_to_test)}")
        for m in models_to_test:
            print(f"    - {m}")
        print(f"\n[>] Tasks: {len(tasks_to_run)}")
        for t in tasks_to_run:
            print(f"    - {t}")
        print(f"\n[>] Total tests: {total_tests}")
        print(f"[>] Estimated time: {estimated_time_quick:.0f} - {estimated_time_min:.0f} minutes")
        print(f"[>] GPU usage: INTENSIVE for entire duration")
        print(f"[>] Mode: {'Interactive' if args.interactive else 'Automatic'}")
        print("\n[>] To run: opti-oignon benchmark --confirm")
        if not args.quick:
            print("[>] For quick test: add --quick flag")
        return
    
    # Check confirmation
    if not args.confirm:
        print("\n" + "=" * 60)
        print("[!] CONFIRMATION REQUIRED")
        print("=" * 60)
        print(f"""
The benchmark will:
  - Test {len(models_to_test)} models on {len(tasks_to_run)} tasks
  - Run {total_tests} tests total
  - Take approximately {estimated_time_quick:.0f}-{estimated_time_min:.0f} minutes
  - Use GPU INTENSIVELY

To avoid interfering with your daily work,
the benchmark NEVER runs automatically.

Options:
  --estimate       See detailed estimate without running
  --confirm        Confirm and run the benchmark
  --quick          Quick version (3 models only)
  --interactive    Enable user scoring
  --update-config  Auto-update routing config
""")
        print("[ERR] Benchmark cancelled (add --confirm to run)")
        sys.exit(0)
    
    # Run benchmark
    print("\n[>] Starting benchmark...")
    results = benchmark.run(
        models=models,
        tasks=tasks_to_run if args.tasks else None,
        verbose=not args.quiet,
    )
    
    if not results:
        print("[ERR] No results")
        sys.exit(1)
    
    # Save results
    benchmark.save_results()
    
    # Check regressions
    if args.check_regression:
        print("\n[>] Checking for regressions...")
        regression_info = benchmark.check_regression()
        
        if regression_info.get("regressions"):
            print("\n[!] REGRESSIONS DETECTED:")
            for reg in regression_info["regressions"]:
                print(f"    - {reg['model']} on {reg['task']}: {reg['previous_score']} -> {reg['current_score']} ({reg['delta']:+d})")
        
        if regression_info.get("improvements"):
            print("\n[OK] IMPROVEMENTS:")
            for imp in regression_info["improvements"]:
                print(f"    - {imp['model']} on {imp['task']}: {imp['previous_score']} -> {imp['current_score']} ({imp['delta']:+d})")
    
    # Update config
    if args.update_config:
        print("\n[>] Updating configuration...")
        benchmark.update_config(dry_run=False)
    else:
        benchmark.update_config(dry_run=True)


# =============================================================================
# RAG COMMANDS
# =============================================================================

def cmd_rag(args):
    """RAG index and search commands."""
    try:
        from .rag import ContexteurRAGIntegration
    except ImportError:
        print("[ERR] RAG module not available.")
        print("      Make sure opti_oignon.rag exists and dependencies are installed.")
        sys.exit(1)
    
    rag = ContexteurRAGIntegration()
    
    if args.rag_action == "index":
        # Index a folder
        folder = Path(args.folder)
        if not folder.exists():
            print(f"[ERR] Folder not found: {folder}")
            sys.exit(1)
        
        print(f"\n[>] Indexing folder: {folder}")
        print(f"    Recursive: {args.recursive}")
        print(f"    Force reindex: {args.force}")
        
        try:
            stats = rag.index_folder(
                folder_path=str(folder),
                recursive=args.recursive,
                force=args.force,
            )
            print(f"\n[OK] Indexing complete!")
            print(f"    Files processed: {stats.get('files_processed', 0)}")
            print(f"    Chunks created: {stats.get('chunks_created', 0)}")
            print(f"    Errors: {stats.get('errors', 0)}")
        except Exception as e:
            print(f"[ERR] Indexing failed: {e}")
            sys.exit(1)
    
    elif args.rag_action == "search":
        # Search the index
        if not args.query:
            print("[ERR] Query required: opti-oignon rag search <query>")
            sys.exit(1)
        
        print(f"\n[>] Searching for: {args.query}")
        print(f"    Max results: {args.n_results}")
        
        try:
            results = rag.search(args.query, n_results=args.n_results)
            
            if not results:
                print("\n[!] No results found.")
                return
            
            print(f"\n[OK] Found {len(results)} results:\n")
            for i, result in enumerate(results, 1):
                source = result.get("source", "unknown")
                score = result.get("score", 0)
                content = result.get("content", "")[:200]
                print(f"{i}. [{score:.3f}] {source}")
                print(f"   {content}...")
                print()
        except Exception as e:
            print(f"[ERR] Search failed: {e}")
            sys.exit(1)
    
    elif args.rag_action == "status":
        # Show RAG status
        try:
            status = rag.get_status()
            print("\n[>] RAG System Status:\n")
            print(f"    Indexed files: {status.get('indexed_files', 0)}")
            print(f"    Total chunks: {status.get('total_chunks', 0)}")
            print(f"    Collection: {status.get('collection_name', 'default')}")
            print(f"    Embedding model: {status.get('embedding_model', 'unknown')}")
        except Exception as e:
            print(f"[ERR] Could not get status: {e}")
            sys.exit(1)
    
    elif args.rag_action == "clear":
        # Clear the index
        if not args.confirm:
            print("[!] This will delete all indexed documents.")
            print("    Add --confirm to proceed.")
            sys.exit(0)
        
        try:
            rag.clear_index()
            print("[OK] Index cleared.")
        except Exception as e:
            print(f"[ERR] Clear failed: {e}")
            sys.exit(1)


# =============================================================================
# CONFIG COMMANDS
# =============================================================================

def cmd_config(args):
    """Configuration management."""
    try:
        from .config import config
    except ImportError:
        print("[ERR] Config module not found.")
        sys.exit(1)
    
    if args.config_action == "show":
        print("\n[>] Current Configuration:\n")
        
        print("   Models by type:")
        for model_type in ["code", "reasoning", "general", "quick"]:
            try:
                primary = config.get_model(model_type, "primary")
                print(f"      {model_type}: {primary}")
            except Exception:
                print(f"      {model_type}: (not configured)")
        
        print("\n   Temperatures:")
        for task in ["code", "debug", "reasoning", "writing", "general"]:
            try:
                temp = config.get_temperature(task)
                print(f"      {task}: {temp}")
            except Exception:
                print(f"      {task}: (default)")
        
        print("\n   System settings:")
        try:
            settings = config.get_system_settings()
            for key, value in settings.items():
                print(f"      {key}: {value}")
        except Exception:
            print("      (no system settings available)")
    
    elif args.config_action == "set":
        if not args.key or args.value is None:
            print("[ERR] Usage: opti-oignon config set <key> <value>")
            sys.exit(1)
        
        try:
            config.set(args.key, args.value)
            print(f"[OK] Set {args.key} = {args.value}")
        except Exception as e:
            print(f"[ERR] Failed to set config: {e}")
            sys.exit(1)
    
    elif args.config_action == "get":
        if not args.key:
            print("[ERR] Usage: opti-oignon config get <key>")
            sys.exit(1)
        
        try:
            value = config.get(args.key)
            print(f"{args.key} = {value}")
        except Exception as e:
            print(f"[ERR] Key not found: {args.key}")
            sys.exit(1)
    
    elif args.config_action == "reset":
        if not args.confirm:
            print("[!] This will reset all configuration to defaults.")
            print("    Add --confirm to proceed.")
            sys.exit(0)
        
        try:
            config.reset()
            print("[OK] Configuration reset to defaults.")
        except Exception as e:
            print(f"[ERR] Reset failed: {e}")
            sys.exit(1)


# =============================================================================
# ASK COMMAND (Quick CLI mode)
# =============================================================================

def cmd_ask(args):
    """CLI mode: quickly ask a question."""
    try:
        from . import analyzer, router, executor
    except ImportError as e:
        print(f"[ERR] Required module not found: {e}")
        sys.exit(1)
    
    # Load document if provided
    document = None
    if args.file:
        file_path = Path(args.file)
        if file_path.exists():
            try:
                document = file_path.read_text(encoding="utf-8")
                print(f"[DOC] Document loaded: {file_path.name} ({len(document)} chars)")
            except Exception as e:
                print(f"[!] Error reading file: {e}")
    
    # Pipeline
    print(f"\n[...] Analyzing question...")
    analysis = analyzer.analyze(args.question, document, args.task)
    print(f"   Detected type: {analysis.task_type} (confidence: {analysis.confidence:.0%})")
    
    print(f"\n[>] Selecting model...")
    routing = router.route(analysis, args.priority, args.model)
    print(f"   Model: {routing.model}")
    print(f"   Temperature: {routing.temperature}")
    print(f"   Reason: {routing.explanation}")
    
    print(f"\n[GEN] Generating response...\n")
    print("-" * 60)
    
    try:
        result = executor.execute(
            args.question, 
            routing, 
            document, 
            refine=args.refine
        )
        
        full_response = ""
        for chunk in result:
            print(chunk, end="", flush=True)
            full_response += chunk
        
        print("\n" + "-" * 60)
        
        # Statistics
        if args.verbose:
            print(f"\n[STATS] Statistics:")
            print(f"   Response length: {len(full_response)} chars")
            print(f"   Words: {len(full_response.split())} words")
            
    except KeyboardInterrupt:
        print("\n\n[!] Generation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERR] Error: {e}")
        sys.exit(1)


# =============================================================================
# PRESETS COMMAND
# =============================================================================

def cmd_presets(args):
    """Preset management."""
    try:
        from .presets import preset_manager
    except ImportError:
        print("[ERR] Presets module not found.")
        sys.exit(1)
    
    if args.preset_action == "list":
        presets = preset_manager.get_ordered()
        print("\n[LIST] Available presets:\n")
        print(f"{'ID':<20} {'Name':<25} {'Model':<25} {'Task':<15}")
        print("-" * 85)
        for p in presets:
            print(f"{p.id:<20} {p.name:<25} {p.model:<25} {p.task:<15}")
        print(f"\nTotal: {len(presets)} presets")
        
    elif args.preset_action == "show":
        if not args.preset_id:
            print("[ERR] Specify preset ID: opti-oignon presets show <preset_id>")
            sys.exit(1)
        preset = preset_manager.get(args.preset_id)
        if preset:
            print(f"\n[>] Preset: {preset.name}\n")
            print(f"   ID:          {preset.id}")
            print(f"   Icon:        {preset.icon}")
            print(f"   Task:        {preset.task}")
            print(f"   Model:       {preset.model}")
            print(f"   Temperature: {preset.temperature}")
            print(f"   Variant:     {preset.prompt_variant}")
            print(f"   Tags:        {', '.join(preset.tags)}")
        else:
            print(f"[ERR] Preset '{args.preset_id}' not found")
            sys.exit(1)


# =============================================================================
# INFO COMMAND
# =============================================================================

def cmd_info(args):
    """Display system information."""
    
    if args.info_type == "models":
        try:
            import ollama
            models = ollama.list()
            
            # Handle both possible API response formats
            if isinstance(models, dict):
                model_list = models.get("models", [])
            else:
                model_list = getattr(models, 'models', models)
                
            print("\n[MODELS] Available Ollama models:\n")
            print(f"{'Name':<35} {'Size':<12} {'Modified':<20}")
            print("-" * 70)
            
            for m in model_list:
                if isinstance(m, dict):
                    name = m.get("name", "?")
                    size = m.get("size", 0)
                    modified = m.get("modified_at", "?")[:10] if m.get("modified_at") else "?"
                else:
                    name = getattr(m, "name", getattr(m, "model", "?"))
                    size = getattr(m, "size", 0)
                    modified = str(getattr(m, "modified_at", "?"))[:10]
                
                size_gb = f"{size / 1e9:.1f} GB" if size else "?"
                print(f"{name:<35} {size_gb:<12} {modified:<20}")
                
            print(f"\nTotal: {len(model_list)} models")
            
        except Exception as e:
            print(f"[ERR] Ollama connection error: {e}")
            print("      Make sure Ollama is running: ollama serve")
            sys.exit(1)
            
    elif args.info_type == "stats":
        try:
            from .history import history
            stats = history.get_stats()
            
            print("\n[STATS] Usage statistics:\n")
            print(f"   Total conversations: {stats.get('total', 0)}")
            print(f"   Average rating: {stats.get('average_rating', 0):.1f}/5")
            
            print("\n   By task:")
            for task, count in stats.get("by_task", {}).items():
                print(f"      {task}: {count}")
                
            print("\n   By model:")
            for model, count in stats.get("by_model", {}).items():
                print(f"      {model}: {count}")
        except Exception as e:
            print(f"[ERR] Could not load statistics: {e}")
            sys.exit(1)
            
    elif args.info_type == "config":
        # Delegate to config show
        args.config_action = "show"
        cmd_config(args)
    
    elif args.info_type == "version":
        print(f"Opti-Oignon {__version__}")
        print("Local LLM Optimization Suite")
        print("Author: Léon")


# =============================================================================
# EXPORT COMMAND
# =============================================================================

def cmd_export(args):
    """Export conversation history."""
    try:
        from .history import history
    except ImportError:
        print("[ERR] History module not found.")
        sys.exit(1)
    
    entries = history.get_recent(args.count)
    if not entries:
        print("[ERR] No conversations to export")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else Path(f"export_opti_oignon_{len(entries)}.md")
    
    history.export_markdown(entries, output_path)
    print(f"[OK] {len(entries)} conversations exported to {output_path}")


# =============================================================================
# MAIN PARSER
# =============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="opti-oignon",
        description="Opti-Oignon - Intelligent optimizer for local LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  opti-oignon ui                              Launch web interface
  opti-oignon ui --port 8080 --share          Launch with custom port and public link
  opti-oignon benchmark --estimate            See benchmark time estimate
  opti-oignon benchmark --quick --confirm     Run quick benchmark (3 models)
  opti-oignon benchmark --interactive --confirm  Full benchmark with user scoring
  opti-oignon rag index ./docs --recursive    Index documents folder
  opti-oignon rag search "Shannon diversity"  Search indexed documents
  opti-oignon config show                     Show current configuration
  opti-oignon config set code.primary qwen3-coder:30b
  opti-oignon ask "How to calculate Shannon in R?"
  opti-oignon info models                     List available Ollama models
  opti-oignon presets list                    List available presets

Documentation: https://github.com/your-username/opti-oignon
        """
    )
    
    parser.add_argument(
        "--version", "-v",
        action="version",
        version=f"Opti-Oignon {__version__}"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # -------------------------------------------------------------------------
    # Command: ui
    # -------------------------------------------------------------------------
    ui_parser = subparsers.add_parser(
        "ui", 
        help="Launch Gradio web interface",
        description="Start the Opti-Oignon web interface for interactive use."
    )
    ui_parser.add_argument(
        "--port", "-p", 
        type=int, 
        default=7860, 
        help="Server port (default: 7860)"
    )
    ui_parser.add_argument(
        "--share", "-s", 
        action="store_true", 
        help="Create public Gradio link"
    )
    ui_parser.add_argument(
        "--debug", "-d", 
        action="store_true", 
        help="Enable debug mode"
    )
    
    # -------------------------------------------------------------------------
    # Command: benchmark
    # -------------------------------------------------------------------------
    bench_parser = subparsers.add_parser(
        "benchmark",
        help="Run model benchmarks",
        description="Benchmark Ollama models on standardized tasks."
    )
    bench_parser.add_argument(
        "--confirm", "-y",
        action="store_true",
        help="REQUIRED: Confirm benchmark launch (uses GPU intensively)"
    )
    bench_parser.add_argument(
        "--estimate",
        action="store_true",
        help="Show time/resource estimate without running"
    )
    bench_parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: prompt for user scores after each test"
    )
    bench_parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick benchmark (3 main models only)"
    )
    bench_parser.add_argument(
        "--models", "-m",
        help="Specific models to test (comma-separated)"
    )
    bench_parser.add_argument(
        "--tasks", "-t",
        help="Specific tasks to test (comma-separated)"
    )
    bench_parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout per test in seconds (default: 300)"
    )
    bench_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for tests (default: 0.7)"
    )
    bench_parser.add_argument(
        "--output-dir", "-o",
        default="routing/benchmarks",
        help="Output directory (default: routing/benchmarks)"
    )
    bench_parser.add_argument(
        "--update-config",
        action="store_true",
        help="Auto-update router config after benchmark"
    )
    bench_parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Compare with previous benchmark results"
    )
    bench_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode (less output)"
    )
    
    # -------------------------------------------------------------------------
    # Command: rag
    # -------------------------------------------------------------------------
    rag_parser = subparsers.add_parser(
        "rag",
        help="RAG (Retrieval Augmented Generation) commands",
        description="Index and search documents for context enrichment."
    )
    rag_subparsers = rag_parser.add_subparsers(dest="rag_action", help="RAG actions")
    
    # rag index
    rag_index = rag_subparsers.add_parser(
        "index",
        help="Index a folder for RAG",
        description="Index documents in a folder for retrieval."
    )
    rag_index.add_argument(
        "folder",
        help="Folder path to index"
    )
    rag_index.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Index subfolders recursively"
    )
    rag_index.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force reindex even if already indexed"
    )
    
    # rag search
    rag_search = rag_subparsers.add_parser(
        "search",
        help="Search indexed documents",
        description="Search the RAG index for relevant content."
    )
    rag_search.add_argument(
        "query",
        help="Search query"
    )
    rag_search.add_argument(
        "--n-results", "-n",
        type=int,
        default=5,
        help="Number of results (default: 5)"
    )
    
    # rag status
    rag_status = rag_subparsers.add_parser(
        "status",
        help="Show RAG index status"
    )
    
    # rag clear
    rag_clear = rag_subparsers.add_parser(
        "clear",
        help="Clear the RAG index"
    )
    rag_clear.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm index deletion"
    )
    
    # -------------------------------------------------------------------------
    # Command: config
    # -------------------------------------------------------------------------
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management",
        description="View and modify Opti-Oignon configuration."
    )
    config_subparsers = config_parser.add_subparsers(dest="config_action", help="Config actions")
    
    # config show
    config_show = config_subparsers.add_parser(
        "show",
        help="Show current configuration"
    )
    
    # config set
    config_set = config_subparsers.add_parser(
        "set",
        help="Set a configuration value"
    )
    config_set.add_argument("key", help="Configuration key (e.g., code.primary)")
    config_set.add_argument("value", help="Value to set")
    
    # config get
    config_get = config_subparsers.add_parser(
        "get",
        help="Get a configuration value"
    )
    config_get.add_argument("key", help="Configuration key")
    
    # config reset
    config_reset = config_subparsers.add_parser(
        "reset",
        help="Reset configuration to defaults"
    )
    config_reset.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm reset"
    )
    
    # -------------------------------------------------------------------------
    # Command: ask
    # -------------------------------------------------------------------------
    ask_parser = subparsers.add_parser(
        "ask",
        help="Ask a question via CLI",
        description="Quick CLI mode to ask a question without the web interface."
    )
    ask_parser.add_argument(
        "question",
        help="The question to ask"
    )
    ask_parser.add_argument(
        "--file", "-f",
        help="Document/code file to include as context"
    )
    ask_parser.add_argument(
        "--task", "-t",
        help="Force task type (code, debug, reasoning, etc.)"
    )
    ask_parser.add_argument(
        "--model", "-m",
        help="Force specific model"
    )
    ask_parser.add_argument(
        "--priority",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Priority mode (default: balanced)"
    )
    ask_parser.add_argument(
        "--refine", "-r",
        action="store_true",
        help="Refine the question before sending"
    )
    ask_parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed statistics"
    )
    
    # -------------------------------------------------------------------------
    # Command: presets
    # -------------------------------------------------------------------------
    presets_parser = subparsers.add_parser(
        "presets",
        help="Preset management",
        description="List and view available presets."
    )
    presets_parser.add_argument(
        "preset_action",
        choices=["list", "show"],
        help="Action to perform"
    )
    presets_parser.add_argument(
        "preset_id",
        nargs="?",
        help="Preset ID (required for 'show')"
    )
    
    # -------------------------------------------------------------------------
    # Command: info
    # -------------------------------------------------------------------------
    info_parser = subparsers.add_parser(
        "info",
        help="System information",
        description="Display system information and status."
    )
    info_parser.add_argument(
        "info_type",
        choices=["models", "stats", "config", "version"],
        help="Type of information to display"
    )
    
    # -------------------------------------------------------------------------
    # Command: export
    # -------------------------------------------------------------------------
    export_parser = subparsers.add_parser(
        "export",
        help="Export conversation history",
        description="Export conversation history to Markdown file."
    )
    export_parser.add_argument(
        "--count", "-n",
        type=int,
        default=20,
        help="Number of entries to export (default: 20)"
    )
    export_parser.add_argument(
        "--output", "-o",
        help="Output file path"
    )
    
    # -------------------------------------------------------------------------
    # Parse and dispatch
    # -------------------------------------------------------------------------
    args = parser.parse_args()
    
    if not args.command:
        # Default: launch UI
        args.command = "ui"
        args.port = 7860
        args.share = False
        args.debug = False
    
    # Command dispatch
    commands = {
        "ui": cmd_ui,
        "benchmark": cmd_benchmark,
        "rag": cmd_rag,
        "config": cmd_config,
        "ask": cmd_ask,
        "presets": cmd_presets,
        "info": cmd_info,
        "export": cmd_export,
    }
    
    if args.command in commands:
        # Handle subcommand validation
        if args.command == "rag" and not hasattr(args, 'rag_action'):
            rag_parser.print_help()
            sys.exit(1)
        if args.command == "config" and not hasattr(args, 'config_action'):
            config_parser.print_help()
            sys.exit(1)
        
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
