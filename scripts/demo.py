#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Opti-Oignon Demo Script
========================

Interactive demonstration of Opti-Oignon's key features.
Runs against the FastAPI backend (API-only, no frontend needed).

Usage:
    # Start the backend first:
    uvicorn opti_oignon.api.app:app --port 8000

    # Then run the demo:
    python scripts/demo.py
    python scripts/demo.py --base-url http://localhost:8000
    python scripts/demo.py --headless        # No prompts, run all demos
    python scripts/demo.py --section routing  # Run a specific section

Sections:
    health        Health check and module status
    conversations Conversation lifecycle (create, list, delete)
    routing       Smart routing and model selection
    pipelines     Pipeline listing and classification
    feedback      Feedback submission and retrieval
    analytics     Performance recording and analytics
    all           Run all sections (default)
"""

import argparse
import json
import sys
import time

try:
    import requests
except ImportError:
    print("Error: 'requests' is required. Install with: pip install requests")
    sys.exit(1)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

BOLD = "\033[1m"
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
CYAN = "\033[96m"
DIM = "\033[2m"
RESET = "\033[0m"

_pass_count = 0
_fail_count = 0


def header(title: str):
    """Print a section header."""
    print(f"\n{BOLD}{BLUE}{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}{RESET}\n")


def step(msg: str):
    """Print a step description."""
    print(f"  {CYAN}> {msg}{RESET}")


def ok(msg: str):
    """Print a success message."""
    global _pass_count
    _pass_count += 1
    print(f"  {GREEN}[OK] {msg}{RESET}")


def fail(msg: str, detail: str = ""):
    """Print a failure message."""
    global _fail_count
    _fail_count += 1
    print(f"  {RED}[FAIL] {msg}{RESET}")
    if detail:
        print(f"    {DIM}{detail}{RESET}")


def info(msg: str):
    """Print an info message."""
    print(f"  {YELLOW}[INFO] {msg}{RESET}")


def show_json(data, indent: int = 4, max_lines: int = 20):
    """Pretty-print JSON data, truncated if too long."""
    text = json.dumps(data, indent=2, default=str)
    lines = text.split("\n")
    prefix = " " * indent
    for line in lines[:max_lines]:
        print(f"{prefix}{DIM}{line}{RESET}")
    if len(lines) > max_lines:
        print(f"{prefix}{DIM}... ({len(lines) - max_lines} more lines){RESET}")


def pause(headless: bool):
    """Wait for user confirmation unless headless."""
    if not headless:
        input(f"\n  {DIM}Press Enter to continue...{RESET}")


# =============================================================================
# API CLIENT
# =============================================================================

class DemoClient:
    """Simple HTTP client for the demo."""

    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()

    def get(self, path: str, **kwargs):
        return self.session.get(f"{self.base}{path}", **kwargs)

    def post(self, path: str, **kwargs):
        return self.session.post(f"{self.base}{path}", **kwargs)

    def patch(self, path: str, **kwargs):
        return self.session.patch(f"{self.base}{path}", **kwargs)

    def delete(self, path: str, **kwargs):
        return self.session.delete(f"{self.base}{path}", **kwargs)

    def alive(self) -> bool:
        """Check if the backend is reachable."""
        try:
            r = self.get("/api/health", timeout=3)
            return r.status_code == 200
        except Exception:
            return False


# =============================================================================
# DEMO SECTIONS
# =============================================================================

def demo_health(client: DemoClient, headless: bool):
    """Demonstrate health check and module status."""
    header("1. Health Check & Module Status")

    step("GET /api/health")
    r = client.get("/api/health")
    if r.status_code == 200:
        data = r.json()
        ok(f"Backend v{data['version']} is running")
        modules = data.get("modules", {})
        available = [k for k, v in modules.items() if v]
        unavailable = [k for k, v in modules.items() if not v]
        info(f"Available modules ({len(available)}): {', '.join(available)}")
        if unavailable:
            info(f"Unavailable ({len(unavailable)}): {', '.join(unavailable)}")
    else:
        fail("Health check failed", f"HTTP {r.status_code}")
        return

    step("GET /api/health/dashboard")
    r = client.get("/api/health/dashboard")
    if r.status_code == 200:
        ok("Dashboard endpoint responds")
    else:
        fail("Dashboard endpoint", f"HTTP {r.status_code}")

    pause(headless)


def demo_conversations(client: DemoClient, headless: bool):
    """Demonstrate conversation lifecycle."""
    header("2. Conversation Lifecycle")

    # Create
    step("POST /api/conversations -- Create")
    r = client.post("/api/conversations", json={"title": "Demo Conversation"})
    if r.status_code in (200, 201):
        conv = r.json()
        conv_id = conv["id"]
        ok(f"Created conversation: {conv_id}")
    else:
        fail("Create conversation", f"HTTP {r.status_code}")
        return

    # List
    step("GET /api/conversations -- List")
    r = client.get("/api/conversations")
    if r.status_code == 200:
        count = len(r.json())
        ok(f"Listed {count} conversation(s)")
    else:
        fail("List conversations", f"HTTP {r.status_code}")

    # Rename
    step(f"PATCH /api/conversations/{conv_id} -- Rename")
    r = client.patch(
        f"/api/conversations/{conv_id}",
        json={"title": "Renamed Demo Conversation"},
    )
    if r.status_code == 200:
        ok(f"Renamed to: {r.json()['title']}")
    else:
        fail("Rename conversation", f"HTTP {r.status_code}")

    # Get detail
    step(f"GET /api/conversations/{conv_id} -- Detail")
    r = client.get(f"/api/conversations/{conv_id}")
    if r.status_code == 200:
        ok("Retrieved conversation detail")
    else:
        fail("Get conversation", f"HTTP {r.status_code}")

    # Delete
    step(f"DELETE /api/conversations/{conv_id} -- Cleanup")
    r = client.delete(f"/api/conversations/{conv_id}")
    if r.status_code in (200, 204):
        ok("Deleted conversation")
    else:
        fail("Delete conversation", f"HTTP {r.status_code}")

    pause(headless)


def demo_routing(client: DemoClient, headless: bool):
    """Demonstrate smart routing and model selection."""
    header("3. Smart Routing & Model Selection")

    # Single model selection
    step_types = ["direct", "code_verify", "reasoning", "consensus"]
    for st in step_types:
        step(f"GET /api/smart-routing/select?step_type={st}")
        r = client.get("/api/smart-routing/select", params={"step_type": st})
        if r.status_code == 200:
            data = r.json()
            model = data.get("model", "?")
            score = data.get("score", 0)
            fallback = data.get("fallback", False)
            tag = " (fallback)" if fallback else ""
            ok(f"{st}: {model} (score={score:.3f}){tag}")
        else:
            fail(f"Select for {st}", f"HTTP {r.status_code}")

    # Pipeline routing (multiple steps)
    step("POST /api/smart-routing/select-pipeline -- Multi-step")
    r = client.post(
        "/api/smart-routing/select-pipeline",
        json=["direct", "code_verify", "reasoning"],
    )
    if r.status_code == 200:
        data = r.json()
        results = data.get("results", data)
        if isinstance(results, dict):
            for stype, result in results.items():
                model = result.get("model", "?") if isinstance(result, dict) else "?"
                info(f"  {stype}: {model}")
        ok("Pipeline routing returned selections for all steps")
    else:
        fail("Pipeline routing", f"HTTP {r.status_code}")

    # Router config
    step("GET /api/smart-routing/config")
    r = client.get("/api/smart-routing/config")
    if r.status_code == 200:
        ok("Router config retrieved")
        data = r.json()
        info(f"  Enabled: {data.get('enabled', '?')}")
        info(f"  Speed preference: {data.get('speed_preference', '?')}")
    else:
        fail("Router config", f"HTTP {r.status_code}")

    pause(headless)


def demo_pipelines(client: DemoClient, headless: bool):
    """Demonstrate pipeline listing and management."""
    header("4. Pipelines")

    # List builtin pipelines
    step("GET /api/pipelines/builtin")
    r = client.get("/api/pipelines/builtin")
    if r.status_code == 200:
        pipelines = r.json()
        ok(f"Found {len(pipelines)} builtin pipeline(s)")
        for p in pipelines:
            name = p.get("name", "?") if isinstance(p, dict) else str(p)
            info(f"  - {name}")
    else:
        fail("List builtin pipelines", f"HTTP {r.status_code}")

    # List custom pipelines
    step("GET /api/pipelines")
    r = client.get("/api/pipelines")
    if r.status_code == 200:
        pipelines = r.json()
        if isinstance(pipelines, list):
            ok(f"Found {len(pipelines)} custom pipeline(s)")
        else:
            ok("Pipelines endpoint responds")
    else:
        fail("List pipelines", f"HTTP {r.status_code}")

    # Pipeline classification demo (using agentic executor heuristics)
    step("Pipeline classification examples (local, no LLM)")
    try:
        from opti_oignon.agentic_executor import _quick_classify
        examples = [
            ("Hello, how are you?", "Simple greeting"),
            ("Write a Python function for quicksort", "Code task"),
            ("Search the web for recent AI news", "Web search"),
            ("Think step by step about climate change impacts", "Complex reasoning"),
            ("Use the calculator tool to compute 2^64", "Tool use"),
        ]
        for query, desc in examples:
            result = _quick_classify(query)
            flags = [k for k, v in result.items() if v]
            flag_str = ", ".join(flags) if flags else "none (direct)"
            ok(f"{desc}: [{flag_str}]")
    except ImportError:
        info("Skipped local classification (import unavailable)")

    pause(headless)


def demo_feedback(client: DemoClient, headless: bool):
    """Demonstrate feedback submission and retrieval."""
    header("5. Feedback System")

    # Submit positive feedback
    step("POST /api/feedback -- Thumbs up")
    r = client.post("/api/feedback", json={
        "conversation_id": "demo-conv-001",
        "message_id": "demo-msg-001",
        "rating_type": "thumbs",
        "rating_value": 1,
        "model_used": "qwen3:32b",
        "pipeline_used": "direct",
        "task_type": "general",
    })
    fid_positive = None
    if r.status_code == 200:
        fid_positive = r.json()["feedback_id"]
        ok(f"Positive feedback: {fid_positive}")
    else:
        fail("Submit positive feedback", f"HTTP {r.status_code}")

    # Submit negative feedback with text
    step("POST /api/feedback -- Thumbs down with text")
    r = client.post("/api/feedback", json={
        "conversation_id": "demo-conv-001",
        "message_id": "demo-msg-002",
        "rating_type": "thumbs",
        "rating_value": 0,
        "feedback_text": "Response was too verbose",
        "model_used": "qwen3:32b",
        "pipeline_used": "code_verify",
        "task_type": "code",
    })
    fid_negative = None
    if r.status_code == 200:
        fid_negative = r.json()["feedback_id"]
        ok(f"Negative feedback: {fid_negative}")
    else:
        fail("Submit negative feedback", f"HTTP {r.status_code}")

    # Get stats
    step("GET /api/feedback/stats")
    r = client.get("/api/feedback/stats")
    if r.status_code == 200:
        stats = r.json()
        ok(f"Stats: {stats.get('total_count', '?')} total, "
           f"{stats.get('positive_count', '?')} positive, "
           f"{stats.get('negative_count', '?')} negative")
    else:
        fail("Feedback stats", f"HTTP {r.status_code}")

    # Export JSON
    step("GET /api/feedback/export/json")
    r = client.get("/api/feedback/export/json")
    if r.status_code == 200:
        ok("JSON export successful")
    else:
        fail("JSON export", f"HTTP {r.status_code}")

    # Cleanup
    for fid in [fid_positive, fid_negative]:
        if fid:
            client.delete(f"/api/feedback/{fid}")
    info("Cleaned up demo feedback entries")

    pause(headless)


def demo_analytics(client: DemoClient, headless: bool):
    """Demonstrate performance tracking and analytics."""
    header("6. Analytics & Performance Tracking")

    # Record some performance data
    step("POST /api/analytics/record -- Recording sample data")
    samples = [
        {"model_used": "qwen3:32b", "pipeline_used": "direct",
         "task_type": "general", "response_time_ms": 320,
         "prompt_tokens": 85, "completion_tokens": 210},
        {"model_used": "qwen3-coder:30b", "pipeline_used": "code_verify",
         "task_type": "code", "response_time_ms": 1450,
         "prompt_tokens": 200, "completion_tokens": 580},
        {"model_used": "qwen3:32b", "pipeline_used": "reasoning",
         "task_type": "reasoning", "response_time_ms": 2800,
         "prompt_tokens": 150, "completion_tokens": 920},
        {"model_used": "deepseek-r1:32b", "pipeline_used": "direct",
         "task_type": "general", "response_time_ms": 450,
         "prompt_tokens": 90, "completion_tokens": 300},
    ]
    recorded = 0
    for sample in samples:
        r = client.post("/api/analytics/record", json=sample)
        if r.status_code == 200:
            recorded += 1
    ok(f"Recorded {recorded}/{len(samples)} performance entries")

    # Overview
    step("GET /api/analytics/overview")
    r = client.get("/api/analytics/overview")
    if r.status_code == 200:
        data = r.json()
        ok(f"Overview: {data.get('total_requests', '?')} requests, "
           f"avg {data.get('avg_response_time_ms', 0):.0f}ms")
        model_dist = data.get("model_distribution", {})
        if model_dist:
            info("Model distribution:")
            for model, count in model_dist.items():
                info(f"  {model}: {count} requests")
    else:
        fail("Analytics overview", f"HTTP {r.status_code}")

    # Trends
    step("GET /api/analytics/trends?window=1h&buckets=4")
    r = client.get("/api/analytics/trends", params={"window": "1h", "buckets": 4})
    if r.status_code == 200:
        data = r.json()
        bucket_count = len(data.get("data", []))
        ok(f"Trends: {bucket_count} time buckets returned")
    else:
        fail("Analytics trends", f"HTTP {r.status_code}")

    # Routing accuracy
    step("GET /api/analytics/routing-accuracy")
    r = client.get("/api/analytics/routing-accuracy")
    if r.status_code == 200:
        ok("Routing accuracy endpoint responds")
    else:
        fail("Routing accuracy", f"HTTP {r.status_code}")

    pause(headless)


# =============================================================================
# MAIN
# =============================================================================

SECTIONS = {
    "health": demo_health,
    "conversations": demo_conversations,
    "routing": demo_routing,
    "pipelines": demo_pipelines,
    "feedback": demo_feedback,
    "analytics": demo_analytics,
}


def main():
    parser = argparse.ArgumentParser(
        description="Opti-Oignon interactive demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--base-url", default="http://localhost:8000",
        help="Backend base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Run without pauses between sections",
    )
    parser.add_argument(
        "--section", default="all",
        choices=["all"] + list(SECTIONS.keys()),
        help="Run a specific section (default: all)",
    )
    args = parser.parse_args()

    print(f"\n{BOLD}{'=' * 60}")
    print("  Opti-Oignon Demo")
    print(f"  Backend: {args.base_url}")
    print(f"{'=' * 60}{RESET}")

    client = DemoClient(args.base_url)

    # Check connectivity
    if not client.alive():
        print(f"\n  {RED}Backend not reachable at {args.base_url}{RESET}")
        print(f"  {YELLOW}Start it with:{RESET}")
        print(f"    uvicorn opti_oignon.api.app:app --port 8000\n")
        sys.exit(1)

    # Run sections
    if args.section == "all":
        for name, func in SECTIONS.items():
            func(client, args.headless)
    else:
        SECTIONS[args.section](client, args.headless)

    # Summary
    print(f"\n{BOLD}{'=' * 60}")
    total = _pass_count + _fail_count
    print(f"  Demo complete: {GREEN}{_pass_count} passed{RESET}, "
          f"{RED if _fail_count else DIM}{_fail_count} failed{RESET} "
          f"({total} checks)")
    print(f"{'=' * 60}{RESET}\n")

    sys.exit(1 if _fail_count > 0 else 0)


if __name__ == "__main__":
    main()
