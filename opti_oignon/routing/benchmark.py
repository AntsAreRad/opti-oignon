#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BENCHMARK - Model Testing Module v2.1
=====================================

Runs benchmarks on Ollama models and updates router preferences.

[!] IMPORTANT: This benchmark NEVER runs automatically!
    It always requires --confirm to avoid consuming
    GPU resources during your daily work.

Features:
- Standardized tests by task type
- Automatic response scoring
- Optional user ratings (interactive mode)
- View full response during scoring
- Prompt recall before each test
- Global ranking with averages
- Auto-configuration generation
- Regression detection
- Configuration update
- Markdown reports

Usage:
    python -m opti_oignon.routing.benchmark --estimate        # See estimate (no run)
    python -m opti_oignon.routing.benchmark --confirm         # Full benchmark (auto scoring)
    python -m opti_oignon.routing.benchmark --interactive --confirm  # With user ratings
    python -m opti_oignon.routing.benchmark --quick --confirm # Quick benchmark (3 models)
    python -m opti_oignon.routing.benchmark --model qwen3-coder:30b --confirm
    python -m opti_oignon.routing.benchmark --confirm --update-config  # Auto-update config

Author: Léon
"""

import argparse
import json
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
import sys

import ollama
import yaml

# =============================================================================
# CONFIGURATION
# =============================================================================

# Benchmark tasks with questions and evaluation criteria
BENCHMARK_TASKS = {
    "simple_question": {
        "name": "Simple Question",
        "description": "Basic factual question",
        "prompt": "What is biodiversity? Answer in 3 sentences maximum.",
        "expected_keywords": ["variety", "species", "living", "ecosystem", "diversity", "life"],
        "max_expected_time": 60,
        "category": "general",
        "task_routing_key": "simple_question",
    },
    "debug_r": {
        "name": "R Code Debug",
        "description": "R code debugging (main use case)",
        "prompt": """I have this R error, help me fix it:

```r
library(dplyr)
df <- data.frame(a = c(1, 2, NA), b = c(4, NA, 6))
result <- df %>% 
  mutate(total = rowSums(.))
```

Error: "Error in rowSums(.) : 'x' must be numeric"

Explain the error and provide the fix.""",
        "expected_keywords": ["select", "numeric", "na.rm", "across", "where", "is.numeric"],
        "max_expected_time": 90,
        "category": "code",
        "task_routing_key": "debug_r",
    },
    "code_python": {
        "name": "Python Code Generation",
        "description": "Functional Python code creation",
        "prompt": """Write a Python function that calculates the Shannon diversity index.

The function should:
1. Take a list of species counts
2. Calculate proportions
3. Apply the formula H' = -Σ(pi * ln(pi))
4. Return the result

Include a usage example with documented code.""",
        "expected_keywords": ["def", "shannon", "log", "math", "sum", "proportion"],
        "max_expected_time": 90,
        "category": "code",
        "task_routing_key": "code_python",
    },
    "explanation": {
        "name": "Concept Explanation",
        "description": "Pedagogical concept explanation",
        "prompt": """Explain the concept of rarefaction in ecology pedagogically.

Include:
1. Simple definition
2. Why it's useful
3. A concrete example
4. Method limitations

Provide a clear and structured answer.""",
        "expected_keywords": ["sampling", "species", "richness", "comparison", "curve", "individuals"],
        "max_expected_time": 180,
        "category": "general",
        "task_routing_key": "explanation",
    },
    "reasoning": {
        "name": "Multi-Step Reasoning",
        "description": "Problem requiring multiple reasoning steps",
        "prompt": """Here is biodiversity data from 3 sites:

| Site | Individuals | Species | Shannon |
|------|-------------|---------|---------|
| A    | 150         | 25      | 2.8     |
| B    | 500         | 20      | 1.9     |
| C    | 200         | 30      | 3.1     |

Analyze this data and answer:
1. Which site has the highest species richness?
2. Why does Site B have lower Shannon despite many individuals?
3. If I do rarefaction to 100 individuals, what will happen?

Reason step by step.""",
        "expected_keywords": ["Site C", "dominance", "evenness", "species", "rarefaction"],
        "max_expected_time": 180,
        "category": "reasoning",
        "task_routing_key": "reasoning",
    },
    "code_r": {
        "name": "R Code Generation",
        "description": "R code generation with tidyverse",
        "prompt": """Write an R function using tidyverse to calculate alpha diversity metrics.

Requirements:
1. Input: dataframe with species (rows) and samples (columns)
2. Calculate: Shannon, Simpson, and Richness for each sample
3. Return: tidy dataframe with results

Include comments and a usage example.""",
        "expected_keywords": ["function", "tidyverse", "shannon", "simpson", "mutate", "summarise"],
        "max_expected_time": 120,
        "category": "code",
        "task_routing_key": "code_r",
    },
}

# Default models to test
DEFAULT_MODELS = [
    "qwen3-coder:30b",
    "qwen3:32b",
    "deepseek-coder:33b",
    "deepseek-r1:32b",
    "qwen2.5-coder:14b",
    "gemma3:27b",
    "devstral-small-2:latest",
    "nemotron-3-nano:30b",
]

# Quick benchmark models
QUICK_MODELS = [
    "qwen3-coder:30b",
    "gemma3:27b",
    "nemotron-3-nano:30b",
]


# =============================================================================
# DATACLASSES
# =============================================================================

@dataclass
class BenchmarkResult:
    """Individual benchmark result."""
    model: str
    task: str
    task_name: str
    response: str
    time_seconds: float
    auto_score: int       # 0-10 automatic score
    user_score: Optional[int] = None  # 0-10 user score (optional)
    final_score: int = 0  # user_score if present, else auto_score
    keywords_found: List[str] = field(default_factory=list)
    keywords_missing: List[str] = field(default_factory=list)
    status: str = "success"  # "success", "timeout", "error", "refused"
    error_message: Optional[str] = None
    prompt: str = ""  # Store the prompt for reference
    
    def __post_init__(self):
        """Calculate final score."""
        self.final_score = self.user_score if self.user_score is not None else self.auto_score
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ModelSummary:
    """Model performance summary."""
    model: str
    avg_score: float
    avg_time: float
    tasks_tested: int
    successes: int
    failures: int
    scores_by_task: Dict[str, int]
    times_by_task: Dict[str, float]
    scores_by_category: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# MAIN CLASS
# =============================================================================

class ModelBenchmark:
    """
    Automated Ollama model benchmark.
    
    Runs standardized tests and generates performance reports.
    """
    
    def __init__(
        self,
        output_dir: str = "routing/benchmarks",
        config_path: str = "routing/config.yaml",
        timeout: int = 300,
        temperature: float = 0.7,
        interactive: bool = False,
    ):
        """
        Initialize the benchmark.
        
        Args:
            output_dir: Results folder
            config_path: Router config path
            timeout: Default timeout in seconds
            temperature: Temperature for tests
            interactive: Enable interactive user scoring
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.config_path = Path(config_path)
        self.timeout = timeout
        self.temperature = temperature
        self.interactive = interactive
        
        self.results: List[BenchmarkResult] = []
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    
    # -------------------------------------------------------------------------
    # EXECUTION
    # -------------------------------------------------------------------------
    
    def run(
        self,
        models: Optional[List[str]] = None,
        tasks: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> List[BenchmarkResult]:
        """
        Run the full benchmark.
        
        Args:
            models: Models to test (all if None)
            tasks: Tasks to test (all if None)
            verbose: Show progress
            
        Returns:
            List of results
        """
        # Get available models
        available = self._get_available_models()
        
        if models:
            models_to_test = [m for m in models if m in available]
            if not models_to_test:
                print(f"[ERR] None of the specified models are available")
                return []
        else:
            models_to_test = [m for m in DEFAULT_MODELS if m in available]
        
        tasks_to_run = tasks or list(BENCHMARK_TASKS.keys())
        
        total = len(models_to_test) * len(tasks_to_run)
        current = 0
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"[TEST] BENCHMARK - {self.timestamp}")
            print(f"{'='*60}")
            print(f"[>] Models: {len(models_to_test)}")
            print(f"[>] Tasks: {len(tasks_to_run)}")
            print(f"[>] Timeout: {self.timeout}s")
            print(f"[>] Mode: {'Interactive' if self.interactive else 'Automatic'}")
            print(f"{'='*60}\n")
        
        for model in models_to_test:
            if verbose:
                print(f"\n[MODEL] Testing {model}")
                print("-" * 40)
            
            for task_id in tasks_to_run:
                current += 1
                task = BENCHMARK_TASKS.get(task_id)
                if not task:
                    continue
                
                if verbose:
                    print(f"   [{current}/{total}] {task['name']}...", end=" ", flush=True)
                
                result = self._run_single_test(model, task_id, task)
                
                # Interactive user scoring
                if self.interactive and result.status == "success":
                    result = self._get_user_score(result, task, verbose)
                
                self.results.append(result)
                
                if verbose:
                    score_str = f"{result.final_score}/10"
                    if result.user_score is not None:
                        score_str = f"{result.final_score}/10 (user)"
                    else:
                        score_str = f"{result.final_score}/10 (auto)"
                    
                    if result.status == "success":
                        print(f"[OK] {score_str} ({result.time_seconds:.1f}s)")
                    elif result.status == "timeout":
                        print(f"[TIMEOUT]")
                    elif result.status == "refused":
                        print(f"[REFUSED]")
                    else:
                        print(f"[ERR]")
        
        if verbose:
            print(f"\n{'='*60}")
            print("[OK] Benchmark complete!")
            print(f"{'='*60}\n")
            
            # Display global ranking
            self._display_global_ranking()
        
        return self.results
    
    def _run_single_test(
        self,
        model: str,
        task_id: str,
        task: Dict,
    ) -> BenchmarkResult:
        """Run a single test."""
        prompt = task["prompt"]
        expected_keywords = task.get("expected_keywords", [])
        max_time = task.get("max_expected_time", self.timeout)
        
        start_time = time.time()
        
        try:
            # Ollama call
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self.temperature,
                    "num_predict": 1000,
                }
            )
            
            elapsed = time.time() - start_time
            response_text = response.get("response", "")
            
            # Check if model refused
            if self._is_refusal(response_text):
                return BenchmarkResult(
                    model=model,
                    task=task_id,
                    task_name=task["name"],
                    response=response_text[:500],
                    time_seconds=elapsed,
                    auto_score=2,
                    keywords_found=[],
                    keywords_missing=expected_keywords,
                    status="refused",
                    prompt=prompt,
                )
            
            # Calculate score
            score, found, missing = self._calculate_score(response_text, expected_keywords)
            
            # Penalty if too slow
            if elapsed > max_time * 2:
                score = max(0, score - 2)
            
            return BenchmarkResult(
                model=model,
                task=task_id,
                task_name=task["name"],
                response=response_text,  # Store full response
                time_seconds=elapsed,
                auto_score=score,
                keywords_found=found,
                keywords_missing=missing,
                status="success",
                prompt=prompt,
            )
            
        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = str(e)
            
            # Detect timeouts
            if "timeout" in error_msg.lower() or elapsed > self.timeout:
                status = "timeout"
            else:
                status = "error"
            
            return BenchmarkResult(
                model=model,
                task=task_id,
                task_name=task["name"],
                response="",
                time_seconds=elapsed,
                auto_score=0,
                keywords_found=[],
                keywords_missing=expected_keywords,
                status=status,
                error_message=error_msg[:200],
                prompt=prompt,
            )
    
    def _get_user_score(self, result: BenchmarkResult, task: Dict, verbose: bool) -> BenchmarkResult:
        """
        Get user score for a result (interactive mode).
        
        Features:
        - Shows test prompt before scoring
        - Allows viewing full response
        - Clear navigation with keyboard shortcuts
        """
        if not verbose:
            return result
        
        # Show prompt first
        print(f"\n{'='*70}")
        print("TEST PROMPT:")
        print("=" * 70)
        print(task["prompt"])
        print("=" * 70)
        
        # Show model info
        print(f"\nMODEL: {result.model}")
        print(f"TIME: {result.time_seconds:.1f}s")
        print(f"AUTO SCORE: {result.auto_score}/10")
        print(f"KEYWORDS FOUND: {', '.join(result.keywords_found) or 'none'}")
        
        # Show response preview
        print(f"\n{'-'*70}")
        print("RESPONSE PREVIEW (first 500 chars):")
        print("-" * 70)
        preview = result.response[:500]
        if len(result.response) > 500:
            preview += "\n... [truncated]"
        print(preview)
        print("-" * 70)
        
        # Interactive loop
        while True:
            try:
                print("\n[v] View full response | [s] Skip (keep auto) | [1-10] Score | [q] Quit")
                user_input = input("> ").strip().lower()
                
                if user_input == 'v':
                    # Show full response
                    print(f"\n{'='*70}")
                    print("FULL RESPONSE:")
                    print("=" * 70)
                    print(result.response)
                    print("=" * 70)
                    # Continue loop for scoring
                    continue
                
                elif user_input == 's' or user_input == '':
                    # Keep auto score
                    print(f"   -> Keeping auto score: {result.auto_score}/10")
                    break
                
                elif user_input == 'q':
                    # Quit interactive mode
                    print("\n   [!] Disabling interactive mode for remaining tests")
                    self.interactive = False
                    break
                
                else:
                    # Try to parse as score
                    try:
                        user_score = int(user_input)
                        if 1 <= user_score <= 10:
                            result.user_score = user_score
                            result.final_score = user_score
                            print(f"   -> User score recorded: {user_score}/10")
                            break
                        else:
                            print("   [!] Enter a number between 1 and 10")
                    except ValueError:
                        print("   [!] Invalid input. Use: v (view), s (skip), 1-10 (score), q (quit)")
                        
            except KeyboardInterrupt:
                print("\n   [!] Skipping user scoring for remaining tests")
                self.interactive = False
                break
        
        return result
    
    def _display_global_ranking(self):
        """Display global ranking at end of benchmark."""
        if not self.results:
            return
        
        summaries = self._calculate_summaries()
        summaries.sort(key=lambda s: s.avg_score, reverse=True)
        
        print(f"\n{'='*70}")
        print("OVERALL RANKING")
        print("=" * 70)
        print()
        
        # Header
        print(f" {'Rank':<5} | {'Model':<25} | {'Avg Score':<10} | {'Tasks':<7} | {'Avg Time':<10}")
        print("-" * 70)
        
        # Rows
        for i, summary in enumerate(summaries):
            rank = f"#{i+1}"
            tasks_str = f"{summary.successes}/{summary.tasks_tested}"
            print(f" {rank:<5} | {summary.model:<25} | {summary.avg_score:>5.1f}/10  | {tasks_str:<7} | {summary.avg_time:>6.1f}s")
        
        print("-" * 70)
        
        # Best by category
        print()
        self._display_best_by_category(summaries)
        
        print("=" * 70)
    
    def _display_best_by_category(self, summaries: List[ModelSummary]):
        """Display best models by category."""
        # Code tasks
        code_tasks = ["code_r", "code_python", "debug_r"]
        code_results = [r for r in self.results if r.task in code_tasks and r.status == "success"]
        if code_results:
            best_code = max(code_results, key=lambda r: r.final_score)
            avg_code = sum(r.final_score for r in code_results if r.model == best_code.model) / max(1, len([r for r in code_results if r.model == best_code.model]))
            print(f"Best for CODE: {best_code.model} (avg {avg_code:.1f}/10)")
        
        # Reasoning tasks
        reasoning_tasks = ["reasoning", "explanation"]
        reasoning_results = [r for r in self.results if r.task in reasoning_tasks and r.status == "success"]
        if reasoning_results:
            best_reasoning = max(reasoning_results, key=lambda r: r.final_score)
            avg_reasoning = sum(r.final_score for r in reasoning_results if r.model == best_reasoning.model) / max(1, len([r for r in reasoning_results if r.model == best_reasoning.model]))
            print(f"Best for REASONING: {best_reasoning.model} (avg {avg_reasoning:.1f}/10)")
        
        # Fastest with acceptable score
        acceptable_results = [r for r in self.results if r.final_score >= 7 and r.status == "success"]
        if acceptable_results:
            fastest = min(acceptable_results, key=lambda r: r.time_seconds)
            print(f"Best for SPEED (score>=7): {fastest.model} ({fastest.time_seconds:.1f}s avg)")
    
    def _is_refusal(self, response: str) -> bool:
        """Detect if model refused to answer."""
        refusal_patterns = [
            r"I'm sorry",
            r"I cannot",
            r"I am not able",
            r"as an AI",
            r"my expertise is",
            r"I don't have",
        ]
        return any(re.search(p, response, re.IGNORECASE) for p in refusal_patterns)
    
    def _calculate_score(
        self,
        response: str,
        expected_keywords: List[str],
    ) -> Tuple[int, List[str], List[str]]:
        """
        Calculate score based on found keywords.
        
        Returns:
            (score, keywords_found, keywords_missing)
        """
        response_lower = response.lower()
        found = []
        missing = []
        
        for keyword in expected_keywords:
            if keyword.lower() in response_lower:
                found.append(keyword)
            else:
                missing.append(keyword)
        
        if not expected_keywords:
            # No criteria = score based on length
            if len(response) > 200:
                return 7, [], []
            elif len(response) > 50:
                return 5, [], []
            return 3, [], []
        
        # Score proportional to keywords found
        ratio = len(found) / len(expected_keywords)
        score = int(ratio * 10)
        
        # Bonus for complete response
        if ratio >= 0.8 and len(response) > 300:
            score = min(10, score + 1)
        
        # Penalty for very short response
        if len(response) < 100:
            score = max(0, score - 2)
        
        return score, found, missing
    
    # -------------------------------------------------------------------------
    # REPORTS
    # -------------------------------------------------------------------------
    
    def generate_report(self) -> str:
        """Generate a complete Markdown report."""
        if not self.results:
            return "# No Results\n\nBenchmark was not run."
        
        # Calculate summaries by model
        summaries = self._calculate_summaries()
        
        # Sort by average score
        summaries.sort(key=lambda s: s.avg_score, reverse=True)
        
        lines = [
            f"# Ollama Model Benchmark",
            f"## Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            f"**Mode:** {'Interactive (with user scores)' if self.interactive else 'Automatic'}",
            "",
            "---",
            "",
            "## Global Ranking",
            "",
            "| Rank | Model | Avg Score | Avg Time |",
            "|------|-------|-----------|----------|",
        ]
        
        for i, summary in enumerate(summaries):
            rank = f"#{i+1}"
            lines.append(
                f"| {rank} | `{summary.model}` | {summary.avg_score:.1f}/10 | {summary.avg_time:.1f}s |"
            )
        
        lines.extend([
            "",
            "---",
            "",
            "## Results by Task",
            "",
        ])
        
        # Results by task
        for task_id, task in BENCHMARK_TASKS.items():
            task_results = [r for r in self.results if r.task == task_id]
            if not task_results:
                continue
            
            # Sort by score
            task_results.sort(key=lambda r: (r.final_score, -r.time_seconds), reverse=True)
            
            lines.extend([
                f"### {task['name']}",
                f"> {task['description']}",
                "",
                "| Model | Score | Time | Status |",
                "|-------|-------|------|--------|",
            ])
            
            for r in task_results:
                score_display = f"{r.final_score}/10"
                if r.user_score is not None:
                    score_display += " (user)"
                
                status_emoji = {
                    "success": "[OK]",
                    "timeout": "[TIMEOUT]",
                    "refused": "[REFUSED]",
                    "error": "[ERR]",
                }.get(r.status, "?")
                
                lines.append(f"| `{r.model}` | {score_display} | {r.time_seconds:.1f}s | {status_emoji} |")
            
            # Winner for this task
            if task_results and task_results[0].status == "success":
                winner = task_results[0]
                lines.append(f"\n**Winner:** `{winner.model}` ({winner.final_score}/10)")
            
            lines.append("")
        
        # Recommendations
        lines.extend([
            "---",
            "",
            "## Recommendations",
            "",
        ])
        
        # Best by category
        code_tasks = ["code_r", "code_python", "debug_r"]
        reasoning_tasks = ["reasoning", "explanation"]
        
        for category, tasks in [("Code", code_tasks), ("Reasoning", reasoning_tasks)]:
            category_results = [r for r in self.results if r.task in tasks and r.status == "success"]
            if category_results:
                best = max(category_results, key=lambda r: r.final_score)
                lines.append(f"- **Best for {category}:** `{best.model}` ({best.final_score}/10 on {best.task})")
        
        # Fastest acceptable
        fast_acceptable = [r for r in self.results if r.final_score >= 7 and r.status == "success"]
        if fast_acceptable:
            fastest = min(fast_acceptable, key=lambda r: r.time_seconds)
            lines.append(f"- **Fastest (score >=7):** `{fastest.model}` ({fastest.time_seconds:.1f}s)")
        
        return "\n".join(lines)
    
    def _calculate_summaries(self) -> List[ModelSummary]:
        """Calculate summaries by model."""
        models = set(r.model for r in self.results)
        summaries = []
        
        for model in models:
            results = [r for r in self.results if r.model == model]
            successful = [r for r in results if r.status == "success"]
            
            if successful:
                avg_score = sum(r.final_score for r in successful) / len(successful)
                avg_time = sum(r.time_seconds for r in successful) / len(successful)
            else:
                avg_score = 0
                avg_time = 0
            
            scores_by_task = {r.task: r.final_score for r in successful}
            times_by_task = {r.task: r.time_seconds for r in results}
            
            # Calculate scores by category
            scores_by_category = {}
            for category, task_list in [("code", ["code_r", "code_python", "debug_r"]), 
                                        ("reasoning", ["reasoning", "explanation"]),
                                        ("general", ["simple_question"])]:
                cat_results = [r for r in successful if r.task in task_list]
                if cat_results:
                    scores_by_category[category] = sum(r.final_score for r in cat_results) / len(cat_results)
            
            summaries.append(ModelSummary(
                model=model,
                avg_score=avg_score,
                avg_time=avg_time,
                tasks_tested=len(results),
                successes=len(successful),
                failures=len(results) - len(successful),
                scores_by_task=scores_by_task,
                times_by_task=times_by_task,
                scores_by_category=scores_by_category,
            ))
        
        return summaries
    
    def save_results(self):
        """Save results."""
        # Detailed JSON
        json_path = self.output_dir / f"benchmark_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        # Markdown report
        md_path = self.output_dir / f"benchmark_{self.timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        # Copy as "latest"
        latest_json = self.output_dir / "benchmark_latest.json"
        latest_md = self.output_dir / "benchmark_latest.md"
        
        with open(latest_json, 'w', encoding='utf-8') as f:
            json.dump(
                [r.to_dict() for r in self.results],
                f,
                indent=2,
                ensure_ascii=False,
            )
        
        with open(latest_md, 'w', encoding='utf-8') as f:
            f.write(self.generate_report())
        
        print(f"[>] Results saved:")
        print(f"   - {json_path}")
        print(f"   - {md_path}")
    
    # -------------------------------------------------------------------------
    # CONFIG UPDATE - AUTOMATIC ROUTING CONFIG GENERATION
    # -------------------------------------------------------------------------
    
    def update_config(self, dry_run: bool = True) -> Dict:
        """
        Update router config based on results.
        
        Args:
            dry_run: If True, only suggest changes
            
        Returns:
            Dictionary of suggested changes
        """
        if not self.results:
            return {"error": "No results"}
        
        summaries = self._calculate_summaries()
        changes = {}
        
        # Find best models by task
        print("\n" + "=" * 70)
        print("AUTO-CONFIGURATION SUGGESTION")
        print("=" * 70)
        print()
        print("Based on benchmark results, recommended config:")
        print()
        
        # Generate task_routing section
        config_lines = ["task_routing:"]
        
        for task_id, task in BENCHMARK_TASKS.items():
            task_results = [r for r in self.results if r.task == task_id and r.status == "success"]
            if not task_results:
                continue
            
            # Best score
            best_score = max(task_results, key=lambda r: r.final_score)
            # Fastest with acceptable score (>= 7)
            fast_acceptable = [r for r in task_results if r.final_score >= 7]
            best_fast = min(fast_acceptable, key=lambda r: r.time_seconds) if fast_acceptable else None
            
            routing_key = task.get("task_routing_key", task_id)
            
            # Build fallback list (other good models)
            fallbacks = [r.model for r in sorted(task_results, key=lambda x: x.final_score, reverse=True) 
                        if r.model != best_score.model and r.final_score >= 6][:2]
            
            changes[routing_key] = {
                "primary": best_score.model,
                "primary_score": best_score.final_score,
                "fallback": fallbacks,
                "fast": best_fast.model if best_fast else None,
                "fast_time": best_fast.time_seconds if best_fast else None,
            }
            
            config_lines.append(f"  {routing_key}:")
            config_lines.append(f'    primary: "{best_score.model}"     # Score: {best_score.final_score}/10')
            if fallbacks:
                fallback_str = ', '.join(f'"{f}"' for f in fallbacks)
                config_lines.append(f'    fallback: [{fallback_str}]')
            if best_fast and best_fast.model != best_score.model:
                config_lines.append(f'    fast: "{best_fast.model}"    # Time: {best_fast.time_seconds:.1f}s')
        
        # Add special routing for categories
        config_lines.append("")
        config_lines.append("  # Category defaults (auto-generated)")
        
        # Best for code category
        code_results = [r for r in self.results if r.task in ["code_r", "code_python", "debug_r"] and r.status == "success"]
        if code_results:
            best_code = max(code_results, key=lambda r: r.final_score)
            config_lines.append(f'  code:')
            config_lines.append(f'    primary: "{best_code.model}"')
        
        # Best for reasoning
        reasoning_results = [r for r in self.results if r.task in ["reasoning", "explanation"] and r.status == "success"]
        if reasoning_results:
            best_reasoning = max(reasoning_results, key=lambda r: r.final_score)
            config_lines.append(f'  reasoning:')
            config_lines.append(f'    primary: "{best_reasoning.model}"')
        
        # Fastest overall
        fast_acceptable = [r for r in self.results if r.final_score >= 7 and r.status == "success"]
        if fast_acceptable:
            fastest = min(fast_acceptable, key=lambda r: r.time_seconds)
            config_lines.append(f'  fast:')
            config_lines.append(f'    primary: "{fastest.model}"     # Fastest: {fastest.time_seconds:.1f}s')
        
        # Display generated config
        for line in config_lines:
            print(line)
        
        print()
        print("-" * 70)
        
        if dry_run:
            # Ask user if they want to apply
            print()
            print("[y] Apply this config | [n] Skip | [e] Edit manually (shows path)")
            try:
                choice = input("> ").strip().lower()
                
                if choice == 'y':
                    self._apply_config_changes(changes)
                    print(f"\n[OK] Config saved to {self.config_path}")
                elif choice == 'e':
                    print(f"\n[>] Edit manually: {self.config_path}")
                    self._generate_config_yaml(changes)
                    print(f"[>] Config template saved to {self.output_dir / 'suggested_config.yaml'}")
                else:
                    print("\n[>] Config not applied. To apply later: --update-config")
            except (KeyboardInterrupt, EOFError):
                print("\n[>] Skipped config update")
        else:
            # Apply changes directly
            self._apply_config_changes(changes)
            print(f"\n[OK] Config saved to {self.config_path}")
        
        return changes
    
    def _generate_config_yaml(self, changes: Dict):
        """Generate a YAML config file suggestion."""
        suggested_path = self.output_dir / "suggested_config.yaml"
        
        config = {
            "task_routing": {},
            "last_benchmark": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "benchmark_mode": "interactive" if self.interactive else "automatic",
        }
        
        for task_id, info in changes.items():
            config["task_routing"][task_id] = {
                "primary": info["primary"],
            }
            if info.get("fallback"):
                config["task_routing"][task_id]["fallback"] = info["fallback"]
            if info.get("fast"):
                config["task_routing"][task_id]["fast"] = info["fast"]
        
        with open(suggested_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    def _apply_config_changes(self, changes: Dict):
        """Apply changes to config file."""
        try:
            # Load existing config or create new
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
            else:
                config = {}
            
            # Ensure task_routing exists
            if "task_routing" not in config:
                config["task_routing"] = {}
            
            task_routing = config["task_routing"]
            
            for task_id, info in changes.items():
                if task_id not in task_routing:
                    task_routing[task_id] = {}
                
                # Update primary model
                if info["primary"]:
                    task_routing[task_id]["primary"] = info["primary"]
                
                # Update fallbacks
                if info.get("fallback"):
                    task_routing[task_id]["fallback"] = info["fallback"]
                
                # Update fast model
                if info.get("fast"):
                    task_routing[task_id]["fast"] = info["fast"]
            
            # Add metadata
            config["last_benchmark"] = datetime.now().strftime("%Y-%m-%d %H:%M")
            config["benchmark_mode"] = "interactive" if self.interactive else "automatic"
            
            # Save
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            
        except Exception as e:
            print(f"[ERR] Config update error: {e}")
    
    # -------------------------------------------------------------------------
    # REGRESSION DETECTION
    # -------------------------------------------------------------------------
    
    def check_regression(self) -> Dict:
        """
        Compare with last benchmark for regressions.
        
        Returns:
            Dictionary of detected regressions
        """
        latest_path = self.output_dir / "benchmark_latest.json"
        if not latest_path.exists():
            return {"status": "no_previous_data"}
        
        try:
            with open(latest_path, 'r', encoding='utf-8') as f:
                previous = json.load(f)
        except:
            return {"status": "error_loading_previous"}
        
        regressions = []
        improvements = []
        
        # Create index of previous results
        prev_by_key = {(r["model"], r["task"]): r for r in previous}
        
        for result in self.results:
            key = (result.model, result.task)
            if key not in prev_by_key:
                continue
            
            prev = prev_by_key[key]
            prev_score = prev.get("final_score", prev.get("auto_score", 0))
            
            # Significant regression (>= 2 points)
            if result.final_score < prev_score - 2:
                regressions.append({
                    "model": result.model,
                    "task": result.task,
                    "previous_score": prev_score,
                    "current_score": result.final_score,
                    "delta": result.final_score - prev_score,
                })
            
            # Significant improvement
            elif result.final_score > prev_score + 2:
                improvements.append({
                    "model": result.model,
                    "task": result.task,
                    "previous_score": prev_score,
                    "current_score": result.final_score,
                    "delta": result.final_score - prev_score,
                })
        
        return {
            "status": "analyzed",
            "regressions": regressions,
            "improvements": improvements,
            "total_compared": len([r for r in self.results if (r.model, r.task) in prev_by_key]),
        }
    
    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------
    
    def _get_available_models(self) -> List[str]:
        """Get available Ollama models."""
        try:
            response = ollama.list()
            models = []
            
            if hasattr(response, 'models'):
                for m in response.models:
                    name = getattr(m, 'model', None) or getattr(m, 'name', None)
                    if name:
                        models.append(name)
            elif isinstance(response, dict):
                for m in response.get("models", []):
                    name = m.get("model") or m.get("name", "")
                    if name:
                        models.append(name)
            
            return models
        except Exception as e:
            print(f"[ERR] Ollama connection error: {e}")
            return []


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Ollama Model Benchmark (MANUAL - requires confirmation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
[!] WARNING: Benchmark consumes significant GPU resources!
    It NEVER runs automatically and always requires --confirm.

Examples:
  python -m opti_oignon.routing.benchmark --estimate        # Estimate time
  python -m opti_oignon.routing.benchmark --confirm         # Full auto benchmark
  python -m opti_oignon.routing.benchmark --interactive --confirm  # With user scoring
  python -m opti_oignon.routing.benchmark --quick --confirm # Quick (3 models)
  python -m opti_oignon.routing.benchmark --confirm --update-config
        """
    )
    
    parser.add_argument(
        "--confirm", "-y",
        action="store_true",
        help="REQUIRED: Confirm benchmark launch (consumes GPU)"
    )
    
    parser.add_argument(
        "--estimate",
        action="store_true",
        help="Estimate time and resources without running"
    )
    
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Interactive mode: prompt for user scores after each test"
    )
    
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick benchmark (3 main models only)"
    )
    
    parser.add_argument(
        "--model", "-m",
        action="append",
        help="Specific model(s) to test (can be repeated)"
    )
    
    parser.add_argument(
        "--task", "-t",
        action="append",
        choices=list(BENCHMARK_TASKS.keys()),
        help="Specific task(s) to test (can be repeated)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds (default: 300)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for tests (default: 0.7)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        default="routing/benchmarks",
        help="Output folder (default: routing/benchmarks)"
    )
    
    parser.add_argument(
        "--update-config",
        action="store_true",
        help="Update router config after benchmark"
    )
    
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Compare with previous benchmark"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Quiet mode"
    )
    
    args = parser.parse_args()
    
    # Determine models to test
    models = args.model
    if args.quick and not models:
        models = QUICK_MODELS
    
    # Create benchmark for estimation
    benchmark = ModelBenchmark(
        output_dir=args.output_dir,
        timeout=args.timeout,
        temperature=args.temperature,
        interactive=args.interactive,
    )
    
    # Get available models for estimation
    available = benchmark._get_available_models()
    if models:
        models_to_test = [m for m in models if m in available]
    else:
        models_to_test = [m for m in DEFAULT_MODELS if m in available]
    
    tasks_to_run = args.task or list(BENCHMARK_TASKS.keys())
    total_tests = len(models_to_test) * len(tasks_to_run)
    
    # Estimate time (based on ~90s avg per test)
    estimated_time_min = (total_tests * 90) / 60
    estimated_time_quick = (total_tests * 45) / 60
    
    # Estimation mode only
    if args.estimate:
        print("\n" + "=" * 60)
        print("[>] BENCHMARK ESTIMATION")
        print("=" * 60)
        print(f"\n[>] Models to test: {len(models_to_test)}")
        for m in models_to_test:
            print(f"   - {m}")
        print(f"\n[>] Tasks: {len(tasks_to_run)}")
        for t in tasks_to_run:
            print(f"   - {t}")
        print(f"\n[>] Total tests: {total_tests}")
        print(f"[>] Estimated time: {estimated_time_quick:.0f} - {estimated_time_min:.0f} minutes")
        print(f"[>] GPU usage: INTENSIVE for entire duration")
        print(f"[>] Mode: {'Interactive' if args.interactive else 'Automatic'}")
        print("\n[>] To run: python -m opti_oignon.routing.benchmark --confirm")
        if not args.quick:
            print("[>] For quick test: add --quick flag")
        if not args.interactive:
            print("[>] For user scoring: add --interactive flag")
        return
    
    # Check REQUIRED confirmation
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
    results = benchmark.run(
        models=models,
        tasks=args.task,
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
                print(f"   - {reg['model']} on {reg['task']}: {reg['previous_score']} -> {reg['current_score']} ({reg['delta']:+d})")
        
        if regression_info.get("improvements"):
            print("\n[OK] IMPROVEMENTS:")
            for imp in regression_info["improvements"]:
                print(f"   - {imp['model']} on {imp['task']}: {imp['previous_score']} -> {imp['current_score']} ({imp['delta']:+d})")
    
    # Update config (shows suggestion and asks for confirmation)
    if args.update_config:
        print("\n[>] Generating config suggestion...")
        benchmark.update_config(dry_run=False)
    else:
        # Show suggestions with interactive apply option
        benchmark.update_config(dry_run=True)


if __name__ == "__main__":
    main()
