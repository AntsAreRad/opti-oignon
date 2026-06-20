#!/usr/bin/env python3
"""
Autonomous Benchmark Evaluator — S88

Measures factual accuracy, code quality, structural quality, and
performance without requiring a judge LLM. Scoring is deterministic
and reproducible.

Scoring dimensions:
  - Factual accuracy: ground truth comparison (exact/fuzzy/keyword)
  - Code quality: sandbox execution with test assertions
  - Structural quality: repetition, lexical diversity, length, format
  - Performance: TTFT, tokens/sec, total time (from performance_monitor)
"""

import json
import logging
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependency: sandbox_manager for code execution
# ---------------------------------------------------------------------------
try:
    from opti_oignon.sandbox_manager import sandbox_manager, CommandResult
    SANDBOX_AVAILABLE = True
except ImportError:
    SANDBOX_AVAILABLE = False
    sandbox_manager = None
    CommandResult = None  # type: ignore[assignment, misc]

# ---------------------------------------------------------------------------
# YAML config loading
# ---------------------------------------------------------------------------
_CONFIG_DIR = Path(__file__).parent / "config"
_QUESTIONS_PATH = _CONFIG_DIR / "benchmark_questions.yaml"
_PROFILES_PATH = _CONFIG_DIR / "benchmark_profiles.yaml"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dict."""
    try:
        import yaml
    except ImportError:
        logger.warning("PyYAML not available, returning empty config")
        return {}
    if not path.exists():
        logger.warning("Config file not found: %s", path)
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class ScoringMethod(str, Enum):
    """How to compare LLM output to expected answer."""
    EXACT = "exact"
    FUZZY = "fuzzy"
    KEYWORD = "keyword"


@dataclass
class Question:
    """A single benchmark question with ground truth."""
    id: str
    category: str
    prompt: str
    expected: list[str]
    scoring: ScoringMethod = ScoringMethod.EXACT
    tolerance: float = 0.0
    keywords: list[str] = field(default_factory=list)
    # Code generation fields
    language: str = ""
    test_code: str = ""
    expected_output: str = ""
    timeout: int = 30


@dataclass
class AccuracyResult:
    """Accuracy evaluation result for a single question."""
    question_id: str
    score: float = 0.0
    matched_answer: str = ""
    method: str = ""
    details: str = ""


@dataclass
class CodeResult:
    """Code execution evaluation result."""
    question_id: str
    compiles: bool = False
    runs: bool = False
    output_matches: bool = False
    tests_pass: bool = False
    score: float = 0.0
    stdout: str = ""
    stderr: str = ""
    return_code: int = -1
    details: str = ""


@dataclass
class StructuralResult:
    """Structural quality evaluation result."""
    repetition_score: float = 0.0
    lexical_diversity: float = 0.0
    length_appropriateness: float = 0.0
    format_compliance: float = 0.0
    composite: float = 0.0
    details: dict = field(default_factory=dict)


@dataclass
class PerformanceResult:
    """Performance metrics for a single generation."""
    ttft_ms: float = 0.0
    tokens_per_second: float = 0.0
    total_time_ms: float = 0.0
    context_utilization: float = 0.0
    score: float = 0.0


@dataclass
class QuestionResult:
    """Complete evaluation result for a single question."""
    question_id: str
    category: str
    prompt: str
    response: str = ""
    accuracy: AccuracyResult | None = None
    code: CodeResult | None = None
    structure: StructuralResult | None = None
    performance: PerformanceResult | None = None
    composite_score: float = 0.0


@dataclass
class WeightPreset:
    """Scoring weights for composite calculation."""
    accuracy: float = 0.35
    code: float = 0.25
    structure: float = 0.25
    speed: float = 0.15


# ---------------------------------------------------------------------------
# Question loader
# ---------------------------------------------------------------------------

def load_questions(path: Path | None = None) -> dict[str, list[Question]]:
    """Load benchmark questions from YAML, grouped by category.

    Returns:
        Dict mapping category name to list of Question objects.
    """
    data = _load_yaml(path or _QUESTIONS_PATH)
    result: dict[str, list[Question]] = {}

    for category, items in data.items():
        if not isinstance(items, list):
            continue
        questions = []
        for item in items:
            scoring = ScoringMethod(item.get("scoring", "exact"))
            q = Question(
                id=item["id"],
                category=category,
                prompt=item["prompt"],
                expected=item.get("expected", []),
                scoring=scoring,
                tolerance=item.get("tolerance", 0.0),
                keywords=item.get("keywords", []),
                language=item.get("language", ""),
                test_code=item.get("test_code", ""),
                expected_output=item.get("expected_output", ""),
                timeout=item.get("timeout", 30),
            )
            questions.append(q)
        result[category] = questions

    return result


def load_profiles(path: Path | None = None) -> dict:
    """Load benchmark profiles from YAML.

    Returns:
        Full profiles config dict with 'profiles', 'weight_presets', 'runner'.
    """
    return _load_yaml(path or _PROFILES_PATH)


def get_weight_preset(name: str, profiles_data: dict | None = None) -> WeightPreset:
    """Get a named weight preset.

    Args:
        name: Preset name (e.g. 'balanced', 'accuracy_first').
        profiles_data: Optional pre-loaded profiles dict.

    Returns:
        WeightPreset with the corresponding weights.
    """
    data = profiles_data or load_profiles()
    presets = data.get("weight_presets", {})
    preset = presets.get(name, presets.get("balanced", {}))
    return WeightPreset(
        accuracy=preset.get("accuracy", 0.35),
        code=preset.get("code", 0.25),
        structure=preset.get("structure", 0.25),
        speed=preset.get("speed", 0.15),
    )


def get_profile_questions(
    profile_name: str,
    profiles_data: dict | None = None,
    questions_data: dict[str, list[Question]] | None = None,
) -> list[Question]:
    """Get all questions for a specific benchmark profile.

    Args:
        profile_name: Profile key (e.g. 'fast_answer', 'all_round').
        profiles_data: Optional pre-loaded profiles config.
        questions_data: Optional pre-loaded questions dict.

    Returns:
        Flat list of Question objects matching the profile categories.
    """
    data = profiles_data or load_profiles()
    questions = questions_data or load_questions()

    profiles = data.get("profiles", {})
    profile = profiles.get(profile_name)
    if not profile:
        logger.warning("Profile '%s' not found, returning empty list", profile_name)
        return []

    categories = profile.get("categories", [])
    result = []
    for cat in categories:
        result.extend(questions.get(cat, []))
    return result


# ---------------------------------------------------------------------------
# Factual accuracy scoring
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip whitespace/punctuation."""
    text = text.lower().strip()
    # Remove common punctuation that doesn't affect meaning
    text = re.sub(r"[.,;:!?\-\"'()\[\]{}]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _extract_answer(response: str) -> str:
    """Extract the core answer from an LLM response.

    Tries to isolate the actual answer from surrounding explanation.
    Looks for patterns like "The answer is X" or short first-line answers.
    """
    response = response.strip()
    if not response:
        return ""

    # Try common answer patterns
    patterns = [
        r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
        r"(?:result is|equals?)\s*(.+?)(?:\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1).strip()

    # If response is short (likely direct answer), use it whole
    first_line = response.split("\n")[0].strip()
    if len(first_line) < 100:
        return first_line

    return first_line


def score_exact(response: str, expected: list[str]) -> tuple[float, str]:
    """Exact match scoring against expected answers.

    Args:
        response: LLM response text.
        expected: List of acceptable answers.

    Returns:
        Tuple of (score 0.0-1.0, matched answer or empty string).
    """
    answer = _normalize_text(_extract_answer(response))
    # S193 BJD-01: an empty extracted answer must never match (the previous
    # reverse containment scored empty responses 1.0), and reverse
    # containment (answer inside the expected string) needs a minimum
    # length so degenerate one-character answers do not match.
    if answer:
        for exp in expected:
            exp_norm = _normalize_text(exp)
            if not exp_norm:
                continue
            if exp_norm in answer or (len(answer) >= 2 and answer in exp_norm):
                return 1.0, exp
    # Also check the full response body for the expected answer
    full_norm = _normalize_text(response)
    if full_norm:
        for exp in expected:
            exp_norm = _normalize_text(exp)
            if exp_norm and exp_norm in full_norm:
                return 1.0, exp
    return 0.0, ""


def score_fuzzy(
    response: str,
    expected: list[str],
    tolerance: float = 0.8,
) -> tuple[float, str]:
    """Fuzzy match scoring using SequenceMatcher.

    Args:
        response: LLM response text.
        expected: List of acceptable answers.
        tolerance: Minimum similarity ratio to count as a match.

    Returns:
        Tuple of (score 0.0-1.0, best matching answer).
    """
    answer = _normalize_text(_extract_answer(response))
    full_norm = _normalize_text(response)

    best_score = 0.0
    best_match = ""

    for exp in expected:
        exp_norm = _normalize_text(exp)
        # Check extracted answer
        ratio = SequenceMatcher(None, answer, exp_norm).ratio()
        if ratio > best_score:
            best_score = ratio
            best_match = exp
        # Check containment in full response
        if exp_norm in full_norm:
            return 1.0, exp
        # Also try SequenceMatcher on full response substrings
        # (sliding window around expected length)
        exp_len = len(exp_norm)
        for i in range(0, max(1, len(full_norm) - exp_len + 1), max(1, exp_len // 2)):
            window = full_norm[i:i + exp_len + 10]
            ratio = SequenceMatcher(None, window, exp_norm).ratio()
            if ratio > best_score:
                best_score = ratio
                best_match = exp

    if best_score >= tolerance:
        return best_score, best_match
    return best_score, best_match


def score_keyword(
    response: str,
    keywords: list[str],
) -> tuple[float, str]:
    """Keyword containment scoring.

    Args:
        response: LLM response text.
        keywords: List of keywords that should appear in the response.

    Returns:
        Tuple of (score 0.0-1.0, comma-separated found keywords).
    """
    if not keywords:
        return 0.0, ""

    resp_norm = _normalize_text(response)
    found = []
    for kw in keywords:
        if _normalize_text(kw) in resp_norm:
            found.append(kw)

    score = len(found) / len(keywords)
    return score, ", ".join(found)


def evaluate_accuracy(question: Question, response: str) -> AccuracyResult:
    """Evaluate factual accuracy of a response against ground truth.

    Dispatches to the appropriate scoring method based on question config.

    Args:
        question: The benchmark question with expected answers.
        response: LLM-generated response text.

    Returns:
        AccuracyResult with score and match details.
    """
    if question.scoring == ScoringMethod.EXACT:
        score, matched = score_exact(response, question.expected)
        return AccuracyResult(
            question_id=question.id,
            score=score,
            matched_answer=matched,
            method="exact",
            details=f"Exact match against {len(question.expected)} expected answers",
        )

    if question.scoring == ScoringMethod.FUZZY:
        score, matched = score_fuzzy(
            response, question.expected, question.tolerance,
        )
        return AccuracyResult(
            question_id=question.id,
            score=score,
            matched_answer=matched,
            method="fuzzy",
            details=f"Fuzzy match (tolerance={question.tolerance})",
        )

    if question.scoring == ScoringMethod.KEYWORD:
        kw_list = question.keywords or question.expected
        score, matched = score_keyword(response, kw_list)
        return AccuracyResult(
            question_id=question.id,
            score=score,
            matched_answer=matched,
            method="keyword",
            details=f"Keyword match ({len(kw_list)} keywords)",
        )

    return AccuracyResult(
        question_id=question.id,
        score=0.0,
        method="unknown",
        details=f"Unknown scoring method: {question.scoring}",
    )


# ---------------------------------------------------------------------------
# Code quality evaluation (sandbox execution)
# ---------------------------------------------------------------------------

def _extract_code_block(response: str, language: str = "python") -> str:
    """Extract code from markdown code blocks in LLM response.

    Tries fenced blocks first (```python ... ```), then falls back
    to the entire response if no blocks found.
    """
    # Try language-specific fenced block
    pattern = rf"```(?:{language})?\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n".join(matches)

    # Try generic fenced block
    pattern = r"```\s*\n(.*?)```"
    matches = re.findall(pattern, response, re.DOTALL)
    if matches:
        return "\n".join(matches)

    # Fallback: if response looks like code, use it directly
    lines = response.strip().split("\n")
    code_lines = [
        l for l in lines
        if not l.startswith("#") or l.startswith("#!")
        or any(kw in l for kw in ["def ", "import ", "print(", "echo ", "for "])
    ]
    if len(code_lines) > len(lines) * 0.3:
        return response.strip()

    return response.strip()


def evaluate_code(
    question: Question,
    response: str,
    sandbox_mgr: Any = None,
) -> CodeResult:
    """Evaluate code quality by executing in a sandbox.

    Creates a temporary sandbox session, writes the extracted code,
    runs it, optionally runs test assertions, and checks output.

    Args:
        question: Code generation question with test_code and expected_output.
        response: LLM response containing generated code.
        sandbox_mgr: Optional sandbox manager override (for testing).

    Returns:
        CodeResult with execution details and score.
    """
    mgr = sandbox_mgr or sandbox_manager
    if mgr is None:
        return CodeResult(
            question_id=question.id,
            details="Sandbox not available, cannot evaluate code",
        )

    code = _extract_code_block(response, question.language)
    if not code.strip():
        return CodeResult(
            question_id=question.id,
            details="No code extracted from response",
        )

    session_id = f"bench-{question.id}-{uuid.uuid4().hex[:8]}"
    result = CodeResult(question_id=question.id)

    try:
        session = mgr.create_sandbox(session_id, allow_degraded=True)
    except Exception as e:
        result.details = f"Sandbox creation failed: {e}"
        return result

    try:
        timeout = question.timeout or 30

        if question.language == "bash":
            # Execute bash directly
            cmd_result = mgr.execute_command(session_id, code, timeout=timeout)
            result.stdout = cmd_result.stdout
            result.stderr = cmd_result.stderr
            result.return_code = cmd_result.return_code
            result.compiles = True  # bash doesn't compile
            result.runs = cmd_result.return_code == 0

            if question.expected_output:
                actual = result.stdout.strip()
                expected = question.expected_output.strip()
                result.output_matches = actual == expected
        else:
            # Python: write code to file, execute
            code_path = os.path.join(session.workspace_path, "solution.py")
            write_cmd = f"cat > {code_path} << 'BENCH_EOF'\n{code}\nBENCH_EOF"
            mgr.execute_command(session_id, write_cmd, timeout=10)

            # Run the code
            cmd_result = mgr.execute_command(
                session_id,
                f"cd {session.workspace_path} && python3 solution.py",
                timeout=timeout,
            )
            result.stdout = cmd_result.stdout
            result.stderr = cmd_result.stderr
            result.return_code = cmd_result.return_code
            result.compiles = "SyntaxError" not in cmd_result.stderr
            result.runs = cmd_result.return_code == 0

            # Check expected output
            if question.expected_output:
                actual = result.stdout.strip()
                expected = question.expected_output.strip()
                result.output_matches = expected in actual

            # Run test assertions if provided
            if question.test_code and result.runs:
                test_path = os.path.join(session.workspace_path, "test_bench.py")
                full_test = f"{code}\n\n{question.test_code}"
                write_test = (
                    f"cat > {test_path} << 'BENCH_EOF'\n{full_test}\nBENCH_EOF"
                )
                mgr.execute_command(session_id, write_test, timeout=10)

                test_result = mgr.execute_command(
                    session_id,
                    f"cd {session.workspace_path} && python3 test_bench.py",
                    timeout=timeout,
                )
                result.tests_pass = (
                    test_result.return_code == 0
                    and "ALL TESTS PASSED" in test_result.stdout
                )

        # Composite code score
        score = 0.0
        if result.compiles:
            score += 0.2
        if result.runs:
            score += 0.3
        if result.output_matches:
            score += 0.25
        if result.tests_pass:
            score += 0.25
        # If no tests or expected output, redistribute weight
        if not question.test_code and not question.expected_output:
            if result.runs:
                score = 1.0 if result.compiles else 0.5
        elif not question.test_code:
            # No tests, weight output more
            if result.output_matches:
                score = min(1.0, score + 0.25)
        elif not question.expected_output:
            # No expected output, weight tests more
            if result.tests_pass:
                score = min(1.0, score + 0.25)

        result.score = min(1.0, score)
        result.details = (
            f"compiles={result.compiles}, runs={result.runs}, "
            f"output_matches={result.output_matches}, "
            f"tests_pass={result.tests_pass}"
        )

    finally:
        try:
            mgr.destroy_sandbox(session_id)
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# Structural quality metrics
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace+punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_repetition_score(text: str) -> float:
    """Compute n-gram repetition ratio (lower = less repetition = better).

    Measures both bigram and trigram repetition, returns average.
    A score of 0.0 means no repetition, 1.0 means fully repetitive.

    Returns:
        Repetition ratio 0.0 to 1.0 (lower is better).
    """
    tokens = _tokenize(text)
    if len(tokens) < 3:
        return 0.0

    def ngram_repetition(toks: list[str], n: int) -> float:
        if len(toks) < n:
            return 0.0
        ngrams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
        if not ngrams:
            return 0.0
        unique = set(ngrams)
        return 1.0 - (len(unique) / len(ngrams))

    bigram_rep = ngram_repetition(tokens, 2)
    trigram_rep = ngram_repetition(tokens, 3)
    return (bigram_rep + trigram_rep) / 2.0


def compute_lexical_diversity(text: str) -> float:
    """Compute type-token ratio (unique tokens / total tokens).

    Higher = more diverse vocabulary = better.

    Returns:
        Ratio 0.0 to 1.0 (higher is better).
    """
    tokens = _tokenize(text)
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def compute_length_appropriateness(
    text: str,
    expected_range: tuple[int, int] = (10, 600),
) -> float:
    """Score response length relative to expected range.

    Returns 1.0 if within range, decreasing score outside.

    Args:
        text: Response text.
        expected_range: (min_words, max_words) tuple.

    Returns:
        Score 0.0 to 1.0.
    """
    word_count = len(_tokenize(text))
    min_len, max_len = expected_range

    if min_len <= word_count <= max_len:
        return 1.0

    if word_count < min_len:
        if min_len == 0:
            return 1.0
        return max(0.0, word_count / min_len)

    # Too long: gentle decay
    overshoot = word_count - max_len
    decay = max(0.0, 1.0 - (overshoot / max(max_len, 1)))
    return max(0.0, decay)


def compute_format_compliance(text: str, expected_format: str = "") -> float:
    """Check if response follows requested format.

    Supported formats:
        - 'json': checks if response contains valid JSON
        - 'markdown': checks for markdown structure
        - '': no format check, returns 1.0

    Returns:
        Score 0.0 to 1.0.
    """
    if not expected_format:
        return 1.0

    if expected_format == "json":
        # Try to find and parse JSON in the response
        json_pattern = r"\{[^{}]*\}|\[[^\[\]]*\]"
        matches = re.findall(json_pattern, text, re.DOTALL)
        for match in matches:
            try:
                json.loads(match)
                return 1.0
            except json.JSONDecodeError:
                continue
        # Try full response
        try:
            json.loads(text.strip())
            return 1.0
        except (json.JSONDecodeError, ValueError):
            return 0.0

    if expected_format == "markdown":
        score = 0.0
        # Check for headers
        if re.search(r"^#{1,6}\s", text, re.MULTILINE):
            score += 0.3
        # Check for lists
        if re.search(r"^[-*]\s", text, re.MULTILINE):
            score += 0.2
        # Check for code blocks
        if "```" in text:
            score += 0.2
        # Check for paragraphs (multiple lines with blank line separator)
        if "\n\n" in text:
            score += 0.15
        # Check for emphasis
        if re.search(r"\*\*.+?\*\*|__.+?__", text):
            score += 0.15
        return min(1.0, score)

    return 1.0


def evaluate_structure(
    response: str,
    expected_length_range: tuple[int, int] = (10, 600),
    expected_format: str = "",
) -> StructuralResult:
    """Evaluate structural quality of a response.

    Combines repetition, lexical diversity, length appropriateness,
    and format compliance into a composite structural score.

    Args:
        response: LLM response text.
        expected_length_range: (min_words, max_words).
        expected_format: Expected format string ('json', 'markdown', '').

    Returns:
        StructuralResult with individual and composite scores.
    """
    repetition = compute_repetition_score(response)
    diversity = compute_lexical_diversity(response)
    length_score = compute_length_appropriateness(response, expected_length_range)
    format_score = compute_format_compliance(response, expected_format)

    # Composite: invert repetition (lower rep = better), average all
    rep_quality = 1.0 - repetition
    composite = (rep_quality + diversity + length_score + format_score) / 4.0

    return StructuralResult(
        repetition_score=repetition,
        lexical_diversity=diversity,
        length_appropriateness=length_score,
        format_compliance=format_score,
        composite=composite,
        details={
            "word_count": len(_tokenize(response)),
            "expected_range": list(expected_length_range),
            "format_checked": expected_format or "none",
        },
    )


# ---------------------------------------------------------------------------
# Performance scoring
# ---------------------------------------------------------------------------

def evaluate_performance(
    ttft_ms: float = 0.0,
    tokens_per_second: float = 0.0,
    total_time_ms: float = 0.0,
    context_utilization: float = 0.0,
    speed_targets: dict | None = None,
) -> PerformanceResult:
    """Score performance metrics against target thresholds.

    Default speed targets:
        - ttft_ms: < 500ms = 1.0, linear decay to 0 at 5000ms
        - tokens_per_second: > 30 = 1.0, linear decay to 0 at 1 tok/s
        - total_time_ms: < 5000ms = 1.0, linear decay to 0 at 60000ms

    Args:
        ttft_ms: Time to first token in milliseconds.
        tokens_per_second: Generation speed.
        total_time_ms: Total response time in milliseconds.
        context_utilization: Fraction of context window used (0.0-1.0).
        speed_targets: Optional custom targets dict.

    Returns:
        PerformanceResult with normalized score.
    """
    targets = speed_targets or {}

    # S193 BJD-02: a generation that never produced a token is a failure,
    # not an instant answer; ttft=0 must not read as "fast" (the previous
    # behaviour scored a failed/empty query 0.6).
    if tokens_per_second <= 0 and ttft_ms <= 0:
        return PerformanceResult(
            ttft_ms=ttft_ms,
            tokens_per_second=tokens_per_second,
            total_time_ms=total_time_ms,
            context_utilization=context_utilization,
            score=0.0,
        )

    # TTFT score
    ttft_good = targets.get("ttft_good_ms", 500)
    ttft_bad = targets.get("ttft_bad_ms", 5000)
    if ttft_ms <= ttft_good:
        ttft_score = 1.0
    elif ttft_ms >= ttft_bad:
        ttft_score = 0.0
    else:
        ttft_score = 1.0 - (ttft_ms - ttft_good) / (ttft_bad - ttft_good)

    # Tokens/sec score
    tps_good = targets.get("tps_good", 30)
    tps_bad = targets.get("tps_bad", 1)
    if tokens_per_second >= tps_good:
        tps_score = 1.0
    elif tokens_per_second <= tps_bad:
        tps_score = 0.0
    else:
        tps_score = (tokens_per_second - tps_bad) / (tps_good - tps_bad)

    # Total time score
    total_good = targets.get("total_good_ms", 5000)
    total_bad = targets.get("total_bad_ms", 60000)
    if total_time_ms <= total_good:
        total_score = 1.0
    elif total_time_ms >= total_bad:
        total_score = 0.0
    else:
        total_score = 1.0 - (total_time_ms - total_good) / (total_bad - total_good)

    # Composite: weighted average
    composite = (ttft_score * 0.3 + tps_score * 0.4 + total_score * 0.3)

    return PerformanceResult(
        ttft_ms=ttft_ms,
        tokens_per_second=tokens_per_second,
        total_time_ms=total_time_ms,
        context_utilization=context_utilization,
        score=composite,
    )


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------

def compute_composite_score(
    accuracy_score: float,
    code_score: float,
    structure_score: float,
    speed_score: float,
    weights: WeightPreset | None = None,
    evaluated: set[str] | None = None,
) -> float:
    """Compute weighted composite score across all dimensions.

    Args:
        accuracy_score: Factual accuracy score (0.0-1.0).
        code_score: Code execution score (0.0-1.0).
        structure_score: Structural quality score (0.0-1.0).
        speed_score: Performance score (0.0-1.0).
        weights: Weight preset for each dimension.
        evaluated: Optional set of axis names actually evaluated
            ('accuracy', 'code', 'structure', 'speed'). When provided,
            non-evaluated axes are excluded and the remaining weights
            renormalized (S193 BJD-03), so e.g. a profile without code
            questions is no longer capped below 1.0 by a dead code axis.
            Note: composites stored before S193 were computed without
            renormalization and are not directly comparable.

    Returns:
        Composite score 0.0 to 1.0.
    """
    w = weights or WeightPreset()
    wa, wc, ws, wp = w.accuracy, w.code, w.structure, w.speed
    if evaluated is not None:
        wa = wa if "accuracy" in evaluated else 0.0
        wc = wc if "code" in evaluated else 0.0
        ws = ws if "structure" in evaluated else 0.0
        wp = wp if "speed" in evaluated else 0.0
    total_weight = wa + wc + ws + wp
    if total_weight == 0:
        return 0.0

    raw = (
        wa * accuracy_score
        + wc * code_score
        + ws * structure_score
        + wp * speed_score
    )
    return raw / total_weight


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class BenchmarkEvaluator:
    """Facade that holds loaded questions and profiles for reuse.

    Merges built-in profiles from benchmark_profiles.yaml with user-defined
    custom profiles from CustomProfileStore (S90). Custom profiles appear
    alongside built-in ones in every listing and lookup method.
    """

    def __init__(
        self,
        questions_path: Path | None = None,
        profiles_path: Path | None = None,
        custom_profile_store: Any = None,
    ):
        self._questions = load_questions(questions_path)
        self._profiles_data = load_profiles(profiles_path)
        self._custom_store = custom_profile_store

    def _get_custom_store(self) -> Any:
        """Lazy-load the custom profile store if not injected."""
        if self._custom_store is not None:
            return self._custom_store
        try:
            from opti_oignon.benchmark_custom_profiles import (
                custom_profile_store as _store,
                CUSTOM_PROFILES_AVAILABLE,
            )
            if CUSTOM_PROFILES_AVAILABLE and _store is not None:
                return _store
        except ImportError:
            pass
        return None

    def _merged_profiles(self) -> dict[str, dict]:
        """Return built-in profiles merged with custom profiles.

        Custom profiles are keyed by their profile_id and include a
        'custom' flag set to True.
        """
        builtin = dict(self._profiles_data.get("profiles", {}))
        store = self._get_custom_store()
        if store is not None:
            try:
                custom = store.as_profiles_dict()
                builtin.update(custom)
            except Exception as exc:
                logger.debug("Failed to merge custom profiles: %s", exc)
        return builtin

    @property
    def questions(self) -> dict[str, list[Question]]:
        return self._questions

    @property
    def profiles_data(self) -> dict:
        return self._profiles_data

    @property
    def available_categories(self) -> list[str]:
        return list(self._questions.keys())

    @property
    def available_profiles(self) -> list[dict]:
        merged = self._merged_profiles()
        return [
            {
                "id": key,
                "name": p.get("name", key),
                "description": p.get("description", ""),
                "categories": p.get("categories", []),
                "weight_preset": p.get("weight_preset", "balanced"),
                "custom": p.get("custom", False),
            }
            for key, p in merged.items()
        ]

    def get_profile_config(self, profile_name: str) -> dict:
        """Get full config for a profile including weight preset.

        Looks up built-in profiles first, then custom profiles.
        """
        merged = self._merged_profiles()
        return merged.get(profile_name, {})

    def get_weights(self, preset_name: str) -> WeightPreset:
        return get_weight_preset(preset_name, self._profiles_data)

    def get_custom_weights(self, profile_name: str) -> WeightPreset | None:
        """Get custom weights for a profile if it has them.

        Returns None if the profile uses a named preset instead.
        """
        config = self.get_profile_config(profile_name)
        cw = config.get("custom_weights")
        if cw and isinstance(cw, dict):
            return WeightPreset(
                accuracy=cw.get("accuracy", 0.35),
                code=cw.get("code", 0.25),
                structure=cw.get("structure", 0.25),
                speed=cw.get("speed", 0.15),
            )
        return None

    def get_weights_for_profile(self, profile_name: str) -> WeightPreset:
        """Get the effective weights for a profile (custom or preset)."""
        custom = self.get_custom_weights(profile_name)
        if custom is not None:
            return custom
        config = self.get_profile_config(profile_name)
        preset_name = config.get("weight_preset", "balanced")
        return self.get_weights(preset_name)

    def get_questions_for_profile(self, profile_name: str) -> list[Question]:
        """Get questions for a profile (built-in or custom)."""
        merged = self._merged_profiles()
        profile = merged.get(profile_name)
        if not profile:
            logger.warning(
                "Profile '%s' not found, returning empty list", profile_name,
            )
            return []
        categories = profile.get("categories", [])
        result = []
        for cat in categories:
            result.extend(self._questions.get(cat, []))
        return result

    def question_count(self) -> int:
        return sum(len(qs) for qs in self._questions.values())

    def reload(self) -> None:
        """Reload questions and profiles from disk."""
        self._questions = load_questions()
        self._profiles_data = load_profiles()
        store = self._get_custom_store()
        if store is not None:
            try:
                store.reload()
            except Exception:
                pass


# Module singleton
try:
    benchmark_evaluator = BenchmarkEvaluator()
    BENCHMARK_EVALUATOR_AVAILABLE = True
except Exception as e:
    logger.warning("BenchmarkEvaluator init failed: %s", e)
    benchmark_evaluator = None  # type: ignore[assignment]
    BENCHMARK_EVALUATOR_AVAILABLE = False
