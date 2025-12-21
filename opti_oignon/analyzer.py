#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANALYZER - OPTI-OIGNON 1.0
==========================

Automatically detects task type from user input.

This module analyzes input text to determine:
- Task type (R code, Python debug, writing, etc.)
- Programming language if applicable
- Complexity level
- Important keywords
- Preset suggestions based on keyword matching

The analysis allows the Router to select the optimal model.

ENHANCED: Integration with preset keywords for automatic routing.

Author: Léon
"""

from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from dataclasses import dataclass
from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

# Avoid circular imports
if TYPE_CHECKING:
    from .presets import PresetManager, Preset

# =============================================================================
# TASK TYPES
# =============================================================================

class TaskType(Enum):
    """Supported task types."""
    CODE_R = "code_r"
    CODE_PYTHON = "code_python"
    DEBUG_R = "debug_r"
    DEBUG_PYTHON = "debug_python"
    SCIENTIFIC_WRITING = "scientific_writing"
    PLANNING = "planning"
    PLANNING_DEEP = "planning_deep"
    LINUX = "linux"
    SIMPLE_QUESTION = "simple_question"
    UNKNOWN = "unknown"


class Complexity(Enum):
    """Detected complexity level."""
    SIMPLE = "simple"       # Quick question, short answer
    MEDIUM = "medium"       # Standard task
    COMPLEX = "complex"     # Requires deep thinking


class Language(Enum):
    """Detected programming languages."""
    R = "R"
    PYTHON = "Python"
    BASH = "Bash"
    SQL = "SQL"
    JAVASCRIPT = "JavaScript"
    OTHER = "Other"
    NONE = "None"


# =============================================================================
# ANALYSIS RESULT
# =============================================================================

@dataclass
class AnalysisResult:
    """Result of question analysis."""
    task_type: TaskType
    confidence: float           # 0.0 to 1.0
    language: Language
    complexity: Complexity
    keywords: List[str]         # Detected keywords
    is_debug: bool              # Is this about debugging?
    is_code: bool               # Is this about code?
    suggested_model_type: str   # "code", "reasoning", "general", "quick"
    explanation: str            # Detection explanation (for debug)
    suggested_preset_id: Optional[str] = None  # NEW: Suggested preset based on keywords
    preset_score: float = 0.0   # NEW: Score from preset matching
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "task_type": self.task_type.value,
            "confidence": self.confidence,
            "language": self.language.value,
            "complexity": self.complexity.value,
            "keywords": self.keywords,
            "is_debug": self.is_debug,
            "is_code": self.is_code,
            "suggested_model_type": self.suggested_model_type,
            "explanation": self.explanation,
            "suggested_preset_id": self.suggested_preset_id,
            "preset_score": self.preset_score,
        }


# =============================================================================
# DETECTION PATTERNS
# =============================================================================

# R patterns (bioinformatics/ecology focus)
R_PATTERNS = {
    "high": [
        r"\blibrary\s*\(", r"<-", r"\btidyverse\b", r"\bdplyr\b", r"\bggplot",
        r"\bvegan\b", r"\bphyloseq\b", r"\bdeseq2\b", r"\bedger\b",
        r"%>%", r"\|>", r"\btibble\b", r"\.R\b", r"\.Rmd\b",
        r"\bdata\.frame\b", r"\bmatrix\b.*\bR\b", r"\bfunction\s*\(",
    ],
    "medium": [
        r"\bR\b(?!eact)", r"\bRstudio\b", r"\bcran\b", r"\bbioconductor\b",
        r"\bshannon\b", r"\bsimpson\b", r"\brarefaction\b", r"\bpcoa\b",
        r"\bnmds\b", r"\bordination\b", r"\babundance\b", r"\bdiversity\b",
        r"\bécologie\b", r"\bbiodiversité\b", r"\bmetagenom\b",
        r"\becology\b", r"\bbiodiversity\b",
    ],
    "low": [
        r"\bstatistique\b", r"\banalyse\b.*\bdonnées\b", r"\bplot\b",
        r"\bstatistics?\b", r"\banalysis\b.*\bdata\b",
    ],
}

# Python patterns
PYTHON_PATTERNS = {
    "high": [
        r"\bimport\s+\w+", r"\bfrom\s+\w+\s+import", r"\bdef\s+\w+\s*\(",
        r"\bclass\s+\w+", r"\.py\b", r"\bpandas\b", r"\bnumpy\b",
        r"\bscikit[-_]?learn\b", r"\bmatplotlib\b", r"\bseaborn\b",
        r"if\s+__name__\s*==", r"\bpip\s+install\b",
    ],
    "medium": [
        r"\bpython\b", r"\bjupyter\b", r"\bnotebook\b", r"\bcolab\b",
        r"\btensorflow\b", r"\bpytorch\b", r"\bkeras\b",
    ],
    "low": [
        r"\bscript\b", r"\bautomatis\b", r"\bautomate\b",
    ],
}

# Debug/error patterns
DEBUG_PATTERNS = {
    "high": [
        r"\berror\b", r"\berreur\b", r"\btraceback\b", r"\bexception\b",
        r"\bfailed\b", r"\béchou[ée]\b", r"\bcrash\b", r"\bbug\b",
        r"\bne\s+(fonctionne|marche)\s+pas\b", r"\bdoesn'?t\s+work\b",
        r"\bwhy\b.*\b(error|fail)\b", r"\bpourquoi\b.*\b(erreur|error)\b",
    ],
    "medium": [
        r"\bproblème\b", r"\bproblem\b", r"\bissue\b",
        r"\bne\s+pas\b.*\bfonctionne\b", r"\bhelp\b.*\b(understand|solve)\b",
        r"\baide\b.*\b(comprendre|résoudre)\b", r"\bwarning\b",
    ],
}

# Scientific writing patterns
WRITING_PATTERNS = {
    "high": [
        r"\babstract\b", r"\brésumé\b", r"\bméthodes?\b", r"\bmethods?\b",
        r"\brésultats?\b", r"\bresults?\b", r"\bmanuscrit\b", r"\bmanuscript\b",
        r"\bpublication\b", r"\breview(er)?\b", r"\barticle\b",
        r"\bintroduction\b", r"\bdiscussion\b",
    ],
    "medium": [
        r"\brédige\b", r"\bwrite\b", r"\bécris\b", r"\bformule\b",
        r"\bexplain\b.*\bscientific\b", r"\bexplique\b.*\bscientifique\b",
        r"\bacadémique\b", r"\bacademic\b", r"\bthèse\b", r"\bthesis\b",
        r"\bmémoire\b", r"\bdissertation\b",
    ],
}

# Planning/reflection patterns
PLANNING_PATTERNS = {
    "high": [
        r"\bplanifier\b", r"\bplan\b", r"\borganiser\b", r"\borganize\b",
        r"\bstratégie\b", r"\bstrategy\b", r"\bétapes?\b", r"\bsteps?\b",
        r"\bbrainstorm\b", r"\bréfléchir\b", r"\bthink\b",
    ],
    "medium": [
        r"\bcomment\s+(faire|procéder|organiser)\b", r"\bhow\s+to\b",
        r"\bmeilleure\s+façon\b", r"\bbest\s+way\b",
        r"\bconseils?\b", r"\badvice\b", r"\brecommand\b", r"\bdécider\b",
    ],
    "complex": [
        r"\bcomparer\b.*\boptions\b", r"\bcompare\b.*\boptions\b",
        r"\banalyse\s+(approfondie|complète)\b", r"\bin-depth\s+analysis\b",
        r"\bpour\s+et\s+contre\b", r"\bpros\s+and\s+cons\b",
        r"\bpeser\b.*\balternatives\b", r"\bweigh\b.*\balternatives\b",
        r"\braisonne\b", r"\breason\b", r"\bréfléchis\b.*\bétapes\b",
    ],
}

# Linux/Bash patterns
LINUX_PATTERNS = {
    "high": [
        r"\bbash\b", r"\bshell\b", r"\bterminal\b", r"\bcommande\b", r"\bcommand\b",
        r"\bsudo\b", r"\bapt\b", r"\bchmod\b", r"\bchown\b",
        r"#!/bin/(ba)?sh", r"\bpipe\b", r"\bgrep\b", r"\bsed\b", r"\bawk\b",
    ],
    "medium": [
        r"\blinux\b", r"\bubuntu\b", r"\bkubuntu\b", r"\bdebian\b",
        r"\bfichier\b.*\bsystème\b", r"\bfile\b.*\bsystem\b",
        r"\bpermission\b", r"\bpath\b",
    ],
}


# =============================================================================
# MAIN CLASS
# =============================================================================

class TaskAnalyzer:
    """
    Intelligent task analyzer.
    
    Analyzes user text to determine task type, programming language,
    complexity, etc. Now with integration for preset keyword matching.
    
    Usage:
        analyzer = TaskAnalyzer()
        result = analyzer.analyze("Why do I have an error with rowSums?")
        print(result.task_type)  # TaskType.DEBUG_R
        print(result.suggested_preset_id)  # "debug_r" (if matching preset found)
    """
    
    def __init__(self):
        """Initialize the analyzer."""
        self._compile_patterns()
        self._preset_manager = None
    
    def _compile_patterns(self):
        """Compile regex for better performance."""
        self._patterns = {
            "r": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                  for k, v in R_PATTERNS.items()},
            "python": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                       for k, v in PYTHON_PATTERNS.items()},
            "debug": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                      for k, v in DEBUG_PATTERNS.items()},
            "writing": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                        for k, v in WRITING_PATTERNS.items()},
            "planning": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                         for k, v in PLANNING_PATTERNS.items()},
            "linux": {k: [re.compile(p, re.IGNORECASE) for p in v] 
                      for k, v in LINUX_PATTERNS.items()},
        }
    
    def _get_preset_manager(self) -> Optional['PresetManager']:
        """Get the preset manager (lazy loading to avoid circular imports)."""
        if self._preset_manager is None:
            try:
                from .presets import preset_manager
                self._preset_manager = preset_manager
            except ImportError:
                logger.warning("Could not import preset_manager")
                return None
        return self._preset_manager
    
    def _score_patterns(self, text: str, patterns: Dict[str, List]) -> Tuple[float, List[str]]:
        """
        Calculate score based on matched patterns.
        
        Returns:
            (score, keywords_found)
        """
        score = 0.0
        keywords = []
        
        weights = {"high": 3.0, "medium": 1.5, "low": 0.5, "complex": 2.0}
        
        for level, pattern_list in patterns.items():
            weight = weights.get(level, 1.0)
            for pattern in pattern_list:
                matches = pattern.findall(text)
                if matches:
                    score += weight * len(matches)
                    keywords.extend([m if isinstance(m, str) else m[0] for m in matches[:2]])
        
        return score, list(set(keywords))[:5]  # Max 5 keywords
    
    def _score_with_presets(self, text: str) -> Tuple[Optional[str], float, int]:
        """
        Score text against preset keywords.
        
        Args:
            text: Text to analyze
            
        Returns:
            (best_preset_id, weighted_score, match_count) or (None, 0, 0)
        """
        pm = self._get_preset_manager()
        if pm is None:
            return None, 0.0, 0
        
        try:
            results = pm.find_by_keywords_with_scores(text, min_matches=1)
            if results:
                best_preset, score, matches = results[0]
                return best_preset.id, score, matches
        except Exception as e:
            logger.warning(f"Error scoring with presets: {e}")
        
        return None, 0.0, 0
    
    def _detect_language(self, text: str, document: Optional[str] = None) -> Language:
        """Detect programming language."""
        # Score for each language
        r_score, _ = self._score_patterns(text + (document or ""), self._patterns["r"])
        py_score, _ = self._score_patterns(text + (document or ""), self._patterns["python"])
        linux_score, _ = self._score_patterns(text + (document or ""), self._patterns["linux"])
        
        # Highest score wins
        scores = {"R": r_score, "Python": py_score, "Bash": linux_score}
        best = max(scores, key=scores.get)
        
        if scores[best] < 1.0:  # Minimum threshold
            return Language.NONE
        
        return Language[best.upper()] if best != "Bash" else Language.BASH
    
    def _detect_complexity(self, text: str) -> Complexity:
        """Detect question complexity."""
        # Complexity indicators
        complex_indicators = [
            r"\bcomplet\b", r"\bcomplete\b", r"\bdétaillé\b", r"\bdetailed\b",
            r"\bapprofondi\b", r"\bin-depth\b", r"\bcomparer\b", r"\bcompare\b",
            r"\banalyse\b", r"\banalyze\b", r"\bexpliquer\s+en\s+détail\b",
            r"\bexplain\s+in\s+detail\b", r"\bétape\s+par\s+étape\b",
            r"\bstep\s+by\s+step\b", r"\bpour\s+et\s+contre\b", r"\bpros\s+and\s+cons\b",
        ]
        
        simple_indicators = [
            r"\brapide\b", r"\bquick\b", r"\bcourt\b", r"\bshort\b",
            r"\bsimple\b", r"\bjuste\b", r"\bjust\b",
            r"^(qu'?est[- ]ce|c'?est quoi|comment|pourquoi|what|how|why)\b",
        ]
        
        # Count indicators
        complex_count = sum(1 for p in complex_indicators if re.search(p, text, re.IGNORECASE))
        simple_count = sum(1 for p in simple_indicators if re.search(p, text, re.IGNORECASE))
        
        # Text length as factor
        text_length = len(text.split())
        
        if complex_count >= 2 or text_length > 100:
            return Complexity.COMPLEX
        elif simple_count >= 1 or text_length < 20:
            return Complexity.SIMPLE
        else:
            return Complexity.MEDIUM
    
    def analyze(
        self, 
        text: str, 
        document: Optional[str] = None,
        force_task: Optional[str] = None,
        use_preset_keywords: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a question/request to determine task type.
        
        Args:
            text: User question
            document: Optional document/code content
            force_task: If specified, force this task type
            use_preset_keywords: Whether to check preset keywords (default: True)
            
        Returns:
            AnalysisResult with all detected information
        """
        full_text = text + "\n" + (document or "")
        
        # Check preset keywords first (if enabled)
        suggested_preset_id = None
        preset_score = 0.0
        
        if use_preset_keywords and not force_task:
            suggested_preset_id, preset_score, _ = self._score_with_presets(full_text)
        
        # If task forced
        if force_task:
            try:
                task_type = TaskType(force_task)
                return AnalysisResult(
                    task_type=task_type,
                    confidence=1.0,
                    language=self._detect_language(full_text),
                    complexity=self._detect_complexity(text),
                    keywords=[],
                    is_debug="debug" in force_task,
                    is_code="code" in force_task or "debug" in force_task,
                    suggested_model_type=self._get_model_type(task_type),
                    explanation=f"Forced task: {force_task}",
                    suggested_preset_id=suggested_preset_id,
                    preset_score=preset_score,
                )
            except ValueError:
                logger.warning(f"Invalid forced task: {force_task}")
        
        # Scores for each category
        scores = {}
        all_keywords = []
        
        # R score
        r_score, r_kw = self._score_patterns(full_text, self._patterns["r"])
        scores["r"] = r_score
        all_keywords.extend(r_kw)
        
        # Python score
        py_score, py_kw = self._score_patterns(full_text, self._patterns["python"])
        scores["python"] = py_score
        all_keywords.extend(py_kw)
        
        # Debug score
        debug_score, debug_kw = self._score_patterns(full_text, self._patterns["debug"])
        scores["debug"] = debug_score
        all_keywords.extend(debug_kw)
        
        # Writing score
        writing_score, writing_kw = self._score_patterns(full_text, self._patterns["writing"])
        scores["writing"] = writing_score
        all_keywords.extend(writing_kw)
        
        # Planning score
        planning_score, planning_kw = self._score_patterns(full_text, self._patterns["planning"])
        scores["planning"] = planning_score
        all_keywords.extend(planning_kw)
        
        # Linux score
        linux_score, linux_kw = self._score_patterns(full_text, self._patterns["linux"])
        scores["linux"] = linux_score
        all_keywords.extend(linux_kw)
        
        # Determine if debug
        is_debug = debug_score >= 2.0
        
        # Determine task type
        task_type, confidence, explanation = self._determine_task(scores, is_debug)
        
        # Determine language
        language = self._detect_language(text, document)
        
        # Adjust task_type if debug + language detected
        if is_debug:
            if language == Language.R:
                task_type = TaskType.DEBUG_R
            elif language == Language.PYTHON:
                task_type = TaskType.DEBUG_PYTHON
        
        # Determine complexity
        complexity = self._detect_complexity(text)
        
        # If high complexity and planning, use planning_deep
        if task_type == TaskType.PLANNING and complexity == Complexity.COMPLEX:
            task_type = TaskType.PLANNING_DEEP
            explanation += " + high complexity -> planning_deep"
        
        # Determine if code
        is_code = task_type in [TaskType.CODE_R, TaskType.CODE_PYTHON, 
                                TaskType.DEBUG_R, TaskType.DEBUG_PYTHON]
        
        # Deduplicate keywords
        unique_keywords = list(dict.fromkeys(all_keywords))[:5]
        
        # Add preset info to explanation if found
        if suggested_preset_id and preset_score > 0:
            explanation += f" | Preset match: {suggested_preset_id} (score={preset_score:.1f})"
        
        return AnalysisResult(
            task_type=task_type,
            confidence=min(confidence, 1.0),
            language=language,
            complexity=complexity,
            keywords=unique_keywords,
            is_debug=is_debug,
            is_code=is_code,
            suggested_model_type=self._get_model_type(task_type),
            explanation=explanation,
            suggested_preset_id=suggested_preset_id,
            preset_score=preset_score,
        )
    
    def _determine_task(
        self, 
        scores: Dict[str, float], 
        is_debug: bool
    ) -> Tuple[TaskType, float, str]:
        """Determine task type from scores."""
        # Find max score (excluding debug which is cross-cutting)
        max_category = max(
            [(k, v) for k, v in scores.items() if k != "debug"],
            key=lambda x: x[1],
            default=("unknown", 0)
        )
        
        category, score = max_category
        total_score = sum(scores.values())
        confidence = score / max(total_score, 1) if total_score > 0 else 0
        
        # Category -> TaskType mapping
        task_map = {
            "r": TaskType.CODE_R,
            "python": TaskType.CODE_PYTHON,
            "writing": TaskType.SCIENTIFIC_WRITING,
            "planning": TaskType.PLANNING,
            "linux": TaskType.LINUX,
        }
        
        # Minimum threshold
        if score < 1.0:
            return TaskType.SIMPLE_QUESTION, 0.5, f"Score too low ({score:.1f}), simple question"
        
        task_type = task_map.get(category, TaskType.SIMPLE_QUESTION)
        explanation = f"{category}={score:.1f}, conf={confidence:.2f}"
        
        return task_type, confidence, explanation
    
    def _get_model_type(self, task_type: TaskType) -> str:
        """Return recommended model type for a task."""
        model_type_map = {
            TaskType.CODE_R: "code",
            TaskType.CODE_PYTHON: "code",
            TaskType.DEBUG_R: "code",
            TaskType.DEBUG_PYTHON: "code",
            TaskType.SCIENTIFIC_WRITING: "general",
            TaskType.PLANNING: "reasoning",
            TaskType.PLANNING_DEEP: "reasoning",
            TaskType.LINUX: "code",
            TaskType.SIMPLE_QUESTION: "quick",
            TaskType.UNKNOWN: "general",
        }
        return model_type_map.get(task_type, "general")


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

analyzer = TaskAnalyzer()


def analyze(text: str, document: Optional[str] = None, force_task: Optional[str] = None) -> AnalysisResult:
    """Convenience function to analyze a question."""
    return analyzer.analyze(text, document, force_task)


def detect_task(text: str) -> str:
    """
    Simply detect task type (backwards compatible).
    
    Returns:
        Task type string (e.g., "code_r", "debug_python")
    """
    result = analyzer.analyze(text)
    return result.task_type.value


def detect_with_preset(text: str, document: Optional[str] = None) -> Tuple[str, Optional[str], float]:
    """
    Detect task type with preset suggestion.
    
    Returns:
        (task_type, suggested_preset_id, preset_score)
    """
    result = analyzer.analyze(text, document)
    return result.task_type.value, result.suggested_preset_id, result.preset_score


# =============================================================================
# TASK ALIASES (for backwards compatibility)
# =============================================================================

TASK_ALIASES = {
    "r": "code_r",
    "R": "code_r",
    "py": "code_python",
    "python": "code_python",
    "debug": "debug_r",  # Default debug = debug R
    "writing": "scientific_writing",
    "redaction": "scientific_writing",
    "plan": "planning",
    "think": "planning_deep",
    "bash": "linux",
    "shell": "linux",
    "quick": "simple_question",
}


def resolve_alias(task: str) -> str:
    """Resolve an alias to full task name."""
    return TASK_ALIASES.get(task, task)


# =============================================================================
# TEST CLI
# =============================================================================

if __name__ == "__main__":
    import sys
    
    test_cases = [
        "How to calculate Shannon index in R?",
        "Error in rowSums(df): 'x' must be numeric",
        "Write a Python function to parse a CSV",
        "Traceback: ImportError pandas",
        "How to organize my thesis project?",
        "Write an abstract about biodiversity",
        "sudo apt install nvidia-driver",
        "What is the capital of France?",
        "Compare pros and cons of ggplot vs base R, detailed analysis",
        "Can you help with ggplot2 and dplyr for ecology analysis?",
    ]
    
    if len(sys.argv) > 1:
        # Analyze the passed argument
        test_cases = [" ".join(sys.argv[1:])]
    
    print("=== Analyzer Test (with Preset Integration) ===\n")
    
    for test in test_cases:
        result = analyze(test)
        print(f"Question: {test[:60]}...")
        print(f"  -> Type: {result.task_type.value}")
        print(f"  -> Language: {result.language.value}")
        print(f"  -> Complexity: {result.complexity.value}")
        print(f"  -> Confidence: {result.confidence:.2f}")
        print(f"  -> Suggested model: {result.suggested_model_type}")
        print(f"  -> Debug: {result.is_debug}, Code: {result.is_code}")
        if result.suggested_preset_id:
            print(f"  -> Preset suggestion: {result.suggested_preset_id} (score={result.preset_score:.1f})")
        print(f"  -> Explanation: {result.explanation}")
        print()
