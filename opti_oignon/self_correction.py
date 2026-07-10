#!/usr/bin/env python3
"""
SELF-CORRECTION ENGINE - OPTI-OIGNON
===================================================

Moteur d'auto-correction pour les sorties non-code.
Checks conformity to instructions, factual coherence,
et la qualite generale des reponses LLM, puis corrige si required.

Trois axes de verification:

1. Instruction Compliance:
   Extract key constraints from the user request
   (format, longueur, ton, contraintes explicites) et verifie
   que la reponse les satisfait. Score 0.0-1.0.

2. Factual Self-Check:
   Demande au model d'identifier les affirmations potentiellement
   incorrectes dans sa propre reponse, et signale les incertitudes.

3. Quality Assessment:
   Checks completeness (all parts of the request are
   traitees), la coherence structurelle, et les marqueurs
   d'hallucination (langage hedging, affirmations non etayees).

4. Self-Repair Loop:
   If the score is below the threshold, generates a corrected version.
   Maximum N iterations (configurable). Returns the best
   version with the correction metadata.
"""

import json
import logging
import re
import time
from collections.abc import Generator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Import conditionnel de la config YAML
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import conditionnel d'Ollama
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

logger = logging.getLogger(__name__)


def _generate_text(result: Any) -> str:
    """Pull the response text from an ollama-style generate response.

    Handles the dict shape and the object shape returned by newer
    ollama-python (a GenerateResponse is not subscriptable and has no
    .get), so an object-form response no longer raises an AttributeError
    that silently degraded every LLM check to heuristics and made
    _generate_correction return None -- i.e. the loop never corrected
    (the dict-vs-object class of failure; mirrors
    memory/legacy._reply_text).
    """
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        return str(result.get("response") or "")
    return str(getattr(result, "response", "") or "")


def _clamp01(value: Any, default: float) -> float:
    """Clamp an LLM-returned score into [0, 1].

    A model returning an out-of-range score (e.g. 5.0) would otherwise
    pass the correction thresholds wrongly.
    """
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


# =============================================================================
# DATACLASSES DE RESULTAT
# =============================================================================

@dataclass
class InstructionCheck:
    """Result de la verification of ae instruction individuelle."""
    instruction: str
    satisfied: bool
    explanation: str = ""
    confidence: float = 1.0


@dataclass
class ComplianceResult:
    """Result de la verification de conformite aux instructions."""
    score: float = 1.0
    instructions_found: list[str] = field(default_factory=list)
    checks: list[InstructionCheck] = field(default_factory=list)
    satisfied_count: int = 0
    total_count: int = 0


@dataclass
class FactualFlag:
    """Un signalement de probleme factuel potentiel."""
    claim: str
    concern: str
    severity: str = "low"  # low, medium, high


@dataclass
class FactCheckResult:
    """Result de la verification factuelle."""
    flags: list[FactualFlag] = field(default_factory=list)
    flag_count: int = 0
    confidence: float = 1.0


@dataclass
class QualityResult:
    """Result de l'evaluation de qualite."""
    completeness_score: float = 1.0
    coherence_score: float = 1.0
    hallucination_risk: float = 0.0
    overall_score: float = 1.0
    issues: list[str] = field(default_factory=list)


@dataclass
class CorrectionIteration:
    """Result of ae iteration de correction."""
    iteration: int
    compliance_score: float
    quality_score: float
    response_text: str
    improvements: list[str] = field(default_factory=list)
    duration_ms: int = 0


@dataclass
class SelfCorrectionResult:
    """Result complet du processus d'auto-correction."""
    original_response: str
    corrected_response: str
    was_corrected: bool = False
    iterations_performed: int = 0
    compliance_before: float = 1.0
    compliance_after: float = 1.0
    quality_before: float = 1.0
    quality_after: float = 1.0
    compliance_result: ComplianceResult | None = None
    fact_check_result: FactCheckResult | None = None
    quality_result: QualityResult | None = None
    iteration_history: list[CorrectionIteration] = field(default_factory=list)
    total_duration_ms: int = 0
    model_used: str = ""


@dataclass
class SelfCorrectionConfig:
    """Configuration du moteur d'auto-correction."""
    enable_auto: bool = False
    max_iterations: int = 2
    compliance_threshold: float = 0.7
    quality_threshold: float = 0.6
    check_instructions: bool = True
    check_facts: bool = True
    check_quality: bool = True
    correction_model: str | None = None
    temperature: float = 0.2


# =============================================================================
# CONSTANTES ET PATTERNS
# =============================================================================

# Mots-cles d'instruction detectables dans les requetes
_FORMAT_KEYWORDS = [
    "format", "formate", "en liste", "as a list", "bullet points",
    "a list of", "une liste",
    "en tableau", "as a table", "markdown", "json", "csv",
    "en paragraphes", "in paragraphs", "numbered",
]

_LENGTH_KEYWORDS = [
    "court", "short", "brief", "bref", "concis", "concise",
    "long", "detaille", "detailed", "en detail", "in detail",
    "mots", "words", "phrases", "sentences", "paragraphes", "paragraphs",
    "maximum", "minimum", "au moins", "at least", "pas plus de", "no more than",
]

_TONE_KEYWORDS = [
    "formel", "formal", "informel", "informal", "casual",
    "professionnel", "professional", "academique", "academic",
    "simple", "technique", "technical", "vulgarise", "simplified",
    "amical", "friendly", "serieux", "serious",
]

_CONSTRAINT_PATTERNS = [
    # "not mentionner X", "do not mention X"
    r"(?:ne\s+pas|do\s+not|don't|n'?\s*(?:utilise|mentionne|inclus))\s+(.+)",
    # "inclure X", "include X"
    r"(?:inclure?|include|ajoute|add)\s+(.+)",
    # "en francais", "in English"
    r"(?:en|in)\s+(fran[cç]ais|english|anglais|spanish|espagnol)",
    # "exactement N", "exactly N"
    r"(?:exactement|exactly|precisely)\s+(\d+)",
]

# Marqueurs d'hallucination potentielle
_HEDGING_MARKERS = [
    "i believe", "je crois", "i think", "je pense",
    "probably", "probablement", "possibly", "possiblement",
    "it seems", "il semble", "approximately", "environ",
    "might be", "pourrait etre", "could be",
]

_OVERCONFIDENCE_MARKERS = [
    "it is certain", "il est certain",
    "without a doubt", "sans aucun doute",
    "absolutely", "absolument",
    "there is no question", "definitively",
    "guaranteed", "garanti",
]


# =============================================================================
# EXTRACTION D'INSTRUCTIONS
# =============================================================================

def extract_instructions(user_message: str) -> list[str]:
    """Extract instructions and key constraints from the user message.

    Analyse le message pour identifier les demandes de format,
    de longueur, de ton, et les contraintes explicites.

    Args:
        user_message: Message original de l'utilisateur

    Returns:
        Liste de chaines decrivant les instructions detectees
    """
    instructions = []
    msg_lower = user_message.lower()

    # Detection de format
    for kw in _FORMAT_KEYWORDS:
        if kw in msg_lower:
            instructions.append(f"format: {kw}")
            break

    # Detection de longueur
    for kw in _LENGTH_KEYWORDS:
        if kw in msg_lower:
            instructions.append(f"length: {kw}")
            break

    # Detection de ton
    for kw in _TONE_KEYWORDS:
        if kw in msg_lower:
            instructions.append(f"tone: {kw}")
            break

    # Detection de contraintes explicites via regex
    for pattern in _CONSTRAINT_PATTERNS:
        matches = re.findall(pattern, msg_lower)
        for match in matches:
            instructions.append(f"constraint: {match.strip()}")

    # Detection de questions multiples (completude)
    questions = [s.strip() for s in user_message.split("?") if s.strip()]
    if len(questions) > 1:
        instructions.append(f"completeness: {len(questions)} questions to answer")

    # Detection de langue demandee
    if "en francais" in msg_lower or "in french" in msg_lower:
        instructions.append("language: french")
    elif "en anglais" in msg_lower or "in english" in msg_lower:
        instructions.append("language: english")

    return instructions


def compute_heuristic_compliance(
    user_message: str,
    response: str,
    instructions: list[str],
) -> ComplianceResult:
    """Compute a heuristic conformity score (without LLM).

    Check detected instructions via simple rules:
    - Format: presence de listes, tableaux, etc.
    - Longueur: taille approximative
    - Ton: indicateurs basiques
    - Langue: detection sommaire

    Args:
        user_message: Message original
        response: Reponse du LLM
        instructions: Instructions extraites

    Returns:
        ComplianceResult avec scores et details
    """
    if not instructions:
        return ComplianceResult(
            score=1.0,
            instructions_found=[],
            checks=[],
            satisfied_count=0,
            total_count=0,
        )

    checks = []
    resp_lower = response.lower()

    for instr in instructions:
        satisfied = True
        explanation = ""

        if instr.startswith("format:"):
            fmt = instr.split(":", 1)[1].strip()
            if fmt in ("en liste", "as a list", "bullet points", "a list of", "une liste"):
                has_list = bool(re.search(r"[\-\*]\s", response) or re.search(r"\d+\.\s", response))
                satisfied = has_list
                explanation = "List format detected" if has_list else "No list format found"
            elif fmt in ("en tableau", "as a table"):
                has_table = "|" in response and "-" in response
                satisfied = has_table
                explanation = "Table format detected" if has_table else "No table format found"
            elif fmt == "json":
                has_json = "{" in response and "}" in response
                satisfied = has_json
                explanation = "JSON structure detected" if has_json else "No JSON found"
            elif fmt == "markdown":
                has_md = "#" in response or "**" in response or "```" in response
                satisfied = has_md
                explanation = "Markdown detected" if has_md else "No markdown formatting"
            else:
                satisfied = True
                explanation = "Format check skipped"

        elif instr.startswith("length:"):
            length_kw = instr.split(":", 1)[1].strip()
            word_count = len(response.split())
            if length_kw in ("court", "short", "brief", "bref", "concis", "concise"):
                satisfied = word_count < 200
                explanation = f"{word_count} words (expected short)"
            elif length_kw in ("long", "detaille", "detailed", "en detail", "in detail"):
                satisfied = word_count > 100
                explanation = f"{word_count} words (expected detailed)"
            else:
                satisfied = True
                explanation = f"{word_count} words"

        elif instr.startswith("language:"):
            lang = instr.split(":", 1)[1].strip()
            if lang == "french":
                # Heuristique simple: mots francais frequents
                fr_markers = ["le", "la", "les", "de", "du", "des", "est", "sont", "dans"]
                fr_count = sum(1 for m in fr_markers if f" {m} " in f" {resp_lower} ")
                satisfied = fr_count >= 3
                explanation = f"French markers: {fr_count}/9"
            elif lang == "english":
                en_markers = ["the", "is", "are", "of", "in", "to", "and", "for"]
                en_count = sum(1 for m in en_markers if f" {m} " in f" {resp_lower} ")
                satisfied = en_count >= 3
                explanation = f"English markers: {en_count}/8"

        elif instr.startswith("completeness:"):
            # Check que la reponse is not trop courte pour le nb de questions
            parts = instr.split(":", 1)[1].strip()
            try:
                n_questions = int(parts.split()[0])
            except (ValueError, IndexError):
                n_questions = 2
            # Heuristique: au moins 30 mots par question
            word_count = len(response.split())
            satisfied = word_count >= n_questions * 30
            explanation = f"{word_count} words for {n_questions} questions"

        elif instr.startswith("constraint:"):
            # Les contraintes explicites sont difficiles a check
            # sans LLM -- on marque comme "unchecked" avec confiance faible
            satisfied = True
            explanation = "Constraint requires LLM verification"

        elif instr.startswith("tone:"):
            # Les tons sont difficiles a check heuristiquement
            satisfied = True
            explanation = "Tone check requires LLM verification"

        checks.append(InstructionCheck(
            instruction=instr,
            satisfied=satisfied,
            explanation=explanation,
            confidence=0.7 if "requires LLM" in explanation else 0.9,
        ))

    satisfied_count = sum(1 for c in checks if c.satisfied)
    total_count = len(checks)
    score = satisfied_count / total_count if total_count > 0 else 1.0

    return ComplianceResult(
        score=score,
        instructions_found=instructions,
        checks=checks,
        satisfied_count=satisfied_count,
        total_count=total_count,
    )


def compute_heuristic_quality(
    user_message: str,
    response: str,
) -> QualityResult:
    """Compute a heuristic quality score (without LLM).

    Evalue:
    - Completude: la reponse est-elle assez substantielle?
    - Coherence: structure basique OK?
    - Risque d'hallucination: marqueurs de hedging/overconfidence

    Args:
        user_message: Message original
        response: Reponse du LLM

    Returns:
        QualityResult avec scores et problemes
    """
    issues = []
    resp_lower = response.lower()
    word_count = len(response.split())

    # -- Completude --
    # Reponse trop courte pour une question substantielle?
    q_word_count = len(user_message.split())
    if q_word_count > 20 and word_count < 30:
        completeness = 0.3
        issues.append("Response seems too short for the question complexity")
    elif q_word_count > 50 and word_count < 80:
        completeness = 0.5
        issues.append("Response may be incomplete for a complex query")
    elif word_count < 5:
        completeness = 0.1
        issues.append("Response is extremely short")
    else:
        completeness = min(1.0, word_count / max(q_word_count * 2, 50))

    # -- Coherence --
    # Check la structure basique
    sentences = [s.strip() for s in re.split(r'[.!?]+', response) if s.strip()]
    if len(sentences) == 0:
        coherence = 0.2
        issues.append("No complete sentences found")
    elif len(sentences) == 1 and word_count > 50:
        coherence = 0.6
        issues.append("Long response with no sentence breaks")
    else:
        coherence = 1.0

    # Repetitions excessives
    words = response.lower().split()
    if len(words) > 20:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            coherence = min(coherence, 0.4)
            issues.append("High word repetition detected")
        elif unique_ratio < 0.5:
            coherence = min(coherence, 0.7)
            issues.append("Moderate word repetition")

    # -- Risque d'hallucination --
    hedging_count = sum(1 for m in _HEDGING_MARKERS if m in resp_lower)
    overconfidence_count = sum(1 for m in _OVERCONFIDENCE_MARKERS if m in resp_lower)

    hallucination_risk = 0.0
    if hedging_count > 3:
        hallucination_risk = min(0.8, hedging_count * 0.15)
        issues.append(f"Excessive hedging language ({hedging_count} markers)")
    if overconfidence_count > 2:
        hallucination_risk = max(hallucination_risk, min(0.7, overconfidence_count * 0.2))
        issues.append(f"Overconfident language ({overconfidence_count} markers)")

    # Score global
    overall = (
        completeness * 0.4
        + coherence * 0.4
        + (1.0 - hallucination_risk) * 0.2
    )

    return QualityResult(
        completeness_score=round(completeness, 3),
        coherence_score=round(coherence, 3),
        hallucination_risk=round(hallucination_risk, 3),
        overall_score=round(overall, 3),
        issues=issues,
    )


# =============================================================================
# MOTEUR D'AUTO-CORRECTION
# =============================================================================

class SelfCorrectionEngine:
    """Moteur d'auto-correction pour les sorties non-code.

    Checks conformity to instructions, factual coherence,
    et la qualite generale, puis corrige si les scores sont
    inferieurs aux seuils configures.
    """

    def __init__(
        self,
        config: SelfCorrectionConfig | None = None,
        config_path: str | None = None,
    ):
        """Initialize the self-correction engine.

        Args:
            config: Configuration explicite (prioritaire)
            config_path: Chemin vers le fichier YAML de configuration
        """
        if config is not None:
            self._config = config
        else:
            self._config = self._load_config(config_path)

        logger.info(
            f"SelfCorrectionEngine: initialized "
            f"(max_iterations={self._config.max_iterations}, "
            f"compliance_threshold={self._config.compliance_threshold}, "
            f"quality_threshold={self._config.quality_threshold})"
        )

    # -----------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------

    def _load_config(self, config_path: str | None = None) -> SelfCorrectionConfig:
        """Load configuration from a YAML file.

        Args:
            config_path: Chemin vers le fichier de config (ou None pour default)

        Returns:
            SelfCorrectionConfig avec les valeurs chargees
        """
        if not YAML_AVAILABLE:
            logger.debug("YAML unavailable, using default config")
            return SelfCorrectionConfig()

        # Chercher le fichier de config
        if config_path is None:
            config_path = str(
                Path(__file__).parent / "config" / "self_correction.yaml"
            )

        try:
            with open(config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

            sc = data.get("self_correction", {})
            return SelfCorrectionConfig(
                enable_auto=sc.get("enable_auto", False),
                max_iterations=sc.get("max_iterations", 2),
                compliance_threshold=sc.get("compliance_threshold", 0.7),
                quality_threshold=sc.get("quality_threshold", 0.6),
                check_instructions=sc.get("check_instructions", True),
                check_facts=sc.get("check_facts", True),
                check_quality=sc.get("check_quality", True),
                correction_model=sc.get("correction_model", None),
                temperature=sc.get("temperature", 0.2),
            )
        except FileNotFoundError:
            logger.debug(f"Config introuvable: {config_path}, default")
            return SelfCorrectionConfig()
        except Exception as e:
            logger.warning(f"Erreur loading config: {e}")
            return SelfCorrectionConfig()

    @property
    def config(self) -> SelfCorrectionConfig:
        """Configuration actuelle."""
        return self._config

    @property
    def available(self) -> bool:
        """Indique si le moteur est operationnel."""
        return OLLAMA_AVAILABLE

    # -----------------------------------------------------------------
    # Checks principaux
    # -----------------------------------------------------------------

    def check_compliance(
        self,
        user_message: str,
        response: str,
        model: str | None = None,
        use_llm: bool = False,
    ) -> ComplianceResult:
        """Check response conformity to instructions.

        By default uses fast heuristics.
        Si use_llm=True et Ollama est disponible, utilise le LLM
        pour une verification plus precise.

        Args:
            user_message: Message original de l'utilisateur
            response: Reponse generee par le LLM
            model: Model pour la verification LLM (optionnel)
            use_llm: Utiliser le LLM pour la verification

        Returns:
            ComplianceResult avec scores et details
        """
        # Extraire les instructions
        instructions = extract_instructions(user_message)

        if not instructions:
            return ComplianceResult(score=1.0)

        # Verification heuristique (toujours disponible)
        heuristic_result = compute_heuristic_compliance(
            user_message, response, instructions,
        )

        if not use_llm or not OLLAMA_AVAILABLE:
            return heuristic_result

        # Verification LLM (plus precise mais plus lente)
        return self._llm_compliance_check(
            user_message, response, instructions, model,
        )

    def check_facts(
        self,
        response: str,
        context: str | None = None,
        model: str | None = None,
    ) -> FactCheckResult:
        """Demande au LLM d'identifier les affirmations douteuses.

        Le model re-examine sa propre reponse pour signaler
        les passages potentiellement incorrects ou incertains.

        Args:
            response: Reponse a check
            context: Contexte additionnel (documents, etc.)
            model: Model pour la verification

        Returns:
            FactCheckResult avec les flags et confiance
        """
        if not OLLAMA_AVAILABLE:
            return FactCheckResult(confidence=0.5)

        _model = model or self._config.correction_model or "qwen3:32b"

        prompt = (
            "Review the following response and identify any claims "
            "that might be factually incorrect, uncertain, or "
            "unsupported. For each concern, provide:\n"
            "- The specific claim\n"
            "- Your concern about it\n"
            "- Severity: low, medium, or high\n\n"
            "Respond ONLY in JSON format:\n"
            '{"flags": [{"claim": "...", "concern": "...", "severity": "low|medium|high"}], '
            '"confidence": 0.0-1.0}\n\n'
            f"Response to review:\n{response[:3000]}"
        )

        if context:
            prompt += f"\n\nOriginal context:\n{context[:1000]}"

        try:
            result = ollama.generate(
                model=_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 1024},
            )
            text = _generate_text(result)
            data = self._parse_json_response(text)

            if data and "flags" in data:
                flags = []
                for f in data["flags"]:
                    flags.append(FactualFlag(
                        claim=f.get("claim", ""),
                        concern=f.get("concern", ""),
                        severity=f.get("severity", "low"),
                    ))
                return FactCheckResult(
                    flags=flags,
                    flag_count=len(flags),
                    confidence=_clamp01(data.get("confidence", 0.7), 0.7),
                )
        except Exception as e:
            logger.warning(f"Fact check LLM echoue: {e}")

        return FactCheckResult(confidence=0.5)

    def check_quality(
        self,
        user_message: str,
        response: str,
        use_llm: bool = False,
        model: str | None = None,
    ) -> QualityResult:
        """Evalue la qualite generale de la reponse.

        Check completeness, coherence and markers
        d'hallucination.

        Args:
            user_message: Message original
            response: Reponse a evaluer
            use_llm: Utiliser le LLM pour l'evaluation
            model: Model pour l'evaluation LLM

        Returns:
            QualityResult avec scores et problemes
        """
        # Toujours calculer les heuristiques
        heuristic = compute_heuristic_quality(user_message, response)

        if not use_llm or not OLLAMA_AVAILABLE:
            return heuristic

        # Evaluation LLM pour affiner
        return self._llm_quality_check(user_message, response, model)

    # -----------------------------------------------------------------
    # Boucle d'auto-correction
    # -----------------------------------------------------------------

    def correct(
        self,
        user_message: str,
        response: str,
        model: str | None = None,
        use_llm: bool = True,
        max_iterations: int | None = None,
    ) -> SelfCorrectionResult:
        """Execute le processus complet d'auto-correction.

        1. Check conformity and quality
        2. If scores are insufficient, generate a correction
        3. Repete jusqu'a satisfaction ou max iterations

        Args:
            user_message: Message original de l'utilisateur
            response: Reponse initiale du LLM
            model: Model pour la correction (ou config)
            use_llm: Utiliser le LLM pour les checks et corrections
            max_iterations: Override du nombre max d'iterations

        Returns:
            SelfCorrectionResult complet
        """
        start_time = time.time()
        _model = model or self._config.correction_model or "qwen3:32b"
        _max_iter = max_iterations or self._config.max_iterations

        # -- Checks initiaux --
        compliance_result = None
        fact_result = None
        quality_result = None

        if self._config.check_instructions:
            compliance_result = self.check_compliance(
                user_message, response, _model, use_llm=use_llm,
            )

        if self._config.check_facts and use_llm:
            fact_result = self.check_facts(response, model=_model)

        if self._config.check_quality:
            quality_result = self.check_quality(
                user_message, response, use_llm=use_llm, model=_model,
            )

        compliance_score = compliance_result.score if compliance_result else 1.0
        quality_score = quality_result.overall_score if quality_result else 1.0

        # Check si correction required
        needs_correction = (
            compliance_score < self._config.compliance_threshold
            or quality_score < self._config.quality_threshold
        )

        if not needs_correction or not use_llm or not OLLAMA_AVAILABLE:
            duration = int((time.time() - start_time) * 1000)
            return SelfCorrectionResult(
                original_response=response,
                corrected_response=response,
                was_corrected=False,
                iterations_performed=0,
                compliance_before=compliance_score,
                compliance_after=compliance_score,
                quality_before=quality_score,
                quality_after=quality_score,
                compliance_result=compliance_result,
                fact_check_result=fact_result,
                quality_result=quality_result,
                total_duration_ms=duration,
                model_used=_model,
            )

        # -- Boucle de correction --
        current_response = response
        iterations = []
        best_response = response
        best_score = (compliance_score + quality_score) / 2

        for i in range(1, _max_iter + 1):
            iter_start = time.time()

            # Generer la correction
            corrected = self._generate_correction(
                user_message, current_response,
                compliance_result, quality_result, fact_result,
                _model,
            )

            if not corrected or corrected == current_response:
                break

            # Re-evaluer
            new_compliance = self.check_compliance(
                user_message, corrected, _model, use_llm=False,
            )
            new_quality = compute_heuristic_quality(user_message, corrected)

            new_score = (new_compliance.score + new_quality.overall_score) / 2
            improvements = []
            if new_compliance.score > compliance_score:
                improvements.append(
                    f"Compliance: {compliance_score:.2f} -> {new_compliance.score:.2f}"
                )
            if new_quality.overall_score > quality_score:
                improvements.append(
                    f"Quality: {quality_score:.2f} -> {new_quality.overall_score:.2f}"
                )

            iter_duration = int((time.time() - iter_start) * 1000)
            iterations.append(CorrectionIteration(
                iteration=i,
                compliance_score=new_compliance.score,
                quality_score=new_quality.overall_score,
                response_text=corrected,
                improvements=improvements,
                duration_ms=iter_duration,
            ))

            # Mettre a jour le meilleur
            if new_score > best_score:
                best_response = corrected
                best_score = new_score

            # Check si les seuils sont atteints
            if (
                new_compliance.score >= self._config.compliance_threshold
                and new_quality.overall_score >= self._config.quality_threshold
            ):
                current_response = corrected
                compliance_result = new_compliance
                quality_result = new_quality
                compliance_score = new_compliance.score
                quality_score = new_quality.overall_score
                break

            current_response = corrected
            compliance_result = new_compliance
            quality_result = new_quality
            compliance_score = new_compliance.score
            quality_score = new_quality.overall_score

        duration = int((time.time() - start_time) * 1000)

        return SelfCorrectionResult(
            original_response=response,
            corrected_response=best_response,
            was_corrected=best_response != response,
            iterations_performed=len(iterations),
            compliance_before=compute_heuristic_compliance(
                user_message, response, extract_instructions(user_message),
            ).score,
            compliance_after=compliance_score,
            quality_before=compute_heuristic_quality(user_message, response).overall_score,
            quality_after=quality_score,
            compliance_result=compliance_result,
            fact_check_result=fact_result,
            quality_result=quality_result,
            iteration_history=iterations,
            total_duration_ms=duration,
            model_used=_model,
        )

    def execute_self_correction(
        self,
        user_message: str,
        response: str,
        model: str | None = None,
    ) -> Generator:
        """Execute l'auto-correction en mode streaming.

        Yields des tuples ou des chaines:
        - str: Tokens de la reponse finale (corrigee ou originale)
        - ("correction_step", dict): Info sur une etape de correction
        - ("correction_done", SelfCorrectionResult): Result final

        Args:
            user_message: Message original
            response: Reponse initiale
            model: Model pour la correction

        Yields:
            Chunks de streaming ou tuples d'evenements
        """
        result = self.correct(
            user_message=user_message,
            response=response,
            model=model,
            use_llm=True,
        )

        # Emettre les etapes de correction
        for iteration in result.iteration_history:
            yield ("correction_step", {
                "iteration": iteration.iteration,
                "compliance_score": iteration.compliance_score,
                "quality_score": iteration.quality_score,
                "improvements": iteration.improvements,
                "duration_ms": iteration.duration_ms,
            })

        # Emettre la reponse finale token par token
        final_response = result.corrected_response
        # Simuler le streaming par chunks
        chunk_size = 50
        for i in range(0, len(final_response), chunk_size):
            yield final_response[i:i + chunk_size]

        # Emettre le result final
        yield ("correction_done", result)

    # -----------------------------------------------------------------
    # Methodes LLM internes
    # -----------------------------------------------------------------

    def _llm_compliance_check(
        self,
        user_message: str,
        response: str,
        instructions: list[str],
        model: str | None = None,
    ) -> ComplianceResult:
        """Verification de conformite via LLM.

        Args:
            user_message: Message original
            response: Reponse a check
            instructions: Instructions detectees
            model: Model pour la verification

        Returns:
            ComplianceResult raffinee par le LLM
        """
        _model = model or self._config.correction_model or "qwen3:32b"

        instr_text = "\n".join(f"- {i}" for i in instructions)
        prompt = (
            "Check if this response follows all these instructions.\n"
            "For each instruction, say if it's satisfied (true/false) "
            "with a brief explanation.\n\n"
            "Respond ONLY in JSON:\n"
            '{"checks": [{"instruction": "...", "satisfied": true/false, '
            '"explanation": "..."}], "overall_score": 0.0-1.0}\n\n'
            f"Instructions:\n{instr_text}\n\n"
            f"User message:\n{user_message[:1000]}\n\n"
            f"Response:\n{response[:2000]}"
        )

        try:
            result = ollama.generate(
                model=_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 1024},
            )
            text = _generate_text(result)
            data = self._parse_json_response(text)

            if data and "checks" in data:
                checks = []
                for c in data["checks"]:
                    checks.append(InstructionCheck(
                        instruction=c.get("instruction", ""),
                        satisfied=c.get("satisfied", True),
                        explanation=c.get("explanation", ""),
                        confidence=0.85,
                    ))
                satisfied = sum(1 for c in checks if c.satisfied)
                fallback_score = satisfied / max(len(checks), 1)
                return ComplianceResult(
                    score=_clamp01(
                        data.get("overall_score", fallback_score),
                        fallback_score,
                    ),
                    instructions_found=instructions,
                    checks=checks,
                    satisfied_count=satisfied,
                    total_count=len(checks),
                )
        except Exception as e:
            logger.warning(f"LLM compliance check echoue: {e}")

        # Fallback sur heuristiques
        return compute_heuristic_compliance(user_message, response, instructions)

    def _llm_quality_check(
        self,
        user_message: str,
        response: str,
        model: str | None = None,
    ) -> QualityResult:
        """Evaluation de qualite via LLM.

        Args:
            user_message: Message original
            response: Reponse a evaluer
            model: Model pour l'evaluation

        Returns:
            QualityResult raffinee par le LLM
        """
        _model = model or self._config.correction_model or "qwen3:32b"

        prompt = (
            "Evaluate the quality of this response on these axes:\n"
            "1. Completeness: Does it address all parts of the query? (0.0-1.0)\n"
            "2. Coherence: Is it well-structured and logical? (0.0-1.0)\n"
            "3. Hallucination risk: Are there unsupported claims? (0.0-1.0, higher = more risk)\n\n"
            "Respond ONLY in JSON:\n"
            '{"completeness": 0.0-1.0, "coherence": 0.0-1.0, '
            '"hallucination_risk": 0.0-1.0, "issues": ["..."]}\n\n'
            f"User question:\n{user_message[:1000]}\n\n"
            f"Response:\n{response[:2000]}"
        )

        try:
            result = ollama.generate(
                model=_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 512},
            )
            text = _generate_text(result)
            data = self._parse_json_response(text)

            if data:
                completeness = _clamp01(data.get("completeness", 0.8), 0.8)
                coherence = _clamp01(data.get("coherence", 0.8), 0.8)
                halluc = _clamp01(data.get("hallucination_risk", 0.2), 0.2)
                issues = data.get("issues", [])
                overall = completeness * 0.4 + coherence * 0.4 + (1.0 - halluc) * 0.2
                return QualityResult(
                    completeness_score=completeness,
                    coherence_score=coherence,
                    hallucination_risk=halluc,
                    overall_score=round(overall, 3),
                    issues=issues,
                )
        except Exception as e:
            logger.warning(f"LLM quality check echoue: {e}")

        return compute_heuristic_quality(user_message, response)

    def _generate_correction(
        self,
        user_message: str,
        current_response: str,
        compliance: ComplianceResult | None,
        quality: QualityResult | None,
        facts: FactCheckResult | None,
        model: str,
    ) -> str | None:
        """Generate a corrected version of the response.

        Build a prompt describing identified problems
        et requested au LLM de produire une version amelioree.

        Args:
            user_message: Message original
            current_response: Reponse actuelle a corriger
            compliance: Result de conformite (ou None)
            quality: Result de qualite (ou None)
            facts: Result de fact-check (ou None)
            model: Model pour la generation

        Returns:
            Texte de la reponse corrigee, ou None si failed
        """
        if not OLLAMA_AVAILABLE:
            return None

        # Construire le feedback
        feedback_parts = []

        if compliance and compliance.score < self._config.compliance_threshold:
            failed = [c for c in compliance.checks if not c.satisfied]
            if failed:
                feedback_parts.append("Instructions not followed:")
                for c in failed:
                    feedback_parts.append(f"  - {c.instruction}: {c.explanation}")

        if quality and quality.overall_score < self._config.quality_threshold:
            if quality.issues:
                feedback_parts.append("Quality issues:")
                for issue in quality.issues:
                    feedback_parts.append(f"  - {issue}")

        if facts and facts.flags:
            feedback_parts.append("Potential factual concerns:")
            for f in facts.flags[:3]:
                feedback_parts.append(f"  - {f.claim}: {f.concern} ({f.severity})")

        if not feedback_parts:
            return None

        feedback_text = "\n".join(feedback_parts)

        prompt = (
            "You previously generated a response that has some issues. "
            "Please produce an improved version that addresses the feedback below.\n\n"
            f"Original question:\n{user_message[:1500]}\n\n"
            f"Your previous response:\n{current_response[:2000]}\n\n"
            f"Feedback:\n{feedback_text}\n\n"
            "Please write ONLY the corrected response, nothing else."
        )

        try:
            result = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    "temperature": self._config.temperature,
                    "num_predict": 4096,
                },
            )
            corrected = _generate_text(result).strip()
            if corrected and len(corrected) > 10:
                return corrected
        except Exception as e:
            logger.warning(f"Generation de correction echouee: {e}")

        return None

    # -----------------------------------------------------------------
    # Utilitaires
    # -----------------------------------------------------------------

    @staticmethod
    def _parse_json_response(text: str) -> dict | None:
        """Parse une reponse JSON du LLM, tolerant au bruit.

        Essaie de trouver et parser le JSON dans le texte,
        meme if it est entoure de bruit (markdown, etc.).

        Args:
            text: Texte brut du LLM

        Returns:
            dict parse ou None
        """
        if not text:
            return None

        # Essayer le parse direct
        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            pass

        # Chercher un bloc JSON dans le texte
        # Pattern: {...} ou ```json ... ```
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Chercher le premier { ... } dans le texte
        brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None


# =============================================================================
# SINGLETON
# =============================================================================

self_correction_engine = SelfCorrectionEngine()
