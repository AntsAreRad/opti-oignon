#!/usr/bin/env python3
"""
Tests pour le moteur de verification de code (S43).

Couvre:
- Verification reussie en premiere execution
- Boucle de correction (fix) reussie
- Echec apres max iterations
- Extraction de blocs de code
- Code R
- Timeout
- Code vide
- Fallbacks gracieux
- Verification de blocs multiples dans une reponse
"""

from dataclasses import dataclass, field
from typing import List
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

# Import du module sous test
from opti_oignon.verification import (
    VerificationEngine,
    VerificationIteration,
    VerificationResult,
    verification_engine,
)

# -- Mocks --

@dataclass
class MockExecutionResult:
    """Mock pour CodeExecutor.execute() -> ExecutionResult."""
    success: bool
    stdout: str = ""
    stderr: str = ""
    return_code: int = 0
    execution_time: float = 0.1
    language: str = "python"
    truncated: bool = False
    error_message: str = ""
    output_files: list[str] = field(default_factory=list)
    working_dir: str = ""


class MockCodeExecutor:
    """Mock du CodeExecutor avec resultats configurable."""

    def __init__(self, results=None):
        """results: liste de MockExecutionResult a retourner par appels successifs."""
        self._results = results or []
        self._call_count = 0
        self._enabled = True

    @property
    def enabled(self):
        return self._enabled

    @enabled.setter
    def enabled(self, value):
        self._enabled = value

    def execute(self, code="", language="python", timeout=30, conv_id=None):
        """Retourne le prochain resultat configure."""
        if self._call_count < len(self._results):
            result = self._results[self._call_count]
        else:
            # Par defaut: echec
            result = MockExecutionResult(
                success=False,
                stderr="Mock: no more results configured",
                return_code=1,
                language=language,
            )
        self._call_count += 1
        result.language = language
        return result


class MockStructuredEngine:
    """Mock du StructuredOutputEngine."""

    def __init__(self, success=True, data=None):
        self._success = success
        self._data = data

    def generate_structured(self, messages=None, schema=None, model=None,
                            temperature=None, **kwargs):
        """Retourne un resultat mock."""
        result = MagicMock()
        result.success = self._success
        result.data = self._data
        return result


# -- Helpers --

def _make_engine(exec_results=None, max_iterations=3,
                 structured_success=True, structured_data=None):
    """Cree un VerificationEngine avec des mocks."""
    code_exec = MockCodeExecutor(results=exec_results or [])
    structured = MockStructuredEngine(
        success=structured_success,
        data=structured_data,
    )
    return VerificationEngine(
        structured_engine=structured,
        code_exec=code_exec,
        max_iterations=max_iterations,
    )


# ===================================================================
# Tests: code passe en premiere execution
# ===================================================================

class TestCodePassesFirst:
    """Test: le code s'execute correctement du premier coup."""

    def test_python_passes_first(self):
        """Code Python passe en premiere execution -> status=passed."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="42\n"),
        ])

        result = engine.verify_and_fix(
            code='print(42)',
            language='python',
            original_question='Print 42',
        )

        assert result.status == "passed"
        assert result.iterations == 1
        assert result.final_code == 'print(42)'
        assert result.execution_output == "42"
        assert len(result.errors_encountered) == 0
        assert len(result.fixes_applied) == 0

    def test_r_passes_first(self):
        """Code R passe en premiere execution -> status=passed."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="[1] 42\n", language="r"),
        ])

        result = engine.verify_and_fix(
            code='cat(42)',
            language='r',
            original_question='Print 42 in R',
        )

        assert result.status == "passed"
        assert result.iterations == 1
        assert result.language == "r"

    def test_passed_has_output(self):
        """Le resultat inclut la sortie standard."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="Hello World\nLine 2\n"),
        ])

        result = engine.verify_and_fix(code='print("Hello World")', language='python')

        assert "Hello World" in result.execution_output
        assert result.total_time >= 0


# ===================================================================
# Tests: boucle de correction reussie
# ===================================================================

class TestFixLoop:
    """Test: le code echoue puis le correctif LLM reussit."""

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_fix_on_second_attempt(self, mock_fix):
        """Echec initial, correctif reussit -> status=fixed, iterations=2."""
        mock_fix.return_value = 'print(42)'

        engine = _make_engine(exec_results=[
            MockExecutionResult(success=False, stderr="NameError: name 'x' is not defined", return_code=1),
            MockExecutionResult(success=True, stdout="42\n"),
        ])

        result = engine.verify_and_fix(
            code='print(x)',
            language='python',
            original_question='Print 42',
        )

        assert result.status == "fixed"
        assert result.iterations == 2
        assert len(result.errors_encountered) == 1
        assert len(result.fixes_applied) == 1
        assert "NameError" in result.errors_encountered[0]

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_fix_on_third_attempt(self, mock_fix):
        """Deux echecs, correctif reussit au 3eme -> status=fixed, iterations=3."""
        # Premier fix genere encore du code qui echoue
        mock_fix.side_effect = ['print(y)', 'print(42)']

        engine = _make_engine(
            exec_results=[
                MockExecutionResult(success=False, stderr="Error 1", return_code=1),
                MockExecutionResult(success=False, stderr="Error 2", return_code=1),
                MockExecutionResult(success=True, stdout="42\n"),
            ],
            max_iterations=3,
        )

        result = engine.verify_and_fix(code='print(x)', language='python')

        assert result.status == "fixed"
        assert result.iterations == 3
        assert len(result.errors_encountered) == 2
        assert len(result.fixes_applied) == 2


# ===================================================================
# Tests: echec apres max iterations
# ===================================================================

class TestMaxIterations:
    """Test: toutes les iterations echouent."""

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_all_iterations_fail(self, mock_fix):
        """Toutes les tentatives echouent -> status=failed."""
        # Retourner du code different a chaque appel pour que la boucle continue
        mock_fix.side_effect = ['attempt_2()', 'attempt_3()']

        engine = _make_engine(
            exec_results=[
                MockExecutionResult(success=False, stderr="Error 1", return_code=1),
                MockExecutionResult(success=False, stderr="Error 2", return_code=1),
                MockExecutionResult(success=False, stderr="Error 3", return_code=1),
            ],
            max_iterations=3,
        )

        result = engine.verify_and_fix(code='broken()', language='python')

        assert result.status == "failed"
        assert result.iterations == 3
        assert len(result.errors_encountered) == 3

    def test_max_iterations_one(self):
        """Avec max_iterations=1, une seule tentative."""
        engine = _make_engine(
            exec_results=[
                MockExecutionResult(success=False, stderr="Error", return_code=1),
            ],
            max_iterations=1,
        )

        result = engine.verify_and_fix(code='broken()', language='python')

        assert result.status == "failed"
        assert result.iterations == 1

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_fix_returns_none_stops_loop(self, mock_fix):
        """Si _attempt_fix retourne None, la boucle s'arrete."""
        mock_fix.return_value = None

        engine = _make_engine(
            exec_results=[
                MockExecutionResult(success=False, stderr="Error", return_code=1),
            ],
            max_iterations=3,
        )

        result = engine.verify_and_fix(code='broken()', language='python')

        assert result.status == "failed"
        # Seulement 1 iteration car le fix a echoue et on sort
        assert result.iterations == 1

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_fix_returns_same_code_stops_loop(self, mock_fix):
        """Si _attempt_fix retourne le meme code, la boucle s'arrete."""
        mock_fix.return_value = 'broken()'

        engine = _make_engine(
            exec_results=[
                MockExecutionResult(success=False, stderr="Error", return_code=1),
            ],
            max_iterations=3,
        )

        result = engine.verify_and_fix(code='broken()', language='python')

        assert result.status == "failed"
        assert result.iterations == 1


# ===================================================================
# Tests: extraction de blocs de code
# ===================================================================

class TestCodeExtraction:
    """Test: extraction de blocs de code depuis du texte."""

    def test_extract_python_block(self):
        """Extrait un bloc Python."""
        engine = VerificationEngine()
        text = "Here is the code:\n```python\nprint(42)\n```\nDone."

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 1
        assert "print(42)" in blocks[0]

    def test_extract_r_block(self):
        """Extrait un bloc R."""
        engine = VerificationEngine()
        text = "```r\ncat(42)\n```"

        blocks = engine._extract_code_blocks(text, "r")

        assert len(blocks) == 1
        assert "cat(42)" in blocks[0]

    def test_extract_no_language_tag(self):
        """Extrait un bloc sans tag de langage."""
        engine = VerificationEngine()
        text = "```\nprint(42)\n```"

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 1

    def test_extract_multiple_blocks(self):
        """Extrait plusieurs blocs."""
        engine = VerificationEngine()
        text = (
            "```python\nprint(1)\n```\n"
            "Some text\n"
            "```python\nprint(2)\n```"
        )

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 2

    def test_extract_ignores_other_language(self):
        """Ignore les blocs d'un autre langage."""
        engine = VerificationEngine()
        text = "```javascript\nconsole.log(42);\n```\n```python\nprint(42)\n```"

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 1
        assert "print(42)" in blocks[0]

    def test_extract_empty_block_skipped(self):
        """Les blocs vides sont ignores."""
        engine = VerificationEngine()
        text = "```python\n\n```"

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 0

    def test_extract_py_alias(self):
        """Gere l'alias 'py' pour Python."""
        engine = VerificationEngine()
        text = "```py\nprint(42)\n```"

        blocks = engine._extract_code_blocks(text, "python")

        assert len(blocks) == 1


# ===================================================================
# Tests: multiple blocs dans une reponse
# ===================================================================

class TestMultipleBlocks:
    """Test: verification de plusieurs blocs dans une reponse."""

    def test_verify_response_multiple_blocks(self):
        """Verifie plusieurs blocs Python dans une reponse."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="1\n"),
            MockExecutionResult(success=True, stdout="2\n"),
        ])

        response = (
            "Here is the code:\n"
            "```python\nprint(1)\n```\n"
            "And another:\n"
            "```python\nprint(2)\n```"
        )

        results = engine.verify_response_code_blocks(response, "Test")

        assert len(results) == 2
        assert all(r.status == "passed" for r in results)

    def test_verify_response_skips_non_executable(self):
        """Ignore les blocs non-executables (json, markdown, etc.)."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="ok\n"),
        ])

        response = (
            "```json\n{\"key\": \"value\"}\n```\n"
            "```python\nprint('ok')\n```"
        )

        results = engine.verify_response_code_blocks(response, "Test")

        assert len(results) == 1

    def test_verify_response_no_code_blocks(self):
        """Reponse sans blocs de code retourne une liste vide."""
        engine = _make_engine()

        results = engine.verify_response_code_blocks("Just text, no code.", "Test")

        assert len(results) == 0


# ===================================================================
# Tests: code R
# ===================================================================

class TestRVerification:
    """Test: verification specifique au code R."""

    def test_r_code_passes(self):
        """Code R correct passe la verification."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="[1] 42\n", language="r"),
        ])

        result = engine.verify_and_fix(
            code='cat(42)',
            language='r',
        )

        assert result.status == "passed"
        assert result.language == "r"

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_r_code_fix(self, mock_fix):
        """Code R en erreur, corrige avec succes."""
        mock_fix.return_value = 'library(stats)\ncat(42)'

        engine = _make_engine(exec_results=[
            MockExecutionResult(success=False, stderr="Error in library(nopackage)", return_code=1),
            MockExecutionResult(success=True, stdout="42", language="r"),
        ])

        result = engine.verify_and_fix(
            code='library(nopackage)\ncat(42)',
            language='r',
        )

        assert result.status == "fixed"


# ===================================================================
# Tests: timeout et gestion d'erreurs
# ===================================================================

class TestEdgeCases:
    """Test: cas limites et gestion d'erreurs."""

    def test_empty_code(self):
        """Code vide -> status=failed, 0 iterations."""
        engine = _make_engine()

        result = engine.verify_and_fix(code='', language='python')

        assert result.status == "failed"
        assert result.iterations == 0
        assert "vide" in result.errors_encountered[0].lower()

    def test_whitespace_only_code(self):
        """Code avec seulement des espaces -> status=failed."""
        engine = _make_engine()

        result = engine.verify_and_fix(code='   \n  \t  ', language='python')

        assert result.status == "failed"
        assert result.iterations == 0

    def test_unsupported_language(self):
        """Langage non supporte -> status=failed."""
        engine = _make_engine()

        result = engine.verify_and_fix(code='echo hello', language='bash')

        assert result.status == "failed"
        assert "non supporte" in result.errors_encountered[0].lower()

    def test_code_executor_unavailable(self):
        """Code executor non disponible -> status=failed."""
        # Creer un engine avec un code_exec explicitement absent
        # En passant un objet None via la propriete
        engine = VerificationEngine(
            code_exec=None,
            structured_engine=None,
        )
        # Patcher la propriete pour s'assurer que le code_exec est absent
        # meme si le singleton module-level existe
        with patch.object(type(engine), 'code_exec', new_callable=PropertyMock, return_value=None):
            result = engine.verify_and_fix(code='print(42)', language='python')

        assert result.status == "failed"
        assert "non disponible" in result.errors_encountered[0].lower()

    def test_execution_exception_handled(self):
        """Exception pendant l'execution -> status=failed gracieux."""
        code_exec = MockCodeExecutor()
        code_exec.execute = MagicMock(side_effect=RuntimeError("Boom"))

        engine = VerificationEngine(
            code_exec=code_exec,
            structured_engine=None,
            max_iterations=3,
        )

        result = engine.verify_and_fix(code='print(42)', language='python')

        assert result.status == "failed"
        assert any("Boom" in e for e in result.errors_encountered)

    def test_language_normalization(self):
        """Les alias de langages sont normalises."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="ok\n"),
        ])

        # python3 -> python
        result = engine.verify_and_fix(code='print("ok")', language='python3')
        assert result.status == "passed"
        assert result.language == "python"


# ===================================================================
# Tests: heuristique _looks_like_code
# ===================================================================

class TestLooksLikeCode:
    """Test: detection heuristique de code."""

    def test_python_code_detected(self):
        """Code Python detecte comme code."""
        assert VerificationEngine._looks_like_code(
            "import os\nfor f in os.listdir('.'):\n    print(f)",
            "python",
        )

    def test_r_code_detected(self):
        """Code R detecte comme code."""
        assert VerificationEngine._looks_like_code(
            "library(ggplot2)\ndata <- data.frame(x=1:10)",
            "r",
        )

    def test_plain_text_not_detected(self):
        """Texte normal non detecte comme code."""
        assert not VerificationEngine._looks_like_code(
            "This is a normal sentence about programming.",
            "python",
        )

    def test_empty_not_detected(self):
        """Texte vide non detecte."""
        assert not VerificationEngine._looks_like_code("", "python")
        assert not VerificationEngine._looks_like_code("short", "python")

    def test_markdown_fenced_not_detected(self):
        """Bloc markdown n'est pas detecte (gere par _extract_code_blocks)."""
        assert not VerificationEngine._looks_like_code(
            "```python\nprint(42)\n```",
            "python",
        )


# ===================================================================
# Tests: iteration_details
# ===================================================================

class TestIterationDetails:
    """Test: le detail de chaque iteration est enregistre."""

    @patch("opti_oignon.verification.VerificationEngine._attempt_fix")
    def test_iteration_details_recorded(self, mock_fix):
        """Chaque iteration est enregistree dans iteration_details."""
        mock_fix.return_value = 'print(42)'

        engine = _make_engine(exec_results=[
            MockExecutionResult(success=False, stderr="Error", return_code=1),
            MockExecutionResult(success=True, stdout="42\n"),
        ])

        result = engine.verify_and_fix(code='broken()', language='python')

        assert len(result.iteration_details) == 2
        assert result.iteration_details[0].success is False
        assert result.iteration_details[0].iteration == 1
        assert result.iteration_details[1].success is True
        assert result.iteration_details[1].iteration == 2

    def test_single_pass_has_one_detail(self):
        """Succes au premier coup -> 1 detail."""
        engine = _make_engine(exec_results=[
            MockExecutionResult(success=True, stdout="ok\n"),
        ])

        result = engine.verify_and_fix(code='print("ok")', language='python')

        assert len(result.iteration_details) == 1
        assert result.iteration_details[0].success is True


# ===================================================================
# Tests: singleton
# ===================================================================

class TestSingleton:
    """Test: le singleton est correctement initialise."""

    def test_singleton_exists(self):
        """Le singleton verification_engine est cree."""
        assert verification_engine is not None
        assert isinstance(verification_engine, VerificationEngine)

    def test_singleton_has_defaults(self):
        """Le singleton a les valeurs par defaut."""
        assert verification_engine._max_iterations == 3


# ===================================================================
# Tests: build_fix_prompt
# ===================================================================

class TestBuildFixPrompt:
    """Test: construction du prompt de correction."""

    def test_fix_prompt_structure(self):
        """Le prompt de fix a la structure correcte."""
        engine = VerificationEngine()

        messages = engine._build_fix_prompt(
            code='print(x)',
            error='NameError: name x is not defined',
            language='python',
            original_question='Print the value of x',
            iteration=1,
        )

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "Python" in messages[0]["content"]
        assert "print(x)" in messages[1]["content"]
        assert "NameError" in messages[1]["content"]
        assert "Print the value of x" in messages[1]["content"]

    def test_fix_prompt_r_language(self):
        """Le prompt mentionne R pour le code R."""
        engine = VerificationEngine()

        messages = engine._build_fix_prompt(
            code='cat(x)',
            error='object x not found',
            language='r',
            original_question='',
            iteration=2,
        )

        assert "R" in messages[0]["content"]
        assert "attempt 2" in messages[1]["content"]

    def test_fix_prompt_with_analysis_hint(self):
        """Le prompt inclut l'analyse structuree si presente."""
        engine = VerificationEngine()

        messages = engine._build_fix_prompt(
            code='print(x)',
            error='NameError',
            language='python',
            original_question='',
            iteration=1,
            analysis_hint='Error analysis: type=runtime, suggested_fix=define x',
        )

        assert "analysis" in messages[1]["content"].lower()
