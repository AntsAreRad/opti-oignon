"""
S194 F7f -- prompt optimization & templates fix lot tests.

Covers:
- POP-01: ollama.show typed-object responses (modern clients) yield the
  real context window; dict path regression-checked.
- POP-02: priority_overrides redistribution uses EFFECTIVE ratios; an
  explicit project_ratio 0.0 opts out and the caller's reserve intent
  survives; the no-override default math is byte-identical to the
  pre-existing suite's expectations.
- POP-03: yaml import and module singletons guarded.
- TC-04 seam re-assertion: the context fingerprint in executor.py is
  computed AFTER opt_result.system_prompt (source-order assertion, the
  heavy-import-chain idiom).

prompt_optimization.py has no relative or package imports beyond the
guarded yaml/ollama, so it loads standalone. The original
test_prompt_optimization.py is a known package-chain collection error
in the container (baseline class D); its redistribution expectations
are unchanged by POP-02 (no-override math identical).
"""

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

_PROJECT = Path(__file__).resolve().parents[1]


def _read(rel):
    return (_PROJECT / rel).read_text(encoding="utf-8")


def _load_module(name, rel_path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(
        name, str(_PROJECT / rel_path)
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


popt = _load_module("s194f_popt", "opti_oignon/prompt_optimization.py")


class _FakeOllama:
    """Context manager installing a fake ollama module."""

    def __init__(self, show_fn):
        self._mod = types.ModuleType("ollama")
        self._mod.show = show_fn
        self._had = None

    def __enter__(self):
        self._had = sys.modules.get("ollama")
        sys.modules["ollama"] = self._mod
        return self

    def __exit__(self, *exc):
        if self._had is None:
            del sys.modules["ollama"]
        else:
            sys.modules["ollama"] = self._had
        return False


class TestPOP01ShowResponseShapes(unittest.TestCase):
    """POP-01: typed-object and dict shapes both detected."""

    def _mgr(self):
        return popt.PromptTokenBudgetManager(config={})

    def test_typed_object_model_info(self):
        show = lambda m: SimpleNamespace(
            model_info={"qwen3.context_length": 32768}, parameters=""
        )
        with _FakeOllama(show):
            self.assertEqual(self._mgr()._query_ollama_show("q"), 32768)

    def test_typed_object_modelinfo_alias(self):
        show = lambda m: SimpleNamespace(
            modelinfo={"llama.context_length": 16384}
        )
        with _FakeOllama(show):
            self.assertEqual(self._mgr()._query_ollama_show("l"), 16384)

    def test_typed_object_parameters_num_ctx(self):
        show = lambda m: SimpleNamespace(
            model_info={}, parameters="num_ctx 24576\nstop <eot>"
        )
        with _FakeOllama(show):
            self.assertEqual(self._mgr()._query_ollama_show("p"), 24576)

    def test_dict_shape_regression(self):
        show = lambda m: {"model_info": {"x.context_length": 8000}}
        with _FakeOllama(show):
            self.assertEqual(self._mgr()._query_ollama_show("d"), 8000)

    def test_window_resolution_uses_typed_value(self):
        show = lambda m: SimpleNamespace(
            model_info={"x.context_length": 32768}
        )
        with _FakeOllama(show):
            self.assertEqual(self._mgr().get_context_window("m"), 32768)

    def test_no_info_falls_back(self):
        show = lambda m: SimpleNamespace(model_info={}, parameters="")
        with _FakeOllama(show):
            mgr = self._mgr()
            self.assertIsNone(mgr._query_ollama_show("n"))
            self.assertEqual(mgr.get_context_window("n"), 8192)


class TestPOP02EffectiveRedistribution(unittest.TestCase):
    """POP-02: overrides redistribute effective ratios."""

    def _mgr(self):
        return popt.PromptTokenBudgetManager(config={})

    def test_default_math_unchanged(self):
        # Mirrors test_redistribution_proportional from the original
        # suite (collection-blocked in the container): 0.60 / 0.15.
        b = self._mgr().calculate_budget(
            "m", project_active=False, context_window_override=10000
        )
        self.assertEqual(b.history_tokens, 6000)
        self.assertEqual(b.user_tokens, 1500)
        self.assertEqual(b.reserve_tokens, 1500)

    def test_project_ratio_zero_opts_out(self):
        b = self._mgr().calculate_budget(
            "m",
            project_active=False,
            context_window_override=10000,
            priority_overrides={
                "system_ratio": 0.10,
                "project_ratio": 0.0,
                "history_ratio": 0.55,
                "user_ratio": 0.20,
                "reserve_ratio": 0.15,
            },
        )
        self.assertEqual(b.history_tokens, 5500)
        self.assertEqual(b.user_tokens, 2000)
        self.assertEqual(b.reserve_tokens, 1500)
        self.assertAlmostEqual(b.utilization, 1.0, places=3)

    def test_override_shares_are_effective(self):
        # Withheld project (default 0.25) split by the OVERRIDDEN
        # history/user ratios (0.50 / 0.25 -> shares 2/3, 1/3).
        b = self._mgr().calculate_budget(
            "m",
            project_active=False,
            context_window_override=12000,
            priority_overrides={
                "system_ratio": 0.10,
                "history_ratio": 0.50,
                "user_ratio": 0.25,
                "reserve_ratio": 0.15,
            },
        )
        # hist = 0.50 + 0.25 * (0.50/0.75) = 0.666...; user = 0.25 +
        # 0.25 * (0.25/0.75) = 0.333... -> clamp trims overflow from
        # reserve then history; user keeps its effective share.
        self.assertEqual(b.user_tokens, int(12000 * (0.25 + 0.25 / 3)))
        self.assertLessEqual(b.total_allocated, 12000)

    def test_active_project_ignores_redistribution(self):
        b = self._mgr().calculate_budget(
            "m", project_active=True, context_window_override=10000
        )
        self.assertEqual(b.project_tokens, 2500)
        self.assertEqual(b.history_tokens, 4000)


class TestPOP03Guards(unittest.TestCase):
    """POP-03: yaml import and singletons guarded."""

    def test_yaml_guard_in_source(self):
        src = _read("opti_oignon/prompt_optimization.py")
        self.assertIn("YAML_AVAILABLE = True", src)
        self.assertIn("except ImportError", src)
        self.assertIn("if YAML_AVAILABLE and path.exists()", src)

    def test_singletons_guarded_and_alive(self):
        src = _read("opti_oignon/prompt_optimization.py")
        self.assertIn("PROMPT_OPTIMIZATION_MODULE_AVAILABLE = True", src)
        self.assertIn("PROMPT_OPTIMIZATION_MODULE_AVAILABLE = False", src)
        self.assertIsNotNone(popt.prompt_budget_manager)
        self.assertIsNotNone(popt.prompt_template_engine)
        self.assertTrue(popt.PROMPT_OPTIMIZATION_MODULE_AVAILABLE)


class TestTC04SeamOrder(unittest.TestCase):
    """The fingerprint stays AFTER the optimizer prompt assignment."""

    def test_fingerprint_after_opt_result(self):
        src = _read("opti_oignon/executor.py")
        opt_pos = src.index("system_prompt = opt_result.system_prompt")
        fp_pos = src.index(
            '_ctx_fp = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()'
        )
        self.assertLess(opt_pos, fp_pos)


class TestTemplateEngineSanity(unittest.TestCase):
    """Resolution order and interpolation invariants on the edited file."""

    def test_resolution_order(self):
        eng = popt.PromptTemplateEngine(config={
            "templates": {
                "code_r": {"system_prompt": "yaml R"},
                "general": {"system_prompt": "yaml general"},
            },
            "project_overrides": {
                "proj1": {"code_r": {"system_prompt": "project R"}},
            },
        })
        self.assertEqual(eng.get_template("code_r").system_prompt, "yaml R")
        self.assertEqual(
            eng.get_template("code_r", project_id="proj1").system_prompt,
            "project R",
        )
        eng.set_runtime_override("code_r", "runtime R")
        self.assertEqual(eng.get_template("code_r").system_prompt, "runtime R")
        self.assertEqual(
            eng.get_template("unknown").system_prompt, "yaml general"
        )

    def test_interpolation_unknown_left_intact(self):
        eng = popt.PromptTemplateEngine(config={})
        tpl = popt.PromptTemplate(
            task_type="t",
            system_prompt="{language_rule} model={model_name} {unknown_var}",
        )
        out = eng.interpolate(tpl, context={"model_name": "m1"})
        self.assertIn("model=m1", out)
        self.assertIn("{unknown_var}", out)
        self.assertNotIn("{language_rule}", out)


if __name__ == "__main__":
    unittest.main()
