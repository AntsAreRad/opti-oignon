"""A stand-in for ``opti_oignon.model_provenance`` in the isolation windows of the suites that drive the Ollama backend
for something else than its provenance (MP-01): its gate passes every model, each call kept in ``asked``. The gate
itself is the contracts' of ``tests/test_ollama_provenance_contracts.py``."""
import types


def provenance_seed(asked=None):
    """``{"opti_oignon.model_provenance": a module whose gate passes}``, to merge into a window's ``seeded``."""
    asked = [] if asked is None else asked
    module = types.ModuleType("opti_oignon.model_provenance")

    def guard_ollama(name, digest, **kwargs):
        asked.append(name)
        return types.SimpleNamespace(allowed=True, reason="verified", model=name, digest=None)

    module.guard_ollama = guard_ollama
    module.asked = asked
    return {"opti_oignon.model_provenance": module}
