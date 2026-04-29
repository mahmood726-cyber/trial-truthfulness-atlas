# Precomputed judge cache

The integration test stubs the ollama client (the `stub_ollama` fixture
in `test_integration_5trial.py`), so no real cache files are needed for
v0.1.0 tests.

When v0.2.0 introduces the full Pairwise70 sweep, real cache JSON files
should be committed here so production reruns hit cache and stay
deterministic.
