from core.router_engine import score_complexity, select_model


def test_simple_query_routes_to_fast_gemini_model():
    score = score_complexity("What is 2+2?")
    decision = select_model(score, "gemini")

    assert score < 0.35
    assert decision.tier == "fast"
    assert decision.model_id == "gemini/gemini-2.5-flash-lite"


def test_complex_query_routes_to_powerful_model():
    query = """
    Write a Python function that implements a binary search tree with
    insert, delete, search, traversal, balancing notes, and tests.
    1. Explain the algorithm step by step.
    2. Compare the time complexity and space complexity.
    3. Analyze edge cases, trade-offs, and pros and cons in detail.
    Include code examples and explain how each function should behave.
    """

    score = score_complexity(query)
    decision = select_model(score, "gemini")

    assert score > 0.70
    assert decision.tier == "powerful"
    assert decision.model_id == "gemini/gemini-2.5-pro"


def test_select_model_falls_back_to_nearest_available_tier(monkeypatch):
    from core import router_engine
    from core.model_registry import ModelSpec

    monkeypatch.setattr(
        router_engine,
        "get_models_for_provider",
        lambda provider: [
            ModelSpec(
                model_id="provider/fast",
                tier="fast",
                cost_per_1k_input=0.001,
                cost_per_1k_output=0.002,
            ),
            ModelSpec(
                model_id="provider/powerful",
                tier="powerful",
                cost_per_1k_input=0.01,
                cost_per_1k_output=0.02,
            ),
        ],
    )

    decision = select_model(0.50, "provider")

    assert decision.tier == "fast"
    assert decision.model_id == "provider/fast"
