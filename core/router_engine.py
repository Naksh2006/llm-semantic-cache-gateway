"""Cost-aware model routing engine.

Scores query complexity using rule-based heuristics (no ML, easy to
explain in an interview) and selects the cheapest model tier that can
handle the query well.

Usage:
    from core.router_engine import route
    decision = await route("Explain quantum physics step by step", "gemini")
    print(decision.model_id, decision.tier, decision.complexity_score)
"""

import re

import structlog
from pydantic import BaseModel

from core.model_registry import ModelSpec, get_models_for_provider

logger = structlog.get_logger(__name__)


# ────────────────────────────────────────────────────────────────────
# Complexity scoring — rule-based, NOT ML
# ────────────────────────────────────────────────────────────────────


def score_complexity(query_text: str) -> float:
    """Score query complexity on a 0.0–1.0 scale using simple heuristics.

    This is intentionally rule-based and transparent — every weight is
    documented so it can be explained in an interview or code review.
    No ML model, no black box.

    Components:
      1. Word count (up to 0.40) — longer queries tend to be more complex
      2. Code signals (up to 0.25) — code fences or programming keywords
      3. Multi-part structure (up to 0.20) — multiple questions or numbered lists
      4. Depth phrases (up to 0.15) — explicit requests for detailed analysis

    Returns:
        float: Complexity score clamped to [0.0, 1.0]
    """
    score = 0.0

    # ── 1. Word count contribution (max 0.40) ──────────────────────
    # Rationale: A 60+ word query is almost certainly complex.
    # Short queries ("What time is it?") score near zero here.
    words = query_text.split()
    word_count = len(words)
    score += min(word_count / 60, 1.0) * 0.40

    # ── 2. Code signals (max 0.25) ─────────────────────────────────
    # Rationale: Queries containing code fences or programming keywords
    # need a model that understands code well (typically a stronger tier).
    code_patterns = ["```", "function", "code", "def ", "class ",
                     "import ", "return ", "async ", "await "]
    if any(pattern in query_text.lower() for pattern in code_patterns):
        score += 0.25

    # ── 3. Multi-part structure (max 0.20) ─────────────────────────
    # Rationale: Multiple questions ("What is X? How does Y work?")
    # or numbered lists ("1. Do this 2. Then that") signal a complex
    # multi-part request that benefits from a stronger model.
    question_marks = query_text.count("?")
    has_numbered_list = bool(re.search(r"\d+\.\s", query_text))

    if question_marks >= 2 or has_numbered_list:
        score += 0.20

    # ── 4. Depth phrases (max 0.15) ────────────────────────────────
    # Rationale: Explicit requests for depth ("explain in detail",
    # "step by step", "analyze", "compare") strongly indicate the
    # user expects a thorough, well-reasoned response.
    depth_phrases = [
        "explain in detail", "step by step", "step-by-step",
        "analyze", "analyse", "compare", "contrast",
        "in depth", "comprehensive", "thorough",
        "pros and cons", "trade-offs", "tradeoffs",
    ]
    query_lower = query_text.lower()
    if any(phrase in query_lower for phrase in depth_phrases):
        score += 0.15

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, score))


# ────────────────────────────────────────────────────────────────────
# Routing decision model
# ────────────────────────────────────────────────────────────────────


class RoutingDecision(BaseModel):
    """The output of model selection — tells the router which model to use."""

    model_id: str
    tier: str
    complexity_score: float
    provider: str
    reasoning: str


# ────────────────────────────────────────────────────────────────────
# Model selection logic
# ────────────────────────────────────────────────────────────────────


def select_model(
    complexity_score: float, provider: str
) -> RoutingDecision:
    """Pick the best model tier for the given complexity score.

    Tier mapping:
      - score < 0.35  → "fast"      (cheapest, simple queries)
      - score > 0.70  → "powerful"  (most capable, complex queries)
      - otherwise     → "balanced"  (middle ground)

    If the provider doesn't have a model for the exact tier (e.g. only
    fast and powerful, no balanced), falls back to the nearest available
    tier rather than crashing.
    """
    models = get_models_for_provider(provider)

    # Determine target tier from score
    if complexity_score < 0.35:
        target_tier = "fast"
    elif complexity_score > 0.70:
        target_tier = "powerful"
    else:
        target_tier = "balanced"

    # Try exact tier match first
    for model in models:
        if model.tier == target_tier:
            return RoutingDecision(
                model_id=model.model_id,
                tier=model.tier,
                complexity_score=complexity_score,
                provider=provider,
                reasoning=(
                    f"complexity={complexity_score:.2f} → "
                    f"{target_tier} tier → {model.model_id}"
                ),
            )

    # Fallback: find nearest tier
    # Priority order for each target tier:
    #   fast     → balanced → powerful (go up)
    #   balanced → fast → powerful     (try cheaper first)
    #   powerful → balanced → fast     (go down)
    fallback_order = {
        "fast": ["balanced", "powerful"],
        "balanced": ["fast", "powerful"],
        "powerful": ["balanced", "fast"],
    }

    for fallback_tier in fallback_order[target_tier]:
        for model in models:
            if model.tier == fallback_tier:
                logger.info(
                    "tier_fallback",
                    target_tier=target_tier,
                    actual_tier=fallback_tier,
                    model_id=model.model_id,
                    provider=provider,
                )
                return RoutingDecision(
                    model_id=model.model_id,
                    tier=model.tier,
                    complexity_score=complexity_score,
                    provider=provider,
                    reasoning=(
                        f"complexity={complexity_score:.2f} → "
                        f"{target_tier} tier (fallback to "
                        f"{fallback_tier}) → {model.model_id}"
                    ),
                )

    # Should never reach here if PROVIDER_MODELS is populated correctly
    raise ValueError(
        f"No models available for provider '{provider}'"
    )


def get_model_spec(model_id: str, provider: str) -> ModelSpec | None:
    """Look up the ModelSpec for a specific model_id.

    Returns None if the model_id isn't in the registry (e.g. user
    overrode with a custom model string).
    """
    models = get_models_for_provider(provider)
    for model in models:
        if model.model_id == model_id:
            return model
    return None


# ────────────────────────────────────────────────────────────────────
# Top-level routing function
# ────────────────────────────────────────────────────────────────────


async def route(
    query_text: str, provider: str
) -> RoutingDecision:
    """Score query complexity and select the appropriate model.

    Async for consistency with the rest of the codebase, even though
    this is pure CPU work with no I/O.
    """
    complexity = score_complexity(query_text)
    decision = select_model(complexity, provider)
    return decision
