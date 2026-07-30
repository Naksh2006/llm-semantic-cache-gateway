"""Model registry mapping providers to available models and pricing.

Each provider has a list of ``ModelSpec`` entries organized by tier
(fast / balanced / powerful). Pricing is approximate and WILL drift
as providers update their APIs — every line is marked with a TODO
reminder to verify periodically.

Usage:
    from core.model_registry import get_models_for_provider
    models = get_models_for_provider("gemini")
"""

from typing import Literal

from pydantic import BaseModel


class ModelSpec(BaseModel):
    """Specification for a single LLM model."""

    model_id: str
    tier: Literal["fast", "balanced", "powerful"]
    cost_per_1k_input: float   # USD per 1,000 input tokens
    cost_per_1k_output: float  # USD per 1,000 output tokens


# ────────────────────────────────────────────────────────────────────
# Provider model catalog
#
# NOTE: Every entry is a best-effort snapshot. Model IDs and pricing
# change frequently. Each line has a TODO to remind you to verify.
# ────────────────────────────────────────────────────────────────────

PROVIDER_MODELS: dict[str, list[ModelSpec]] = {
    "gemini": [
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="gemini/gemini-2.0-flash",
            tier="fast",
            cost_per_1k_input=0.0001,
            cost_per_1k_output=0.0004,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="gemini/gemini-2.5-flash",
            tier="balanced",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="gemini/gemini-2.5-pro",
            tier="powerful",
            cost_per_1k_input=0.00125,
            cost_per_1k_output=0.005,
        ),
    ],
    "openai": [
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="gpt-4o-mini",
            tier="fast",
            cost_per_1k_input=0.00015,
            cost_per_1k_output=0.0006,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="gpt-4o",
            tier="balanced",
            cost_per_1k_input=0.0025,
            cost_per_1k_output=0.01,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="o3-mini",
            tier="powerful",
            cost_per_1k_input=0.0011,
            cost_per_1k_output=0.0044,
        ),
    ],
    "anthropic": [
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="anthropic/claude-3-5-haiku-latest",
            tier="fast",
            cost_per_1k_input=0.0008,
            cost_per_1k_output=0.004,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="anthropic/claude-sonnet-4-20250514",
            tier="balanced",
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
        ),
        ModelSpec(  # TODO: verify current model id + pricing
            model_id="anthropic/claude-opus-4-20250514",
            tier="powerful",
            cost_per_1k_input=0.015,
            cost_per_1k_output=0.075,
        ),
    ],
}


def get_models_for_provider(provider: str) -> list[ModelSpec]:
    """Return the model catalog for the given provider.

    Raises ``ValueError`` with a clear message if the provider is not
    in the registry.
    """
    provider = provider.lower()
    if provider not in PROVIDER_MODELS:
        available = ", ".join(sorted(PROVIDER_MODELS.keys()))
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available providers: {available}"
        )
    return PROVIDER_MODELS[provider]
