"""Builds a scenario registry around a test config, for SimulationManager /
create_app (which serve backend/config.Scenario registries)."""

from __future__ import annotations

from dataclasses import replace

from backend.config import DEFAULT_SCENARIO, SCENARIOS, Scenario, SimulationConfig


def single_scenario(config: SimulationConfig, **overrides) -> dict[str, Scenario]:
    """The default scenario (same name and model files unless overridden,
    e.g. mappo_model_path=...) with `config` swapped in."""
    return {DEFAULT_SCENARIO: replace(SCENARIOS[DEFAULT_SCENARIO], config=config, **overrides)}


def two_scenarios(small_config: SimulationConfig, large_config: SimulationConfig, **large_overrides) -> dict[str, Scenario]:
    """"small" and "large" with test configs swapped in."""
    return {
        "small": replace(SCENARIOS["small"], config=small_config),
        "large": replace(SCENARIOS["large"], config=large_config, **large_overrides),
    }
