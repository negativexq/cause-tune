#!/usr/bin/env python3
"""Run the bounded Experiment 03C fake-provider pilot, or an explicit live pilot."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from causetune.incident_telemetry import (
    FakeTelemetryProvider,
    OpenAICompatibleConfig,
    OpenAICompatibleProvider,
    PilotProviderConfig,
    build_pilot_manifest,
    run_rendering_pilot,
)


def _config(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("experiment") != "03C" or value.get("network_calls_allowed_by_default") is not False:
        raise ValueError("invalid 03C configuration")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/incident_telemetry_03c"))
    parser.add_argument("--config", type=Path, default=Path("configs/incident_telemetry_03c.json"))
    parser.add_argument("--provider-live", action="store_true", help="opt in to the OpenAI-compatible network adapter")
    args = parser.parse_args()
    config = _config(args.config)
    pilot_config = config["pilot"]
    fake_config = config["fake_provider"]
    if not isinstance(pilot_config, dict) or not isinstance(fake_config, dict):
        raise ValueError("03C pilot configuration sections are invalid")
    families = tuple(pilot_config["renderer_families"])
    live_config = config["live_provider"]
    if not isinstance(live_config, dict):
        raise ValueError("live provider configuration is invalid")
    provider_config_id = str(live_config["config_id"] if args.provider_live else config["provider_config_id"])
    manifest, scenarios, assignments, pairs = build_pilot_manifest(
        reference_seed=int(config["source_reference_seed"]),
        render_seed=int(config["render_seed"]),
        provider_config_id=provider_config_id,
        renderer_families=families,
    )
    if args.provider_live:
        model_env = str(live_config["model_identifier_env"])
        model = os.environ.get(model_env)
        if not model:
            raise ValueError(f"missing live model environment variable: {model_env}")
        provider = OpenAICompatibleProvider(
            OpenAICompatibleConfig(
                model_identifier=model,
                api_key_env=str(live_config["api_key_env"]),
                base_url_env=str(live_config["base_url_env"]),
                timeout_seconds=float(live_config["timeout_seconds"]),
                max_retries=int(live_config["max_retries"]),
                temperature=float(live_config["temperature"]),
                max_output_tokens=int(live_config["max_output_tokens"]),
            )
        )
    else:
        provider = FakeTelemetryProvider(str(fake_config["model_identifier"]))
    generation_config = live_config if args.provider_live else fake_config
    summary = run_rendering_pilot(
        args.output_dir,
        provider=provider,
        manifest=manifest,
        scenarios=scenarios,
        assignments=assignments,
        pairs=pairs,
        provider_config=PilotProviderConfig(
            config_id=provider_config_id,
            temperature=float(generation_config["temperature"]),
            max_output_tokens=int(generation_config["max_output_tokens"]),
            max_attempts=int(pilot_config["max_attempts"]),
        ),
    )
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
