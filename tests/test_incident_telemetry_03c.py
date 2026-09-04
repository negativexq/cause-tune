from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from causetune.incident_telemetry import (
    FakeTelemetryProvider,
    PilotManifest,
    PilotProviderConfig,
    ProviderRawResponse,
    RenderContractError,
    ScenarioGenerator,
    ScenarioRequest,
    build_pilot_manifest,
    build_render_plan,
    build_render_prompt,
    build_reference_corpus,
    pilot_manifest_fingerprint,
    render_provider_payload,
    run_rendering_pilot,
)
from causetune.incident_telemetry.fingerprints import rendered_telemetry_fingerprint, scenario_fingerprint
from causetune.incident_telemetry.rendering import RENDERER_FAMILY_BY_ID


class RenderingContractTests(unittest.TestCase):
    def setUp(self) -> None:
        scenarios, manifest, pairs = build_reference_corpus()
        self.scenario = next(item for item in scenarios if item.difficulty == "HARD")
        self.manifest_hash = manifest.manifest_fingerprint
        self.pairs = pairs
        self.plan = build_render_plan(
            self.scenario,
            renderer_family="concise_ops_v1",
            render_seed=303,
            split_manifest_hash=self.manifest_hash,
        )
        self.provider = FakeTelemetryProvider()
        self.payload = json.loads(self.provider.generate(build_render_prompt(self.plan)).content)

    def test_prompt_is_label_free_and_families_are_distinct(self) -> None:
        prompt = build_render_prompt(self.plan)
        self.assertNotIn(self.scenario.root_cause_id, prompt)
        self.assertNotIn(self.scenario.expected_runbook_id, prompt)
        self.assertNotIn(self.scenario.provenance.archetype_id, prompt)
        other = build_render_plan(self.scenario, renderer_family="verbose_enterprise_v1", render_seed=303)
        self.assertNotEqual(prompt, build_render_prompt(other))
        self.assertEqual(RENDERER_FAMILY_BY_ID["concise_ops_v1"].prompt_template_version, self.plan.prompt_template_version)

    def test_fake_provider_is_deterministic(self) -> None:
        first = self.provider.generate(build_render_prompt(self.plan))
        second = self.provider.generate(build_render_prompt(self.plan))
        self.assertEqual(first.content, second.content)
        self.assertEqual(first.content_hash(), second.content_hash())

    def test_valid_payload_preserves_authority_and_hashes(self) -> None:
        rendered = render_provider_payload(
            self.scenario,
            self.plan,
            self.payload,
            provider="fake",
            model_identifier="fake-telemetry-renderer-v1",
            split_manifest_hash=self.manifest_hash,
        )
        self.assertEqual(rendered.canonical_scenario_hash, scenario_fingerprint(self.scenario))
        self.assertEqual(rendered.split_manifest_hash, self.manifest_hash)
        mutated = replace(rendered, observations=(replace(rendered.observations[0], text="different surface"),))
        self.assertNotEqual(rendered_telemetry_fingerprint(rendered), rendered_telemetry_fingerprint(mutated))

    def test_label_leakage_is_rejected(self) -> None:
        leaked = json.loads(json.dumps(self.payload))
        leaked["observations"][0]["text"] = f"root cause: {self.scenario.root_cause_id}"
        with self.assertRaisesRegex(RenderContractError, "forbidden labels"):
            render_provider_payload(self.scenario, self.plan, leaked, provider="fake", model_identifier="fake")
        leaked["observations"][0]["text"] = f"use {self.scenario.expected_runbook_id}"
        with self.assertRaisesRegex(RenderContractError, "forbidden labels"):
            render_provider_payload(self.scenario, self.plan, leaked, provider="fake", model_identifier="fake")

    def test_unknown_missing_duplicate_component_and_kind_fail_closed(self) -> None:
        unknown = json.loads(json.dumps(self.payload))
        unknown["observations"][0]["evidence_id"] = "log_999"
        with self.assertRaisesRegex(RenderContractError, "unsupported evidence"):
            render_provider_payload(self.scenario, self.plan, unknown, provider="fake", model_identifier="fake")

        missing = json.loads(json.dumps(self.payload))
        missing["observations"] = missing["observations"][1:]
        with self.assertRaisesRegex(RenderContractError, "required semantic evidence"):
            render_provider_payload(self.scenario, self.plan, missing, provider="fake", model_identifier="fake")

        duplicate = json.loads(json.dumps(self.payload))
        duplicate["observations"].append(dict(duplicate["observations"][0]))
        with self.assertRaisesRegex(RenderContractError, "duplicate evidence"):
            render_provider_payload(self.scenario, self.plan, duplicate, provider="fake", model_identifier="fake")

        wrong_component = json.loads(json.dumps(self.payload))
        wrong_component["observations"][0]["component_id"] = "unknown_component"
        with self.assertRaisesRegex(RenderContractError, "source component"):
            render_provider_payload(self.scenario, self.plan, wrong_component, provider="fake", model_identifier="fake")

        wrong_kind = json.loads(json.dumps(self.payload))
        wrong_kind["observations"][0]["kind"] = "ALERT"
        with self.assertRaisesRegex(RenderContractError, "permitted telemetry surface"):
            render_provider_payload(self.scenario, self.plan, wrong_kind, provider="fake", model_identifier="fake")

    def test_intentionally_missing_evidence_cannot_reappear(self) -> None:
        generator = ScenarioGenerator()
        incomplete = generator.generate(ScenarioRequest("container_memory_exhaustion", "WEB_DB", "python_fastapi", "INCOMPLETE", 9))
        plan = build_render_plan(incomplete, renderer_family="concise_ops_v1", render_seed=9)
        payload = json.loads(self.provider.generate(build_render_prompt(plan)).content)
        missing_id = next(item.evidence_id for item in plan.entries if item.may_be_omitted)
        payload["observations"].append(
            {
                "evidence_id": missing_id,
                "kind": next(item.permitted_surface_types[0] for item in plan.entries if item.evidence_id == missing_id),
                "component_id": next(item.source_component for item in plan.entries if item.evidence_id == missing_id),
                "timestamp_offset_seconds": 1,
                "text": "reappeared evidence",
                "value": None,
            }
        )
        with self.assertRaisesRegex(RenderContractError, "missing evidence"):
            render_provider_payload(incomplete, plan, payload, provider="fake", model_identifier="fake")

    def test_counterfactual_render_keeps_pair_identity(self) -> None:
        # The reference pair is the last two scenarios in corpus order.
        scenarios, _manifest, pairs = build_reference_corpus()
        pair = pairs[0]
        first = next(item for item in scenarios if item.scenario_id == pair.source_scenario_id)
        second = next(item for item in scenarios if item.scenario_id == pair.counterfactual_scenario_id)
        for scenario in (first, second):
            plan = build_render_plan(scenario, renderer_family="ecosystem_native_v1", render_seed=44)
            payload = json.loads(self.provider.generate(build_render_prompt(plan)).content)
            render_provider_payload(scenario, plan, payload, provider="fake", model_identifier="fake")
            self.assertEqual(scenario.counterfactual_pair_id, pair.pair_id)


class PilotExecutionTests(unittest.TestCase):
    def test_manifest_is_stable_and_excludes_sealed_holdout(self) -> None:
        first, scenarios, assignments, pairs = build_pilot_manifest()
        second, _scenarios, _assignments, _pairs = build_pilot_manifest()
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.manifest_fingerprint, pilot_manifest_fingerprint(first))
        self.assertEqual(len(first.entries), 144)
        self.assertTrue(all(item.split != "SEALED_HOLDOUT" for item in first.entries))
        self.assertEqual(len(scenarios), 48)

    def test_malformed_response_retries_and_permanent_failure_is_retained(self) -> None:
        manifest, scenarios, assignments, pairs = build_pilot_manifest()
        entry = manifest.entries[0]
        tiny = PilotManifest(
            manifest.pilot_version,
            manifest.reference_seed,
            manifest.render_seed,
            manifest.source_manifest_fingerprint,
            manifest.provider_config_id,
            (entry,),
        )
        tiny = PilotManifest(**{**tiny.__dict__, "manifest_fingerprint": pilot_manifest_fingerprint(tiny)})

        class FlakyProvider:
            provider_name = "test"
            model_identifier = "flaky"
            adapter_version = "test-provider-v1"

            def __init__(self) -> None:
                self.calls = 0
                self.good = FakeTelemetryProvider()

            def generate(self, prompt: str) -> ProviderRawResponse:
                self.calls += 1
                if self.calls == 1:
                    return ProviderRawResponse("not-json")
                return self.good.generate(prompt)

        with tempfile.TemporaryDirectory() as directory:
            summary = run_rendering_pilot(
                directory,
                provider=FlakyProvider(),
                manifest=tiny,
                scenarios=scenarios,
                assignments=assignments,
                pairs=pairs,
                provider_config=PilotProviderConfig(max_attempts=2),
            )
            self.assertEqual(summary["accepted"], 1)
            self.assertEqual(summary["rejected"], 0)
            self.assertEqual(summary["retries"], 1)
            self.assertTrue(list((Path(directory) / "rejected").glob("*.json")))

        class BrokenProvider(FlakyProvider):
            def generate(self, prompt: str) -> ProviderRawResponse:
                self.calls += 1
                return ProviderRawResponse("not-json")

        with tempfile.TemporaryDirectory() as directory:
            summary = run_rendering_pilot(
                directory,
                provider=BrokenProvider(),
                manifest=tiny,
                scenarios=scenarios,
                assignments=assignments,
                pairs=pairs,
                provider_config=PilotProviderConfig(max_attempts=2),
            )
            self.assertEqual(summary["accepted"], 0)
            self.assertEqual(summary["rejected"], 1)
            self.assertEqual(len(list((Path(directory) / "rejected").glob("*.json"))), 2)
            self.assertEqual(json.loads((Path(directory) / "audit.json").read_text())["status"], "fail")
