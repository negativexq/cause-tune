from __future__ import annotations

import copy
import json
import unittest
from dataclasses import replace

from causetune.incident_telemetry import (
    CanonicalScenario,
    CausalEvidenceDefinition,
    CausalStateVariable,
    DependencyState,
    DiagnosticOutput,
    DistractorDefinition,
    FAILURE_MODE_IDS,
    OUTPUT_CONTRACT_VERSION,
    ONTOLOGY_VERSION,
    Provenance,
    RecentChange,
    RenderedTelemetry,
    RendererAuthorityError,
    RendererMetadata,
    RuntimeEnvironment,
    SCENARIO_SCHEMA_VERSION,
    ServiceComponent,
    ServiceTopology,
    SplitIntegrityError,
    TELEMETRY_SCHEMA_VERSION,
    TelemetryObservation,
    TopologyEdge,
    audit_dataset,
    build_split_manifest,
    evaluate_diagnostic_outputs,
    parse_diagnostic_output,
    render_scenario,
    rendered_telemetry_fingerprint,
    scenario_fingerprint,
    split_manifest_fingerprint,
    validate_diagnostic_output,
    validate_ontology,
)


def make_scenario(
    scenario_id: str = "scenario-001",
    *,
    case_type: str = "STANDARD",
    answerability: str = "ANSWERABLE",
    root_cause_id: str | None = "memory_leak",
    root_cause_family: str | None = "resource",
    split_group_id: str | None = None,
    topology_group_id: str | None = None,
    template_family: str = "template-a",
    generator_family: str = "generator-a",
    counterfactual_pair_id: str | None = None,
) -> CanonicalScenario:
    component = ServiceComponent("app-1", "app", "checkout")
    database = ServiceComponent("db-1", "database", "orders-db")
    topology = ServiceTopology(
        topology_id="topology-1",
        topology_family="commerce",
        components=(component, database),
        edges=(TopologyEdge("app-1", "db-1", "calls"),),
    )
    provenance = Provenance(
        experiment_version="03A",
        ontology_version=ONTOLOGY_VERSION,
        scenario_schema_version=SCENARIO_SCHEMA_VERSION,
        telemetry_schema_version=TELEMETRY_SCHEMA_VERSION,
        output_contract_version=OUTPUT_CONTRACT_VERSION,
        archetype_id="archetype-a",
        generator_family=generator_family,
        template_family=template_family,
        seed=42,
    )
    causal_evidence = (
        CausalEvidenceDefinition("log_001", "application_log", "heap growth is monotonic"),
        CausalEvidenceDefinition("metric_001", "metric", "resident memory crosses its limit"),
    )
    if case_type == "INCOMPLETE":
        causal_evidence = (
            CausalEvidenceDefinition(
                "log_001", "application_log", "the decisive heap trace was removed", True, True
            ),
        )
    if case_type == "HEALTHY_CONTROL":
        causal_evidence = ()
        root_cause_id = None
        root_cause_family = None
    return CanonicalScenario(
        scenario_id=scenario_id,
        schema_version=SCENARIO_SCHEMA_VERSION,
        root_cause_id=root_cause_id,
        fault_domain="RESOURCE" if root_cause_id else None,
        root_cause_family=root_cause_family,
        affected_component="app-1" if root_cause_id else None,
        topology=topology,
        runtime=RuntimeEnvironment("python", "kubernetes", "1.30"),
        causal_state=(CausalStateVariable("memory_growth", True),),
        causal_evidence=causal_evidence,
        distractors=(DistractorDefinition("alert_001", "alert", "a downstream warning"),),
        recent_changes=(RecentChange("change_001", "app-1", "deploy", "2026-09-04T10:00:00Z"),),
        dependency_state=(DependencyState("dependency_001", "db-1", "healthy", "normal"),),
        difficulty=case_type,
        case_type=case_type,
        answerability=answerability,
        expected_runbook_id="rb-memory" if root_cause_id else None,
        provenance=provenance,
        split_group_id=split_group_id or f"group-{scenario_id}",
        topology_group_id=topology_group_id or f"topology-group-{scenario_id}",
        counterfactual_pair_id=counterfactual_pair_id,
    )


def make_rendered(scenario: CanonicalScenario) -> RenderedTelemetry:
    observations = [
        TelemetryObservation("alert_001", "alert", "2026-09-04T10:01:02Z", "db-1", "downstream warning"),
    ]
    if "log_001" in scenario.declared_evidence_ids:
        observations.insert(0, TelemetryObservation("log_001", "application_log", "2026-09-04T10:01:00Z", "app-1", "heap grows"))
    if "metric_001" in scenario.declared_evidence_ids:
        observations.insert(1, TelemetryObservation("metric_001", "metric", "2026-09-04T10:01:01Z", "app-1", "memory high", 0.99))
    return RenderedTelemetry(
        scenario_id=scenario.scenario_id,
        schema_version=TELEMETRY_SCHEMA_VERSION,
        observations=tuple(observations),
        renderer=RendererMetadata("test", "fixture", "fixture-v1", 42),
        canonical_scenario_hash=scenario_fingerprint(scenario),
    )


class OntologyAndModelTests(unittest.TestCase):
    def test_ontology_is_versioned_and_bounded(self) -> None:
        validate_ontology()
        self.assertEqual(ONTOLOGY_VERSION, "incident-telemetry-ontology-v1")
        self.assertEqual(len(FAILURE_MODE_IDS), 28)

    def test_fingerprint_and_serialization_are_stable(self) -> None:
        scenario = make_scenario()
        self.assertEqual(scenario_fingerprint(scenario), scenario_fingerprint(copy.deepcopy(scenario)))
        self.assertEqual(scenario.to_dict(), json.loads(json.dumps(scenario.to_dict())))
        rendered = make_rendered(scenario)
        self.assertEqual(rendered_telemetry_fingerprint(rendered), rendered_telemetry_fingerprint(copy.deepcopy(rendered)))

    def test_unknown_taxonomy_and_invalid_answerability_fail_closed(self) -> None:
        with self.assertRaises(ValueError):
            replace(make_scenario(), root_cause_id="not-a-root-cause")
        with self.assertRaises(ValueError):
            replace(make_scenario(), answerability="INSUFFICIENT_EVIDENCE")
        with self.assertRaises(ValueError):
            make_scenario(case_type="INCOMPLETE", answerability="INSUFFICIENT_EVIDENCE", root_cause_id=None, root_cause_family=None)


class RendererTests(unittest.TestCase):
    def test_renderer_is_checked_against_authoritative_scenario(self) -> None:
        scenario = make_scenario()

        class GoodRenderer:
            def render(self, value: CanonicalScenario) -> RenderedTelemetry:
                return make_rendered(value)

        self.assertEqual(render_scenario(GoodRenderer(), scenario).scenario_id, scenario.scenario_id)

        class BadRenderer:
            def render(self, value: CanonicalScenario) -> RenderedTelemetry:
                result = make_rendered(value)
                return replace(
                    result,
                    observations=result.observations
                    + (TelemetryObservation("log_999", "application_log", "t", "app-1", "invented"),),
                )

        with self.assertRaises(RendererAuthorityError):
            render_scenario(BadRenderer(), scenario)

    def test_diagnostic_evidence_references_must_be_rendered(self) -> None:
        scenario = make_scenario()
        output = DiagnosticOutput("memory_leak", "app-1", ("log_999",), "rb-memory", False)
        with self.assertRaises(ValueError):
            validate_diagnostic_output(output, scenario, make_rendered(scenario))

    def test_renderer_output_cannot_contain_authoritative_label_fields(self) -> None:
        scenario = make_scenario()

        class LabelRenderer:
            def render(self, value: CanonicalScenario) -> dict[str, object]:
                return {"root_cause_id": "cpu_throttling"}

        with self.assertRaises(RendererAuthorityError):
            render_scenario(LabelRenderer(), scenario)  # type: ignore[arg-type]

    def test_incomplete_case_allows_only_explicit_evidence_removal(self) -> None:
        scenario = make_scenario(case_type="INCOMPLETE", answerability="INSUFFICIENT_EVIDENCE")
        rendered = replace(make_rendered(scenario), observations=(make_rendered(scenario).observations[-1],))
        # The removed causal evidence is explicitly marked may_be_removed.
        render_scenario(type("Renderer", (), {"render": lambda _self, _scenario: rendered})(), scenario)


class SplitAndAuditTests(unittest.TestCase):
    def test_grouped_split_and_ood_invariants(self) -> None:
        first = make_scenario("scenario-001", split_group_id="shared", topology_group_id="topology-train")
        second = make_scenario("scenario-002", split_group_id="shared", topology_group_id="topology-test")
        with self.assertRaises(SplitIntegrityError):
            build_split_manifest((first, second), {first.scenario_id: "TRAIN", second.scenario_id: "VALIDATION"})

        train = make_scenario("scenario-003", template_family="template-train", generator_family="generator-train", topology_group_id="topology-train")
        template_ood = make_scenario("scenario-004", template_family="template-train", generator_family="generator-ood", topology_group_id="topology-ood")
        with self.assertRaises(SplitIntegrityError):
            build_split_manifest((train, template_ood), {train.scenario_id: "TRAIN", template_ood.scenario_id: "TEMPLATE_OOD"})

        pair_a = make_scenario("scenario-005", case_type="COUNTERFACTUAL", counterfactual_pair_id="pair-1", split_group_id="pair-1", topology_group_id="pair-a")
        pair_b = make_scenario("scenario-006", case_type="COUNTERFACTUAL", counterfactual_pair_id="pair-1", split_group_id="pair-1", topology_group_id="pair-b")
        with self.assertRaises(SplitIntegrityError):
            build_split_manifest((pair_a, pair_b), {pair_a.scenario_id: "COUNTERFACTUAL_TEST", pair_b.scenario_id: "SEALED_HOLDOUT"})

    def test_counterfactual_pair_and_holdout_are_explicit(self) -> None:
        pair_a = make_scenario("scenario-007", case_type="COUNTERFACTUAL", counterfactual_pair_id="pair-2", split_group_id="pair-2", topology_group_id="pair-a")
        pair_b = make_scenario("scenario-008", case_type="COUNTERFACTUAL", counterfactual_pair_id="pair-2", split_group_id="pair-2", topology_group_id="pair-b")
        manifest = build_split_manifest(
            (pair_a, pair_b),
            {pair_a.scenario_id: "COUNTERFACTUAL_TEST", pair_b.scenario_id: "COUNTERFACTUAL_TEST"},
        )
        self.assertFalse(manifest.sealed_holdout_consumed)
        self.assertEqual(split_manifest_fingerprint(manifest), split_manifest_fingerprint(copy.deepcopy(manifest)))

        holdout = make_scenario("scenario-009", topology_group_id="topology-holdout")
        manifest = build_split_manifest(
            (pair_a, pair_b, holdout),
            {
                pair_a.scenario_id: "COUNTERFACTUAL_TEST",
                pair_b.scenario_id: "COUNTERFACTUAL_TEST",
                holdout.scenario_id: "SEALED_HOLDOUT",
            },
        )
        self.assertTrue(manifest.consume_sealed_holdout().sealed_holdout_consumed)

    def test_audit_reports_duplicates_and_fails_closed(self) -> None:
        first = make_scenario("scenario-010")
        duplicate = replace(first, scenario_id="scenario-011")
        with self.assertRaises(ValueError):
            audit_dataset((first, duplicate))


class EvaluationTests(unittest.TestCase):
    def test_output_contract_and_sliceable_metrics_are_deterministic(self) -> None:
        answerable = make_scenario("scenario-012")
        incomplete = make_scenario("scenario-013", case_type="INCOMPLETE", answerability="INSUFFICIENT_EVIDENCE")
        healthy = make_scenario("scenario-014", case_type="HEALTHY_CONTROL", answerability="HEALTHY_CONTROL")
        outputs = {
            answerable.scenario_id: DiagnosticOutput(
                "memory_leak", "app-1", ("log_001", "metric_001"), "rb-memory", False
            ),
            incomplete.scenario_id: DiagnosticOutput(None, None, (), None, True, ("log_001",)),
            healthy.scenario_id: DiagnosticOutput(None, None, (), None, False),
        }
        rendered = {scenario.scenario_id: make_rendered(scenario) for scenario in (answerable, incomplete, healthy)}
        result = evaluate_diagnostic_outputs((answerable, incomplete, healthy), outputs, rendered)
        self.assertEqual(result["diagnosis_joint_exact"]["count"], 1)
        self.assertEqual(result["abstention_recall"], 1.0)
        self.assertEqual(result["false_diagnosis_rate_on_insufficient_evidence"]["count"], 0)
        self.assertEqual(result["slices"]["evaluation_split"]["UNASSIGNED"]["count"], 3)

        parsed, category, valid = parse_diagnostic_output(json.dumps(outputs[answerable.scenario_id].to_dict()))
        self.assertEqual(category, "valid JSON")
        self.assertTrue(valid)
        self.assertIsNotNone(parsed)
