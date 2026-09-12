"""Thin command-line surface for CauseTune application workflows."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .application import (
    compare_workflow,
    doctor_workflow,
    evaluate_workflow,
    prepare_train_workflow,
    verify_workflow,
)
from .doctor import render_doctor_report
from .experiment_contract import ExperimentContractError
from .verify import render_verification_report


EXIT_SUCCESS = 0
EXIT_GENERAL = 1
EXIT_CONTRACT = 2
EXIT_PREFLIGHT = 3
EXIT_VERIFICATION = 4


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def _common_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="causetune", description="CauseTune laboratory workflows")
    commands = parser.add_subparsers(dest="command", required=True)

    doctor = commands.add_parser("doctor", help="run CPU-safe or opt-in hardware preflight")
    _common_config(doctor)
    doctor.add_argument("--hardware", action="store_true")

    train = commands.add_parser("train", help="validate and initialize an evidence-backed run")
    _common_config(train)
    train.add_argument("--run-dir")
    train.add_argument("--hardware", action="store_true")

    evaluate = commands.add_parser("evaluate", help="score persisted predictions")
    evaluate.add_argument("--predictions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--json", action="store_true", dest="as_json")

    compare = commands.add_parser("compare", help="compare persisted base and tuned evaluations")
    compare.add_argument("--base", required=True)
    compare.add_argument("--tuned", required=True)
    compare.add_argument("--json", action="store_true", dest="as_json")

    verify = commands.add_parser("verify", help="verify a persisted evidence bundle")
    verify.add_argument("run_dir")
    verify.add_argument("--offline", action="store_true", default=True)
    verify.add_argument("--json", action="store_true", dest="as_json")
    return parser


def _emit(value: Any, *, as_json: bool, human: str | None = None) -> None:
    if as_json or human is None:
        sys.stdout.write(_json(value))
    else:
        sys.stdout.write(human)


def _preflight_exit_code(report: dict[str, Any]) -> int:
    if report["summary"]["status"] != "FAIL":
        return EXIT_SUCCESS
    return (
        EXIT_CONTRACT
        if any(check["layer"] == "Contract" and check["status"] == "FAIL" for check in report["checks"])
        else EXIT_PREFLIGHT
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "doctor":
            report = doctor_workflow(args.config, hardware=args.hardware)
            _emit(report, as_json=args.as_json, human=render_doctor_report(report))
            return _preflight_exit_code(report)
        if args.command == "train":
            result = prepare_train_workflow(args.config, run_dir=args.run_dir, hardware=args.hardware)
            _emit(result, as_json=args.as_json)
            return EXIT_SUCCESS
        if args.command == "evaluate":
            result = evaluate_workflow(args.predictions, args.output)
            _emit(result, as_json=args.as_json)
            return EXIT_SUCCESS
        if args.command == "compare":
            result = compare_workflow(args.base, args.tuned)
            _emit(result, as_json=args.as_json)
            return EXIT_SUCCESS
        if args.command == "verify":
            report = verify_workflow(args.run_dir, offline=args.offline)
            _emit(report, as_json=args.as_json, human=render_verification_report(report))
            return EXIT_SUCCESS if report["summary"]["status"] != "FAIL" else EXIT_VERIFICATION
    except ExperimentContractError as exc:
        sys.stderr.write(f"contract error: {exc}\n")
        return EXIT_CONTRACT
    except Exception as exc:
        if args.command == "train" and hasattr(exc, "report"):
            report = exc.report
            _emit(report, as_json=args.as_json, human=render_doctor_report(report))
            return _preflight_exit_code(report)
        if args.command == "verify":
            sys.stderr.write(f"verification error: {exc}\n")
            return EXIT_VERIFICATION
        sys.stderr.write(f"causeTune error: {exc}\n")
        return EXIT_GENERAL
    return EXIT_GENERAL


if __name__ == "__main__":
    raise SystemExit(main())
