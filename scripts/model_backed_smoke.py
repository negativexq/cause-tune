#!/usr/bin/env python3
"""Run the isolated real-dependency model-backed training smoke."""

from __future__ import annotations

import argparse
import json

from causetune.training_smoke import dependency_versions, run_model_backed_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="hf-internal-testing/tiny-random-GPT2")
    parser.add_argument("--output-dir", default="outputs/model_backed_smoke")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()
    result = run_model_backed_smoke(
        model_id=args.model_id,
        output_dir=args.output_dir,
        local_files_only=args.local_files_only,
    )
    print(json.dumps({"dependencies": dependency_versions(), **result.to_dict()}, sort_keys=True))


if __name__ == "__main__":
    main()
