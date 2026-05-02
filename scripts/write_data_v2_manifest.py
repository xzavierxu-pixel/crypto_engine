from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def _optional_path(value: str | None) -> str | None:
    return str(Path(value).resolve()) if value else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a data_v2 rebuild manifest for a complete data/model run.")
    parser.add_argument("--data-root", default="artifacts/data_v2", help="Unified data root.")
    parser.add_argument("--run-name", required=True, help="Manifest run name.")
    parser.add_argument("--raw-input", action="append", default=[], help="Raw input path. Can be repeated.")
    parser.add_argument("--normalized-output", action="append", default=[], help="Normalized output path. Can be repeated.")
    parser.add_argument("--second-level-store", help="Second-level feature store path.")
    parser.add_argument("--training-frame", help="Training frame parquet path.")
    parser.add_argument("--experiment-output", help="Experiment output directory.")
    parser.add_argument("--config-hash", help="Config hash for the run.")
    parser.add_argument("--feature-count", type=int, help="Model feature count.")
    parser.add_argument("--start", help="Inclusive data start timestamp/date.")
    parser.add_argument("--end", help="Inclusive data end timestamp/date.")
    parser.add_argument("--extra", help="Optional JSON object to merge into the manifest.")
    args = parser.parse_args()

    data_root = Path(args.data_root)
    output_path = data_root / "manifests" / f"{args.run_name}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_name": args.run_name,
        "generation_timestamp": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root.resolve()),
        "raw_inputs": [_optional_path(path) for path in args.raw_input],
        "normalized_outputs": [_optional_path(path) for path in args.normalized_output],
        "second_level_feature_store": _optional_path(args.second_level_store),
        "training_frame": _optional_path(args.training_frame),
        "experiment_output": _optional_path(args.experiment_output),
        "config_hash": args.config_hash,
        "feature_count": args.feature_count,
        "start": args.start,
        "end": args.end,
    }
    if args.extra:
        extra = json.loads(args.extra)
        if not isinstance(extra, dict):
            raise ValueError("--extra must be a JSON object.")
        payload.update(extra)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(output_path.resolve()), **payload}, indent=2))


if __name__ == "__main__":
    main()
