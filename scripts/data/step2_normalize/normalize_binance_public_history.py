from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.binance_public.normalizer import normalize_binance_public_history
from src.data.binance_public.qa import run_binance_public_qa


def run_normalize(settings_path: str | Path, output_root: str | Path | None = None) -> dict[str, object]:
    settings = load_settings(settings_path)
    resolved_output_root = (
        Path(output_root) if output_root else Path(settings.paths.artifacts_dir) / "data" / "binance_public"
    )
    normalize_manifest = normalize_binance_public_history(resolved_output_root)
    qa_manifest = run_binance_public_qa(resolved_output_root)
    return {
        "output_root": resolved_output_root,
        "normalize_manifest": normalize_manifest,
        "qa_manifest": qa_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize Binance public raw history into parquet outputs and run QA."
    )
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--output-root", help="Binance public output root. Defaults to artifacts/data/binance_public.")
    args = parser.parse_args()

    result = run_normalize(args.settings, output_root=args.output_root)
    output_root = Path(result["output_root"])
    manifest = result["normalize_manifest"]
    qa_manifest = result["qa_manifest"]

    print(f"output_root={output_root.resolve()}")
    print(f"normalized_outputs={len(manifest['normalized_outputs'])}")
    print(f"unsupported_files={len(manifest['unsupported_files'])}")
    print(f"schema_manifest={(output_root / 'manifests' / 'schema_manifest.json').resolve()}")
    print(f"qa_table_count={qa_manifest['summary']['table_count']}")
    print(f"qa_table_fail_count={qa_manifest['summary']['table_fail_count']}")
    print(f"qa_manifest={(output_root / 'manifests' / 'qa_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
