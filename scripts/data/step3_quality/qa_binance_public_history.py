from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / 'src').is_dir())
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.config import load_settings
from src.data.binance_public.qa import run_binance_public_qa


def main() -> None:
    parser = argparse.ArgumentParser(description="Run QA checks over normalized Binance public history outputs.")
    parser.add_argument("--settings", default="config/settings.yaml", help="Path to settings YAML.")
    parser.add_argument("--output-root", help="Binance public output root. Defaults to artifacts/data/binance_public.")
    args = parser.parse_args()

    settings = load_settings(args.settings)
    output_root = Path(args.output_root) if args.output_root else Path(settings.paths.artifacts_dir) / "data" / "binance_public"
    manifest = run_binance_public_qa(output_root)

    print(f"output_root={output_root.resolve()}")
    print(f"table_count={manifest['summary']['table_count']}")
    print(f"table_fail_count={manifest['summary']['table_fail_count']}")
    print(f"cross_table_check_count={manifest['summary']['cross_table_check_count']}")
    print(f"qa_manifest={(output_root / 'manifests' / 'qa_manifest.json').resolve()}")


if __name__ == "__main__":
    main()
