#!/usr/bin/env bash
set -euo pipefail

. .venv/bin/activate
python execution_engine/run_once.py --config execution_engine/config.yaml --mode paper --print-json

