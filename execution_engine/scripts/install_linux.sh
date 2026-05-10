#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r execution_engine/requirements.txt
python -m pip install git+https://github.com/Polymarket/py-clob-client-v2.git

if [ ! -f execution_engine/config.yaml ]; then
  cp execution_engine/config.example.yaml execution_engine/config.yaml
fi

mkdir -p artifacts/logs/execution_engine artifacts/state/execution_engine
