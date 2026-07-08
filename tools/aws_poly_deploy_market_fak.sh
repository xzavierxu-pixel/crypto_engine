#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <staging_dir>" >&2
  exit 2
fi

REPO_DIR="/home/ubuntu/opt/crypto_engine"
STAGING_DIR="$1"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="$REPO_DIR/execution_engine/deploy_backups/$TIMESTAMP"
REQUIRED_KEYS=(
  POLYMARKET_PRIVATE_KEY
  CLOB_API_KEY
  CLOB_SECRET
  CLOB_PASS_PHRASE
  POLYMARKET_SIGNATURE_TYPE
  DEPOSIT_WALLET_ADDRESS
)

cd "$REPO_DIR"

echo "branch=$(git branch --show-current)"
echo "staging_dir=$STAGING_DIR"

missing=0
for key in "${REQUIRED_KEYS[@]}"; do
  if grep -q "^${key}=" execution_engine/secrets.env; then
    echo "secret_key_present:$key=true"
  else
    echo "secret_key_present:$key=false"
    missing=1
  fi
done

if grep -q '^POLYMARKET_SIGNATURE_TYPE=3$' execution_engine/secrets.env; then
  echo "signature_type_3=true"
else
  echo "signature_type_3=false"
  missing=1
fi

if [[ $missing -ne 0 ]]; then
  echo "remote preflight failed" >&2
  exit 1
fi

sudo systemctl stop execution-engine.timer execution-engine.service execution-engine-prewarm.timer execution-engine-prewarm.service || true

mkdir -p "$BACKUP_DIR"
for rel in \
  execution_engine/config.py \
  execution_engine/order_plan.py \
  execution_engine/polymarket_v2.py \
  execution_engine/config.example.yaml \
  execution_engine/config.yaml \
  execution_engine/deploy/price_estimator_expected_return_h14/artifact_manifest.json \
  execution_engine/deploy/price_estimator_expected_return_h14/expected_return_hazard.npz \
  execution_engine/deploy/price_estimator_expected_return_h14/feature_columns.json
do
  if [[ -f "$rel" ]]; then
    install -D "$rel" "$BACKUP_DIR/$rel"
  fi
done

. .venv/bin/activate
python -m pip install --upgrade \
  git+https://github.com/Polymarket/py-clob-client-v2.git \
  git+https://github.com/Polymarket/py-builder-relayer-client.git

for rel in \
  execution_engine/config.py \
  execution_engine/order_plan.py \
  execution_engine/polymarket_v2.py \
  execution_engine/config.example.yaml \
  execution_engine/config.yaml \
  execution_engine/deploy/price_estimator_expected_return_h14/artifact_manifest.json \
  execution_engine/deploy/price_estimator_expected_return_h14/expected_return_hazard.npz \
  execution_engine/deploy/price_estimator_expected_return_h14/feature_columns.json
do
  install -D "$STAGING_DIR/$rel" "$REPO_DIR/$rel"
done

python - <<'PY'
from execution_engine.config import load_execution_config
from py_clob_client_v2 import OrderType

cfg = load_execution_config("execution_engine/config.yaml")
assert cfg.orders.first.price_mode == "best_ask_market"
assert cfg.orders.first.order_type == "FAK"
assert cfg.orders.first.size == 5.0
assert cfg.price_estimator.enabled is False
assert cfg.execution_edge.enabled is True
assert abs(cfg.execution_edge.min_edge - 0.01) < 1e-12
assert hasattr(OrderType, "FAK")
with open("execution_engine/deploy/price_estimator_expected_return_h14/artifact_manifest.json", "r", encoding="utf-8"):
    pass
with open("execution_engine/deploy/price_estimator_expected_return_h14/feature_columns.json", "r", encoding="utf-8"):
    pass
print("config_validation=ok")
PY

sudo systemctl enable --now execution-engine-prewarm.timer execution-engine.timer
sudo systemctl is-active execution-engine-prewarm.timer execution-engine.timer
sudo systemctl list-timers --all --no-pager | grep 'execution-engine'
