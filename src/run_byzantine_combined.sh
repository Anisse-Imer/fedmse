#!/usr/bin/env bash
# Run the combined Byzantine attack experiment (data poisoning + model corruption,
# 30% ratio, 30 clients) on both IID and non-IID distributions back-to-back.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IID_CFG="Configuration/scen2-nba-iot-byzantine-30clients-iid.json"
NONIID_CFG="Configuration/scen2-nba-iot-byzantine-30clients-non-iid.json"

echo "========================================"
echo "  Byzantine combined — IID (30 clients)"
echo "========================================"
python byzantine_combined_main.py --file "$IID_CFG"

echo ""
echo "=============================================="
echo "  Byzantine combined — non-IID (30 clients)"
echo "=============================================="
python byzantine_combined_main.py --file "$NONIID_CFG"

echo ""
echo "All runs complete."
