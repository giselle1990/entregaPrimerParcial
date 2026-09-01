#!/usr/bin/env bash
set -euo pipefail
python -m src.training.train --data data/raw/customer_churn_historical.csv --experiment customer-churn-entrega-1 --register-best
