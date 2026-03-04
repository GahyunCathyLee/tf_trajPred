#!/usr/bin/env bash
set +e

python3 -m scripts.train --config configs/gate_exp1.yaml

python3 -m scripts.train --config configs/gate_exp2.yaml