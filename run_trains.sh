#!/usr/bin/env bash
set +e

python3 -m scripts.train --config configs/enc_dec_eh2.yaml
python3 -m scripts.eval --config configs/enc_dec_eh2.yaml --ckpt ckpts/combined_T2_Tf5_hz3/run_002/best.pt

python3 -m scripts.train --config configs/enc_dec_eh4.yaml
python3 -m scripts.eval --config configs/enc_dec_eh4.yaml --ckpt ckpts/combined_T2_Tf5_hz3/run_004/best.pt