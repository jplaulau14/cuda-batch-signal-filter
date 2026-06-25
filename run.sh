#!/usr/bin/env bash
set -euo pipefail

mkdir -p output
make clean build
./batch_signal_filter --input data/input --output output/filtered --window 7 --limit 100 --generate 100 --length 2048 | tee output/run.log
