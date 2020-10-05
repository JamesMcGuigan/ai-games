#!/usr/bin/env bash

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../"

parallel -k --ungroup --jobs 6 "
(
  rm -vf ./neural_networks/models/GameOfLifeForward_{1}.pth;
  PYTHONPATH='.' PYTHONUNBUFFERED=1 time -p timeout 20m python3 ./neural_networks/hardcoded/GameOfLifeForward_{1}.py
) 2>&1 | tee ./neural_networks/hardcoded/logs/GameOfLifeForward_{1}.{2}.log
" ::: 1 1N 2 2N 4 128 ::: {1..10}
