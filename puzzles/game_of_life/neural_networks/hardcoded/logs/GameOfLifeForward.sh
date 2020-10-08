#!/usr/bin/env bash

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../"

parallel -k --ungroup --jobs 6 "
(
  rm -vf ./neural_networks/models/GameOfLifeForward_{1}.pth;
  PYTHONPATH='.' PYTHONUNBUFFERED=1 time -p timeout 30m python3 ./neural_networks/hardcoded/GameOfLifeForward_{1}.py
) 2>&1 | tee ./neural_networks/hardcoded/logs/GameOfLifeForward_{1}.{2}.log
" ::: 11N 1N 1 2N 2 4 128 ::: {1..10}
