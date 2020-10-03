#!/usr/bin/env bash

cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/../../../"
for n in 1 1N 2 2N 4 128; do
(
  for i in `seq 8`; do
  (
      rm -vf ./neural_networks/models/GameOfLife_$n.pth
      PYTHONUNBUFFERED=1 time python3 ./neural_networks/hardcoded/GameOfLifeForward_$n.py
  ) 2>&1 | tee ./neural_networks/hardcoded/logs/GameOfLifeForward_$n.$i.log
  done
) &
done
wait
