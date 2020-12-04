#!/usr/bin/env bash

kaggle c download conways-reverse-game-of-life-2020 -p input
unzip -f input/conways-reverse-game-of-life-2020.zip -d input/
rm input/conways-reverse-game-of-life-2020.zip
