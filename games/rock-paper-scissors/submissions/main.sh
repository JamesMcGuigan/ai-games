#!/usr/bin/env bash
# Kaggle BUG: multi directory submissions - https://www.kaggle.com/product-feedback/215421

cd $(dirname $(readlink -f ${BASH_SOURCE[0]}));  # OSX requires: brew install coreutils
set -x

find ./ -name '__pycache__' -or -name '*.py[cod]' -print -delete
find ./ -name '*.py' | xargs tar cfvz main.py.tar.gz

echo "kaggle competitions submit -c rock-paper-scissors -f main.py.tar.gz -m 'Simulations()'"
