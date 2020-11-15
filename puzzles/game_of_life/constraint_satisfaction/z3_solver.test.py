#!/usr/bin/env python3
#
# Merge Submission Files
#   find ./ -name 'submission*.csv' | xargs cat | sort -nr | awk -F',' '!a[$1]++' | sort -n | sponge > output/submission.csv
#   grep ',1,' output/submission.csv | wc -l   # count number of non-zero entries
#
# Run Main Script:
#   PYTHONUNBUFFERED=1 time -p nice ././constraint_satisfaction/z3_solver.run.py | tee -a submission.log
#

import numpy as np

from constraint_satisfaction.solve_dataframe import solve_board_idx
from utils.datasets import test_df
from utils.plot import plot_3d
from utils.util import csv_to_delta, csv_to_numpy

if __name__ == '__main__':
    # noinspection PyRedeclaration
    for idx in [
        50000,
        53758,  # 53758 | delta = 3 | cells =  56 -> 275 | Solved!   537.1s
        90081,  # breaks with distance=1, requires distance=2
        58151,  # | delta = 5 | cells =  30 ->   0 | unsolved 5441.7s
        98531,  # | delta = 5 | cells =  30 ->   0 | unsolved 5782.9s
        59711,  # | delta = 5 | cells =  29 ->   0 | unsolved 5783.9s
        61671,  # | delta = 5 | cells =  29 ->   0 | unsolved 5784.1s
        55371,  # | delta = 4 | cells =  30 ->   0 | unsolved 3569.8s
        66341,  # | delta = 5 | cells =  30 ->   0 | unsolved 5792.2s
        91641,  # | delta = 5 | cells =  30 ->   0 | unsolved 5766.7s
        85851,  # | delta = 5 | cells =  30 ->   0 | unsolved 5789.5s
        85291,  # | delta = 5 | cells =  30 ->   0 | unsolved 5085.8s
        54891,  # | delta = 4 | cells =  30 ->   0 | unsolved 3437.6s
        66931,  # | delta = 5 | cells =  30 ->   0 | unsolved 5194.4s
        71601,  # | delta = 5 | cells =  30 ->   0 | unsolved 5791.1s
        58921,  # | delta = 5 | cells =  31 ->   0 | unsolved 5796.2s
    ]:
        df    = test_df
        delta = csv_to_delta(df, idx)
        board = csv_to_numpy(df, idx, key='stop')

        solution_3d, idx, time_taken = solve_board_idx(board=board, delta=delta, idx=idx, exact=False, timeout=120, verbose=True)
        solution_3d = np.append(solution_3d, [board], axis=0)
        plot_3d(solution_3d)
