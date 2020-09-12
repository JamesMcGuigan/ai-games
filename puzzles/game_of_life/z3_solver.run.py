#!/usr/bin/env python3
# PYTHONUNBUFFERED=1 time -p nice ./z3_solver.run.py | tee -a submission.log

from datasets import test_df
from z3_solver import solve_dataframe


if __name__ == '__main__':
    solve_dataframe(test_df, save='submission.csv')