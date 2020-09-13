#!/usr/bin/env python3
# PYTHONUNBUFFERED=1 time -p nice ././constraint_satisfaction/z3_solver.run.py | tee -a submission.log

from constraint_satisfaction.z3_solver import solve_dataframe
from util.datasets import test_df

if __name__ == '__main__':
    solve_dataframe(test_df, save='submission.csv', modulo=(2,1), sort_cells=True, sort_delta=False)
