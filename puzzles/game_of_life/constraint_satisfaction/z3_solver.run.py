#!/usr/bin/env python3
# PYTHONUNBUFFERED=1 time -p nice ././constraint_satisfaction/z3_solver.run.py | tee -a submission.log
from constraint_satisfaction.solve_dataframe import solve_dataframe
from utils.datasets import submission_file
from utils.datasets import test_df

if __name__ == '__main__':
    solve_dataframe(test_df, savefile=submission_file, modulo=(2, 1), sort_cells=True, sort_delta=False)
