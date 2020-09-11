#!/usr/bin/env python3

from datasets import test_df
from z3_solver import solve_dataframe

if __name__ == '__main__':
    solve_dataframe(test_df, save='submission.csv')