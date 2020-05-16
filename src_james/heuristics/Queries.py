from typing import List, Tuple, Union

import numpy as np

from src_james.core.DataModel import Problem, ProblemSet, Task



class Query:

    @classmethod
    def grid_size_ratio_problem(cls, problem: Problem) -> Tuple[int,int]:
        return ( problem['input'].shape[0] / problem['output'].shape[0],
                 problem['input'].shape[1] / problem['output'].shape[1] )

    @classmethod
    def grid_size_ratio_task(cls, task: Task) -> Union[Tuple[int,int], None]:
        return cls.grid_size_ratio_problemset(task['train'])

    @classmethod
    def grid_size_ratio_problemset(cls, problemset: ProblemSet) -> Union[Tuple[int,int], None]:
        ratios = cls.grid_size_ratios_problemset(problemset)
        return ratios[0] if len(ratios) == 1 else None

    @classmethod
    def grid_size_ratios_problemset(cls, problemset: ProblemSet) -> List[Tuple[int,int]]:
        return list({ cls.grid_size_ratio_problem(problem) for problem in problemset })

    @classmethod
    def count_nonzero(cls, input: np.ndarray) -> int:
        return np.count_nonzero(input)

    @classmethod
    def unique_colors(cls, input: np.ndarray) -> int:
        return np.count_nonzero(np.bincount(input.flatten())[1:])
