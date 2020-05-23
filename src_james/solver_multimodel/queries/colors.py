from src_james.solver_multimodel.queries.grid import grid_unique_colors
from src_james.solver_multimodel.queries.ratio import is_task_shape_ratio_consistant


def task_is_singlecolor(task):
    if not is_task_shape_ratio_consistant(task): return False
    return all([ len(grid_unique_colors(spec['output'])) == 1 for spec in task['train'] ])

