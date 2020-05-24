import json

import pandas as pd

from src_james.ensemble.sample_sub.example_grid import example_grid
from src_james.ensemble.sample_sub.path import data_path, test_path, output_dir
from src_james.ensemble.solvers.Solve_split_shape import Solve_mul_color_bound_negative, Solve_juxt, Solve_split_shape
from src_james.ensemble.solvers.Solve_split_shape import Solve_split_shape_negative, solve_cross_map
from src_james.ensemble.solvers.Solve_split_shape import solve_cross_map_line, Solve_mul_color_negative
from src_james.ensemble.util import Create, flattener

sample_sub4 = pd.read_csv(data_path / 'sample_submission.csv')
sample_sub4.head()

# display(example_grid)
print(flattener(example_grid))

Solved = []
Problems = sample_sub4['output_id'].values
Proposed_Answers = []

for i in range(len(Problems)):
    preds = [example_grid, example_grid, example_grid]
    predict_solution = []
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:
        task = json.load(read_file)

    basic_task = Create(task, pair_id)
    try:
        predict_solution.append(Solve_split_shape_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_cross_map(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_cross_map_line(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_bound_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_juxt(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_split_shape(basic_task))
    except:
        predict_solution.append(-1)

    for j in range(len(predict_solution)):
        if predict_solution[j] != -1 and predict_solution[j] not in preds:
            preds.append(predict_solution[j])

    pred = ''
    if len(preds) > 3:
        Solved.append(i)
        pred1 = flattener(preds[-1])
        pred2 = flattener(preds[-2])
        pred3 = flattener(preds[-3])
        pred = pred + pred1 + ' ' + pred2 + ' ' + pred3 + ' '

    if pred == '':
        pred = flattener(example_grid)

    Proposed_Answers.append(pred)

sample_sub4['output'] = Proposed_Answers
sample_sub4.to_csv(output_dir/'submission4.csv', index=False)