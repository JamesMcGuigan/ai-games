import json

import pandas as pd

from src.ensemble.sample_sub.example_grid import example_grid
from src.ensemble.sample_sub.path import data_path
from src.ensemble.sample_sub.path import output_dir
from src.ensemble.sample_sub.path import test_path
from src.ensemble.solvers.Match_crop_mode import Match_crop_mode
from src.ensemble.solvers.Match_crop_mode import Match_crop_mode_1
from src.ensemble.solvers.Recolor import Recolor
from src.ensemble.solvers.Recolor0 import Recolor0
from src.ensemble.solvers.Recolor0 import Recolor0_bound
from src.ensemble.solvers.Solve_color_crop import Solve_color_crop
from src.ensemble.solvers.Solve_connect import solve_connect
from src.ensemble.solvers.Solve_filing import Solve_filling
from src.ensemble.solvers.Solve_inoutmap import solve_inoutmap
from src.ensemble.solvers.Solve_inoutmap import solve_inoutmap_colormap
from src.ensemble.solvers.Solve_logtic import Solve_logtic
from src.ensemble.solvers.Solve_mode_dict import Solve_mode_dict
from src.ensemble.solvers.Solve_mul_color_bound import Solve_mul_color_bound
from src.ensemble.solvers.Solve_negative import Solve_negative
from src.ensemble.solvers.Solve_object_mode_color import solve_object_mode
from src.ensemble.solvers.Solve_object_mode_color import solve_object_mode_color
from src.ensemble.solvers.Solve_output_color_change_mode import solve_output_color_change_mode
from src.ensemble.solvers.Solve_patch import solve_task
from src.ensemble.solvers.Solve_period import Solve_period
from src.ensemble.solvers.Solve_resize import Solve_resize
from src.ensemble.solvers.Solve_resize import Solve_resize_bound
from src.ensemble.solvers.Solve_resize import Solve_resizec
from src.ensemble.solvers.Solve_resize import Solve_resizec_bound
from src.ensemble.solvers.Solve_train_test_map import Solve_train_test_map
from src.ensemble.solvers.Solve_trans import Solve_trans
from src.ensemble.solvers.Solve_trans import Solve_trans_bound
from src.ensemble.solvers.Solve_trans_negative import Solve_trans_negative
from src.ensemble.util import Create
from src.ensemble.util import flattener

sample_sub2 = pd.read_csv(data_path/'sample_submission.csv')
sample_sub2 = sample_sub2.set_index('output_id', drop=False)
sample_sub2.head()

Solved = []
Problems = sample_sub2['output_id'].values
Proposed_Answers = []

for i in range(len(Problems)):
    # print(i)
    preds=[example_grid,example_grid,example_grid]
    predict_solution=[]
    output_id = Problems[i]
    task_id = output_id.split('_')[0]
    pair_id = int(output_id.split('_')[1])
    f = str(test_path / str(task_id + '.json'))

    with open(f, 'r') as read_file:
        task = json.load(read_file)

    basic_task = Create(task, pair_id)

    try:
        predict_solution.append(Solve_mode_dict(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor0(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor0_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Recolor(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_train_test_map(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_color_crop(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_filling(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_mul_color_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_period(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap(basic_task,0,0,0,0))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap_colormap(basic_task,1,1,1,1))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_inoutmap_colormap(basic_task,2,2,2,2))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_logtic(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_task(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_trans_negative(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_object_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_object_mode_color(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_output_color_change_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resize(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resizec(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resize_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Solve_resizec_bound(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(solve_connect(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Match_crop_mode(basic_task))
    except:
        predict_solution.append(-1)
    try:
        predict_solution.append(Match_crop_mode_1(basic_task))
    except:
        predict_solution.append(-1)


    for j in range(len(predict_solution)):
        if predict_solution[j]!=-1 and predict_solution[j] not in preds:
            preds.append(predict_solution[j])


    pred = ''
    if len(preds)>3:
        Solved.append(i)
        pred1 = flattener(preds[-1])
        pred2 = flattener(preds[-2])
        pred3 = flattener(preds[-3])
        pred  = pred+pred1+' '+pred2+' '+pred3+' '

    if pred == '':
        pred = flattener(example_grid)

    Proposed_Answers.append(pred)

sample_sub2['output'] = Proposed_Answers
sample_sub2.to_csv(output_dir/'submission2.csv', index=False)