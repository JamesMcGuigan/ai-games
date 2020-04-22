import os    
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()

def color_count(grid):
    height, width = len(grid), len(grid[0])
    coords = [(r,c) for r in range(height) for c in range(width)]

    color_dict = dict()
    for rc in coords:
        r, c = rc
        if grid[r][c] in color_dict.keys():
            color_dict[grid[r][c]] += 1
        else:
            color_dict[grid[r][c]] = 1
    return color_dict, len(coords)

def convert_grid_bw(grid, major, minor):
    grid_bw = grid[:]
    height, width = len(grid), len(grid[0])
    coords = [(r,c) for r in range(height) for c in range(width)]
    for rc in coords:
        r, c = rc
        if grid_bw[r][c] in major:
            grid_bw[r][c] = 0
        else:
            grid_bw[r][c] = 1
    return grid_bw

def convert_task_bw(input, output):
    color_dict, total = color_count(output)
    height, width = len(output), len(output[0])
    major = []
    minor = []
    is_background_exists = (0 in color_dict.keys())
    if is_background_exists == False:    
        for color in color_dict.keys():
            count = color_dict[color]
            if count > 0.3*total:
                major.append(color)
            else:
                minor.append(color)
    else:
        major.append(0)
        for color in color_dict.keys():
            minor.append(color)
    input_bw = convert_grid_bw(input, major, minor)
    output_bw = convert_grid_bw(output, major, minor)
    return input_bw, output_bw

def plot_task(task, bw=False):
    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(2, n, figsize=(4*n,8), dpi=50)
    plt.subplots_adjust(wspace=0, hspace=0)
    fig_num = 0
    
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        if bw == True:
            t_in, t_out = convert_task_bw(t_in, t_out)
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Train-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Train-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    for i, t in enumerate(task["test"]):
        t_in, t_out = np.array(t["input"]), np.array(t["output"])
        if bw == True:
            t_in, t_out = convert_task_bw(t_in, t_out)
        axs[0][fig_num].imshow(t_in, cmap=cmap, norm=norm)
        axs[0][fig_num].set_title(f'Test-{i} in')
        axs[0][fig_num].set_yticks(list(range(t_in.shape[0])))
        axs[0][fig_num].set_xticks(list(range(t_in.shape[1])))
        axs[1][fig_num].imshow(t_out, cmap=cmap, norm=norm)
        axs[1][fig_num].set_title(f'Test-{i} out')
        axs[1][fig_num].set_yticks(list(range(t_out.shape[0])))
        axs[1][fig_num].set_xticks(list(range(t_out.shape[1])))
        fig_num += 1
    plt.tight_layout()
    plt.show()   

    
def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()
    
def load_data(input_path):
    #data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
    data_path = Path(f'{input_path}')
    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'
    training_tasks = sorted(os.listdir(training_path))
    eval_tasks = sorted(os.listdir(evaluation_path))

    solved_id=set()
    solved_eva_id=set()

    T = training_tasks
    Trains = []
    for i in range(400):
        task_file = str(training_path / T[i])
        task = json.load(open(task_file, 'r'))
        Trains.append(task)
        
    E = eval_tasks
    Evals= []
    for i in range(400):
        task_file = str(evaluation_path / E[i])
        task = json.load(open(task_file, 'r'))
        Evals.append(task)
        
    print('data loaded')
    return Trains, Evals