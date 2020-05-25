# Source: https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers/
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np
from fastcache._lrucache import clru_cache
from matplotlib import colors

# Modified from: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated
from src_james.core.DataModel import Task


@clru_cache()
def invert_hexcode(hexcode):
    hexcode = hexcode.replace('#','0x')
    number  = (16**len(hexcode)-1) - int(hexcode, 16)
    return hex(number).replace('0x','#')

def plot_one(task, ax, i,train_or_test,input_or_output):
    hexcodes = [
        '#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
        '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25',
    ]
    inverted_hexcodes = list(map(invert_hexcode,hexcodes))
    cmap  = colors.ListedColormap(hexcodes)
    icmap = colors.ListedColormap(inverted_hexcodes)
    norm = colors.Normalize(vmin=0, vmax=9)

    try:
        input_matrix = task[train_or_test][i][input_or_output]
        matrix_size  = np.sqrt(input_matrix.shape[0] * input_matrix.shape[1])
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
        # DOC: https://stackoverflow.com/questions/33828780/matplotlib-display-array-values-with-imshow
        for (j,i),label in np.ndenumerate(input_matrix):
            ax.text(i,j,label,ha='center',va='center', fontsize=50/matrix_size, color='black')
        ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
        ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
        ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_title(train_or_test + ' '+input_or_output)
    except: pass  # mat throw on tests, as they have not "output"

def plot_task(task: Task, scale=2):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    if isinstance(task, str): task = Task(task)
    filename = task.filename
    task_solutions = {
        "solutions": list(chain(*task['solutions']))  # this is a 2D array now
    }
    num_train      = len(task['train']) + len(task['test']) + 1
    if task.solutions_count: num_train += task.solutions_count + 1

    fig, axs = plt.subplots(2, num_train, figsize=(scale*num_train,scale*2))
    if filename: fig.suptitle(filename)

    i = 0
    for i in range(len(task['train'])):
        plot_one(task, axs[0,i],i,'train','input')
        plot_one(task, axs[1,i],i,'train','output')

    axs[0,i+1].axis('off'); axs[1,i+1].axis('off')
    j = 0
    for j in range(len(task['test'])):
        plot_one(task, axs[0,i+2+j],j,'test','input')
        plot_one(task, axs[1,i+2+j],j,'test','output')

    if task.solutions_count:
        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')
        for k in range(len(task_solutions)):
            plot_one(task_solutions, axs[0,i+j+4+k],k,'solutions','input')
            plot_one(task_solutions, axs[1,i+j+4+k],k,'solutions','output')

    for ax in chain(*axs): ax.axis('off')
    plt.show()
