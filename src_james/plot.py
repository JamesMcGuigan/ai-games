# Source: https://www.kaggle.com/jamesmcguigan/arc-geometry-solvers/

import matplotlib.pyplot as plt
from matplotlib import colors

# Modified from: https://www.kaggle.com/zaharch/visualizing-all-tasks-updated
from src_james.core.DataModel import Task


def plot_one(task, ax, i,train_or_test,input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    try:
        input_matrix = task[train_or_test][i][input_or_output]
        ax.imshow(input_matrix, cmap=cmap, norm=norm)
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

    num_train = len(task['train']) + len(task['test']) + 1
    if len(task['solutions']): num_train += len(task['solutions']) + 1

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

    if len(task['solutions']):
        axs[0,i+j+3].axis('off'); axs[1,i+j+3].axis('off')
        for k in range(len(task['solutions'])):
            plot_one(task, axs[0,i+j+4+k],k,'solutions','input')
            plot_one(task, axs[1,i+j+4+k],k,'solutions','output')

    plt.show()
