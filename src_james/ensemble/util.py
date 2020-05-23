import numpy as np


def Defensive_Copy(A):
    n = len(A)
    k = len(A[0])
    L = np.zeros((n, k), dtype=np.int8)
    for i in range(n):
        for j in range(k):
            L[i, j] = 0 + A[i][j]
    return L.tolist()


def Create(task, task_id=0):
    n = len(task['train'])
    Input  = [Defensive_Copy(task['train'][i]['input']) for i in range(n)]
    Output = [Defensive_Copy(task['train'][i]['output']) for i in range(n)]
    Input.append(Defensive_Copy(task['test'][task_id]['input']))
    return Input, Output


def flattener(pred):
    pred = np.array(pred).astype(np.int8).tolist()
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred
