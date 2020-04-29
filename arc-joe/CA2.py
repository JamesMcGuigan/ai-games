import os
import json
import numpy as np
from pathlib import Path

from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import animation, rc
from skimage.util.shape import view_as_windows
from skimage.feature import greycomatrix
from itertools import combinations
import copy
data_path = Path('./abstraction-and-reasoning-challenge')
train_path = data_path / 'training'
valid_path = data_path / 'evaluation'
test_path = data_path / 'test'

train_tasks = { task.stem: json.load(task.open()) for task in train_path.iterdir() } 
valid_tasks = { task.stem: json.load(task.open()) for task in valid_path.iterdir() } 
test_tasks=   { task.stem: json.load(task.open()) for task in test_path.iterdir() } 
cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
    
def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()
    
def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])
        
def inp2img(inp):
    inp = np.array(inp)
    img = np.full((10, inp.shape[0], inp.shape[1]), 0, dtype=np.uint8)
    for i in range(10):
        img[i] = (inp==i)
    return img

def input_output_shape_is_same(task):
    return all([np.array(el['input']).shape == np.array(el['output']).shape for el in task['train']])


def calk_score(task_test, predict):
    return [int(np.equal(sample['output'], pred).all()) for sample, pred in zip(task_test, predict)]
def pad_with(vector, pad_width, iaxis, kwargs):
    pad_value = kwargs.get('padder', -1)
    vector[:pad_width[0]] = pad_value
    vector[-pad_width[1]:] = pad_value

def plot_picture(x):
    plt.imshow(np.array(x), cmap = cmap, norm = norm)
    plt.show()

def mat_to_key(arr):
    return tuple(map(tuple, arr))
def key_to_mat(k):
    return np.asarray(k)
def generateConsistentCandidates(tin,tout):
    candidates={}
    ruled_out={}
    locs=np.where(tin!=tout)
    m,n=tin.shape
    tin_wdw = view_as_windows(tin, (3,3), step=1).reshape((m-2)*(n-2),3,3)
    tout_wdw=view_as_windows(tout, (3,3), step=1).reshape((m-2)*(n-2),3,3)
    for i,j in zip(locs[0],locs[1]):
        wdw=tin[i-1:i+2,j-1:j+2]
        out=tout[i,j]
        rout=False
        if ruled_out.get(mat_to_key(wdw))==1 or candidates.get(mat_to_key(wdw))==1:
            continue
        for in_wdw,out_wdw in zip(tin_wdw,tout_wdw):
            if np.equal(wdw, in_wdw).all() and out!=out_wdw[1,1]:
                ruled_out[mat_to_key(wdw)]=1
                rout=True
                break
        if not rout:
            candidates[mat_to_key(wdw)]=out
    return candidates

def clashInconsistent(candidates_array):
    candidates_clashed=copy.deepcopy(candidates_array)
    for  idx,sett in enumerate(candidates_array):
        for idx2,sett2 in enumerate(candidates_array):
            for key in sett:
                if sett2.get(key) and sett2.get(key)!=sett.get(key) and candidates_clashed[idx].get(key) :
                    del candidates_clashed[idx][key]
                    del candidates_clashed[idx2][key]
    return candidates_clashed

class CA:

    def __init__(self,task):
        self.task=task
        self.trManager=TransitionRuleManager()
        inWindows=[]
        outWindows=[]
        idx=0
        for sample in task["train"]:
            tp = np.pad(np.array(sample["input"]), 1, pad_with)
            to = np.pad(np.array(sample["output"]), 1, pad_with)
            m,n=tp.shape
            if (tp.shape != to.shape):
                self.buildSuccess = False
                return
            tin_wdw = view_as_windows(tp, (3, 3), step=1).reshape((m - 2) * (n - 2), 3, 3)
            tout_wdw = view_as_windows(to, (3, 3), step=1).reshape((m - 2) * (n - 2), 3, 3)
            if idx==0:
                inWindows=tin_wdw
                outWindows=tout_wdw
            else:
                inWindows=np.concatenate((inWindows,tin_wdw),axis=0)
                outWindows=np.concatenate((outWindows,outWindows),axis=1)
            idx+=1
        self.inWindows=inWindows
        self.outWindows=outWindows
    def build_succeeded(self):
        return self.buildSuccess
    def run(self,image,timestep):
        for i in range(timestep):
            image=self.evaluate(image)
        return image
    def updateTrain(self):
        task=self.task
        for idx,sample in enumerate(task["train"]):
            task["train"][idx]["input"]=self.evaluate(task["train"][idx]["input"])
        self.task=task
        inWindows = []
        idx = 0
        for sample in task["train"]:
            tp = np.pad(np.array(sample["input"]), 1, pad_with)
            m, n = tp.shape
            tin_wdw = view_as_windows(tp, (3, 3), step=1).reshape((m - 2) * (n - 2), 3, 3)
            if idx == 0:
                inWindows = tin_wdw
            else:
                inWindows = np.concatenate((inWindows, tin_wdw), axis=0)
            idx += 1
        self.inWindows = inWindows
    def evaluate(self,image):
        tp = np.pad(image, 1, pad_with)
        tout=np.copy(tp)
        for i in range(1,tp.shape[0]-1):
            for j in range(1,tp.shape[1]-1):
                wdw=tp[i-1:i+2,j-1:j+2]
                out=self.trManager.matchWindow(wdw)
                if out!=-1:
                    tout[i, j]=out
        # plot_picture(tout[1:tp.shape[0]-1,1:tp.shape[1]-1])
        return tout[1:tp.shape[0]-1,1:tp.shape[1]-1]
    def build(self):

        timestep_candiates={}
        timestep_ruled_out={}
        for lop in range(7):
            for sample in task["train"]:
                inp=sample["input"]
                out=sample["output"]

                tp = np.pad(inp, 1, pad_with)
                to = np.pad(out, 1, pad_with)
                if (tp.shape != to.shape):
                    self.buildSuccess = False
                    return
                locs = np.where(tp != to)
                for i, j in zip(locs[0], locs[1]):
                    wdw=tp[i-1:i+2,j-1:j+2]
                    out = to[i, j]
                    hasRule=self.trManager.matchWindow(wdw)
                    if hasRule==-1:
                        rout = False
                        if timestep_ruled_out.get(mat_to_key(wdw)) == 1 or timestep_candiates.get(mat_to_key(wdw)) == 1:
                            continue
                        for in_wdw, out_wdw in zip(self.inWindows, self.outWindows):
                            if np.equal(wdw, in_wdw).all() and out != out_wdw[1, 1]:
                                timestep_ruled_out[mat_to_key(wdw)] = 1
                                rout = True
                                break
                        if not rout:
                            timestep_candiates[mat_to_key(wdw)] = out
                    else:
                        continue
                for key in timestep_candiates.keys():
                    out=timestep_candiates[key]
                    rule_wdw=key_to_mat(key)
                    exists=self.trManager.matchWindow(rule_wdw)
                    if exists ==-1:
                        #refineRule
                        new_rule=TransitionRule(rule_wdw,out)
                        self.trManager.addRule(new_rule)
                    else:
                        continue
            self.updateTrain()
            #plot_sample(self.task["train"][1])
        self.buildSuccess=True

    def printRules(self):
        self.trManager.printRules()

    def refineRule(self,tp,to,rule):
        pass
#0-9 colors
#10-13 borders
#15 don't care
def match(wdw1,wdw2):
    match_map=np.equal(wdw1,wdw2)
    wdw3=wdw2!=15
    return np.equal(match_map,wdw3).all()
class TransitionRuleManager:
    def __init__(self):
        self.rules=[]
    def addRule(self,rule):
        self.rules.append(rule)
    def matchWindow(self,wdw):
        for rule in self.rules:
            output=rule.apply(wdw)
            if output !=-1:
                return output
        return -1
    def printRules(self):
        for rule in self.rules:
            rule.print()
class TransitionRule:
    def __init__(self, neigh,out):
        # neigh[neigh==0]=15
        # neigh[neigh==-1]=15
        if(np.count_nonzero(neigh==15)==9):
            neigh[neigh==15]=20
        self.neigh=neigh
        self.out=out
    def apply(self,wdw,mode=1):
        if mode==1:
            doesMatch=match(wdw,self.neigh)
            if doesMatch:
                return self.out
            else:
                return -1
        else:
            candidates=[self.neigh,np.fliplr(self.neigh),np.fliplr(self.neigh)]
            for c in candidates:
                doesMatch = match(wdw, c)
                if doesMatch:
                    return self.out
                else:
                    continue
            return -1
    def getNeigh(self):
        return self.neigh
    def setNeigh(self,n):
        self.neigh=n
    def print(self):
        print(self.neigh,'------->',self.out)




# CA@
# tp=np.array(task[0]["input"])
# to=np.array(task[0]["output"])
# tp1=np.array(task[1]["input"])
# to1=np.array(task[1]["output"])
# tp=np.pad(tp, 1, pad_with)
# to=np.pad(to, 1, pad_with)
# tp1=np.pad(tp1, 1, pad_with)
# to1=np.pad(to1, 1, pad_with)
# cands=generateConsistentCandidates(tp,to)
# cands2=generateConsistentCandidates(tp1,to1)
# print([cands,cands2])
# print(clashInconsistent([cands,cands2]))
# elem=[[0,1,2],[3,4,5],[6,7,8]]
# elem2=[[15,15,15],[15,15,15],[6,7,8]]
# elem=np.array(elem)
# elem2=np.array(elem2)
# elem3=elem>3
# print(match(elem,elem2))

# myarray=np.arange()
#task = train_tasks["db3e9e38"]
# task = train_tasks["3aa6fb7a"]
# CA2=CA(task)
# CA2.build()
training_solved=[]
idx=0
# test_tasks=["af902bf9","b6afb2da","dc1df850","d4f3cd78","db3e9e38","23581191","32597951","d364b489","3aa6fb7a","4258a5f9","0ca9ddb6","41e4d17e","508bd3b6"]
#test_tasks=["69889d6e","f0afb749","f0df5ff0","fe9372f3","c87289bb","c97c0139"]
#test_tasks=["a85d4709","794b24be","6f8cd79b","c0f76784"]
test_tasks=["28e73c20"]
for t in test_tasks:
    task=train_tasks[t]
    CA2 = CA(task)
    CA2.build()
    #CA2.printRules()
    if(not CA2.build_succeeded()):
        continue
    a=CA2.run(task['test'][0]['input'],1).tolist()
    plot_picture(a)
    a2 = CA2.run( task['test'][0]['input'],2).tolist()
    plot_picture(a2)
    first_solved=False
    plot_sample(task["train"][1])
    if a != -1 and task['test'][0]['output'] == a:
        #plot_picture(a)
        training_solved.append(a)
        first_solved=True
    if a2 != -1 and task['test'][0]['output'] == a2 and not first_solved:
        #plot_picture(a2)
        training_solved.append(a2)
    idx+=1
print(len(training_solved))