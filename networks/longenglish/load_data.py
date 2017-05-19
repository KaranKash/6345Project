import numpy as np
from grid_mnist import *
import random

label_map = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'o':0,'z':0}
choices = [[0],[0,1],[0,1,2,2],[0,1,2,2,3,3,3],[0,1,2,3,3,4,4,4,4],[0,1,2,3,3,4,4,4,5,5,5,5,5],[0,1,2,3,4,4,5,5,6,6,6,6,6],[0,1,2,3,4,5,6,6,7,7,7,7,7]]

def generate_mnist_set(labels,copy,train=True):
    out = []
    matches = []
    for i in range(copy):
        for label in labels:
            label = str(label) # python 2.7
            # label = str(label)[2:-1] # python3
            overlap = random.choice(choices[len(label)])
            grid = generate_mnist_grid(label,overlap,train)
            out.append(grid)
            matches.append(overlap)
    out = np.array(out)
    matches = np.array(matches)
    out = np.expand_dims(out, axis=4)
    return out, matches

def generate_mnist_grid(label,overlap,train=True):
    vals = ""
    orig = [label_map[c] for c in label]
    for i in range(overlap):
        c = np.random.randint(0,len(label))
        vals += str(label_map[label[c]])
        label = label[:c] + label[c+1:]
    for i in range(9-len(vals)):
        c = np.random.randint(0,10)
        if c not in orig:
            vals += str(c)
    return makeGrid(vals,train=train)
