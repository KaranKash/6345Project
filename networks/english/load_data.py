import os
import argparse
import gzip
import sys
import time
import numpy as np
from multiprocessing import Pool
from contextlib import closing
import csv
import tensorflow as tf
from itertools import permutations
from grid_mnist import *

bin_freq = 23
spect_width = bin_freq  # Don't add one pixel of zeros on either side of the image
dim_Y = 11
window_size = 400

label_map = {'1':1,'2':2,'3':3,'4':4,'5':5,'6':6,'7':7,'8':8,'9':9,'o':0,'z':0}
perms = [p for p in permutations('0123456789')]

def load_from_file(f):
    '''Given a file, returns a list of the string values in that value'''
    data = []
    for line in f:
        vector = []
        line = line.replace("[", "")
        line = line.replace("]", "")
        line_chars = line.split()
        for char in line_chars:
            vector.append(float(char))
        try:
            assert len(vector) == bin_freq
            data.append(vector)
        except AssertionError:
            if len(vector) == 0:
                pass
            else:
                # print len(vector)
                raise AssertionError

    # Now we have a list of length-23 vectors which we need to trim/pad to
    # window_size
    if len(data)>window_size:
        #cut excess rows
        cut = 1.*(len(data) - window_size)
        data = data[int(np.floor(cut/2)):-int(np.ceil(cut/2))]
    else:
        # pad data with excess rows of zeros about center
        cut = 1.*(window_size - len(data))
        data = [[0]*bin_freq]*int(np.floor(cut/2)) + data + [[0]*bin_freq]*int(np.ceil(cut/2))
    #Convert data to a numpy array and invert it
    data = np.flipud(np.array(data,dtype=np.float32))
    return data.flatten().tolist()

# def load_from_file(f):
#     '''Given a file, returns a list of the string values in that value'''
#     data = []
#     for line in f:
#         vector = []
#         line = line.replace("[", "")
#         line = line.replace("]", "")
#         line_chars = line.split()
#         for char in line_chars:
#             # vector.append(float(char)-MEAN_SPEC)
#             vector.append(float(char))
#         try:
#             assert len(vector) == bin_freq
#             data.append(vector)
#         except AssertionError:
#             if len(vector) == 0:
#                 pass
#             else:
#                 # print len(vector)
#                 raise AssertionError
#
#     #Convert data to a numpy array and invert it
#     data = np.flipud(np.array(data,dtype=np.float32))
#     return data.flatten().tolist()

def ld(rootdir,target):
    print("Couldn't find target file, creating it...")
    with open(target, 'wb') as datafile:
        writer = csv.writer(datafile)
        for subdir, dirs, files in os.walk(rootdir):
            for filename in files:
                tmp = filename.split("_")
                chars = tmp[1][:-5]
                if len(chars) <= 4:
                    f = open(os.path.join(subdir, filename))
                    row = load_from_file(f)
                    f.close()
                    writer.writerow([chars] + row)

def generate_mnist_set(labels,train=True):
    out = []
    matches = []
    for label in labels:
        i = np.random.randint(0,len(perms))
        j = np.random.randint(1,5)
        grid = perms[i][0:j]
        count = countOverlap(label,grid)
        out.append(makeGrid(grid,train=train))
        matches.append(count)
    out = np.array(out)
    matches = np.array(matches)
    out = np.expand_dims(out, axis=3)
    return out, matches

def countOverlap(string1,string2):
    count = 0
    for i in range(len(string1)):
        c = string1[i]
        if c == 'o' or c == 'z':
            c = '0'
        if c in string2:
            count += 1
    return count
