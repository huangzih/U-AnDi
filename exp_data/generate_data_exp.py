import argparse
import os

import andi
AD = andi.andi_datasets()

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--N', type=str)
arg('--l', type=str)
arg('--s', type=str)
args = parser.parse_args()

N_traj = int(args.N)
l = int(args.l)

import numpy as np
import pandas as pd
import csv
import random
from tqdm.auto import tqdm
from matplotlib import pyplot as plt

Tsum = l
dimension = 2
Tmin=10
Tmax=l
sigma = round(float(args.s), 3)
print(l, sigma)

file_path = './origin_data/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = file_path + 'data-exp-2d-{}-{}.csv'.format(l, sigma)

ls = None
def generate_exponent(model):
    global ls
    if model == 0 or model ==1:
        ls = [0.01*i for i in range(5,101)]
    elif model == 2:
        ls = [0.01*i for i in range(5,200)]
    elif model ==3:
        ls = [0.01*i for i in range(101,201)]
    elif model ==4:
        ls = [0.01*i for i in range(5,201)]
    n = len(ls)
    idx = np.random.randint(0,n)
    return ls[idx]

def generate_T(M,Tmin,Tmax,Tsum):
    # brute force
    assert M in (2,3,4,5)
    randval = lambda : random.randint(Tmin, Tmax)
    randseq = lambda : [randval() for i in range(M-1)]
    lastval = lambda seq : (Tsum - sum(seq))

    seq = randseq()
    while not ((Tmin <= lastval(seq) <= Tmax) and (np.unique(seq + [lastval(seq)]).shape[0] == M)) :
        seq = randseq()
    return seq + [lastval(seq)]

#return the name of the model
def model_name(model):
    model_name = AD.avail_models_name[int(model)]
    return model_name

T,exponents,models = None,None,None
dataset = []

def segmented_trajectories_std(M, Tsum, Tmin=50, Tmax=100, dimension=1):
    
    global T,exponents,models
    global dataset
     
    exponents = [0]*M
    
    
    models = random.choices(range(1,3),k=M) 
    #print('model list',models)
    
    for i in range(1,M):
        while (np.all(models[i] == models[i-1])):
            models[i] = random.choices(range(1,3))[0]
    #print('model list',models)
    for i in range(0,M):
        if not models[i] == 5:
            exponents[i] = generate_exponent(models[i])  
        elif models[i] == 5:
            exponents[i] = 1.
            
    
    T = generate_T(M,Tmin,Tmax,Tsum)  
    
    
    T1 = [T[0]]*M
    for i in range(1,M):
        T1[i] = T[i]+1
    
    Tcnt = 0
    labelsum = np.zeros([1,3*M]) 
    datasetsum = np.zeros([1,dimension*Tsum])
    datasetsum_x = np.zeros([1,Tsum])
    datasetsum_y = np.zeros([1,Tsum])
    datasetsum_z = np.zeros([1,Tsum])
    
    for i in range(M):
        dataset = AD.create_noisy_localization_dataset(
            dataset=np.array([]),
            T= T1[i],
            N= 1,
            exponents= exponents[i],
            models= models[i],
            dimension= dimension,
            sigma=sigma)
        label = dataset[0,:2]
        label = np.append(label,[Tcnt+T[i]])
        labelsum[0,3*i:3*i+3] = label
        if i == 0:
            data_x = dataset[0,2:T[i]+2]
            data_y = dataset[0,T[i]+2:2*T[i]+2]
            datasetsum[0,Tcnt:Tcnt+T[i]] = data_x
            datasetsum[0,Tsum+Tcnt:Tsum+Tcnt+T[i]] = data_y
        else:
            data_x = dataset[0,3:T1[i]+2]
            data_y = dataset[0,T1[i]+3:2*T1[i]+2]
            datasetsum[0,Tcnt:Tcnt+T[i]] = data_x+datasetsum[0,Tcnt-1]
            datasetsum[0,Tsum+Tcnt:Tsum+Tcnt+T[i]] = data_y+datasetsum[0,Tsum+Tcnt-1]
        
        Tcnt = Tcnt+T[i]
    datasetsum = np.append(labelsum,datasetsum)
    datasetsum = np.append([M],datasetsum)
    datasetsum = datasetsum.reshape(-1,3*M+1+dimension*Tsum)
    
    return datasetsum

with open(filename,"w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pos_x","pos_y","length","N_diffmodel","label","changepoint"])
    for i in tqdm(range(N_traj)):
        M = random.choice(range(2,6))

        datasetsum = segmented_trajectories_std(M,Tsum,Tmin,Tmax,dimension)

        data_x = datasetsum[0,(3*M+1):(3*M+Tsum+1)]
        data_x2str = ['%f'%i for i in data_x]
        data_x2str = ','.join(data_x2str)

        data_y = datasetsum[0,(3*M+Tsum+1):]
        data_y2str = ['%f'%i for i in data_y]
        data_y2str = ','.join(data_y2str)


        label = datasetsum[0,1:(3*M+1)]
        label_new = []
        for i in range(label.shape[0]):
            if (i-2)%3 == 1:
                label_new.append(int(label[i]))
            if (i-2)%3 == 2:
                label_new.append(round(label[i],2))
        label2str = [str(i) for i in label_new]
        label2str = ','.join(label2str)

        chp = [datasetsum[0,3*i] for i in range(1,M)]
        chp2str = ['%d'%i for i in chp]
        chp2str = ','.join(chp2str)

        writer.writerow([data_x2str,data_y2str,Tsum,int(datasetsum[0,0]),label2str,chp2str])
f.close()
