import numpy as np
import pandas as pd
import lcgen
import matplotlib.pyplot as plt
from scipy.special import erfc
from mittag_leffler.mittag_leffler import ml
import argparse
import os
import gc
import csv
import random
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--N', default=1e6,type=int)
arg('--l', default=500,type=int)
arg('--c', required=True,type=int)  # type of long-time correlation
arg('--tau',default=10,type=lambda x: int(x) if ',' not in x else [int(i) for i in x.split(',')])
arg('--omega',default=0.1,type=float)

args = parser.parse_args()
c = args.c
N = args.N
l = args.l
tau = args.tau
omega = args.omega

c_map={
    1:"expo",
    2:"ml",
    3:"multiexp",
    4:"expcos"
}
cc = c_map[c]

file_path = './origin_data/'
if not os.path.exists(file_path):
    os.makedirs(file_path)
filename = file_path+'data-{}-2d-{}.csv'.format (cc,l)

def generate_T(M,Tmin,Tsum):
    Tmax = Tsum
    randval = lambda : random.randint(Tmin, Tmax)
    randseq = lambda : [randval() for i in range(M-1)]
    lastval = lambda seq : (Tsum - sum(seq))

    seq = randseq()
    while not (Tmin <= lastval(seq) <= Tmax):
        seq = randseq()
    return seq + [lastval(seq)]

def generate_exponent():
    global ls
    ls = [0.01*i for i in range(5,200)]
    n = len(ls)
    idx = np.random.randint(0,n)
    return ls[idx]

T,exponents,models = None,None,None
dataset = []

def segmented_trajectories(M, Tsum, Tmin, dimension,c):
    global T,exponents,models
    global dataset
    
    coef,a = [0]*M,[0]*M
    coef[0] = 0.01*random.randint(5,200)
    for i in range(1,M):
        coef[i] = 0.01*random.randint(5,200)
        while abs(coef[i]-coef[i-1])<=0.5:
            coef[i] = 0.01*random.randint(5,200)
    corr_map={
        1:lcgen.fbm_expo,
        2:lcgen.fbm_ml,
        3:lcgen.fbm_multiexp,
        4:lcgen.fbm_expcos,
    }
    
    if c == 1: #expo
        a = [i/tau for i in coef]
    elif c == 2: #ml
        exponents = [0]*M
        exponents[0] = generate_exponent()
        for i in range(1,M):
            exponents[i] = generate_exponent()
            while(abs(exponents[i]-exponents[i-1])<=0.5):
                exponents[i] = generate_exponent()        
    elif c == 3: #multiexp
        a = [i/(0.5*tau[0]+0.5*tau[1]) for i in coef]
    elif c == 4: # expcos
        a = [i*(1+tau**2 * omega**2)/tau for i in coef]

    T = generate_T(M,Tmin,Tsum)  
    
    T1 = [T[0]]*M
    for i in range(1,M):
        T1[i] = T[i]+1
        
    Tcnt = 0
    labelsum = np.zeros([1,3*M]) 
    datasetsum = np.zeros([1,dimension*Tsum])

    for i in range(M):
        if c == 1:
            dataset = np.concatenate((corr_map[c](T1[i]-1,a[i],tau),
                                     corr_map[c](T1[i]-1,a[i],tau)))
        elif c == 2:
            dataset = np.concatenate((corr_map[c](T1[i]-1,1,0.5*(2-exponents[i]),tau),
                                     corr_map[c](T1[i]-1,1,0.5*(2-exponents[i]),tau)))          
        elif c == 3:
            dataset = np.concatenate((corr_map[c](T1[i]-1,a[i],tau),
                                     corr_map[c](T1[i]-1,a[i],tau)))
        elif c == 4:
            dataset = np.concatenate((corr_map[c](T1[i]-1,a[i],tau,omega),
                                     corr_map[c](T1[i]-1,a[i],tau,omega)))
        if c == 2:
            dataset = np.append([c,exponents[i]],dataset)
        else:
            dataset = np.append([c,coef[i]],dataset)
        dataset = dataset.reshape(1,dataset.shape[0])        
        
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
        Tcnt+=T[i]
    datasetsum = np.append(labelsum,datasetsum)
    datasetsum = np.append([M],datasetsum)
    datasetsum = datasetsum.reshape(-1,3*M+1+dimension*Tsum)
    return datasetsum

N_traj = int(args.N)
Tsum = int(args.l)


Tmin = 10
dimension = 2

with open(filename,"w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pos_x","pos_y","length","N_seg","label","changepoint"])
    for i in tqdm(range(N_traj)):
        M = random.choice(range(2,6))
        datasetsum = segmented_trajectories(M,Tsum,Tmin,dimension,c)
        
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
