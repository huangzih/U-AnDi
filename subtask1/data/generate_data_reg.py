from andi_datasets.models_phenom import models_phenom 
import numpy as np
import pandas as pd
import argparse
import os
import gc
import csv
import random
from tqdm.auto import tqdm

parser = argparse.ArgumentParser()
arg = parser.add_argument
arg('--N', type=int)
arg('--l', type=int)
args = parser.parse_args()

N = args.N
l = args.l

file_path = './origin_data/'
if not os.path.exists(file_path):
    os.makedirs(file_path)

filename = file_path + 'data-reg-2d-{}.csv'.format(l)

MP = models_phenom()

L = 1.5*12800

#['attm', 'ctrw', 'fbm', 'lw', 'sbm'] correspond to 0,1,2,3,4
ls = None
def generate_exponent():
    global ls
    ls = [0.01*i for i in range(5,200)]
    n = len(ls)
    idx = np.random.randint(0,n)
    return ls[idx]

def generate_T(M,Tmin,Tsum):
    # brute force
    
    assert M in (2,3,4,5)
    Tmax = Tsum
    randval = lambda : random.randint(Tmin, Tmax)
    randseq = lambda : [randval() for i in range(M-1)]
    lastval = lambda seq : (Tsum - sum(seq))

    seq = randseq()
    while not (Tmin <= lastval(seq) <= Tmax):
        seq = randseq()
    return seq + [lastval(seq)]

T,exponents,models = None,None,None
dataset = []

def task1_generate(M,Tsum,Tmin=50):
    global T,exponents,models
    global dataset
    
    
    models = random.choices(range(2,3),k=M)
    
    exponents = [0]*M
    exponents[0] = generate_exponent()
    for i in range(1,M):
        exponents[i] = generate_exponent()
        while(abs(exponents[i]-exponents[i-1])<=0.5):
            exponents[i] = generate_exponent()
            
    
    T = generate_T(M,Tmin,Tsum)  
    
    T1 = [T[0]]*M
    for i in range(1,M):
        T1[i] = T[i]+1
    
    Tcnt = 0
    stch_lab = np.zeros([1,3*M]) 
    stch_traj = np.zeros([1,2*Tsum])
    
    for i in range(M):
        traj,label = MP.single_state( 
                        N = 1,
                        L = L,
                        T = T1[i],
                        Ds = [1.0,0.0],
                        alphas = [exponents[i],0])
        
        lab = [2]
        lab = np.append(lab,[exponents[i]])
        lab = np.append(lab,[Tcnt+T[i]])
        stch_lab[0,3*i:3*i+3] = lab
        if i == 0:
            traj_x = traj[:,0,0]-traj[0,0,0]
            traj_y = traj[:,0,1]-traj[0,0,1]
            stch_traj[0,Tcnt:Tcnt+T[i]] = traj_x
            stch_traj[0,Tsum+Tcnt:Tsum+Tcnt+T[i]] = traj_y
        else:
            traj_x = traj[1:,0,0]-traj[0,0,0]
            traj_y = traj[1:,0,1]-traj[0,0,1]
            stch_traj[0,Tcnt:Tcnt+T[i]] = traj_x+stch_traj[0,Tcnt-1]
            stch_traj[0,Tsum+Tcnt:Tsum+Tcnt+T[i]] = traj_y+stch_traj[0,Tsum+Tcnt-1]
        
        
        Tcnt = Tcnt+T[i]
        
    stch_traj = np.append(stch_lab,stch_traj)
    stch_traj = np.append([M],stch_traj)
    stch_traj = stch_traj.reshape(-1,3*M+1+2*Tsum)
    
    return stch_traj

Tmin = 10
N_traj = int(args.N)
Tsum = int(args.l)
dimension = 2

with open(filename,"w",newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pos_x","pos_y","length","N_seg","label","changepoint"])
    for i in tqdm(range(N_traj)):
        M = random.choice(range(2,6))

        datasetsum = task1_generate(M,Tsum,Tmin=Tmin)

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
