import numpy as np 
import random
import csv
from random import randint
from matplotlib import pyplot as plt

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

filename = file_path + 'data-bio-2d-L{}.csv'.format(l)

def Brownian_Motion(T,D=1.,dimension=2):
    x = 0
    y = 0
    z = 0
    if dimension == 1:
        x_ls = [0]
        disp = [0]
        for i in range(T-1):
            rn = np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            disp.append(rn)
            x = x + rn
            x_ls.append(x)
        x_ls = np.array(x_ls)
        x_ls = x_ls.reshape(-1,T)
        return x_ls,disp
    elif dimension == 2:
        x_ls = [0]
        for i in range(T-1):
            x = x + np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            x_ls.append(x)
        x_ls.append(0)
        for i in range(T-1):
            y = y + np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            x_ls.append(y)
    elif dimension == 3:
        x_ls = [5,1,0]
        for i in range(T-1):
            x = x + np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            x_ls.append(x)
        x_ls.append(0)
        for i in range(T-1):
            y = y + np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            x_ls.append(y)    
        x_ls.append(0)
        for i in range(T-1):
            z = z + np.sqrt(2*np.abs(D))*np.random.normal(0.,1.)
            x_ls.append(z)  
    return x_ls


def free_diffusion(T,D=1.):
    x_ls = Brownian_Motion(T,D,dimension=2)
    x_ls.insert(0,1)
    x_ls.insert(0,0)
    return np.array(x_ls)


def confined_diffusion(T,R,D=1.):       
    pos = np.zeros((T, 2))
    dispx, dispy = Brownian_Motion(T,D=1.,dimension=1)[1], Brownian_Motion(T,D=1.,dimension=1)[1]
    angle = np.random.rand() * 2 * np.pi
    pos[0, :] = [R * np.cos(angle), R * np.sin(angle)]   
    for t in range(1, T):
        pos[t, :] = [pos[t-1, 0]+dispx[t], pos[t-1, 1]+dispy[t]]  
        distance = np.sqrt(pos[t, 0]**2 + pos[t, 1]**2)
        if distance > R:
            angle = np.arctan2(pos[t, 1], pos[t, 0])
            pos[t, 0] = R * np.cos(angle)
            pos[t, 1] = R * np.sin(angle)
    pos = pos-pos[0,:]
    pos = np.concatenate((np.array([1,1]),pos[:,0],pos[:,1]))
    return pos

def directed_diffusion(T,v,theta,D=1.):
    pos = np.zeros((T, 2))
    dispx_r, dispy_r = Brownian_Motion(T,D=1.,dimension=1)[1], Brownian_Motion(T,D=1.,dimension=1)[1]
    dispx_d, dispy_d = np.array([v*np.cos(theta)]*T), np.array([v*np.sin(theta)]*T)
    dispx, dispy = np.add(dispx_d,dispx_r), np.add(dispy_d,dispy_r)
    pos[0,:] = [0.,0.]
    for t in range(1, T):
        pos[t, :] = [pos[t-1, 0]+dispx[t], pos[t-1, 1]+dispy[t]] 
    pos_label = np.array([2,1])
    pos = np.concatenate((pos_label,pos[:,0],pos[:,1]))
    return pos


def immobility(T,sigma_p):
    pos = np.zeros((T, 2))
    pos[0,:] = [0.,0.]
    pos[1:,0] = np.random.normal(pos[0,0],sigma_p,T-1)
    pos[1:,1] = np.random.normal(pos[0,1],sigma_p,T-1)
    pos_label = np.array([3,1])
    pos = np.concatenate((pos_label,pos[:,0],pos[:,1])) 
    return pos


def generate_T(M,Tmin,Tsum):
    # brute force
    assert M in (2,3,4,5,6)
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
def segmented_trajectories_std(M,Tsum,Tmin=50,dimension=1):
    global T,models
    global dataset
    
    models = random.choices(range(0,4),k=M)
    for i in range(1,M):
        while (np.all(models[i] == models[i-1])):
            models[i] = random.choices(range(0,4))[0]
            
    T = generate_T(M,Tmin,Tsum)  
    T1 = [T[0]]*M
    for i in range(1,M):
        T1[i] = T[i]+1
    
    Tcnt = 0
    labelsum = np.zeros([1,3*M]) 
    datasetsum = np.zeros([1,dimension*Tsum])
    for i in range(M):
        if models[i]==0:
            dataset = free_diffusion(T=T1[i],D=random.uniform(1,3))
        elif models[i]==1:
            dataset = confined_diffusion(T=T1[i],R=random.uniform(3,5),D=random.uniform(1,3))
        elif models[i]==2:
            theta = np.random.rand() * 2 * np.pi
            dataset = directed_diffusion(T=T1[i],v=random.uniform(2,3),theta=theta,D=random.uniform(1,3))
        else:
            dataset = immobility(T=T1[i],sigma_p=random.uniform(0.8,1.6))
        
        dataset = dataset.reshape(-1,1).transpose()
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

Tmin = 10
N_traj = int(args.N)
Tsum = int(args.l)
dimension = 2

with open(filename, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["pos_x","pos_y","length","N_diffmodel","label","changepoint"])
    for i in tqdm(range(N_traj)):
        M = random.choice(range(2,6))

        datasetsum = segmented_trajectories_std(M,Tsum,Tmin,dimension)

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