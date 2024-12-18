import numpy as np
import torch
import scipy.stats as st
import matplotlib.pyplot as plt
import random
import sys
from scipy.io import savemat, loadmat
import itertools
import time

def sample_discrete(probs):
    """Randomly sample an index with probability given by probs."""
    # Generate random number
    q = np.random.rand()
    
    # Find index
    i = 0
    p_sum = 0.0
    while p_sum < q:
        p_sum += probs[i]
        i += 1
    return i - 1  

lst2 = np.asarray(list(itertools.product([0, 1], repeat=2)))
lst3 = np.asarray(list(itertools.product([0, 1], repeat=3)))
lst4 = np.asarray(list(itertools.product([0, 1], repeat=4)))

def combination(con,nR):
    lst = np.asarray(list(itertools.product([0, 1], repeat=nR)))
    con = con.T
    sum_ = 0;
    for i in range(len(lst)):
        sum_ = sum_ + np.prod(con**lst[i,:],axis=1)
        
    return sum_

def two_list(c0,c1,lst2):
    sum_ = 0;
    for i in range(len(lst2)):
        sum_ = sum_ + (c0**lst2[i,0])*(c1**lst2[i,1])
    return sum_

def three_list(c0,c1,c2,lst3):
    sum_ = 0;
    for i in range(len(lst3)):
        sum_ = sum_ + (c0**lst3[i,0])*(c1**lst3[i,1])*(c2**lst3[i,2])
    return sum_

def four_list(c0,c1,c2,c3,lst4):
    sum_ = 0;
    for i in range(len(lst4)):
        sum_ = sum_ + (c0**lst4[i,0])*(c1**lst4[i,1])*(c2**lst4[i,2])*(c3**lst4[i,3])
    return sum_

def generate_samples_mCAD(tend,m,K,n,lx,r,lp,vol,Ndim,seed_):

    np.random.seed(seed_);
    G0 = [5*vol+np.random.normal(0,1)]; G1 = [5*vol+np.random.normal(0,1)]; G2 = [5*vol+np.random.normal(0,1)]; 
    G3 = [5*vol+np.random.normal(0,1)]; G4 = [5*vol+np.random.normal(0,1)]; 
    P0 = [5*vol+np.random.normal(0,1)]; P1 = [5*vol+np.random.normal(0,1)]; P2 = [5*vol+np.random.normal(0,1)];
    P3 = [5*vol+np.random.normal(0,1)]; P4 = [5*vol+np.random.normal(0,1)]; 
    t = [0]
    
    while t[-1] < tend:
        
        x0 = G0[-1]/vol; x1 = G1[-1]/vol; x2 = G2[-1]/vol; x3 = G3[-1]/vol; x4 = G4[-1]/vol; 
        x0_ = (r/lp)*x0;  x1_ = (r/lp)*x1;  x2_ = (r/lp)*x2;  x3_ = (r/lp)*x3;  x4_ = (r/lp)*x4;  # this is essential using proteins as markers
        x0n = (x0_/K)**n; x1n = (x1_/K)**n; x2n = (x2_/K)**n; x3n = (x3_/K)**n; x4n = (x4_/K)**n; 
        
        props = [ m*x1n/three_list(x1n,x3n,x4n,lst3), \
                  m*x2n/two_list(x2n,x3n,lst2), \
                  m*(x1n*x2n)/three_list(x1n,x2n,x3n,lst3), \
                  m*x4n/four_list(x0n,x1n,x2n,x4n,lst4), \
                  m/two_list(x1n,x2n,lst2),\
                  lx*x0, lx*x1, lx*x2, lx*x3, lx*x4] 

        ind = sample_discrete(np.asarray(props)/np.sum(np.asarray(props)))
        #print(ind)
        prop_sum = vol*sum(props)
        tau = np.random.exponential(scale=1/prop_sum)

        t.append(t[-1] + tau)
        
        if((G0[-1] + updates_G0[ind]) < 0): G0.append(G0[-1])
        else: G0.append(G0[-1] + updates_G0[ind])
            
        if((G1[-1] + updates_G1[ind]) < 0): G1.append(G1[-1])
        else: G1.append(G1[-1] + updates_G1[ind])
        
        if((G2[-1] + updates_G2[ind]) < 0): G2.append(G2[-1])
        else: G2.append(G2[-1] + updates_G2[ind])
        
        if((G3[-1] + updates_G3[ind]) < 0): G3.append(G3[-1])
        else: G3.append(G3[-1] + updates_G3[ind])
        
        if((G4[-1] + updates_G4[ind]) < 0): G4.append(G4[-1])
        else: G4.append(G4[-1] + updates_G4[ind])
            
    return np.asarray(t), np.asarray(G0), np.asarray(G1), np.asarray(G2), np.asarray(G3), np.asarray(G4)

up = 1.0;
updates_G0 = np.array([ up, 0, 0, 0, 0,  -up, 0, 0, 0, 0])
updates_G1 = np.array([  0,up, 0, 0, 0,   0,-up, 0, 0, 0])
updates_G2 = np.array([  0, 0,up, 0, 0,   0, 0,-up, 0, 0])
updates_G3 = np.array([  0, 0, 0,up, 0,   0, 0, 0,-up, 0])
updates_G4 = np.array([  0, 0, 0, 0,up,   0, 0, 0, 0,-up])

Ndim = 5;
m = 20; K = 10; n = 10;
lp = 1; r = 10; lx = 5;

vol  = np.double(sys.argv[1])
tend = np.double(sys.argv[2]); nsamples = 6000;
seed_  = int(sys.argv[3]);

start_time = time.time()
samples = np.zeros((nsamples,Ndim))
for i in range(0,nsamples):
    t1, G0, G1, G2, G3, G4 = generate_samples_mCAD(tend,m,K,n,lx,r,lp,vol,Ndim,i+nsamples*seed_+int((tend)/0.04)*nsamples);
    samples[i,0] = G0[-1]; samples[i,1] = G1[-1]; samples[i,2] = G2[-1]; samples[i,3] = G3[-1]; samples[i,4] = G4[-1];
    
    if(i%500==0):
        print(" done with sample ", i)
        
print(" done with tmax ", tend)
print(" total-simulation time ", time.time() - start_time)

mdic = {"tend": tend, "samples": samples, "m":m, "K":K, "n":n, "lp":lp, "r":r, "lx":lx}
savemat('/mnt/ceph/users/smaddu/stochastic_inference/mCAD_qssa_vol'+str(vol)+'_Nsamples'+str(nsamples)+'_tend'+str(tend)+'_seed'+str(seed_)+'.mat', mdic)