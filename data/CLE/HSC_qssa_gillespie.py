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

def combination_two(con,lst2,truth):
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst2)):
        temp = (con[0]**lst2[i,0])*(con[1]**lst2[i,1])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp

    return  weight_sum/sum_

def combination_three(con,lst3,truth):
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst3)):
        temp = (con[0]**lst3[i,0])*(con[1]**lst3[i,1])*(con[2]**lst3[i,2])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp
        
    return  weight_sum/sum_

def combination_four(con,lst4,truth):
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst4)):
        temp = (con[0]**lst4[i,0])*(con[1]**lst4[i,1])*(con[2]**lst4[i,2])*(con[3]**lst4[i,3])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp
    return  weight_sum/sum_

Gn = {'G1':0,'G2':1,'Fg':2,'E':3,
      'Fli':4,'S':5,'Ceb':6,'P':7,
      'cJ':8, 'Eg':9,'G':10
     }

def generate_samples_HSC(tend,m,K,n,lx,r,lp,vol,Ndim,seed_):

    np.random.seed(seed_);
    G0 = [5*vol+np.random.normal(0,1)]; G1 = [5*vol+np.random.normal(0,1)]; G2 = [5*vol+np.random.normal(0,1)]; 
    G3 = [5*vol+np.random.normal(0,1)]; G4 = [5*vol+np.random.normal(0,1)]; G5 = [5*vol+np.random.normal(0,1)]; 
    G6 = [5*vol+np.random.normal(0,1)]; G7 = [5*vol+np.random.normal(0,1)]; G8 = [5*vol+np.random.normal(0,1)]; 
    G9 = [5*vol+np.random.normal(0,1)]; G10 = [5*vol+np.random.normal(0,1)]; 

    t = [0]
    while t[-1] < tend:
        
        x0 = G0[-1]/vol; x1 = G1[-1]/vol; x2 = G2[-1]/vol; x3 = G3[-1]/vol; x4 = G4[-1]/vol; 
        x5 = G5[-1]/vol; x6 = G6[-1]/vol; x7 = G7[-1]/vol; x8 = G8[-1]/vol; x9 = G9[-1]/vol; 
        x10 = G10[-1]/vol; 
        
        x0_ = (r/lp)*x0;  x1_ = (r/lp)*x1;  x2_ = (r/lp)*x2;  x3_ = (r/lp)*x3;  x4_ = (r/lp)*x4;  # this is essential using proteins as markers
        x5_ = (r/lp)*x5;  x6_ = (r/lp)*x6;  x7_ = (r/lp)*x7;  x8_ = (r/lp)*x8;  x9_ = (r/lp)*x9; 
        x10_ = (r/lp)*x10;  
        
        x0n = (x0_/K)**n; x1n = (x1_/K)**n; x2n = (x2_/K)**n; x3n = (x3_/K)**n; x4n = (x4_/K)**n; 
        x5n = (x5_/K)**n; x6n = (x6_/K)**n; x7n = (x7_/K)**n; x8n = (x8_/K)**n; x9n = (x9_/K)**n; 
        x10n = (x10_/K)**n; 

        # G1 <-- (G1 | G2 | Fli) & ~P
        truth = np.where((lst4[:, 0] | lst4[:, 1] | lst4[:, 2]) & ~lst4[:, 3], 1, 0)
        p0 = m*combination_four([x0n, x1n, x4n, x7n],lst4,truth)

        # G2 <-- G2 & ~(G1 & Fg) & ~P
        truth = np.where(lst4[:, 0] & ~(lst4[:, 1] & lst4[:, 2]) & ~lst4[:, 3], 1, 0)
        p1 = m*combination_four([x1n, x0n, x2n, x7n],lst4,truth)

        # Fg <-- G1
        p2 = m*x0n/(1+x0n)

        # E <-- G1 & ~Fli
        truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
        p3 = m*combination_two([x0n,x4n],lst2,truth)

        # Fli <-- G1 & ~E
        truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
        p4 = m*combination_two([x0n,x3n],lst2,truth)

        # S <-- G1 & ~P      
        truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
        p5 = m*combination_two([x0n,x7n],lst2,truth)

        # Ceb <-- Ceb & ~(G1 & Fg & S)     
        truth = np.where(lst4[:, 0] & ~(lst4[:, 1] & lst4[:, 2] & lst4[:, 3]), 1, 0)
        p6 = m*combination_four([x6n,x0n,x2n,x5n],lst4,truth)

        # P <-- (Ceb | P) & ~(G1 | G2)   
        truth = np.where( (lst4[:, 0] | lst4[:, 1]) & ~(lst4[:, 2] | lst4[:, 3]), 1, 0)
        p7 = m*combination_four([x6n,x7n,x0n,x1n],lst4,truth)

        # cJ <-- (P & ~ G)     
        truth = np.where( lst2[:, 0] & ~lst2[:, 1], 1, 0)
        p8 = m*combination_two([x7n,x10n],lst2,truth)

        # Eg <-- (P & cJ) & ~G   
        truth = np.where( (lst3[:, 0] & lst3[:, 1]) & ~lst3[:,2], 1, 0)
        p9 = m*combination_three([x7n,x8n,x10n],lst3,truth)

        # G <-- (Ceb & ~Eg)
        truth = np.where( lst2[:, 0] & ~lst2[:, 1], 1, 0)
        p10 = m*combination_two([x6n,x9n],lst2,truth)
        
        props = [ p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, \
                  lx*x0, lx*x1, lx*x2, lx*x3, lx*x4, lx*x5, lx*x6, lx*x7, lx*x8, lx*x9, lx*x10] 

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

        if((G5[-1] + updates_G5[ind]) < 0): G5.append(G5[-1])
        else: G5.append(G5[-1] + updates_G5[ind])

        if((G6[-1] + updates_G6[ind]) < 0): G6.append(G6[-1])
        else: G6.append(G6[-1] + updates_G6[ind])

        if((G7[-1] + updates_G7[ind]) < 0): G7.append(G7[-1])
        else: G7.append(G7[-1] + updates_G7[ind])

        if((G8[-1] + updates_G8[ind]) < 0): G8.append(G8[-1])
        else: G8.append(G8[-1] + updates_G8[ind])

        if((G9[-1] + updates_G9[ind]) < 0): G9.append(G9[-1])
        else: G9.append(G9[-1] + updates_G9[ind])

        if((G10[-1] + updates_G10[ind]) < 0): G10.append(G10[-1])
        else: G10.append(G10[-1] + updates_G10[ind])
    
    return np.asarray(t), np.asarray(G0), np.asarray(G1), np.asarray(G2), np.asarray(G3), np.asarray(G4), np.asarray(G5), np.asarray(G6), np.asarray(G7), np.asarray(G8), np.asarray(G9), np.asarray(G10)

up = 1.0;
updates_G0  = np.array([ up, 0, 0, 0, 0, 0, 0, 0, 0, 0,  0,  -up, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
updates_G1  = np.array([  0,up, 0, 0, 0, 0, 0, 0, 0, 0,  0,  0, -up, 0, 0, 0, 0, 0, 0, 0, 0, 0])
updates_G2  = np.array([  0, 0,up, 0, 0, 0, 0, 0, 0, 0,  0,  0,  0,-up, 0, 0, 0, 0, 0, 0, 0, 0])
updates_G3  = np.array([  0, 0, 0,up, 0, 0, 0, 0, 0, 0,  0,  0, 0, 0, -up, 0, 0, 0, 0, 0, 0, 0])
updates_G4  = np.array([  0, 0, 0, 0,up, 0, 0, 0, 0, 0,  0,  0, 0, 0, 0, -up, 0, 0, 0, 0, 0, 0])
updates_G5  = np.array([  0, 0, 0, 0, 0, up, 0, 0, 0, 0, 0,  0, 0, 0, 0, 0, -up, 0, 0, 0, 0, 0])
updates_G6  = np.array([  0, 0, 0, 0, 0, 0, up, 0, 0, 0, 0,  0, 0, 0, 0, 0, 0, -up, 0, 0, 0, 0])
updates_G7  = np.array([  0, 0, 0, 0, 0, 0, 0, up, 0, 0, 0,  0, 0, 0, 0, 0, 0, 0, -up, 0, 0, 0])
updates_G8  = np.array([  0, 0, 0, 0, 0, 0, 0, 0, up, 0, 0,  0, 0, 0, 0, 0, 0, 0, 0, -up, 0, 0])
updates_G9  = np.array([  0, 0, 0, 0, 0, 0, 0, 0, 0, up, 0,  0, 0, 0, 0, 0, 0, 0, 0, 0, -up, 0])
updates_G10 = np.array([  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, up,  0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -up])

Ndim = 11;
m = 20; K = 10; n = 10;
lp = 1; r = 10; lx = 5;

vol  = np.double(sys.argv[1])
tend = np.double(sys.argv[2]); nsamples = 2000;
seed_  = int(sys.argv[3]);

start_time = time.time()
samples = np.zeros((nsamples,Ndim))
for i in range(0,nsamples):
    t1, G0, G1, G2, G3, G4, G5, G6, G7, G8, G9, G10 = generate_samples_HSC(tend,m,K,n,lx,r,lp,vol,Ndim,i+nsamples*seed_+int((tend)/0.04)*nsamples);
    samples[i,0] = G0[-1]; samples[i,1] = G1[-1]; samples[i,2] = G2[-1]; samples[i,3] = G3[-1]; samples[i,4] = G4[-1];
    samples[i,5] = G5[-1]; samples[i,6] = G6[-1]; samples[i,7] = G7[-1]; samples[i,8] = G8[-1]; samples[i,9] = G9[-1];
    samples[i,10] = G10[-1]; 
    
    if(i%500==0):
        print(" done with sample ", i)
        
print(" done with tmax ", tend)
print(" total-simulation time ", time.time() - start_time)

mdic = {"tend": tend, "samples": samples, "m":m, "K":K, "n":n, "lp":lp, "r":r, "lx":lx}
savemat('/mnt/ceph/users/smaddu/stochastic_inference/HSC_qssa_vol'+str(vol)+'_Nsamples'+str(nsamples)+'_tend'+str(tend)+'_seed'+str(seed_)+'.mat', mdic)