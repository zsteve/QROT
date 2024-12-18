import numpy as np
# from pinnutils import *
from scipy.interpolate import griddata
from itertools import product, combinations
from scipy.io import savemat, loadmat

import math
import scipy.io as sio
# from pinnutils import *
import sys
from tqdm import tqdm_notebook as tqdm
import os
import time as timeit
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:32768"
import itertools


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

def combination_two(con,lst2,truth):
    con = con.T
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst2)):
        temp = (con[:,0]**lst2[i,0])*(con[:,1]**lst2[i,1])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp

    return  weight_sum, sum_

def combination_three(con,lst3,truth):
    con = con.T
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst3)):
        temp = (con[:,0]**lst3[i,0])*(con[:,1]**lst3[i,1])*(con[:,2]**lst3[i,2])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp
        
    return  weight_sum, sum_

def combination_four(con,lst4,truth):
    con = con.T
    sum_ = 0; weight_sum = 0;
    for i in range(len(lst4)):
        temp = (con[:,0]**lst4[i,0])*(con[:,1]**lst4[i,1])*(con[:,2]**lst4[i,2])*(con[:,3]**lst4[i,3])
        sum_ = sum_ + temp
        weight_sum = weight_sum + truth[i]*temp
    return  weight_sum, sum_


Gn = {'G1':0,'G2':1,'Fg':2,'E':3,
      'Fli':4,'S':5,'Ceb':6,'P':7,
      'cJ':8, 'Eg':9,'G':10
     }

def interact_HSC(x,Ndim,mvec,kvec,nvec):
    x_ = np.zeros_like(x)
    for i in range(0,Ndim):
        x_[i,:] = (x[i,:]/kvec[i])**nvec[i]
 
    f = np.zeros_like(x);
    
    # G1 <-- (G1 | G2 | Fli) & ~P
    keys = ['G1', 'G2', 'Fli', 'P']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where((lst4[:, 0] | lst4[:, 1] | lst4[:, 2]) & ~lst4[:, 3], 1, 0)
    num_, den_ = combination_four(x_[arr_ind,:],lst4,truth)
    f[0,:] = mvec[0]*(num_/den_);
    
    # G2 <-- G2 & ~(G1 & Fg) & ~P
    keys = ['G2', 'G1', 'Fg', 'P']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where(lst4[:, 0] & ~(lst4[:, 1] & lst4[:, 2]) & ~lst4[:, 3], 1, 0)
    num_, den_ = combination_four(x_[arr_ind,:],lst4,truth)
    f[1,:] = mvec[0]*(num_/den_);
    
    # Fg <-- G1
    f[2,:] = mvec[0]*x_[Gn['G1'],:]/(1+x_[Gn['G1'],:])
    
    # E <-- G1 & ~Fli
    keys = ['G1', 'Fli']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
    num_, den_ = combination_two(x_[arr_ind,:],lst2,truth)
    f[3,:] = mvec[0]*(num_/den_);
                     
    # Fli <-- G1 & ~E
    keys = ['G1', 'E']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
    num_, den_ = combination_two(x_[arr_ind,:],lst2,truth)
    f[4,:] = mvec[0]*(num_/den_);
                     
    # S <-- G1 & ~P      
    keys = ['G1', 'P']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where(lst2[:, 0] & ~lst2[:, 1], 1, 0)
    num_, den_ = combination_two(x_[arr_ind,:],lst2,truth)
    f[5,:] = mvec[0]*(num_/den_);
                     
    # Ceb <-- Ceb & ~(G1 & Fg & S)     
    keys = ['Ceb', 'G1', 'Fg','S']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where(lst4[:, 0] & ~(lst4[:, 1] & lst4[:, 2] & lst4[:, 3]), 1, 0)
    num_, den_ = combination_four(x_[arr_ind,:],lst4,truth)
    f[6,:] = mvec[0]*(num_/den_);
                     
    # P <-- (Ceb | P) & ~(G1 | G2)     
    keys = ['Ceb', 'P', 'G1','G2']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where( (lst4[:, 0] | lst4[:, 1]) & ~(lst4[:, 2] | lst4[:, 3]), 1, 0)
    num_, den_ = combination_four(x_[arr_ind,:],lst4,truth)
    f[7,:] = mvec[0]*(num_/den_);

    # cJ <-- (P & ~ G)     
    keys = ['P', 'G']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where( lst2[:, 0] & ~lst2[:, 1], 1, 0)
    num_, den_ = combination_two(x_[arr_ind,:],lst2,truth)
    f[8,:] = mvec[0]*(num_/den_);
                     
    # Eg <-- (P & cJ) & ~G   
    keys = ['P', 'cJ', 'G']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where( (lst3[:, 0] & lst3[:, 1]) & ~lst3[:,2], 1, 0)
    num_, den_ = combination_three(x_[arr_ind,:],lst3,truth)
    f[9,:] = mvec[0]*(num_/den_);
                     
    # G <-- (Ceb & ~Eg)
    keys = ['Ceb', 'Eg']; arr_ind = np.array([Gn[key] for key in keys])
    truth = np.where( (lst2[:, 0] & ~lst2[:, 1]), 1, 0)
    num_, den_ = combination_two(x_[arr_ind,:],lst2,truth)
    f[10,:] = mvec[0]*(num_/den_);

    return f

def simulate_HSC(nsamples,Nsnaps,Ndim,seed_,maxiter,c,lx,r,lp,nm='add'):
    # P = 0; S = 1; F = 2; E = 3; C = 4;
    dt = 0.001; 
    np.random.seed(seed_)
    xold =  np.random.normal(5,1,(Ndim,nsamples))
    xnew =  xold + 0;

    samples_fullx = np.zeros((Nsnaps,nsamples,Ndim+1))
    
    count = 0; 
    tt = np.zeros((Nsnaps,))
    
    for i in range(0,maxiter-1):
        if((i%int(maxiter/Nsnaps)==0)):
            samples_fullx[count,:,0:Ndim] = xold.T
            samples_fullx[count,:,Ndim]   = dt*i
            tt[count] = dt*i
            count = count + 1;
            
        if(nm=='add'): 
            noisex = np.sqrt(dt)*np.random.normal(0,1,(Ndim,nsamples))
        elif(nm=='mult'): 
            noisex = np.sqrt(xold)*np.sqrt(dt)*np.random.normal(0,1,(Ndim,nsamples))
        else:
            noisex = np.sqrt(interact_HSC((r/lp)*xold,Ndim,mvec,kvec,nvec) + lx*xold)*np.sqrt(dt)*np.random.normal(0,1,(Ndim,nsamples))
            
        xnew  = xold + dt*(interact_HSC((r/lp)*xold,Ndim,mvec,kvec,nvec) - lx*xold) + c*noisex

        ind = np.where(xnew<0)
        xnew[ind[0],ind[1]] = xold[ind[0],ind[1]] # -(xnew[ind[0],ind[1]])
        
        xold  = xnew + 0; 
        
    print(count,dt*i, " SFI dt ",tt[1] - tt[0])
    
    return samples_fullx, tt

if __name__ == "__main__":
    print( " lst2 ", len(lst2))
    print( " lst3 ", len(lst3))
    print( " lst4 ", len(lst4))
    
    Ndim = 11;
    mvec = 20*np.ones((Ndim,)); 
    kvec = 10*np.ones((Ndim,)); 
    nvec = 10*np.ones((Ndim,));

    dt = 0.001; lp = 1;  r = 10;  lx = 5; 
    # control lx to balance between drift and degradation
    c = np.double(sys.argv[1]); 
    noise_model = str(sys.argv[2]);

    nsamples = 20_000; Nsnaps = 200; seed = 10; maxiter = 4001; 
    samples_full, tt = simulate_HSC(nsamples,Nsnaps,Ndim,seed,maxiter,c,lx,r,lp,nm=noise_model);
    dt = tt[1] - tt[0];

    tind = np.arange(0,Nsnaps,1); nsnaps = len(tind);
    print(" nsnaps ", len(tind), " tind ", tt[tind], tind, " dtt ", np.round(tt[tind][1] - tt[tind][0],2))

    samples = samples_full[tind,:,:]
    mdic = {"samples":samples,"tt":tt[tind],"c":c,"maxiter":maxiter,"seed":seed,
           "dt":dt,",lp":lp,"lx":lx,"r":r}
    # savemat("/mnt/ceph/users/smaddu/stochastic_inference/data/longtraj_HSC_"+str(noise_model)+"c"+str(c)+"_dim_"+str(Ndim)+".mat", mdic)
    savemat("longtraj_HSC_"+str(noise_model)+"c"+str(c)+"_dim_"+str(Ndim)+".mat", mdic)

    print(" samples_full shape ", samples_full.shape, " SFI dt ", tt[1] - tt[0], tt.shape)
    print(" done with simulations.... ")
