# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 06:26:27 2021

@author: Francisco
"""

import learning_algorithms as la
import numpy as np

np.random.seed(100)
n_samples = 10000

#Example 1
#y(t) = z^(-2)D(z)v(t)
#D(z) = 1 + 0.412z^(-1) + .309z^(-2)
#y(t) = v(t-2) + .412v(t-3) + .309v(t-4)

#v_t_ex1 = np.random.normal(0,1,n_samples+4)
#data_ex1 = v_t_ex1[2:-2] + .412*v_t_ex1[1:-3] + .309*v_t_ex1[0:-4]
#
#k = 0
#q = 4
#p = 10
#data = data_ex1
#[v_t_1_ex1, theta_ex1] = la.arma_ls(k,q,p,data)
#
##Example 2
##y(t) = D(z)v(t)
##y(t) = v(t) + 1.32v(t-1) + 1v(t-2)
#
#v_t_ex2 = np.random.normal(0,1,n_samples+2)
#data_ex2 = v_t_ex2[2:] + 1.32*v_t_ex2[1:-1] + v_t_ex2[0:-2]
#
#k = 0
#q = 2
#p = 6
#data = data_ex2
#[v_t_1_ex2, theta_ex2] = la.arma_ls(k,q,p,data)
#
##Example 3
##A(z)y(t)=D(z)v(t)
##A(z)=1-1.6z^(-1)+.8z^(-2)
##D(z)=1+.412z^(-1)+.309z^(-2)
##y(t) = 1.6*y(t-1)-.8y(t-2)+v(t)+.412v(t-1)+.309v(t-2)
#v_t_ex3 = np.random.normal(0,1,n_samples+2)
#data_ex3 = np.zeros(n_samples)
#for i in range(2,data_ex3.shape[0]):
#    data_ex3[i] = 1.6*data_ex3[i-1] - .8*data_ex3[i-2] + v_t_ex3[i] + .412*v_t_ex3[i-1] + .309*v_t_ex3[i-2]
    
    
#Example 4
#v_t_ex4 = np.random.normal(0,1,n_samples)
#data_ex4 = np.zeros(n_samples)
#data_ex4[0] = 2
#for i in range(1,data_ex4.shape[0]):
#    data_ex4[i] = 0.9*data_ex4[i-1] + v_t_ex4[i]
#
#k = 1
#q = 0
#p = 4
#data = data_ex4
#[v_t_1_ex4_ls, theta_ex4_ls] = la.arma_ls(k,q,p,data)
#[v_t_1_ex4_rls, theta_ex4_rls] = la.arma_rls(k,q,p,data)
#[v_t_1_ex4_rsg, theta_ex4_rsg] = la.arma_rsg(k,q,p,data)

#Example 5
v_t_ex5 = np.random.normal(0,1,n_samples)
data_ex5 = np.zeros(n_samples)
data_ex5[0] = 2
for i in range(1,data_ex5.shape[0]):
    data_ex5[i] = 0.9*data_ex5[i-1] + v_t_ex5[i] + .6*v_t_ex5[i-1]

k = 1
q = 1
p = 10
data = data_ex5
[v_t_1_ex5_ls, theta_ex5_ls] = la.arma_ls(k,q,p,data)
[v_t_1_ex5_rls, theta_ex5_rls] = la.arma_rls(k,q,p,data)
[v_t_1_ex5_rsg, theta_ex5_rsg] = la.arma_rsg(k,q,p,data)