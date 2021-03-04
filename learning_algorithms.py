# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 06:57:35 2021

@author: Francisco
"""

import numpy as np
import pandas as pd

def square_loss_arma_ons(x_tilde,x):
    return (x - x_tilde)**2

def squared_loss_gradient_arma_ons(x_tilde,x,gamma):
    return -2 * (x - x_tilde) * gamma
    
def sherman_morrison_inv(A,u,v):
    numerator = np.matmul(np.matmul(A,np.matmul(u,v.T)),A)
    denominator = 1 + np.matmul(np.matmul(v.T,A),u)
    return A - (numerator/denominator)

def projection_K(gamma):
    

def arma_ons(k, q, eta, L, M_max, epsilon, loss, gradient, data):
    T = data.shape[0]
    m = q * (np.log(1/(T*L*M_max)) / np.log(1-epsilon))
    A_t = np.random.rand(m+k , m+k)
    gamma_matrix = np.zeros((T,m + k))
    for t in range(m+k,T):
        x_data = data['Close'].iloc[t-(m+k):t].values
        x_tilde = np.matmul(x_data,gamma_matrix[t,:])
        observed_loss = loss(x_tilde,x_data.iloc[t])
        observed_gradient = gradient(x_tilde,x,gamma_matrix[t,:])
        A_t = A_t + np.matmul(observed_gradient.reshape(-1,1),observed_gradient.reshape(-1,1).T)
        
        if t == m+k:
            A_inv_t = np.linalg.inv(A_t)
        else:
            A_inv_t = sherman_morrison_inv(A_t,observed_gradient.reshape(-1,1),observed_gradient.reshape(-1,1))
        
        gamma_matrix[t+1,:] = projection_K(gamma_matrix[t,:].reshape(-1,1) - (1/eta)*np.matmul(A_inv_t,observed_gradient.reshape(-1,1)))
        