# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 06:57:35 2021

@author: Francisco
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

np.random.seed(100)

def square_loss_arma_ons(x_tilde,x):
    return (x - x_tilde)**2

def squared_loss_gradient_arma_ons(x_tilde,x,gamma):
    return -2 * (x - x_tilde) * gamma
    
def sherman_morrison_inv(A,u,v):
    numerator = np.matmul(np.matmul(A,np.matmul(u,v.T)),A)
    denominator = 1 + np.matmul(np.matmul(v.T,A),u)
    return A - (numerator/denominator)

def projection_K(gamma,c,A):
    
    def distance(x):
        return np.matmul(np.matmul(gamma-x,A),gamma-x)
    
    bounds = ((-c,c),)*gamma.shape[0]
    x0 = np.random.rand(gamma.shape[0]) * c
    x = minimize(distance,x0,bounds = bounds)
    return x

def Phi_t(t,k,q,p,y,v_t_1):
    """
    This function is used in the models ARMA-LS, ARMA-RLS and ARMA-RSG
    Input:
        t: time
        theta_t: theta at time t in order to compute error,
        k: AR order
        q: MA order
        y: time series
        v_t_1: current computations of error    
    """

    n = k + q
    Phi_mat = np.zeros((n,p))
    j = p-1
    for i in range(t-p,t+1):
        y_part = np.flip(y[np.maximum(i-k,0):i])
        y_part = (-1)*np.concatenate((y_part,np.zeros(k-y_part.shape[0])))
        v_part = np.flip(v_t_1[np.maximum(i-q,0):i])
        v_part = np.concatenate((v_part,np.zeros(q-v_part.shape[0])))
        
        Phi_mat[:,j] = np.concatenate((y_part,v_part))
        j = j - 1
    return Phi_mat

def arma_ls(k,q,p,data):
    """
    k: order of AR
    q: order MA
    p: selected length for iterative estimation
    
    """
    v_t_1 = np.zeros(1)
    v_t_1[0] = data[0]
    theta = np.zeros((data.shape[0],k+q))

    theta[0,:] = (0.000001) * np.ones(k+q)

    for t in range(1,data.shape[0]):
        y = data[0:(t+1)]
        Y = np.flip(y[np.maximum(t+1-p,0):(t+1)])
        Y = np.concatenate((Y,np.zeros(np.maximum(p-Y.shape[0],0))))
        Phi_t_mat = Phi_t(t,k,q,p,y,v_t_1)
        
        if t >= p:
            theta[t,:] = np.matmul(np.linalg.inv(np.matmul(Phi_t_mat,Phi_t_mat.T)),np.matmul(Phi_t_mat,Y))
        else:
            theta[t,:] = (0.000001) * np.ones(k+q)
            
        v_t = y[t] - np.matmul(Phi_t_mat[:,0],theta[t,:])
        v_t_1 = np.concatenate((v_t_1,v_t))
    return [v_t_1, theta]

def arma_rls(k,q,p,data):
    """
    k: order of AR
    q: order MA
    p: selected length for iterative estimation
    
    """
    v_t_1 = np.zeros(1)
    v_t_1[0] = data[0]
    theta = np.zeros((data.shape[0],k+q))
    theta[0,:] = (0.000001) * np.ones(k+q)
    P_t_inv = (0.000001)*np.eye(k+q)

    for t in range(1,data.shape[0]):
        y = data[0:(t+1)]
        Y = np.flip(y[np.maximum(t+1-p,0):(t+1)])
        Y = np.concatenate((Y,np.zeros(np.maximum(p-Y.shape[0],0))))
        Phi_t_mat = Phi_t(t,k,q,p,y,v_t_1)
        
        theta[t,:] = theta[t-1,:] + np.matmul(np.matmul(P_t_inv,Phi_t_mat),Y - np.matmul(Phi_t_mat.T,theta[t-1,:]))
        P_t_inv = P_t_inv + np.matmul(Phi_t_mat,Phi_t_mat.T)
            
        v_t = y[t] - np.matmul(Phi_t_mat[:,0],theta[t,:])
        v_t_1 = np.concatenate((v_t_1,v_t))
    return [v_t_1, theta]

    
def arma_rsg(k,q,p,data):
    """
    inputs:
    k: AR order
    q: MA order
    p: data length
    """
    v_t_1 = np.zeros(1)
    v_t_1[0] = data[0]
    theta = np.zeros((data.shape[0],k+q))
    r = np.ones(data.shape[0])
    for t in range(1,data.shape[0]):
        y = data[0:(t+1)]
        Y = np.flip(y[np.maximum(t+1-p,0):(t+1)])
        Y = np.concatenate((Y,np.zeros(np.maximum(p-Y.shape[0],0))))
        Phi_t_mat = Phi_t(t,k,q,p,y,v_t_1)
        r[t] = r[t-1] + np.trace(np.matmul(Phi_t_mat,Phi_t_mat.T))
        theta[t,:] = theta[t-1,:] + (1/r[t]) * np.matmul(Phi_t_mat, Y - np.matmul(Phi_t_mat,theta[t-1,:]))
        v_t = y[t] - np.matmul(Phi_t_mat[:,0],theta[t,:])
        v_t_1 = np.concatenate((v_t_1,v_t))
    return [v_t_1, theta]

def arma_ons(k, q, eta, L, M_max, epsilon, loss, gradient, data):
    """
    inputs
    k: AR order
    q: MA order
    eta:
    L: Lipschitz continuity constant 
    M_max: absolute value of time series upper bound 
    epsilon: 1-epsilon upper bound of the sum of absolute value MA coefficients
    loss: loss function
    gradient: gradient of the loss function
    data: time series
    """
    T = data.shape[0]
    m = q * (np.log(1/(T*L*M_max)) / np.log(1-epsilon))
    A_t = np.random.rand(m+k , m+k)
    gamma_matrix = np.zeros((T,m + k))
    for t in range(m+k,T):
        x_data = data[t-(m+k):t]
        x_tilde = np.matmul(x_data,gamma_matrix[t,:])
        observed_loss = loss(x_tilde,x_data[t])
        observed_gradient = gradient(x_tilde,x_data[t],gamma_matrix[t,:])
        A_t = A_t + np.matmul(observed_gradient.reshape(-1,1),observed_gradient.reshape(-1,1).T)
        
        if t == m+k:
            A_inv_t = np.linalg.inv(A_t)
        else:
            A_inv_t = sherman_morrison_inv(A_t,observed_gradient.reshape(-1,1),observed_gradient.reshape(-1,1))
        
        un_projected_gamma = (gamma_matrix[t,:].reshape(-1,1) - (1/eta)*np.matmul(A_inv_t,observed_gradient.reshape(-1,1))).flatten()
        gamma_matrix[t+1,:] = projection_K(un_projected_gamma)
    return gamma_matrix