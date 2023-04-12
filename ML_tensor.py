#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module with functions for machine learning with tensor networks.
In future releases the various functions will be combined and organised in 
a few classes. Functions are based on [1] https://arxiv.org/abs/2006.02516

@author: lcoopmans
"""

import jax
jax.config.update('jax_enable_x64', True) # enable 64-bit precision
import jax.numpy as np
from jax import grad, random, jit

import tensornetwork as tn
tn.set_default_backend("jax")     # different backends are possible, see the tensornetwork documentation guide

##############################################################################
### Data preparation functions for the feature map ###
##############################################################################

def one_label_subset(x, y, label):
    """ 
        Params: trainingset x (numpy array), traininglabels y (numpy vector)
               and the label for the subset.
        Return: the set x with only 1 specific label.
    """
    return x[np.where(y==label)]

@jit                              # the jit decorator from jax helps speeding up function evualtions 
def Maxpool(image, K=2, L=2):
    """ 
        Applies a max pooling layer with kernel (KxL)
        to an input image (2d numpy array). For this implementation the 
        kernel dimensions K (int) and L (int) need to be set to 2.
        This can be later generalized to more generic pooling layers.
        Return: new_image (2d numpy array)
    """
    M, N = image.shape
    MK = M // K         # compute the number of kernel shifts
    NL = N // L
    new_image = image[:MK*K, :NL*L].reshape(MK, K, NL, L).max(axis=(1, 3))
    return new_image

def feature_map(x_data, Pooling=False):
    """ 
        Params: x_data (numpy array), Pooling (boolean) 
        Return: feature vectors after the feature map as defined in ref [1]
        and possible max pooling.
    """
    
    # apply max pooling
    if Pooling:
        x = []
        for i in range(np.shape(x_data)[0]):
            x.append(Maxpool(x_data[i])) 
        
        x_data = np.array(x)

    # reshape the pooled data into vectors
    x_data = x_data.reshape(np.shape(x_data)[0],
                            np.shape(x_data)[1]*np.shape(x_data)[2]) 
        
    # apply the feature map
    fvecs = np.stack((np.cos(np.pi/2*x_data),np.sin(np.pi/2*x_data)),axis=1)
    return fvecs

##############################################################################
### Tensor network and contraction functions for the loss and decision     ###
##############################################################################

def rand_anomdet_MPO(N, d, b, p, S, mean, std):
    """ 
        Function to make a random anomaly detection MPO for which the 
        topology of the network is given in figure 3 of ref [1].
        
        Params: N (int) number of tensors in the matrix product operator
                d (int) dimension of the local features in the feature vector
                b (int) bond dimension between neighbouring tensors
                p (int) MPO output leg dimension
                S (int) number of out output legs -2 (for the boundaries)
                mean (float64) mean for the random distribution
                std (float64) standard deviation for the random distribution
                
        Return: list of random tensors (numpy arrays) making up the MPO
    """
    
    MPO = [] 
    
    for i in range(N):       
        key = random.PRNGKey(i)
        
        if i == 0:
            MPO.append(mean + std*random.normal(key, shape=(d,b,p,1))) # adds left boundary tensor
        elif i == (N-1):                                              
            MPO.append(mean + std*random.normal(key, shape=(d,p,b))) # adds right boundary tensor
        elif i%S == 0: 
            MPO.append(mean + std*random.normal(key, shape=(d,b,p,b))) # adds a node to the list with 4 bonds
        else:
            MPO.append(mean + std*random.normal(key, shape=(d,b,b))) # adds a node to the list with 3 bonds          
    return MPO

@jit
def apply_MPO_to_fvec(fvec, MPO):
    """ 
        Function that applies our tensor network operator to a feature vector.
        The connected bonds are contracted with the tensor network function 
        ncon.
        
        Params: fvec (numpy array) one single feature vector
               MPO (list of numpy arrays) matrix product operator
               
        Return: new contracted MPS (list of numpy arrays) with dimensions of 
                the form of figure 6b in [1].
                
        TO DO: Include check that dimensions should line up 
    """
    MPS = []  
               
    for i in range(len(MPO)): # we loop over all the tensors and contract them with ncon
        
        if len(np.shape(MPO[i])) == 3:
            MPS.append(tn.ncon([fvec[:,i], MPO[i]], [[1],[1,-1,-2]]))
        else: 
            MPS.append(tn.ncon([fvec[:,i], MPO[i]], [[1],[1,-1,-2,-3]]))
    return MPS


@jit
def reduce_MPS(MPS):
    """ 
        Reduces the size of the MPS by contracting every horizontal 
        (.1,...S-1) set of bonds (the tensors without any outgoing leg)
        as in Fig. 6c.
        
        Params: MPS (list of numpy arrays) matrix product state
        
        Return: Red_MPS (list of numpy arrays) size reduced matrix product 
                state in which each tensor has an outgoing leg.
                
        TO DO: include right boundary tensor in bulk loop
    
    """
    
    Red_MPS = []
    C = MPS[0]   # we start contracting from the left most tensor
    
    # loop over the bulk tensors
    for i in range(1,len(MPS)-1):
        
        # if the tensor has no outgoing leg we contract it
        if len(np.shape(MPS[i])) == 2: 
            C = tn.ncon([C,MPS[i]], [[1,-2,-3],[-1,1]]) 
        else:
            Red_MPS.append(C)
            C = MPS[i]
    Red_MPS.append(C) # C is still part of the bulk    
    
    # add right boundary tensor
    Red_MPS.append(MPS[len(MPS)-1])  
    
    return Red_MPS

@jit
def L2norm_MPS(MPS):
    """ Fully contracts the squared  F 2 norm of an MPS
        Uses zig-zag contraction starting from the left as in figure 7 of [1]
        
        Params: MPS (list of numpy arrays) matrix product state
        
        Return: P (float64) F 2 norm value
    """
    
    # start contracting from the left boundary
    P = tn.ncon([MPS[0],MPS[0]], [[-1,1,2],[-2,1,2]])
    
    # contract the bulk tensors with zig-zag
    for i in range(1,len(MPS)-1):
        P = tn.ncon([P, MPS[i]], [[1, -3],[-1,-2,1]])
        P = tn.ncon([P, MPS[i]],[[-1,1,2],[-2,1,2]])
    
    # finally contract the right boundary tensor
    P = tn.ncon([P, MPS[len(MPS)-1]],[[1,-2],[-1,1]])
    P = tn.ncon([P, MPS[len(MPS)-1]],[[1,2],[1,2]])   
      
    return P


def decision_fun(eps, MPO, fvec):
    """
        Applies MPO to MPS and computes the anomaly decision value.
        
        Params: eps (float64) decision radius
                MPO (list of numpy arrays) tensors defining matrix product 
                                           operator
                fvec (numpy 1d array) feature vector
                
        Return: int 0 anomaly int 1 normal instance
    """
    
    # compute the F 2 normal of MPO applied to the feature vector
    MPS = apply_MPO_to_fvec(fvec, MPO)
    red_MPS = reduce_MPS(MPS)
    dec_value = L2norm_MPS(red_MPS)
    
    if dec_value > eps: 
        return 1 
    
    else:
        return 0

def anomaly_detection(MPO, fvecs):
    """
        computes the predictions of the MPO applied to the feature vectors.
        
        Params: MPO (list of numpy arrays) matrix product operator
                fvecs (numpy array) feature vectors
        
        Return: numpy array of the prediction values
    """
    y_predict = []
    
    for i in range(len(fvecs)):
        y_predict.append(decision_fun(0.5, MPO, fvecs[i]))
        
    return np.array(y_predict)

@jit
def loss_function(MPO, fvec):
    """ 
        Computes the loss of a single training instance (without penalty).
        
        Params: MPO (list of numpy arrays) matrix product operator
                fvec (numpy array) feature vector for which to compute the loss 
                
        Return: L (float64) value of the loss
    """
    
    mps = apply_MPO_to_fvec(fvec, MPO) 
    red_mps = reduce_MPS(mps)
    L = (np.log(L2norm_MPS(red_mps))-1)**2
    
    return L

# def the gradient of the loss function
loss_grad = jit(grad(loss_function))

@jit
def L2Norm_MPO(MPO):
    """ 
        Fully contracts the squared  F 2 norm of an MPO
        Uses zig-zag contraction starting from the left as in figure 7 of [1]
        Note: now we have some double bonds compared to the MPS contractor.
        
        Params: MPO (list of numpy arrays) matrix product operator
        
        Return: P (float64) F 2 norm value
    """
    
    # start contracting from the left boundary 
    P = tn.ncon([MPO[0],MPO[0]], [[1,-1,1,2],[1,-2,1,2]]) 
    
    # contract the bulk tensors with zig-zag
    for i in range(1,len(MPO)-1):
        if len(np.shape(MPO[i])) == 4:
            P = tn.ncon([P, MPO[i]], [[1, -4],[-2,-1,-3, 1]])
            P = tn.ncon([P, MPO[i]], [[-1,1,2,3],[1,-2,2,3]])
        else:
            P = tn.ncon([P, MPO[i]], [[1, -3],[-2,-1,1]])
            P = tn.ncon([P, MPO[i]],[[-1,1,2],[1,-2,2]])
    
    # finally contract the right boundary tensor
    P = tn.ncon([P, MPO[len(MPO)-1]],[[1,-3],[-1,-2,1]])
    P = tn.ncon([P, MPO[len(MPO)-1]],[[1,2,3],[1,2,3]]) 
    
    return P

@jit
def penalty(MPO):
    """
        Computes the penalty for a trivial MPO solution. 
        
        Params: MPO (list of numpy arrays) matrix product operator
        
        Return: value of ReLU[log(F2 norm MPO)] (float64)
    """
    return np.max(np.array([0.0, np.log(L2Norm_MPO(MPO))]))


# define the gradient of the penalty function
penalty_grad = jit(grad(penalty))

@jit
def sum_grads(gradient_a, gradient_b, batch_size):
    """
        function to average two gradients of the MPO in the batch
        
        Params: gradient_a (list of numpy arrays) 
                gradient_b (list of numpy arrays)
                batch_size (float64) 
                
        Return: newly averaged gradient_a (list of numpy arrays)
    """
    for j in range(len(gradient_a)):
        gradient_a[j] = gradient_a[j] + gradient_b[j]/batch_size
    return gradient_a


def batch_loss_and_gradient(MPO, fvecs, alpha):
    """ 
        Computes the loss and gradient for a batch of feature vectors.
        
        Params: MPO (list of numpy arrays) matrix product operator
                fvecs (numpy array) batch of feature vectors
                alpha (float64) hyperparameter for importance of penalty
                
        Return: batch averaged loss (float64), penalty (float64), 
                batch averaged gradient
    
        TO DO: Can easily be parallelized, perhaps with jax's vmap function
       
    """
    Loss = 0 
    Gradient = 0
    
    # run over the feature vectors and average the losses and gradients
    for i in range(len(fvecs)):
        fvec = fvecs[i]
        Loss += loss_function(MPO, fvec)
        
        if i != 0:
            Gradient  = sum_grads(Gradient, loss_grad(MPO,fvec), len(fvecs))
        else:
            Gradient = loss_grad(MPO,fvec)
    
    return Loss/len(fvecs), penalty(MPO), sum_grads(Gradient, penalty_grad(MPO),1/alpha)