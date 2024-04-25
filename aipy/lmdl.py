import os,sys, shutil, random, copy
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn

class LNet(nn.Module):
  '''

  1 Linear Layer, with Quadratic Optimization
  
  Example:
    >>> n = 2
    >>> mdl = QSSLNet(in_dim=n,out_dim=n)
    >>> x = torch.ones((n,1))/n # c[t-1]
    >>> y = mdl(x)
    >>> c = mdl.csoftmax(y) # c[t]
    >>> print(y)
    >>> print(p)
    >>> print(mdl)
  '''
  def __init__(self, dim=2, eps=1E-15):
    super().__init__()
    self.dim = dim
    ones_vec = torch.ones((dim,1), dtype=torch.float)

    # sytem's data matrix and scaling matrix
    self.register_buffer("A",torch.empty((dim,dim), dtype=torch.float))
    self.register_buffer("M",torch.eye(dim, dtype=torch.float))
    self.register_buffer("eps",torch.tensor(eps,dtype=torch.float))
    
    # quadratic cost parameters A, b
    # sum to one constraint
    self.register_buffer("ones_vec",ones_vec)
    self.register_buffer("zeros_vec",torch.zeros((dim,1), dtype=torch.float))
    self.register_buffer("b",ones_vec)
    
    # model's parameters:
    # linear layer
    self.weight = nn.Parameter(torch.empty((dim, 1)))
    self.learner = None
    
    
    self.last_cost = torch.tensor(torch.inf, dtype=torch.float)    
        
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters 
        # & if it's callable called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters): m.reset_parameters()

    # Applies fn recursively to every submodule
    self.apply(fn=weight_reset)
    
  def set_learner(self, learner_obj):
    # attach optimizer to weight.
    self.learner = learner_obj
        
  def reset_parameters(self) -> None:
      """
      Initialize parameters randomly
      """
      for p in self.parameters():
          # nn.init.normal_(p, std=self.eps)
          nn.init.constant_(p, 1/self.dim)

  
  @torch.no_grad()
  def data_matrix(self, A, use_corr=True) -> None:
    '''
    __init__ _summary_
    Args:
        A (Tensor): system's data matrix, symmetric covariance 
        use_corr (bool): transform covariance to correlation
    '''
    # when use_corr is True: preconditioned system 
    # normalizes/transform (cov. to corr. form)
    
    # expects all entries in A to be positively correlated.
    if A.shape[0] != A.shape[1]:
      raise Exception(f"Expected data matrix should be {self.out_dim} x {self.out_dim}")
    
    b = self.ones_vec
        
    # if abs(min(A[A<0].tolist())) < 0.1:
      
    dd = 1/(torch.diag(A).sqrt())
    # skip normalizing with diagonal, if diag elements are ~ 1
    if (1 - (dd).mean()).abs() < 0.1: use_corr = False
    # print('.looks like matrix diagonal is already scaled.')
    
    if use_corr:
      self.M = torch.diag(dd)
        
    self.A = (self.M.mm(A.mm(self.M)))
    self.b = (self.M.mm(b))
      
      
  def quadcost(self, w):
    '''
    Quadratic cost function
    '''
    # f = 0.5*y'Ay - b'y
    return ((0.5*(w.T.mm(self.A.mm(w)))).sub(((self.b.T.mm(w)))))
  
  def uncquadcost(self, w):
    '''
    Quadratic cost function
    '''
    # f = 0.5*y'Ay - b'y + 1
    return ((0.5*(w.T.mm(self.A.mm(w)))).sub(((self.b.T.mm(w)))).add(1))

    
  @torch.no_grad()
  def delta_cost(self, f):
      ''' Abs. fractional change in cost '''
      delta = torch.abs( (f+self.eps-self.last_cost)/(f+self.eps) )
      self.last_cost = 1*f
      return delta
    
  