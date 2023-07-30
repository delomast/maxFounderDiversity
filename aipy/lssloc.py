import os,sys, shutil, random, copy
from pathlib import Path

import math
# import numpy as np

import torch
import torch.nn as nn


from aipy.asgmlin import AutoSGMLin

class QuadObj(nn.Module):
  '''
  QuadObj Quadratic Objective Function
  '''
  
  def __init__(self) -> None:
    super().__init__()
    
  def forward(self, y, A, b):

    # f = 0.5*y'Ay - b'y
    f = (0.5*y.T.mm(A.mm(y))).sub(b.T.mm(y))
    # f = (0.5*torch.mm(y.T,torch.mm(A,y))).sub(torch.mm(b.T,y))
  
    return f


class LSSLNet(nn.Module):
  '''
  LSSLNet Linear Self Supervised Neural Network

  One Linear Layer, with Quadratic Optimization
  
  Example:
    >>> n = 2
    >>> mdl = LSSLNet(in_dim=n,out_dim=n)
    >>> x = torch.ones((n,1))/n # p[t-1]
    >>> y = mdl(x)
    >>> p = mdl.softmax(y) # p[t]
    >>> print(y)
    >>> print(p)
    >>> print(mdl)
  '''
  def __init__(self, in_dim=2, out_dim=2):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    
    # input: previous belief
    x = torch.ones((self.in_dim,1), dtype=torch.float32).div(in_dim)
    self.register_buffer("x",x)
    
    # objective fcn's parameters A, b
    # sum to one constraint
    ones_vec = torch.ones((out_dim,1), dtype=torch.float32)
    self.register_buffer("ones_vec",ones_vec)
    self.register_buffer("b",ones_vec)
    
    # sytem's corr. matrix
    A = torch.empty((out_dim,out_dim), dtype=torch.float32)
    self.register_buffer("A",A)
    Di = torch.eye(out_dim, dtype=torch.float32)
    self.register_buffer("Di",Di)
    
    # model's parameters:
    # linear layer
    self.weight = nn.Parameter(torch.empty((in_dim, out_dim)))
    # self.weight2 = nn.Parameter(torch.empty((out_dim, out_dim)))
    self.learner = None
    self.criterion = QuadObj()
    
    eps = torch.tensor(1e-12,dtype=torch.float32)
    self.register_buffer("eps",eps)
    
    self.last_loss = torch.tensor(0, dtype=torch.float32)
    
        
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters 
        # & if it's callable called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters): m.reset_parameters()

    # Applies fn recursively to every submodule
    self.apply(fn=weight_reset)
    
  def set_learner(self, learner):
    # attach optimizer to weight.
    self.learner = learner
    
  def reset_parameters(self) -> None:
      """
      Initialize parameters randomly
      """
      for p in self.parameters():
          # normal + sparsity
          nn.init.normal_(p, std=1e-16)
          # nn.init.zeros_(p)
          # nn.init.sparse_(p,sparsity=0.1,std=1e-16)
    
  def __repr__(self):
    return f"input belief vector:={self.x.transpose(0,1)},\nweight matrix:={self.weight}"

  def forward(self, input=None):
    ''' Forward Pass
    '''
    # y = Wx
    if input is not None:
      self.x = (input)
      
    # y = self.weight.mm(self.x)
    
    return self.weight.mm(self.x)

  @torch.no_grad()
  def fcloss(self, f):
      ''' abs. fractional change in loss '''
      fcloss = torch.abs(1 - (self.last_loss/(f+(self.eps))))
      self.last_loss = 1*f
      return fcloss
  
  @torch.no_grad()
  def csoftmax(self, y):
    ''' The output head of the network.
    
    Computes output belief vector at iteration t, p[t]
    '''
    # inp = self.Di.mm(y)
    # out = inp.softmax(dim=0)
    '''
    Observation:
    Floating-point numerical issues 
    could lead to slightly different answers among the smaller values of out.
    
    '''
    # out = inp/inp.sum()
    # out = self.softmax_vanilla(inp)
    # out = self.softmax_addone(inp)
    # out = self.softmax_def(inp) # closest
    # out = self.softmax_addmax(inp)
    # outr = torch.softmax(inp,dim=0)
    # print(torch.allclose(outr,out))
    # print(torch.sum(torch.abs(outr-out)))
    
    return (self.Di.mm(y)).softmax(dim=0)
  
  
  # def softmax_vanilla(self, inp):
  #     num = (inp).exp()
  #     den = num.sum()
  #     outp = num.div(den)
  #     return outp
      
  # def softmax_addmax(self, inp):
  #     m = torch.amax(inp, keepdim=True)
  #     num = (inp-m).exp()
  #     den = num.sum().add((-m).exp())
  #     outp = num.div(den)
  #     return outp
    
  # def softmax_addone(self, inp):
  #     num = (inp).exp()
  #     den = num.sum().add(1)
  #     outp = num.div(den)
  #     return outp
    
  # def softmax_def(self, inp):
  #     m = torch.amax(inp, keepdim=True)
  #     num = (inp-m).exp()
  #     den = num.sum()
  #     outp = num.div(den)
  #     return outp
    
  @torch.no_grad()
  def data_matrix(self, A, use_corr=True) -> None:
    '''
    __init__ _summary_
    Args:
        A (Tensor): system's data matrix, symmetric covariance 
        use_corr (bool): transform covariance to correlation
    '''
    # normalize: transform (cov. to corr. form)
    # preconditioned system 
    
    
    if A.shape[0] != A.shape[1]:
      raise Exception(f"Expected data matrix should be {self.out_dim} x {self.out_dim}")
    
    if use_corr:
      Dinv = torch.diag(1/(torch.diag(A).sqrt()))
      self.Di = Dinv
      self.A = (Dinv.mm(A.mm(Dinv)))
      self.b = (Dinv.mm(self.ones_vec))
    


  def sparsing(self, sparsity, std=0.01):
      r"""Fills the 2D input `Tensor` as a sparse matrix, where the
      non-zero elements will be drawn from the normal distribution
      :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
      Hessian-free optimization` - Martens, J. (2010).

      Args:
          tensor: an n-dimensional `torch.Tensor`
          sparsity: The fraction of elements in each column to be set to zero
          std: the standard deviation of the normal distribution used to generate
              the non-zero values

      Examples:
          >>> w = torch.empty(3, 5)
          >>> nn.init.sparse_(w, sparsity=0.1)
      """
      if self.weight.ndimension() != 2:
          raise ValueError("Only tensors with 2 dimensions are supported")

      rows, cols = self.weight.shape
      num_zeros = int(math.ceil(sparsity * rows))

      with torch.no_grad():
          # self.weight.normal_(0, std)
          for col_idx in range(cols):
              row_indices = torch.randperm(rows)
              zero_indices = row_indices[:num_zeros]
              self.weight[zero_indices, col_idx] = 0
