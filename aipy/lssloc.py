import os,sys, shutil, random, copy
from pathlib import Path

import math
# import numpy as np

import torch
import torch.nn as nn

class Linear(nn.Module):
  def __init__(self, weight) -> None:
     super().__init__()
     
     self.weight = weight
     
  def forward(self, input):
    ''' Forward Pass
    '''
    # y = Wx
    return self.weight.mm(input)

class QSSLNet(nn.Module):
  '''
  SSLNet Self Supervised Neural Network repreentation of a Quadratic Cost.

  1 Layer, with Quadratic Optimization
  
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
  def __init__(self, in_dim=2, out_dim=2, eps=1E-15):
    super().__init__()
    self.in_dim = in_dim
    self.out_dim = out_dim
    
    # input: previous belief
    self.register_buffer("x",torch.ones((self.in_dim,1), dtype=torch.float32).div(in_dim))
    
    # quadratic cost parameters A, b
    # sum to one constraint
    ones_vec = torch.ones((out_dim,1), dtype=torch.float32)
    self.register_buffer("ones_vec",ones_vec)
    self.register_buffer("zeros_vec",0*ones_vec)
    self.register_buffer("b",ones_vec)
    
    # sytem's data matrix and scaling matrix
    self.register_buffer("A",torch.empty((out_dim,out_dim), dtype=torch.float32))
    self.register_buffer("M",torch.eye(out_dim, dtype=torch.float32))
    
    # model's parameters:
    # linear layer
    self.weight_W = nn.Parameter(torch.empty((in_dim, out_dim)))
    self.Lin_W = Linear(self.weight_W)
    self.learnerW = None
    
    self.register_buffer("eps",torch.tensor(eps,dtype=torch.float32))
    
    self.last_cost = torch.tensor(torch.inf, dtype=torch.float32)
    self.last_cost_noc = torch.tensor(torch.inf, dtype=torch.float32)
    
        
    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters 
        # & if it's callable called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters): m.reset_parameters()

    # Applies fn recursively to every submodule
    self.apply(fn=weight_reset)
    
  def set_learner(self, learnerW):
    # attach optimizer to weight.
    self.learnerW = learnerW
        
  def reset_parameters(self) -> None:
      """
      Initialize parameters randomly
      """
      for p in self.parameters():
          nn.init.normal_(p, std=self.eps)
    
  def __repr__(self):
    return f"prior belief vector:={self.x.transpose(0,1)},\nstate weight matrix:={self.weight_W}"
  
  
  def act(self, xinp=None):
    ''' Forward Pass
    '''
    if xinp is not None:
      self.x = 1*xinp # +/- c_{t-1} works
    y =  self.Lin_W(self.x)  
    return y
  # def forward(self, inp=None):
  #   ''' Forward Pass
  #   '''
  #   if inp is not None:
  #     self.x = 1*inp
      
  #   return self.Lin_W(self.x)
  
  
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
    
    b = self.ones_vec+0
    
    # fix for ~0 value, any semidefiniteness in matrix
    # and small negative entries.
    if A.abs().min() < 1e-3:
      # A.add_(1e-1*torch.eye(self.in_dim)) 
      # A.add_(1e-4)
      if abs(min(A[A<0].tolist())) < 0.1:
        A.abs_()
    
    dd = 1/(torch.diag(A).sqrt())
    # skip normalizing with diagonal, if diag elements are ~ 1
    if (1 - (dd).mean()).abs() < 0.1: #todo
      use_corr = False
      print('.looks like matrix diagonal is already scaled.')
    if use_corr:
      self.M = torch.diag(dd)
        
    self.A = (self.M.mm(A.mm(self.M)))
    self.b = (self.M.mm(b))
      
      
  def quadcost(self, y):
    '''
    Quadratic cost function
    '''
    # f = 0.5*y'Ay - b'y  + 1
    return (0.5*y.T.mm(self.A.mm(y))).sub(((self.b.T.mm(y))-1))
  
  # def quadcost_noc(self, y):
  #   '''
  #   Quadratic cost function
  #   '''
  #   # f = 0.5*y'Ay
  #   return (0.5*y.T.mm(self.A.mm(y)))
    
  @torch.no_grad()
  def delta_cost(self, f):
      ''' Abs. fractional change in cost '''
      delta = torch.abs( (f+self.eps-self.last_cost)/(f+self.eps) )
      self.last_cost = 1*f
      return delta
    
  
  @torch.no_grad()
  def csoftmax(self, inp):
    ''' The output head of the network.
    - transforms to the original co-ordinate space with matrix M.
    - computes output belief vector at iteration t, c_t
    '''
    return (self.M.mm(inp)).softmax(dim=0)

    

      

    


  # def sparsing(self, sparsity, std=0.01):
  #     r"""Fills the 2D input `Tensor` as a sparse matrix, where the
  #     non-zero elements will be drawn from the normal distribution
  #     :math:`\mathcal{N}(0, 0.01)`, as described in `Deep learning via
  #     Hessian-free optimization` - Martens, J. (2010).

  #     Args:
  #         tensor: an n-dimensional `torch.Tensor`
  #         sparsity: The fraction of elements in each column to be set to zero
  #         std: the standard deviation of the normal distribution used to generate
  #             the non-zero values

  #     Examples:
  #         >>> w = torch.empty(3, 5)
  #         >>> nn.init.sparse_(w, sparsity=0.1)
  #     """
  #     if self.weight.ndimension() != 2:
  #         raise ValueError("Only tensors with 2 dimensions are supported")

  #     rows, cols = self.weight.shape
  #     num_zeros = int(math.ceil(sparsity * rows))

  #     with torch.no_grad():
  #         # self.weight.normal_(0, std)
  #         for col_idx in range(cols):
  #             row_indices = torch.randperm(rows)
  #             zero_indices = row_indices[:num_zeros]
  #             self.weight[zero_indices, col_idx] = 0



  #   # return (y).softmax(dim=0)
  #   # '''
  #   # Observation:
  #   # floating-point issues 
  #   # could lead to slightly different answers among the smaller values of out.
  #   # '''
  #   # out = inp/inp.sum()
  #   # out = self.softmax_vanilla(inp)
  #   # out = self.softmax_addone(inp)
  #   # out = self.softmax_def(inp) # closest
  #   # out = self.softmax_addmax(inp)
  #   # outr = torch.softmax(inp,dim=0)
  #   # print(torch.allclose(outr,out))
  #   # print(torch.sum(torch.abs(outr-out)))
    
    
  
  
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