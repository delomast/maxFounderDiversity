import os, sys, shutil, random, copy
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn


class ILinear(nn.Module):
    """ """

    def __init__(self, dim=2, eps=1E-15):
        super().__init__()
        self.dim = dim
        ones_vec = torch.ones((dim, 1), dtype=torch.float)

        # sytem's data matrix and scaling matrix
        self.register_buffer("A", torch.empty((dim, dim), dtype=torch.float))
        self.register_buffer("M", torch.eye(dim, dtype=torch.float))
        self.register_buffer("eps", torch.tensor(eps, dtype=torch.float))

        self.register_buffer("ones_vec", ones_vec)
        self.register_buffer("zeros_vec", torch.zeros((dim, 1), dtype=torch.float))
        self.register_buffer("b", ones_vec)

        self.param_u = nn.Parameter(torch.empty((dim, 1)))
        self.param_c = nn.Parameter(torch.empty((dim, 1)))
        self.register_buffer("lmda", torch.tensor(1, dtype=torch.float))

        self.last_cost_u = torch.tensor(torch.inf, dtype=torch.float)
        self.last_cost_c = torch.tensor(torch.inf, dtype=torch.float)

        @torch.no_grad()
        def weight_reset(m: nn.Module):
            # check if the current module has reset_parameters method,
            # if it's callable, call it.
            reset_parameters = getattr(m, "reset_parameters", None)
            if callable(reset_parameters):
                m.reset_parameters()

        # apply fn recursively to every submodule
        self.apply(fn=weight_reset)

    def reset_parameters(self) -> None:
        """
        Initialize parameters randomly
        """
        for p in self.parameters():
            # nn.init.normal_(p, std=self.eps)
            nn.init.constant_(p, 1/self.dim)

    def __repr__(self):
        return f"(states) param. (u):= {self.param_u.flatten()},\nparam. (c):= {self.param_c.flatten()}"

    def forward(self, inp=1):
        """Forward Pass"""
        return inp*self.param_u

    @torch.no_grad()
    def data_matrix(self, A, use_corr=True) -> None:
        """
        assign data matrix and vector to model

        Args:
            `A` (Tensor). a symmetric covariance data matrix.

            `use_corr` (bool, optional). transform covariance to correlation (default=True).
        """
        # when use_corr is True: preconditioned system
        # normalizes/transform (cov. to corr. form)

        # expects all entries in A to be positively correlated.
        self.check_pos_corr(A)

        # check diagonal elements of A matrix
        self.M = self.check_diag(A, use_corr)
        self.Ao = A
        self.A = self.M.mm(A.mm(self.M))
        self.b = self.M.mm(self.ones_vec)
    
    @torch.no_grad()
    def check_pos_corr(self, A):
        '''
        check if entries in `A` matrix are positively correlated.
        '''
        if A.shape[0] != A.shape[1]:
            raise Exception(
                f"Expected data matrix should be {self.out_dim} x {self.out_dim}"
            )
    @torch.no_grad()
    def check_diag(self, A, use_corr, epsln=0.1):
        """
        check if the `A` matrix diagonal is already scaled,
        compute diagonal preconditioner
        """
        dd = 1 / (torch.diag(A).sqrt())
        # skip normalizing with diagonal, if diag elements are ~ 1
        if (1 - (dd).mean()).abs() < epsln:
            use_corr = False

        if use_corr:
            return torch.diag(dd)
        else:
            return self.M

    @torch.no_grad()
    def quad_cost_sum1(self, c=None):
        """
        quadratic cost (with sum to 1 constraint)
        """
        # f = 0.5*c'Ac - lmbda*(b'c  + 1)
        if c is None: c = self.param_c
        return (0.5*(c.T.mm(self.A.mm(c)))) - self.lmda*((self.b.T.mm(c)) - 1)
    
    @torch.no_grad()
    def quad_cost_sum1_tf(self, c=None):
        """
        quadratic cost (with sum to 1 constraint)
        """
        # f = 0.5*c'Ac - lmbda*(b'c  + 1)
        if c is None: c = self.tf(self.param_c)
        return (0.5*(c.T.mm(self.Ao.mm(c)))) - self.lmda*((self.ones_vec.T.mm(c)) - 1)

    def quad_cost_unc(self, u=None):
        """
        quadratic cost (unconstrained)
        """
        # f = 0.5*u'Au - b'u
        if u is None: u = self.param_u
        return (0.5*(u.T.mm(self.A.mm(u)))) - (self.b.T.mm(u)) + 1

    @torch.no_grad()
    def coan_metric(self, A=None, c=None):
        """
        quadratic objective
        """
        # f = 0.5*c'Ac
        if A is None: A = self.A
        if c is None: c = self.param_c
        return 0.5*(c.T.mm(A.mm(c)))
    
    @torch.no_grad()
    def coan_metric_tf(self, A=None, c=None):
        """
        quadratic objective
        """
        # f = 0.5*c'Ac
        if A is None: A = self.Ao
        if c is None: c = self.tf(self.param_c)
        return 0.5*(c.T.mm(A.mm(c)))

    @torch.no_grad()
    def delta_cost_u(self, cost_u):
        """absolute (fractional) change in cost"""
        delta, self.last_cost_u = self.df_cost(cost_u, self.last_cost_u)
        return delta
    
    @torch.no_grad()
    def delta_cost_c(self, cost_c):
        """absolute (fractional) change in cost"""
        delta, self.last_cost_c = self.df_cost(cost_c, self.last_cost_c)
        return delta

    def df_cost(self, f, last_f):
        delta = torch.abs((f + self.eps - last_f) / (f + self.eps))
        return delta, 1*f

    @torch.no_grad()
    def tf(self, inp=None):
        """The output head of the network.
        - transforms to the original co-ordinate space with matrix M.
        - computes parameter vector at iteration t, c_t
        """
        if inp is None:
            inp = self.param_c
        return (self.M.mm(inp)).relu()
