r"""AutoSGM"""


import math
import numpy as np

import torch
from torch.optim.optimizer import Optimizer, _use_grad_for_differentiable
from torch.optim import _functional as Fcn
from torch import Tensor
from typing import Any, List, Union, Optional


'''
LPF
'''
# General Unbiased First-order Low Pass Filter Structure 
# (Recursive Weighted Average)
# somefuno@oregonstate.edu

class LPF():

    def __init__(self, inplace:bool=True, cdevice=None):
        self.inplace = inplace
        self.cdevice = cdevice

    '''
    ***
    A General First-order Low Pass Filter Structure
    (Recursive Weighted Average Function)
    '''
    @torch.no_grad()
    def torch_ew_compute(self, in_k:Tensor, x:Tensor, 
                        beta:Tensor, step:Tensor=torch.ones((1,)), mode=1) -> tuple[Tensor, Tensor]:
        
        '''
        in_k: input at current time
        x: state at previous time
        beta: LPF pole at current time
        step: current discrete time
        mode: [default: mode=1] unbiased (all, except 2) | asympt. unbiased (2)

        out_k : output at current time, k
        x : updated state for next time, k+1
        '''
        # temps. (avoid repititions)
        k, betak = step, beta
        
        if not mode in [3, 30]:
            one_minus_betak = (1 - betak)
            betak_pow_k = betak.pow(k)
            one_minus_betak_pow_k = (1-betak_pow_k)

        if mode == 10:  # exponential. (stable if 0 \le \beta < 1) 
            # temps.
            betak_pow_kp1 = betak.pow(k+1)
            one_minus_betak_pow_kp1 = (1-betak_pow_kp1)

            # forward:
            ((x.mul_((betak - betak_pow_kp1))).add_(one_minus_betak*in_k)).div_(one_minus_betak_pow_kp1)
            out_k = 1*x
            
        elif mode == 11: # exponential. (stable if 0 \le \beta < 1)   
            
            # forward:
            ((x.mul_((betak - betak_pow_k))).add_(one_minus_betak*in_k)).div_(one_minus_betak_pow_k)
            out_k = 1*x
            
        elif mode == 1: # exponential. (stable if 0 \le \beta < 1)
        
            # forward:
            (x.mul_(betak)).add_(in_k)
            out_k = x*(one_minus_betak/one_minus_betak_pow_k)    
                        
        elif mode == 12: # exponential. (stable if 0 \le \beta < 1)
            betak_pow_kp1 = betak.pow(k+1)
            one_minus_betak_pow_kp1 = (1-betak_pow_kp1)
            
            # forward:
            (x.mul_(betak)).add_(in_k)
            out_k = x*(one_minus_betak/one_minus_betak_pow_kp1)            
                
        elif mode == 2: # exponential. (stable if 0 \le \beta < 1)

            # forward:
            (x.mul_(betak)).add_(one_minus_betak*in_k)
            out_k = 1*x

        elif mode == 30: # uniform (unstable as k \to infinity, as \beta \to 1)
        
            # forward:
            (x.mul_(k)).add_(in_k).div_(k+1)
            out_k = 1*x
            
        elif mode == 3: # uniform (unstable as k \to infinity, as \beta \to 1)
        
            # forward:
            (x.mul_(k-1)).add_(in_k).div_(k)
            out_k = 1*x   

        '''
        out_k : output at current time, k
        x : updated state for next time, k+1
        '''
        return out_k, x


'''
ASGM
PyTorch Backend.
'''
class aSGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, auto:bool, mode:int, lpf:LPF,
                betas:tuple, lr_inits:Tensor, wdecay_cte:Tensor, eps:Tensor, steps_per_epoch:int, maximize:bool, joint:bool) -> None:
        self.lpf = lpf
        self.auto = auto
        self.mode = mode
        self.maximize = maximize
        self.eps = eps
        self.device = eps.device
        self.beta_in_init = betas[0]
        self.beta_out = betas[1]
        self.beta_den = betas[2]
        self.beta_num = betas[3]
        self.beta_diff = betas[4]
        self.lr_inits = lr_inits
        self.joint = joint 
        self.f_low = eps
        self.f_one = torch.tensor(1., dtype=torch.float, device=eps.device)
        self.f_high = 0.99999*self.f_one
        self.weightdecay_cte = wdecay_cte
        self.est_numels = 0
        self.spe = steps_per_epoch #number of batches
    
    @torch.no_grad()
    def log_stats(self):        
        # logging.
        txt = "="
        infostr = "aSGM info:"          
        strlog = f"{infostr} [auto={self.auto}, inits: lr={self.lr_inits:.5g} | lowpass poles [i,n,d,o] : {self.beta_in_init:.4g}, {self.beta_num:.4g}, {self.beta_den:.4g}, {self.beta_out:.4g} | digital diff. : {self.beta_diff:.4g} | weight-decay: {self.weightdecay_cte:.4g}]"
        # debug  
        print(f"{txt * (len(strlog)) }") 
        # print(f"[p={self.p}]")
        print(f"[total params, d={self.est_numels}]")
        print(strlog) #      
        print(f"{txt * (len(strlog)) }\n")

            
    @torch.no_grad()
    def compute_opt(self,step:Tensor|int,step_c:Tensor|int,        
                param:Tensor, param_grad:Tensor,grad_regs:Tensor,
                qk:Tensor,wk:Tensor,mk:Tensor,sk:Tensor,
                ak:Tensor,bk:Tensor,lrk:Tensor,
                beta_out_k:Tensor,beta_in_k:Tensor,beta_den_k:Tensor, beta_num_k:Tensor
                ):
        
        # -1. input grad.          
        # -2. gain fcn. (prop.+deriv.) 
        # -3. output param. update. (integ.)

        # input error, err = 0-grad
        err = param_grad.neg()
        regerror = grad_regs.neg()
        if self.maximize: err.neg_(); regerror.neg_()
        if self.joint: err.add_(regerror)
                
        err_smth_old = 1*mk
        # input: smoothing    
        err_smth, mk = self.lpf.torch_ew_compute(in_k=err, x=mk, beta=beta_in_k, step=step)
        
        # input: digital differentiation.
        v = err + self.beta_diff*(err-err_smth_old) # => err
        v_smth = err_smth + self.beta_diff*(err_smth-err_smth_old) # => err_smth
        
        # - Bayes optimal propotional gain or step-size:

        # variance estimation: (also a robust alternative to time derivative ops. in classic PID formulations)   (averaging + smoothing) 
        v_var, sk = self.lpf.torch_ew_compute(in_k=(v*v.conj()), x=sk,beta=beta_den_k, step=step)
        v_norm = (v_smth).div_(((v_var.sqrt_()).add_(self.eps)))  
        vr_norm = (v).div_(((v_var.sqrt_()).add_(self.eps)))  

        # linear correlation estimation: eff. step-size or learning rate estimation
        if self.auto:
            # linear correlation estimate update  (averaging + smoothing)
            lcavgk, bk = self.lpf.torch_ew_compute(in_k= v_norm.mul(wk), beta=self.f_high, x=bk, step=step)       
            #          
            lck, ak= self.lpf.torch_ew_compute(in_k=lcavgk, x=ak, beta=beta_num_k, step=step, mode=2)    
            # projection.
            alpha_hat = (lck.abs_()) 
            # alpha_hat = (lck.abs_().mean()) 
            # alpha_hat = (lck.max().mean_()) 
        else:
            # use externally supplied linear correlation estimate 
            alpha_hat = self.lr_inits
            
        # update out: integrate P-D contribution
        wk.add_(v_norm.mul_(alpha_hat))
            
            
        # optional: Et[w] output smoothing 
        param_est, qk = self.lpf.torch_ew_compute(in_k=wk, x=qk,beta=beta_out_k, step=step)

        # pass param est. values to network's parameter placeholder.
        param.copy_(param_est)  
        lrk.copy_(alpha_hat)
        # END

        
        
'''
PyTorch Frontend.
'''                     
class AutoSGM(Optimizer):

    """Implements: AutoSGM learning algorithm.
    
    The automatic SGM "Stochastic" Gradient Method is a discrete-time PID structure, with lowpass regularizing components.
    
    PID structure is a discrete-time structure with Proportional + Integral + Derivative components
    
    The Bayes optimal proportional gain (or step-size) component contains the effective step-size (or learning rate) which represents a linear correlation variable; and a variance estimation component (or normalizing component)
    
    The derivative gain component is a digital differentiator, that is most always sensitive to noise, so it is usually turned off.
    
    The integral component is the digital integrator, which does parameter updates or adaptation using the first-order gradient of a scalar-valued objective-function.
    
    Author: OAS
        
    Date: (Changes)

        2022. Nov. (initial code.)

        2023. March. (refactored code.)

    Args:
        params(iterable, required): iterable of parameters to optimize or dicts defining parameter groups
        
        auto (bool, optional, default: True) Bayes optimal step-size (full proportional gain) or Bayes optimal learning rate (a linear correlation estimate) in (0, 1)
        
        mode (int, optional, default: 1) switches between many possible implementations of AutoSGM (not yet implemented.)
        
        steps_per_epoch (int, required): iterations per epoch or number of minibatches >= 1 (default: 1)

        lr_init (float, optional, default=1e-3): If auto=True, then this initializes the state of the lowpass filter used for computing the learning rate. If auto=False, then this is a fixed learning rate;

        beta_in_smooth (float, optional, default = 0.9): for input smoothing. lowpass filter pole in (0, 1). 
        
        beta_den (float, optional, default = 0.999): for averaging, input variance est. lowpass filter pole in (0, 1)    
        
        beta_num (float, optional, default = 0.999): for averaging, linear correlation est. lowpass filter pole in (0, 1)       
        
        beta_out (float, optional, default = 0): for output averaging. lowpass filter pole in (0, 1).

        beta_diff (float, optional, default=0): for digital differentiation, should be left to its default, in most cases.
        
        maximize (bool, optional, default: False): maximize the params based on the objective, instead of minimizing

        weight_decay_cte (float, optional, default = 0): for L2 regularization or simply weight decay. Should be in (0, 1).
        
        joint (bool, optional, default: True): weight decay not decoupled. Decoupled implementation will be done later.
        
        usecuda (bool, optional, default: False): set to True, if your machine is cuda enabled or if you want to use cuda.
        
        .. AutoSGM: Automatic (Stochastic) Gradient Method _somefuno@oregonstate.edu

    Example:
        >>> from asgm import AutoSGM
        >>> ...
        
        >>> optimizer = AutoSGM(model.parameters())
        
        >>> optimizer = AutoSGM(model.parameters(), auto=False) 
        
        >>> optimizer = AutoSGM(model.parameters(), usecuda=True)
        
        >>> optimizer = AutoSGM(model.parameters(), beta_in_smooth=0.9, beta_out=0., beta_den = 0.999, beta_num=0.999, weight_decay_cte=1e-5)

        >>> optimizer = AutoSGM(model.parameters(), auto=False, lr_init=1e-3, weight_decay_cte=1e-5)
        
        >>> ...
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step
    """
    # 
    def __init__(self, params, *, steps_per_epoch=1, lr_init=1e-3, beta_in_smooth=0.9, beta_out=0., beta_den= 0.999, beta_num=0.999, beta_diff=0., weight_decay_cte=0, mode=1, maximize=False, auto=True, joint=True, usecuda=False):

        if not 0.0 <= lr_init:
            raise ValueError(f"Invalid value: lr_init={lr_init} must be in [0,1) for tuning")
        if not 0.0 <= beta_in_smooth and not 1.0 > beta_in_smooth:
            raise ValueError(f"Invalid input lowpass pole: {beta_in_smooth} must be in [0,1) for tuning")        
        if not 0.0 <= beta_den and not 1.0 > beta_den:
            raise ValueError(f"Invalid step-size denominator lowpass pole: {beta_den} must be in [0,1) for tuning")        
        if not 0.0 <= beta_num and not 1.0 > beta_num:
            raise ValueError(f"Invalid step-size numerator lowpass pole: {beta_num} must be in [0,1) for tuning")
        if not 0.0 <= beta_out and not 1.0 > beta_out:
            raise ValueError(f"Invalid output lowpass pole: {beta_out} must be in [0,1) for tuning")        
        if not 0.0 <= beta_diff and not 1.0 > beta_diff:
            raise ValueError(f"Invalid differentiator parameter: {beta_out}recommend to be in [0,1) for tuning") 
        if not 0.0 <= weight_decay_cte and not 1.0 > weight_decay_cte:
            raise ValueError(f"Invalid weight decay value: {weight_decay_cte}")
                    
        if maximize: weight_decay_cte = -weight_decay_cte

        self.debug = True
        
        # -pick computation device
        devcnt = torch.cuda.device_count()
        if devcnt > 0 and usecuda: self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')
            
        # eps added: div. by zero.
        eps = torch.tensor(1e-8, dtype=torch.float, device=self.device)
        # effective step-size.
        lr_init = torch.tensor(lr_init, dtype=torch.float, device=self.device)
        
        # compute: one-pole filter gain
        #
        beta_num = torch.tensor(beta_num, dtype=torch.float, device=self.device)
        beta_den = torch.tensor(beta_den, dtype=torch.float, device=self.device)
        #
        beta_in_smooth = torch.tensor(beta_in_smooth, dtype=torch.float, device=self.device)
        beta_out = torch.tensor(beta_out, dtype=torch.float, device=self.device)  
        #
        beta_diff = torch.tensor(beta_diff, dtype=torch.float, device=self.device)
        wdecay_cte = torch.tensor(weight_decay_cte, dtype=torch.float, device=self.device)  
        
        betas = (beta_in_smooth, beta_out, beta_den, beta_num, beta_diff)
        
        # init. LPF object
        self.lpf = LPF(cdevice=self.device)

        defaults = dict(
            lr_inits=lr_init, betas=betas, wdecay_cte=wdecay_cte, steps_per_epoch=steps_per_epoch, auto=auto, mode=mode, maximize=maximize, joint=joint, eps=eps
            )

        super(AutoSGM, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(AutoSGM, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
    
    @torch.no_grad()
    # @_use_grad_for_differentiable
    def step(self, cost=None, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates 
                the model grad and returns the loss.
        """
        loss = cost
        if closure is not None:
            with torch.enable_grad():
                # loss = closure()
                loss = closure
                

        for group in self.param_groups:
            if 'asgm' not in group: # at first step.
                group['asgm'] = aSGM(group['auto'], group['mode'], 
                    self.lpf, group['betas'], group['lr_inits'], group['wdecay_cte'], group['eps'], group['steps_per_epoch'],  group['maximize'], group['joint']) 
            
            asgm = group['asgm']
            
            # list of parameters, gradients, gradient notms
            params_with_grad = []
            grads = []

            # list to hold step count
            state_steps = []

            # lists to hold previous state memory
            wk, lrk = [],[]
            qk, mk, sk, ak, bk = [],[],[],[],[]
            beta_out_k, beta_in_k, beta_den_k, beta_num_k = [],[],[],[]
                        
            for p in group['params']:

                if p.grad is not None:
                    # get parameter with grad.
                    params_with_grad.append(p)
                    
                    if p.grad.is_sparse:
                        raise RuntimeError(
                            'This implementation does not support sparse gradients.')

                    # get its gradient
                    grads.append(p.grad)

                    state = self.state[p]
                    # initialize state, if empty
                    if len(state) == 0:
                        # -out 
                        # first step count
                        state['step'] = torch.zeros((1,), dtype=torch.float, device=self.device)
                        # controller state
                        state['w'] = p.clone(memory_format=torch.preserve_format).detach()   
                        # filter state 
                        state['q'] = p.clone(memory_format=torch.preserve_format).detach() 
                        state['beta_out'] = group['betas'][1]*torch.ones((1,), dtype=torch.float, device=self.device)  
                        
                        # -in
                        # filter state 
                        state['m'] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)
                        
                        state['beta_in'] = group['betas'][0]* torch.ones((1,), dtype=torch.float, device=self.device)
                        
                        # -den
                        # filter state 
                        state[f"s"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)
                        state[f"beta_den"] = group['betas'][2]*torch.ones((1,), dtype=torch.float, device=self.device)
                        
                        #-num
                        # filter state 
                        state[f"a"] = group["lr_inits"]*torch.ones_like(p, memory_format=torch.preserve_format, device=self.device)    
                        # optional filter state 
                        state[f"b"] = group["lr_inits"]*torch.ones_like(p, memory_format=torch.preserve_format, device=self.device) 
                        state[f"beta_num"] = group['betas'][3]* torch.ones((1,), dtype=torch.float, device=self.device)
                        
                        #-for logging step-size or learning rate
                        state[f"lr"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format, device=self.device)
                        
                    wk.append(state['w'])    
                    qk.append(state['q']) 
                    beta_out_k.append(state['beta_out'])
                    mk.append(state['m']) 
                    beta_in_k.append(state['beta_in'])
                    sk.append(state['s'])
                    beta_den_k.append(state['beta_den']) 
                    ak.append(state['a'])
                    bk.append(state['b'])
                    beta_num_k.append(state['beta_num'])
                    lrk.append(state['lr'])
                    
                    # update the step count by 1.
                    state['step'] += 1

                    # record the step
                    state_steps.append(state['step'])

            # Actual Learning Event: 
            control_event(asgm, 
                params_with_grad, grads,
                wk, qk, mk, sk, ak, bk, lrk,
                beta_out_k, beta_in_k, beta_den_k, beta_num_k,
                state_steps
            )

        return loss, state['step']


'''
One step/iteration
'''
@torch.no_grad()
def control_event(asgm:aSGM,
        params: List[Tensor], grads: List[Tensor], 
        wk: List[Tensor], qk: List[Tensor], mk: List[Tensor], 
        sk: List[Tensor], ak: List[Tensor], bk: List[Tensor], lrk: List[Tensor],
        beta_out_k: List[Tensor], beta_in_k: List[Tensor], beta_den_k: List[Tensor], beta_num_k: List[Tensor], state_steps: List[Tensor]):
    
    r'''Functional API that computes the AutoSGM control/learning algorithm for each parameter in the model.

    See : class:`~torch.optim.AutoSGM` for details.
    '''
    # lrs = []
    step = state_steps[0][0]
    # cyclic step in each epoch
    step_c = (((step-1) % asgm.spe) + 1)
    
    #- At each step, adapt parameters (weights of the neural network)
    # in <- Dt{Et{-g}} or Et{Dt{-g}} 
    # state <- state + alphap*in
    # out <- Et{state}
    
    # UPDATE MAIN PARAMETER ESTIMATES.
    for i, param in enumerate(params):
        
        if step == 1:         
            if i == 0: asgm.est_numels = param.nelement()
            else: asgm.est_numels += param.nelement()   
        
        grad_regs = wk[i].mul(asgm.weightdecay_cte/asgm.est_numels)
        asgm.compute_opt(
                        step,step_c,param,grads[i],grad_regs,
                        qk[i],wk[i],mk[i],sk[i],ak[i],bk[i],lrk[i],
                        beta_out_k[i], beta_in_k[i], beta_den_k[i], beta_num_k[i]
                    )
        # lrs = None
        # lrs.append(lr)
    
    # LOGGING.
    if step == 1: asgm.log_stats()
        
    # return alphaps
