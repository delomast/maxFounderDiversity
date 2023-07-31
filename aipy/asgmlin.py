r"""AutoSGMLin"""
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
                        beta:Tensor, step:Tensor=torch.ones((1,)), mode=1):
        
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
        
        one_minus_betak = (1 - betak)
        betak_pow_k = betak.pow(k)
        one_minus_betak_pow_k = (1-betak_pow_k)
        gamma_k = (one_minus_betak/one_minus_betak_pow_k) 
        
        if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
            
            # forward:
            ((x.mul_((betak - betak_pow_k))).add_(one_minus_betak*in_k)).div_(one_minus_betak_pow_k)
            out_k = 1*x #x.detach().clone()
            # gamma_k = torch.ones_like(x)
            
        elif mode == 1: # exponential. (stable if 0 \le \beta < 1)
        
            # forward:
            (x.mul_(betak)).add_(in_k)
            out_k = gamma_k*x       
            # out_k = 1*x # x.detach().clone()     
                
        elif mode == 2: # exponential. (stable if 0 \le \beta < 1)

            # forward:
            (x.mul_(betak)).add_(one_minus_betak*in_k)
            out_k = 1*x #x.detach().clone() 
            
        elif mode == 3: # uniform (unstable as k \to infinity, as \beta \to 1)
        
            # forward:
            (x.mul_(k-1)).add_(in_k).div_(k)
            out_k = 1*x #x.detach().clone()
            
        '''
        out_k : output at current time, k
        x : updated state for next time, k+1
        '''
        return out_k, x, gamma_k


'''
ASGM
PyTorch Backend.
'''
class aSGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, auto:bool, mode:int, lpf:LPF,
                betas:tuple, lr_inits:Tensor, eps:Tensor, steps_per_epoch:int, maximize:bool) -> None:
        self.lpf = lpf
        self.auto = auto
        self.mode = mode
        self.maximize = maximize
        self.eps = eps
        self.device = eps.device
        self.beta_in_init = betas[0]
        self.beta_out = betas[1]
        self.beta_diff = betas[2]
        self.lr_inits = lr_inits
        self.f_low = eps 
        self.f_one = torch.tensor(1., dtype=torch.float, device=eps.device)
        self.f_high = torch.tensor(0.99999, dtype=torch.float32, device=eps.device)
        self.est_numels = 0
        self.spe = steps_per_epoch #number of batches
    
    @torch.no_grad()
    def log_stats(self):        
        # logging.
        txt = "="
        infostr = "aSGM info:"          
        strlog = f"{infostr} [auto={self.auto}, inits: lr={self.lr_inits:.5g} | lowpass poles [i,o] : {self.beta_in_init:.4g}, {self.beta_out:.4g} | digital diff. : {self.beta_diff:.4g} ]"
        # debug  
        print(f"{txt * (len(strlog)) }") 
        # print(f"[p={self.p}]")
        print(f"[total params, d={self.est_numels}]")
        print(strlog) #      
        print(f"{txt * (len(strlog)) }\n")

            
    @torch.no_grad()
    def compute_opt(self,step:Tensor,        
                x_mat:Tensor, A_mat:Tensor, param:Tensor, param_grad:Tensor,
                qk:Tensor,wk:Tensor,gk:Tensor,
                mk:Tensor,dk:Tensor,sk:Tensor,lrk:Tensor,
                beta_out_k:Tensor,beta_in_k:Tensor
                ):
        
        # -1. input grad.          
        # -2. gain fcn. (prop.+deriv.) 
        # -3. output param. update. (integ.)
        
        # dke = 1*dk #.add(0)
        # mke = mk.add(self.eps2)

        # input error, err = 0-grad
        err = param_grad.neg()
        if self.maximize: err.neg_()
        
        # input: digital differentiation.
        d = err + self.beta_diff*(err-gk) # => err  

        # set pole of input lowpass filter
        if self.auto and step > 1:
            # usually goes to a very small value, 
            # for the linear problem under consideration, far less noisy.
            beta_new = torch.abs((d.T.mm(d-dk)).div_((mk.T.mm(dk)).add_(self.eps)))
            #   beta_in_k.copy_(beta_new)
            beta_in_k.copy_(beta_new*sk)

        # input: smoothing    
        err_smth, mk, gamk = self.lpf.torch_ew_compute(in_k=d, x=mk, beta=beta_in_k, step=step)
        
        
        gk.copy_(err)
        dk.copy_(d)
        if step > 1: sk.copy_(gamk)
        vk = err_smth
        # - optimal propotional gain (step-size): for lin model + quad. opt.
        if self.auto:
            Avxx = (A_mat.mm(vk)).mm(x_mat) 
            alpha_k = (vk.T.mm(d)).div_((vk.T.mm(Avxx)).add_(self.eps))
            # out: update parameter wk: by integrating P-D component
            wk.add_(vk.mul_(alpha_k))
        else:
            # use externally supplied linear correlation estimate 
            alpha_k = self.lr_inits
            # out: update parameter wk: by integrating P-D component
            wk.add_(vk.mul_(alpha_k))
                
            
        # optional: Et[w] output smoothing 
        param_est, qk, _ = self.lpf.torch_ew_compute(in_k=wk, x=qk,beta=beta_out_k, step=step)

        # pass param est. values to network's parameter placeholder.
        param.copy_(param_est)  
        lrk.copy_(alpha_k)
        # END

        
        
'''
PyTorch Frontend.
'''                     
class AutoSGMLin(Optimizer):

    """Implements: AutoSGM (Linear Layer) learning algorithm.
    
    The automatic SGM "Stochastic" Gradient Method is a discrete-time PID structure, with lowpass regularizing components.
    
    PID structure is a discrete-time structure with Proportional + Integral + Derivative components
    
    The Bayes optimal proportional gain (or step-size) component contains the effective step-size (or learning rate) which represents a linear correlation variable; and a variance estimation component (or normalizing component)
    
    The derivative gain component is a digital differentiator, that is most always sensitive to noise, so it is usually turned off.
    
    The integral component is the digital integrator, which does parameter updates or adaptation using the first-order gradient of a scalar-valued objective-function.
    
    Author: OAS
        
    Date: (Changes)

        2023. July. (init. code.)

    Args:
        params(iterable, required): iterable of parameters to optimize or dicts defining parameter groups
        
        auto (bool, optional, default: True) Bayes optimal step-size (full proportional gain) or Bayes optimal learning rate (a linear correlation estimate) in (0, 1)
        
        mode (int, optional, default: 1) switches between many possible implementations of AutoSGM (not yet implemented.)
        
        steps_per_epoch (int, required): iterations per epoch or number of minibatches >= 1 (default: 1)

        lr_init (float, optional, default=1e-3): If auto=True, then this initializes the state of the lowpass filter used for computing the learning rate. If auto=False, then this is a fixed learning rate;

        beta_in_smooth (float, optional, default = 0.9): for input smoothing. lowpass filter pole in (0, 1).   
        
        beta_out (float, optional, default = 0): for output averaging. lowpass filter pole in (0, 1).

        beta_diff (float, optional, default=0): for digital differentiation, should be left to its default, in most cases.
        
        maximize (bool, optional, default: False): maximize the params based on the objective, instead of minimizing

        usecuda (bool, optional, default: False): set to True, if your machine is cuda enabled or if you want to use cuda.
        
        .. AutoSGM: Automatic (Stochastic) Gradient Method _somefuno@oregonstate.edu

    Example:
        >>> from asgmlin  import AutoSGMLin
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
    def __init__(self, params, *, steps_per_epoch=1, lr_init=1e-1, beta_in_smooth=0.9, beta_out=0, beta_diff=0, mode=1, maximize=False, auto=True, usecuda=False):

        if not 0.0 <= lr_init:
            raise ValueError(f"Invalid value: lr_init={lr_init} must be in [0,1) for tuning")
        if not 0.0 <= beta_in_smooth and not 1.0 > beta_in_smooth:
            raise ValueError(f"Invalid input lowpass pole: {beta_in_smooth} must be in [0,1) for tuning")        
        if not 0.0 <= beta_out and not 1.0 > beta_out:
            raise ValueError(f"Invalid output lowpass pole: {beta_out} must be in [0,1) for tuning")        
        if not 0.0 <= beta_diff and not 1.0 > beta_diff:
            raise ValueError(f"Invalid differentiator parameter: {beta_out}recommend to be in [0,1) for tuning") 
                    
        self.debug = True
        
        # -pick computation device
        devcnt = torch.cuda.device_count()
        if devcnt > 0 and usecuda: self.device = torch.device('cuda')
        else: self.device = torch.device('cpu')
            
        # eps added: div. by zero.
        if usecuda:
            eps = torch.tensor(1e-12, dtype=torch.float32, device=self.device)
        else:
            eps = torch.tensor(1e-12, dtype=torch.float32, device=self.device)
        # effective step-size.
        lr_init = torch.tensor(lr_init, dtype=torch.float32, device=self.device)
        
        # compute: one-pole filter gain
        beta_in_smooth = torch.tensor(beta_in_smooth, dtype=torch.float32, device=self.device)
        beta_out = torch.tensor(beta_out, dtype=torch.float32, device=self.device)  
        #
        beta_diff = torch.tensor(beta_diff, dtype=torch.float32, device=self.device)

        betas = (beta_in_smooth, beta_out, beta_diff)
        
        # init. LPF object
        self.lpf = LPF(cdevice=self.device)

        defaults = dict(
            lr_inits=lr_init, betas=betas, steps_per_epoch=steps_per_epoch, auto=auto, mode=mode, maximize=maximize, eps=eps
            )

        super(AutoSGMLin, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(AutoSGMLin, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
    
    @torch.no_grad()
    # @_use_grad_for_differentiable
    def step(self, xinput:Tensor, A_mat:Tensor, gradin:Tensor=None, closure=None):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates 
                the model grad and returns the loss.
        """
        if closure is not None:
            with torch.enable_grad():
                # loss = closure()
                loss = closure
                

        for group in self.param_groups:
            if 'asgm' not in group: # at first step.
                group['asgm'] = aSGM(group['auto'], group['mode'], 
                    self.lpf, group['betas'], group['lr_inits'], group['eps'], group['steps_per_epoch'],  group['maximize']) 
            
            asgm = group['asgm']
            
            # list of parameters, gradients, inputs to parameter's layer
            params_with_grad = []
            grads = []
            x_mat = []
            # list to hold step count
            state_steps = []

            # lists to hold previous state memory
            wk, lrk = [],[]
            qk, mk, gk, dk, sk = [],[],[],[],[]
            beta_out_k, beta_in_k = [],[]
                        
            p = group['params'][0]

            if p.grad is not None:
                # get parameter with grad.
                params_with_grad.append(p)
                
                if p.grad.is_sparse:
                    raise RuntimeError(
                        'This implementation does not support sparse gradients.')

                # get its gradient
                grads.append(p.grad)
                # grads.append(gradin)
                
                # get its input
                x_mat.append(xinput.mm(xinput.T))

                state = self.state[p]
                # initialize state, if empty
                if len(state) == 0:
                    # -out 
                    # first step count
                    state['step'] = torch.zeros((1,), dtype=torch.int, device=self.device)
                    
                    # control parameter state
                    state['w'] = p.clone(memory_format=torch.preserve_format).detach()   
                    # filter state 
                    state['q'] = p.clone(memory_format=torch.preserve_format).detach() 
                    state['beta_out'] = group['betas'][1]*torch.ones((1,), dtype=torch.float32, device=self.device)  
                    
                    # -in
                    # gradient state
                    state['g'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=self.device)
                    # filter state 
                    state['m'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=self.device)
                    state['d'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=self.device)                    
                    state['s'] = torch.ones_like(
                        p, memory_format=torch.preserve_format, device=self.device)
                    state['beta_in'] = group['betas'][0]*torch.ones_like(p, dtype=torch.float32, device=self.device)

                    #-for logging step-size or learning rate
                    state[f"lr"] = group["lr_inits"]*torch.ones_like(torch.matmul(xinput,xinput.T), memory_format=torch.preserve_format, device=self.device)
                    
                wk.append(state['w'])    
                qk.append(state['q']) 
                beta_out_k.append(state['beta_out'])
                gk.append(state['g']) 
                beta_in_k.append(state['beta_in'])
                mk.append(state['m'])
                dk.append(state['d'])
                sk.append(state['s'])
                lrk.append(state['lr'])
                
                # update the step count by 1.
                state['step'] += 1

                # record the step
                state_steps.append(state['step'])

            # Actual Learning Event: 
            
            control_event(asgm, x_mat, A_mat,
                params_with_grad, grads,
                wk, qk, gk, mk, dk, sk, lrk,
                beta_out_k, beta_in_k,
                state_steps
            )

        return state['step'].item(), lrk, beta_in_k


'''
One step/iteration
'''
@torch.no_grad()
def control_event(asgm:aSGM, x_mat:List[Tensor], A_mat:Tensor,
        params: List[Tensor], grads: List[Tensor], 
        wk: List[Tensor], qk: List[Tensor], gk: List[Tensor], 
        mk: List[Tensor], dk: List[Tensor], sk: List[Tensor], lrk: List[Tensor],
        beta_out_k: List[Tensor], beta_in_k: List[Tensor], state_steps: List[Tensor]):
    
    r'''Functional API that computes the AutoSGMLin control/learning algorithm for each parameter in the model.

    See : class:`~torch.optim.AutoSGMLin` for details.
    '''
    # lrs = []
    step = state_steps[0][0]
    
    #- At each step, adapt parameters (weights of the neural network)
    # in <- Dt{Et{-g}} or Et{Dt{-g}} 
    # state <- state + alphap*in
    # out <- Et{state}
    
    # UPDATE MAIN PARAMETER ESTIMATES.
    for i, param in enumerate(params):
        
        if step == 1:         
            if i == 0: asgm.est_numels = param.nelement()
            else: asgm.est_numels += param.nelement()   
        
        asgm.compute_opt(
                        step,
                        x_mat[i], A_mat, 
                        param,grads[i],
                        qk[i],wk[i],gk[i],mk[i],dk[i],sk[i],lrk[i],
                        beta_out_k[i], beta_in_k[i],
                    )
        # lrs = None
        # lrs.append(lr)
    
    # LOGGING.
    if step == 1: asgm.log_stats()
        
    # return alphaps
