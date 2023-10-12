r"""AutoSGMQuad"""
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
    def torch_ew_compute(self, in_t:Tensor, x:Tensor, 
                        beta:Tensor, step:Tensor=torch.ones((1,)), mode=1):
        
        '''
        in_t: input at current time
        x: state at previous time
        beta: LPF pole at current time
        step: current discrete time
        mode: [default: mode=1] unbiased (all, except 2) | asympt. unbiased (2)

        out_t : output at current time, t
        x : updated state for next time, t+1
        '''
        # temps. (avoid repititions)
        k, betak = step, beta
        
        one_minus_betak = (1 - betak)
        betak_pow_k = betak.pow(k)
        one_minus_betak_pow_k = (1-betak_pow_k)
        gamma_t = 1
        
        
        if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
            
            # forward:
            ((x.mul_((betak - betak_pow_k))).add_(one_minus_betak*in_t)).div_(one_minus_betak_pow_k)
            out_k = 1*x
            
        elif mode == 1: # exponential. (stable if 0 \le \beta < 1)
        
            # forward:
            gamma_t = (one_minus_betak/one_minus_betak_pow_k) 
            (x.mul_(betak)).add_(in_t)
            out_k = gamma_t*x       
                
        elif mode == 2: # exponential. (stable if 0 \le \beta < 1)

            # forward:
            (x.mul_(betak)).add_(one_minus_betak*in_t)
            out_k = 1*x #/one_minus_betak_pow_k#
            
        elif mode == 3: # uniform (unstable as k \to infinity, as \beta \to 1)
        
            # forward:
            (x.mul_(k-1)).add_(in_t).div_(k)
            out_k = 1*x
            
        '''
        out_k : output at current time, t
        x : updated state for next time, t+1
        '''
        return out_k, x, gamma_t


'''
ASGM
PyTorch Backend.
'''
class aSGM():
    ''' Gradient Method: automatic algorithm for learning/control/estimation

    Core algorithm implementation (Torch)
    '''
    def __init__(self, auto:bool, mode:int, lpf:LPF,
                betas:tuple, ss_inits:Tensor, eps:Tensor, steps_per_epoch:int, maximize:bool) -> None:
        self.lpf = lpf
        self.auto = auto
        self.mode = mode
        self.maximize = maximize
        self.eps = eps
        self.device = eps.device
        self.beta_in_init = betas[0]
        self.beta_out = betas[1]
        self.beta_diff = betas[2]
        self.ss_inits = ss_inits
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
        strlog = f"{infostr} [auto={self.auto}, inits: lr={self.ss_inits:.5g} | lowpass poles [i,o] : {self.beta_in_init:.4g}, {self.beta_out:.4g} | digital diff. : {self.beta_diff:.4g} ]"
        # debug  
        print(f"{txt * (len(strlog)) }") 
        # print(f"[p={self.p}]")
        print(f"[total params, d={self.est_numels}]")
        print(strlog) #      
        print(f"{txt * (len(strlog)) }\n")

            
    @torch.no_grad()
    def compute_opt(self,step:Tensor,        
                A_mat:Tensor, b_vec:Tensor, x_vec:Tensor, c_t_vec:Tensor, param:Tensor, param_grad:Tensor,
                q_t:Tensor,w_t:Tensor,g_k:Tensor,
                m_t:Tensor,d_t:Tensor, gbar_t:Tensor, gamma_t:Tensor,ss_t:Tensor,
                beta_out_t:Tensor,beta_in_t:Tensor
                ):
        
        # -1. input grad.          
        # -2. gain fcn. (prop.+deriv.) 
        # -3. output param. update. (integ.)
        
        x_mat = x_vec.mm(x_vec.T)
        c_t_mat = c_t_vec.mm(c_t_vec.T)

        # input g_t, err = -g_t
        g_t = 1*param_grad
        if self.maximize: g_t.neg_()
        
        # on input: arbitrary noise addition
        # prop term + time-derivative term => High-Pass Filter (1st order)
        if self.beta_diff > 0:
            pd_t = g_t + self.beta_diff*(g_t-g_k)
        else: pd_t = 1*g_t
        
        # set pole of input lowpass filter
        if self.auto and step > 1:
            # mode 1
            beta_new = ((pd_t.T.mm(g_t-gbar_t)).div_((m_t.T.mm(gbar_t)).add_(self.eps)))
            # mode 2
            # beta_new = (((d.T.mm(g_t-gbar_t)).add_(self.eps)).div_(((d.T.mm(g_t-gbar_t))+(m_t.T.mm(gbar_t))).add_(self.eps)))
            beta_in_t.copy_(beta_new*gamma_t)

        # on input: smoothing    
        # Low-pass Filter (1st order)
        v_t, m_t, gam_t = self.lpf.torch_ew_compute(in_t=pd_t, x=m_t, beta=beta_in_t, step=step)

        g_k.copy_(g_t)
        d_t.copy_(pd_t)
        if step > 1: gamma_t.copy_(gam_t)
        
        # - optimal propotional gain (step-size): for lin model + quad. opt.
        if self.auto:
            e_gt=A_mat.mm(w_t.mm(c_t_mat-x_mat)) - 1*b_vec.mm(c_t_vec.T-x_vec.T)
            gbar_t.copy_(g_t + e_gt)
            Avcc = (A_mat.mm(v_t)).mm(c_t_mat)
            alpha_t = ((v_t.T.mm(gbar_t))).div_((v_t.T.mm(Avcc)).add_(self.eps))
        else:
            # use externally supplied linear correlation estimate 
            alpha_t = self.ss_inits
        
        # out: update parameter w_t: by integrating P-D component
        w_t.sub_(v_t.mul_(alpha_t))
                
            
        # optional: Et[w] output smoothing 
        if beta_out_t > 0:
            param_est, q_t, _ = self.lpf.torch_ew_compute(in_t=w_t, x=q_t,beta=beta_out_t, step=step, mode=2)
        else: param_est = 1*w_t

        # pass param est. values to network's parameter placeholder.
        param.copy_(param_est)  
        ss_t.copy_(alpha_t)
        # END

        
        
'''
PyTorch Frontend.
'''                     
class AutoSGMQuad(Optimizer):

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

        ss_init (float, optional, default=1e-3): If auto=True, then this initializes the state of the lowpass filter used for computing the learning rate. If auto=False, then this is a fixed learning rate;

        beta_in_smooth (float, optional, default = 0.9): for input smoothing. lowpass filter pole in (0, 1).   
        
        beta_out (float, optional, default = 0): for output averaging. lowpass filter pole in (0, 1).

        beta_diff (float, optional, default=0): for digital differentiation, should be left to its default, in most cases.
        
        maximize (bool, optional, default: False): maximize the params based on the objective, instead of minimizing

        usecuda (bool, optional, default: False): set to True, if your machine is cuda enabled or if you want to use cuda.
        
        .. AutoSGM: Automatic (Stochastic) Gradient Method _somefuno@oregonstate.edu

    Example:
        >>> from asgmlin  import AutoSGMQuad
        >>> ...
        
        >>> optimizer = AutoSGMQuad(model.parameters())
        
        >>> optimizer = AutoSGMQuad(model.parameters(), auto=False) 
        
        >>> optimizer = AutoSGMQuad(model.parameters(), usecuda=True)
        
        >>> optimizer = AutoSGMQuad(model.parameters(), beta_in_smooth=0.9, beta_out=0.)

        >>> optimizer = AutoSGM(model.parameters(), auto=False, ss_init=1e-3)
        
        >>> ...
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step
    """
    # 
    def __init__(self, params, *, steps_per_epoch=1, ss_init=1e-1, beta_in_smooth=0.9, beta_out=0, beta_diff=0, eps=1e-8, mode=1, maximize=False, auto=True, usecuda=False):

        if not 0.0 <= ss_init:
            raise ValueError(f"Invalid value: ss_init={ss_init} must be in [0,1) for tuning")
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
            eps = torch.tensor(eps, dtype=torch.float32, device=self.device)
        else:
            eps = torch.tensor(eps, dtype=torch.float32, device=self.device)
        # effective step-size.
        ss_init = torch.tensor(ss_init, dtype=torch.float32, device=self.device)
        
        # compute: one-pole filter gain
        beta_in_smooth = torch.tensor(beta_in_smooth, dtype=torch.float32, device=self.device)
        beta_out = torch.tensor(beta_out, dtype=torch.float32, device=self.device)  
        #
        beta_diff = torch.tensor(beta_diff, dtype=torch.float32, device=self.device)

        betas = (beta_in_smooth, beta_out, beta_diff)
        
        # init. LPF object
        self.lpf = LPF(cdevice=self.device)

        defaults = dict(
            ss_inits=ss_init, betas=betas, steps_per_epoch=steps_per_epoch, auto=auto, mode=mode, maximize=maximize, eps=eps
            )

        super(AutoSGMQuad, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(AutoSGMQuad, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('maximize', False)
    
    @torch.no_grad()
    # @_use_grad_for_differentiable
    def step(self,  A_mat:Tensor, b_vec:Tensor, cx_vec:Tensor, cc_t_vec:Tensor,gradin:Tensor=None, closure=None):
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
                    self.lpf, group['betas'], group['ss_inits'], group['eps'], group['steps_per_epoch'],  group['maximize']) 
            
            asgm = group['asgm']
            
            # list of parameters, gradients, inputs to parameter's layer
            params_with_grad = []
            grads = []
            x_vec = []
            c_t_vec = []
            # list to hold step count
            state_steps = []

            # lists to hold previous state memory
            w_t, ss_t = [],[]
            q_t, m_t, g_k, d_t, gbar_t, gamma_t = [],[],[],[],[],[]
            beta_out_t, beta_in_t = [],[]
                        
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
                c_t_vec.append(cc_t_vec)
                x_vec.append(cx_vec)

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
                    state['gbar'] = torch.zeros_like(
                        p, memory_format=torch.preserve_format, device=self.device)                 
                    state['gam'] = torch.ones_like(
                        p, memory_format=torch.preserve_format, device=self.device)
                    state['beta_in'] = group['betas'][0]*torch.ones_like(p, dtype=torch.float32, device=self.device)

                    #-for logging step-size or learning rate
                    state[f"lr"] = group["ss_inits"]*torch.ones_like(p, memory_format=torch.preserve_format, device=self.device)
                    
                w_t.append(state['w'])    
                q_t.append(state['q']) 
                beta_out_t.append(state['beta_out'])
                g_k.append(state['g']) 
                beta_in_t.append(state['beta_in'])
                m_t.append(state['m'])
                d_t.append(state['d']) 
                gbar_t.append(state['gbar'])
                gamma_t.append(state['gam'])
                ss_t.append(state['lr'])
                
                # update the step count by 1.
                state['step'] += 1

                # record the step
                state_steps.append(state['step'])

            # Actual Learning Event: 
            
            control_event(asgm, A_mat, b_vec, x_vec, c_t_vec, 
                params_with_grad, grads,
                w_t, q_t, g_k, m_t, d_t, gbar_t, gamma_t, ss_t,
                beta_out_t, beta_in_t,
                state_steps
            )

        return state['step'].item(), 1*ss_t[0], 1*beta_in_t[0]


'''
One step/iteration
'''
@torch.no_grad()
def control_event(asgm:aSGM, A_mat:Tensor, b_vec:Tensor, 
                  x_vec:List[Tensor], c_t_vec:List[Tensor], 
        params: List[Tensor], grads: List[Tensor], 
        wk: List[Tensor], qk: List[Tensor], g_k: List[Tensor], 
        mk: List[Tensor], dk: List[Tensor], gbark: List[Tensor], sk: List[Tensor], lrk: List[Tensor],
        beta_out_k: List[Tensor], beta_in_k: List[Tensor], state_steps: List[Tensor]):
    
    r'''Functional API that computes the AutoSGMQuad control/learning algorithm for each parameter in the model.

    See : class:`~torch.optim.AutoSGMQuad` for details.
    '''
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
                        A_mat,b_vec,x_vec[i],c_t_vec[i],  
                        param,grads[i],
                        qk[i],wk[i],g_k[i],mk[i],dk[i],gbark[i],sk[i],lrk[i],
                        beta_out_k[i], beta_in_k[i]
                    )
    
    # LOGGING.
    if step == 1: asgm.log_stats()