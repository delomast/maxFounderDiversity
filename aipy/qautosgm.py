"""
:mod:`qautosgm` is a package implementing the stochastic gradient learning algorithm for quadratic functions.
"""

# Common doc strings among pytorch's optimizer impl.
_foreach_doc = r"""foreach (bool, optional): whether foreach implementation of optimizer
            is used. If unspecified by the user (so foreach is None), we will try to use
            foreach over the for-loop implementation on CUDA, since it is usually
            significantly more performant. (default: None)"""

_differentiable_doc = r"""differentiable (bool, optional): whether autograd should
            occur through the optimizer step in training. Otherwise, the step()
            function runs in a torch.no_grad() context. Setting to True can impair
            performance, so leave it False if you don't intend to run autograd
            through this instance (default: False)"""

_maximize_doc = r"""maximize (bool, optional): maximize the params based on the
            objective, instead of minimizing (default: False)"""

_email_doc = r"""somefuno@oregonstate.edu"""



from dataclasses import dataclass

import math, torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer, required, 
    _use_grad_for_differentiable,
    _default_to_fused_or_foreach,
    _get_value, _stack_if_compiling, _dispatch_sqrt,
)
from typing import Any, Dict, List, Optional

from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union

from torch.autograd.grad_mode import no_grad

__all__ = ['LPF', 'AutoSGM']

# Forked 'group tensors' from old pytorch install: from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

# This util function splits tensors into groups by device and dtype, which is useful before sending
# tensors off to a foreach implementation, which requires tensors to be on one device and dtype.
# If tensorlistlist contains more than one tensorlist, the following assumptions are made BUT NOT verified:
#   - tensorlists CAN be None
#   - all tensors in the first specified list cannot be None
#   - given an index i, all specified tensorlist[i]s match in dtype and device
# with_indices (bool, optional): whether to track previous indices as the last list per dictionary entry.
#   It comes in handy if there are Nones or literals in the tensorlists that are getting scattered out.
#   Whereas mutating a tensor in the resulting split-up tensorlists WILL propagate changes back to the
#   original input tensorlists, changing up Nones/literals WILL NOT propagate, and manual propagation
#   may be necessary.
@no_grad()
def _group_tensors_by_device_and_dtype(tensorlistlist: List[List[Tensor]],
                                       with_indices: Optional[bool] = False) -> \
        Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]]:
    assert all([not x or len(x) == len(tensorlistlist[0]) for x in tensorlistlist]), (
           "all specified tensorlists must match in length")
    per_device_and_dtype_tensors: Dict[Tuple[torch.device, torch.dtype], List[List[Union[Tensor, int]]]] = defaultdict(
        lambda: [[] for _ in range(len(tensorlistlist) + (1 if with_indices else 0))])
    for i, t in enumerate(tensorlistlist[0]):
        key = (t.device, t.dtype)
        for j in range(len(tensorlistlist)):
            # a tensorlist may be empty/None
            if tensorlistlist[j]:
                per_device_and_dtype_tensors[key][j].append(tensorlistlist[j][i])
        if with_indices:
            # tack on previous index
            per_device_and_dtype_tensors[key][j + 1].append(i)
    return per_device_and_dtype_tensors

def _has_foreach_support(tensors: List[Tensor], device: torch.device) -> bool:
    if device.type not in ['cpu', 'cuda'] or torch.jit.is_scripting():
        return False
    return all([t is None or type(t) == torch.Tensor for t in tensors])


def cmplx2real(lols):
    "input: List of Lists"
    return [[torch.view_as_real(tsr) 
            if torch.is_complex(tsr) else tsr 
                for tsr in lst] 
                    for lst in lols]

@dataclass
class Props():
    def __init__(self, **kwds):
        self.__dict__.update(kwds)
        
    

# LPF
# The `LPF` class defines a generic first-order low pass filter structure for routine smoothing and averaging operations, with methods for exponential lowpass filtering, and raised cosine window generation.
class LPF():
    """ (Generic Digital) First-order Low Pass Filter Structure (Linear System)
        
        Recursively computes a weighted average (exponential or uniform).
        
        Main Use: routine smoothing, averaging operations.
    """

    def __init__(self, inplace:bool=True, foreach:bool=False, fused:bool=False):
        self.inplace = inplace
        self.foreach = foreach
        self.fused = fused
        self.tensor_lists = fused or foreach

    @torch.no_grad()
    def btrng(self, bt:Tensor|float, p:int=1):
        return (1 - (10**(-p))*(1-bt)) - bt
        
    
    @torch.no_grad()
    def compute(self, in_t:Tensor, x:Tensor, 
                beta:Tensor, step:Tensor, mode:int=1, fix:bool=False, mix:bool=False, sq:bool=False, beta_d:Tensor|float=0, epp:Tensor|float=1):
        
        '''Computes in_t -> LPF -> out_t
        
            in_t: input at current time
            x: state at previous time
            beta: LPF pole at current time
            step: current discrete time
            mode: [default: mode = 1] 
            fix: add one to the iteration
            mix: shelve
            beta_d: HPF param.
            epp: weighting order, depends on mode

        out_t : output at current time, t
        x : updated state for next time, t+1
        '''
        
        if not self.tensor_lists:
            u_t = 1*in_t
        elif self.tensor_lists:
            u_t = torch._foreach_mul(in_t,1) 
        else:
            return in_t
        
        t = 1*step
        if fix: t = t + 1
        beta_t = beta       
        
        one_minus_beta_t = (1-beta_t).pow(epp)
        beta_t_pow_t = beta_t.pow(t)
        one_minus_beta_t_pow_t = (1-beta_t_pow_t)
                      
        if not self.tensor_lists:
            if sq:
                in_t = in_t.pow(2)
                
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                # forward:
                ((x.mul_((beta_t - beta_t_pow_t))).add_(one_minus_beta_t*in_t)).div_(one_minus_beta_t_pow_t)
                out_t = 1*x
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                # instead of mode=2, use if init x_t = 0
                (x.mul_(beta_t)).add_(in_t)
                out_t = x*(one_minus_beta_t/one_minus_beta_t_pow_t)     
                
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # instead of mode=4, use if init x_t is likely not 0
                # forward:
                (x.mul_(beta_t)).add_((one_minus_beta_t)*in_t)
                out_t = 1*x      
                    
            elif mode == 4: # exponential. (stable if 0 \le \beta < 1) 
                # instead of mode=3, use if init x_t = 0
                # forward:
                (x.mul_(beta_t)).add_(one_minus_beta_t*in_t)
                out_t = x.div(one_minus_beta_t_pow_t)     
                
            elif mode == 5: # uniform (constant as t \to infinity or as \beta \to 1)
                # useful for averaging, even when t is not time but a sample instance
                # forward:
                (x.mul_(t-1)).add_(in_t).div_(t)
                out_t = 1*x   

            elif mode == 6: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                # trusts memory as t increases
                # forward:
                b = beta_t.mul((t-1)/t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = 1*x  
                
            elif mode == 7: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                #
                # forward:
                b = beta_t.mul((t-1)/t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = x/(1-b.pow(t))                
            
            elif mode == 8: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging, 
                # trusts input as t increases
                # forward:
                b = beta_t.div(t)
                (x.mul_(b)).add_((1-b)*in_t)
                out_t = 1*x  
                  
            elif mode == -1: # exponential. (as beta_t -> 1) 
                # often: use mode = 1 instead of this. 
                # forward:
                (x.mul_(beta_t)).add_(in_t)
                out_t = 1*x     
                    
        elif self.tensor_lists:
            if sq:
                in_t = torch._foreach_mul(in_t, in_t)
            if mode == 0: # exponential. (stable if 0 \le \beta < 1)   
                # forward:
                torch._foreach_mul_(x, (beta_t - beta_t_pow_t))
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                torch._foreach_div_(x, one_minus_beta_t_pow_t)
                out_t = torch._foreach_mul(x, 1)
                
            elif mode == 1: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_div(torch._foreach_mul(x, one_minus_beta_t),one_minus_beta_t_pow_t)      
                               
            elif mode == 3: # exponential. (stable if 0 \le \beta < 1) k \to infty
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                out_t = torch._foreach_mul(x,1)    
                                     
            elif mode == 4: # exponential. (stable if 0 \le \beta < 1) 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, torch._foreach_mul(in_t, one_minus_beta_t))
                out_t = torch._foreach_div(x,one_minus_beta_t_pow_t) 

                                   
            elif mode == 5: # uniform (constant as k \to infinity or as \beta \to 1)
            
                # forward:
                torch._foreach_mul_(x, t-1)
                torch._foreach_add_(x, in_t)
                torch._foreach_div_(x, t)
                out_t = torch._foreach_mul(x,1) 
                
                
            elif mode == 6: # hybrid (stable if 0 \le \beta < 1)
                # useful for smoothing/averaging,
                # trusts memory as t increases
                # forward:
                # ct = beta_t.mul((1-(1/t)))
                ct = beta_t.mul((t-1)/t)
                in_t = torch._foreach_mul(in_t, 1-ct)
                #
                torch._foreach_mul_(x, ct)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x,1)         
       
            elif mode == -1: # exponential. (as beta_t -> 1) 
                # often: use mode = 1 instead of this. 
                # forward:
                torch._foreach_mul_(x, beta_t)
                torch._foreach_add_(x, in_t)
                out_t = torch._foreach_mul(x,1)                                           

        # MIX: 
        # shelve (enhanced smoothing (~ stateless, 2nd order) 
        # than the highpass addition below)
        if mix:
            self.shelve(u_t, out_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t)
                 
        '''
        out_t : output at current time, k
        x : updated state for next time, k+1
        '''

        return out_t, x

    def shelve(self, u_t, y_t, mode, beta_t, one_minus_beta_t, beta_t_pow_t, one_minus_beta_t_pow_t):
        ''' enhanced smoothing
        '''
        if not self.tensor_lists:
            if mode in [0, 1, 4]: 
                ((y_t.mul_((beta_t - beta_t_pow_t))).add_(one_minus_beta_t*u_t)).div_(one_minus_beta_t_pow_t)
            elif mode in [0, 1, 4, 3, 5]: 
                (y_t.mul_((beta_t)).add_(one_minus_beta_t*u_t))
            else:
                pass
        else:  # foreach impl.       
            if mode in [0, 1, 4]: 
                torch._foreach_mul_(y_t, (beta_t - beta_t_pow_t))
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
                torch._foreach_div_(y_t, one_minus_beta_t_pow_t)
            elif mode in [3, 5]:                  
                torch._foreach_mul_(y_t, beta_t)
                torch._foreach_add_(y_t, torch._foreach_mul(u_t, one_minus_beta_t))
            else:
                pass

       
    # Exponential Lowpass Parameter
    @torch.no_grad()
    def expbeta(self, t:Tensor|float, dn:int=1, u:float=1, l:float=0):
        '''
        expbeta Exponential beta for Averaging

        Makes (Uniformly Exponentially-Weighted) LPF handle tracking better

        Args:
            t (Tensor | float): current window step
            dn (int, optional): sampling time or size. Defaults to 1.
            u (float, optional): max. freq. param. Defaults to 1.
            l (float, optional): min. freq. param. Defaults to 0.

        Returns:
           beta (float): beta. parameter (t-dn)/t
        '''
        # alpha = 1-beta = freq param = dn/t : from  1 to 0
        
        return (1 - ( l + ( (u-l)*torch.abs(dn/t) ) )).item()
       
       
    # Raised Cosine Window
    @torch.no_grad()
    def rcf(self, freq:Tensor|float, a:float=0.5, rho:float=1, n:int=2, rhlf_win:Optional[bool|int]=True, not_reverse:bool=True):
        '''
        rcf (Digital) Raised Cosine Function (Lowpass Filter)

        Y(z)/X(z) = |H(z)|: Frequency (freq) to LPF Magnitude Mapping

        Args:
            freq (Tensor |float, required): R.H.S unit-circle frequency in [0,1]
            rho (float, optional): damping factor in [0,1]. Defaults to 1 (critically damped binomial filter).
            n (int, optional): all-pole polynomial order, n=1 is first order, n=2 is second-order. Defaults to 2.
            a0 (float, optional): a step input. Defaults to 1.

            freq: (float, required) current normalized frequency value must be in [-1, 1].
            
            n:(float, default=2) filter order. higher order means more smoothing, and delay.
            
            rho: (float, default=0.5) configures the roll-off factor of the filter's gain OR number of oscillating modes (1/rho) in one sinusodial cycle.  
            
            a: (float, default=0.5) configures cosine mangnitude range
            raised cosine (shifted chebyshev), a >= 0.5 -> gain-magnitude in [2a-1, 1]
            
            hann (n=2, a = 0.5) -> gain-magnitude in [0, 1], 
            
            hamming (n=2, a = 0.54) -> gain-magnitude in [4/46, 1], 
            
            pass-through (a = 1 or n=0) -> gain-magnitude in [1, 1], 
            
            chebyshev (cosine), a = 0 -> gain-magnitude in [-1, 1].

            rhlf_win: (bool|int|None, default=True) if True: right-half window. if False: left-half window. if None: full window. if int >=1
            
            not_reverse: (bool, default=True) if False: (set n as int) reverses the window upside down.          
        Returns:
            rcf(f): gain-magnitude, at freq: f
        '''        

        # window's gain mappings
        low = 2*a - 1 # smallest value in the window
        vrng = 1-low # range
        
        assert 0 <= rho <= 1, f"{rho} is not in (0,1]"
        assert 0 <= a <= 1, f"{a} is not in [0,1]."
        assert -1 <= low <= 1, f"{low} is not in [-1,1]."
        
        # window's f-interval mappings
        # [1, N], f:[0,1] -> full [-1, 1]
        # [1, N], f:[0,1] -> left-half [-1, 0]

        if rhlf_win is None: freq = 2*freq - 1 # full window 
        elif not rhlf_win: freq -= 1 # left-half window    
        elif type(rhlf_win) is int:
            # if int: rhlf_win=10, f:[0,1] -> full [-1/10, 1]
            freq = (-1/rhlf_win) + (1+(1/rhlf_win))*freq # a window 
                
        
        # fmax = 1 # max freq.
        fnq = 0.5 # nyquist freq
        fstop = fnq*(1+rho)
        fpass = fnq*(1-rho)
        
        if 0 <= torch.abs(freq) <= fpass:
            if not_reverse: return 1*torch.ones((1,)).item()
            else: return 0*torch.ones((1,)).item()

        elif fpass < torch.abs(freq) < fstop:
            
            # teta = (torch.pi/rho)*(torch.abs(f) - fnq)
            # s = torch.sin(teta)
            # return a0*0.5*(1-s) # n = 2
            
            # 2*phi = teta_plus_halfpi 
            # 2*phi = sc*(teta + fnq*torch.pi)     
            # => sc*(torch.pi/rho)*(torch.abs(f) - fpass)
            # return a0*0.5*(1+c) # n = 2 
            
            phi = 0.5*(torch.pi/rho)*(torch.abs(freq) - fpass)
            cn = torch.cos(phi).pow(n)
            # safeguards: floating-point issues
            if torch.isnan(cn): 
                if not_reverse: cn = low*torch.ones((1,))
                else: cn = 1*torch.ones((1,))
            
            # phi.cos_().pow_(n).mul_(vrng).add_(low)    
            if not_reverse: return (low + (vrng*cn)).item() 
            else: return (1-(low + (vrng*cn))).item() 
        
        else:  # torch.abs(f) > fstop:
            if not_reverse and n > 0: return low*torch.ones((1,)).item()
            else: return vrng*torch.ones((1,)).item()


# Backend    
# The `CommonSets` class defines methods for handling 
# common configurations and computations
class CommonSets():
    """ Commons 
    """
    
    def __init__(self, lpf:LPF, gen_cfg:Props, 
                 misc_cfg:Props, beta_cfg:Props, 
                 rcf_cfg:Props, mdl) -> None:

        self.lpf = lpf
        #
        self.gen_cfg = gen_cfg
        self.eps = gen_cfg.eps
        self.p = gen_cfg.p
        self.maximize = gen_cfg.maximize
        #
        self.misc_cfg = misc_cfg
        self.down = misc_cfg.down
        self.lrlogstep = misc_cfg.lrlogstep
        self.epp = misc_cfg.epp
        self.fwd = misc_cfg.fwd
        #
        self.beta_cfg = beta_cfg
        #        
        self.rcf_cfg = rcf_cfg
        self.mdl = mdl
        #       
        self.est_numels = 0
        
        
                   
    @torch.no_grad()
    def grp_devdt(self, device, dtype):
        fzero = torch.tensor(0., dtype=dtype, device=device)
        fone = torch.tensor(1., dtype=dtype, device=device)
        self.fone = fone
        self.fzero = fzero
        self.dev_lr_init = self.gen_cfg.lr_init*fone
        self.dev_beta_i = self.beta_cfg.beta_i*fone
        self.dev_beta_o = self.beta_cfg.beta_o*fone
        self.dev_beta_lr = self.beta_cfg.beta_lr*fone
        self.dev_eps = self.gen_cfg.eps*fone
        self.dev_epp = self.epp*fone
        self.levels = self.p    
        #
        self.dev_fone = 1*fone
        if not self.down: self.dev_fone.neg_()
              
    @torch.no_grad()
    def log_stats(self, params=None):  
        
        if params is not None:
            self.est_numels = sum(p.numel() for p in params)  
        # logging.
        dtxt = '-'
        eqtxt = "."
        pltxt = "+"
        sptxt = " "
        vsltxt = "|"
        
        infostr = f"AutoSGM info:\t [total params, d={self.est_numels}]"  
           
        if self.rcf_cfg.auto:
            rststr = f"[rcw={self.rcf_cfg.win}, half={self.rcf_cfg.half}, auto={self.rcf_cfg.auto}, upfact={self.rcf_cfg.upfact}, n={self.rcf_cfg.n}, l={2*self.rcf_cfg.a - 1} ]"
        else:
            rststr = f"[rcw={self.rcf_cfg.win}, half={self.rcf_cfg.half}, auto={self.rcf_cfg.auto}, init-width={self.rcf_cfg.width}, up={self.rcf_cfg.upfact}, n={self.rcf_cfg.n}, l={2*self.rcf_cfg.a - 1} ]"
        
        autostr = f"[init_lr={self.gen_cfg.lr_init:.5g}, forward={self.fwd}, step_fact={self.misc_cfg.epp}]"
        
        filtstr = f"[lpfs. [in, out, lr] : {self.beta_cfg.beta_i:.9g}, {self.beta_cfg.beta_o:.9g}, {self.beta_cfg.beta_lr:.9g}]"
        
        othstr = f"[eps: {self.gen_cfg.eps:.4g},]"
        
        strlogs = []
        strlogs.append(infostr)
        strlogs.append(autostr)
        strlogs.append(rststr)
        strlogs.append(filtstr)
        strlogs.append(othstr)   
        maxlen = 0
        for astr in strlogs:
            maxlen = max(len(astr), maxlen)
        
        print(f"{pltxt}{dtxt*(maxlen+2)}{pltxt}") # account for two spaces
        for i, astr in enumerate(strlogs):
            splen = ((maxlen) - len(astr))
            if i > 0: print(f"{pltxt}{eqtxt*(maxlen+2)}{pltxt}")
            print(f"{vsltxt} {astr}{sptxt*splen} {vsltxt}")    
        print(f"{pltxt}{dtxt*(maxlen+2)}{pltxt}\n")
                   
    # Trace Gradient inputs for each level (back-prop/autodiff)
    def grader(self, step, pl, grad_lev, grad_smth_lev, grad_in_lev_t1):
        '''
        trace gradient values for all levels (back-prop/auto-diff)
        '''            
        if not self.lpf.tensor_lists:
            # get nn. gradient for current level
            gin_t = 1*grad_lev[pl]              

            # compute bt_in.
            betain = self.bt_compute(step, grad_smth_lev[pl], gin_t, grad_in_lev_t1[pl]) 

            # smooth gradient input [lowpass]  
            m_t, grad_smth_lev[pl] = self.lpf.compute(in_t=gin_t, x=grad_smth_lev[pl], beta=betain, step=step, mode=1)

            # flip sign, if maximizing. 
            if pl == 0 and (self.down or self.maximize): 
                m_t.neg_()

        
        elif self.lpf.tensor_lists: # operating on lists
            # get gradient for this 'pl' level from all nn layers
            gpl = [ allist[pl] for allist in grad_lev]
            gsmthpl = [ allist[pl] for allist in grad_smth_lev]
            gpl1 = [ allist[pl] for allist in grad_in_lev_t1]

            # get nn. gradient for current level                           
            gin_t = torch._foreach_mul(gpl,1)
            
            # compute bt_in.
            self.bt_compute(step, gsmthpl, gin_t, gpl1)            
            betain = self.dev_beta_i   

            # smooth gradient input [lowpass]            
            m_t, gsmthpl = self.lpf.compute(in_t=gin_t, x=gsmthpl, beta=betain, step=step, mode=6)
                       
            # flip sign, if maximizing.   
            if pl == 0 and self.down or self.maximize: 
                torch._foreach_neg_(m_t)
                                  
        return m_t, gin_t
  
    # Back Trace SGM input for next iteration
    def back_grade(self, rpl, grad_in_t, g_t):
        '''
        store current SGM (smooth) gradient input for the next iteration
        '''
        if not self.lpf.tensor_lists:
            grad_in_t[rpl].mul_(0).add_(g_t[rpl])
        else:
            grad_in_rpl = [allist[rpl] for allist in grad_in_t]
            torch._foreach_zero_(grad_in_rpl)
            torch._foreach_add_(grad_in_rpl, g_t[rpl])
                   
    def bt_compute(self, step, m_t, g_t, g_to):# -> Tensor | Any | List[Tensor]:
        '''
        computes an iteration-dependent lowpass parameter 
        '''
        # input lowpass parameter estimation
        
        if self.beta_cfg.auto and step > 1:
            if not self.lpf.tensor_lists:
                num_val = g_t.T.mm(g_t-g_to)   
                den_val = m_t.T.mm(g_to)
                # approxs.
                num_val.div_((num_val + den_val).add_(self.dev_eps))
                return num_val[0].abs()
            
            elif self.lpf.tensor_lists:
            
                pass
        else:
            return self.dev_beta_i

    # Compute lr (gain)
    def lr_compute(self, rpl, step, m_t, g_t):# -> Tensor | Any | List[Tensor]:
        '''
        computes an iteration-dependent learning rate that approximates an optimal choice of step-size.
        '''
        # learning rate estimation
        # linear correlation estimate update  
        # can we estimate this more accurately?

        if not self.lpf.tensor_lists:
            A = self.mdl.A
            numa_val = m_t[rpl].T.mm(g_t[rpl])
            dena_val = m_t[rpl].T.mm(A.mm(m_t[rpl]))

            numb_val = m_t[rpl].T.mm(A.mm(g_t[rpl]))
            denb_val = m_t[rpl].T.mm((A*A).mm(m_t[rpl]))        
            # approxs.
            numa_val.div_(dena_val.add(self.dev_eps))
            numb_val.div_(denb_val.add(self.dev_eps))
            

            if self.dev_beta_lr in [0, 1]:
                bt_lr = self.dev_beta_lr 
            else:   
                pt = (self.dev_epp)/(step + (self.dev_epp-self.fone))
                bt_lr = self.dev_beta_lr*(1-pt)

            lrat = (bt_lr*numa_val) + ((1-bt_lr)*numb_val) 

            # abs. val projection. to ensure positive rates.
            alpha_hat_t = lrat #.abs()    

        elif self.lpf.tensor_lists:
                   
            pass
 
        return alpha_hat_t 

    # Integrator
    def integrator(self, w_t, m_t, rpl, alpha_hat_t, a_t=1):
        '''
        a state-space function: digital integration
        '''
        # [optional] rcf smoothing, a_t 
        if not self.lpf.tensor_lists:                   
            w_t[rpl].addcmul_(m_t[rpl].mul(a_t), alpha_hat_t, value=self.dev_fone)
        elif self.lpf.tensor_lists:
            wrpl = [ allist[rpl] for allist in w_t]
            #
            torch._foreach_mul_(m_t[rpl], alpha_hat_t)
            torch._foreach_mul_(m_t[rpl], a_t*self.dev_fone)
            torch._foreach_add_(wrpl, m_t[rpl])

    # Smooth/Averaged output
    def smooth_avg_out(self, step, rpl, w_t, w_smth):
        '''
        Smooth/Averaged output
        '''
        if rpl == 0:
            # smooth out/average. [lowpass]
            if self.dev_beta_o > 0:                    
                
                beta_o = self.dev_beta_o
                
                if not self.lpf.tensor_lists: 
                    if isinstance(w_t, list):
                        wrplin = 1*w_t[rpl]
                    else:
                        wrplin = 1*w_t

                    wst, w_smth[rpl] = self.lpf.compute(in_t=wrplin, x=w_smth[rpl], beta=beta_o, step=step, mode=6)
                    param_val = 1*wst
                    # param_val = 0.1*wst + 0.9*wrplin

                else:
                    wrpl = [ allist[rpl] for allist in w_t]          
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    wrplin = torch._foreach_mul(wrpl, 1)

                    wst, wsmthrpl = self.lpf.compute(in_t=wrplin, x=wsmthrpl, beta=beta_o, step=step, mode=6)
                    param_val = torch._foreach_mul(wst, 1)
                    
            else:
                if isinstance(w_t, list):
                        wrplin = 1*w_t[rpl]
                else:
                        wrplin = 1*w_t

                if not self.lpf.tensor_lists: 
                    w_smth[rpl].mul_(0).add_(wrplin)
                    param_val = w_smth[rpl] 
                else:
                    wrpl = [ allist[rpl] for allist in w_t]          
                    wsmthrpl = [ allist[rpl] for allist in w_smth]
                    
                    torch._foreach_zero_(wsmthrpl)
                    torch._foreach_add_(wsmthrpl, wrpl)
                    param_val = wsmthrpl  
        else:
            param_val = None 
        
        return param_val                             
                    
    # Pass state (instantaneous/averaged) to network
    def pass_to_nn(self, rpl, param, w_out):
        '''
        Copy state (instantaneous/averaged) to neural network's placeholder.
        '''
        if rpl == 0:
            if not self.lpf.tensor_lists:
                if isinstance(w_out, list):
                    param.mul_(0).add_(w_out[rpl])
                else:
                   param.mul_(0).add_(w_out) 
            else: 
                if isinstance(w_out[0], list):
                    wrplin = [ allist[rpl] for allist in w_out]  
                else:
                    wrplin = w_out                   
                torch._foreach_zero_(param)
                torch._foreach_add_(param, wrplin)   
   

    # RCF computation
    def rcf_cmp(self, step, cfg):
        '''
        for current step, returns the modulation value of the configured raised cosine window
        '''
        a_t = 1
        if cfg.win:           
            fone = torch.tensor(1., dtype=step.dtype, device=step.device)
        
            # RCF
            # self.epoch_movwin, first moving window width,  epochs >=1
            # self.movwin_upfact, moving window width upsampling factor
            
            #   
            if cfg.half is None:
                maxiters = math.ceil((1)/(self.gen_cfg.lr_init))
            else:
                maxiters = math.ceil((2)/(self.gen_cfg.lr_init))

            
            denm = maxiters*(cfg.upfact**(cfg.cnt-1))
            tc = ((step-cfg.last_t) % denm ) + 1
            tc_end = denm
            fc = fone.mul(tc/denm) 
            
            #
            a_t = self.lpf.rcf(freq=fc, a=cfg.a, rho=cfg.rho, n=cfg.n, rhlf_win=cfg.half)      
            #         
            if tc == torch.floor(fone.mul(tc_end)):
                cfg.cnt += 1
                cfg.last_t = step+1  
  
        return a_t
    
    # Log LR or SS                    
    def logginglr(self, rpl, lrm, alpha_hat_t):
        '''
        logs the learning-rate for each parameter
        '''
        if not self.lpf.tensor_lists:
            if self.lrlogstep:
                # we want to do this per step
                lrm[rpl].mul_(0).add_(alpha_hat_t)
          
        elif self.lpf.tensor_lists:            
            if self.lrlogstep:
                lrmrpl = [ allist[rpl] for allist in lrm]
                # we want to do this per step
                torch._foreach_zero_(lrmrpl)
                torch._foreach_add_(lrmrpl, alpha_hat_t)
                           
    # Projection
    @torch.no_grad()
    def project(self, rpl, step, param, w_t, w_smth):
        
        # smooth-projection
        if not self.lpf.tensor_lists: 
            # proj.
            param_val = self.smooth_avg_out(step, rpl, w_t[rpl].relu(), w_smth)

            self.mdl.lmda = 1/(self.mdl.ones_vec.T.mm(self.mdl.M.mm(param_val)))

        else:
            wrpl = [ allist[rpl] for allist in w_t]          
            wsmthrpl = [ allist[rpl] for allist in w_smth]
    
        # pass update to the neural network's placeholder.
        self.pass_to_nn(rpl, param, w_t[rpl]) # self.mdl.param_u

        self.pass_to_nn(rpl, self.mdl.param_c, param_val.mul(self.mdl.lmda))

        

# PyTorch Front   
class AutoSGM(Optimizer):
    
    r""".. _AutoSGM: for Quadratic Functionals
        paper link here.
    
    """.format(maximize=_maximize_doc, foreach=_foreach_doc, differentiable=_differentiable_doc) + r"""
    Example:
        >>> # xdoctest: +SKIP
        >>> from opts.autosgml import AutoSGM
        >>> optimizer = AutoSGM(model.parameters(), weight_decay=5e-4)
        >>> ....
        >>> ....
        >>> optimizer.zero_grad()
        >>> loss_fcn(model(input), target).backward()
        >>> optimizer.step()
        
        .. note::
            Below is just one implementation. There can be any number of specialized implementations, the structure or idea remains the same.
        .. math::
            \begin{aligned}
            v_t &= E{g_t}
            x_t &= x_{t-1} - \alpha_{t}*F_{t}*v_{t}
            w_t &= E{x_t}
            \end{aligned}
            
        where :math:`w`, denote the parameters to adapt, :math:`g` is its gradient, :math:`F*v`, is its smooth gradient by lowpass-filtering.
    """
    
    def __init__(self, params, mdl, *, 
                 lr_mode=(True,1),
                 lr_init=1e-3, eps=1e-15, 
                 beta_cfg=(True, 0., 0.9, 1),
                 rcf_cfg=((False), (True,True), (1, 1, 0)), 
                 loglr_step:Optional[bool]=None,
                 maximize:bool=False, foreach:bool=False):  
        """
        Implements the Stochastic Gradient Method with approximations of an automatic, optimal learning rate function for quadratic minimization.
        
        AutoSGM is a unified learning framework for the gradient method.
        
        Args:
         `params` (`iterable`): iterable of parameters to optimize or dicts defining parameter groups.
         
         `mdl` (`object`) model object with `A`, `b`, `ones_vec`, `lmda`, `param_u`, `param_c` as properties. `params` must be the iterable of `param_u` in `mdl`.

         `lr_mode` (`tuple`, optional). (forward lr computation, stepping factor) (default: `(True, 1)`)
             
         `lr_init` (`float`, optional): used as initial learning rate value, it will be varied iteratively when `autolr` is `True`. (default: `1e-3`).
         
         `eps` (`float`, optional): a small positive constant used to condition/stabilize the gradient normalization operation (default: `1e-8`).

         `beta_cfg` (`tuple`, optional): configures lowpass parameters (default: `(True, 0.9, 0.9, 1)`) => (`auto, beta_i, beta_o, beta_lr`). 
         `auto` (`bool`): auto tune beta_i. (default: `True`).
         `beta_i` (`float`): smoothing lowpass pole param for input gradient, often less or equal to `0.9`. (default:`0.9`).
         `beta_o` (`float`): smoothing lowpass pole param for output projection, often greater or equal to `0`. (default:`0.9`).  
         `beta_lr` (`int`|`float`, optional): use to select the lr approximation (default: `1`). expects `0` or `1` or a real value in (0,1) to switch between two approximations.         

         `rcf_cfg` (`tuple`, optional) use a raised cosine window to spectrally smooth the input. (default: `((False),(True,True),(1, 1, 0))` => `((active), (half_win, auto_init_width), (up, order, min))`
             active (`bool`): use to activate or deactivate the window function. (default: `False`).
             half_win (`bool`|`None`): full or half window (default: `True`). if `True`: right-half window. if `False`: left-half window. if `None`: full window.         
             auto_init_width (`bool`|`int`): automates the initial window width using the `lr_init`. (default: `True`).        
             set as `int` (e.g:`30`) to manually configure the initial window iteration-width (often in epochs). if set as `int`, needs `spe` to convert iterations to epochs.
             up (`int`): window width increase factor >= 1. often set to `1` or `2`. (default:`1`). 
             order (`int`): order of the raised cosine function. (default: `2`). Often `>= 1`.
             min: (`float`)  configures smallest mangnitude range. (default: `0`).
        
        `loglr_step`:(`bool`, optional) how to log learning-rates: per step (True) or per epoch (False) or don't log to make training faster (None)  (default: `None`).
        
        `maximize` (`bool`, optional): whether the objective is being maximized
            (default: `False`).    
                        
        `foreach` (`bool`, optional): fast cuda operation on lists instead of looping. (default: `False`). 

        .. note:: 
                foreach and fused implementations are typically faster than the for-loop, single-tensor implementation.
        """
    
        # Inits: 1 level (Auto)SGM for quadratic minimization 
        # (think of levels like layers)

        # if not hasattr(cfg, 'x'): cfg.x = 0
        # init. lowpass filter obj.
        self.lpf = LPF(foreach=foreach)
        self.nodes = 1

        misc_cfg = Props(down=False, lrlogstep=loglr_step, 
                       fwd=lr_mode[0], epp=lr_mode[1])
            
        beta_cfg = Props(auto=beta_cfg[0],beta_i=beta_cfg[1], beta_o=beta_cfg[2], 
                       beta_lr=beta_cfg[3])
        
        rcf_cfg = Props(win=rcf_cfg[0], half=rcf_cfg[1][0], auto=rcf_cfg[1][1],
                      upfact=rcf_cfg[2][0], rho=1, 
                      n=rcf_cfg[2][1], a=0.5*(rcf_cfg[2][2]+1), 
                      last_t=1, cnt=1,)
        
            
        defaults = dict(p=1, lr_init=lr_init,               
                        eps=eps,         
                        maximize=maximize, foreach=foreach,
                        beta_cfg=beta_cfg, misc_cfg=misc_cfg, 
                        rcf_cfg=rcf_cfg, mdl=mdl,
                        com_sets=None)
        
        
        super().__init__(params, defaults)
        
    def __setstate__(self, state):
        super().__setstate__(state)
        '''
        Set defaults for parameter groups
        '''
        
        for group in self.param_groups:
            group.setdefault('maximize', False)
            group.setdefault('foreach', None)
            group.setdefault('p',1)
            group.setdefault('com_sets', None)   
            
    @torch.no_grad()
    def zero_logged_lr(self):
        """zero lr logged in last epoch.
        
            This will help save time in plotting logged lrs
            since we average the whole lrs logged in an epoch by the total steps in that epoch.
        """
        
        for group in self.param_groups:
          #  lev = group['p']
          for lev in range(1, self.nodes+1):
            lrm_save_list = [
                self.state[p]['levels'][f'{lev}']["lr_m_save"] 
                for p in group['params'] 
                if p.grad is not None
            ]
            torch._foreach_zero_(lrm_save_list)
                
                
    def _init_group(self, group):
        '''
        Inits state of params
        '''
        
        if 'com_sets' not in group or group['com_sets'] is None:
            
            gen_cfg = Props(p=group['p'], lr_init=group['lr_init'], eps=group['eps'], maximize=group['maximize'])

            group['com_sets'] = CommonSets(
                self.lpf, gen_cfg, group['misc_cfg'], group['beta_cfg'], 
                group['rcf_cfg'],  group['mdl']
            ) 
        
        com_sets = group['com_sets']
        has_sparse_grad = False       
        
        params_with_grad_list, steps = [], []
        weight_list, weight_smth_list = [], []
        grad_list, grad_smth_list, grad_in_list = [],[],[]
        lr_avga_list, lr_avgb_list, bt_in_list = [],[],[]
        lrm_save_list = []    
            
        for p in group['params']:
            if p.grad is not None:
                params_with_grad_list.append(p)
                if p.grad.is_sparse: has_sparse_grad = True
                
                state = self.state[p]
                # Lazy state init.
                if len(state)==0:
                    
                    state['step'] = torch.tensor(0, dtype=torch.int, device=p.device)
                    
                    state['levels'] = dict()
                    for pl in range(group['p']):
                        lev = pl+1
                        
                        state['levels'][f'{lev}'] = dict()
                        
                        # - for all levels
                        state['levels'][f'{lev}']['grad_smth'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device) 
                        state['levels'][f'{lev}']['grad_in'] = torch.zeros_like(p.real, memory_format=torch.preserve_format, device=p.device)   
                        
                        # if lev == 1:
                        state['levels'][f'{lev}']['grads'] = None
          
                        # if lev == 1:
                        # weight of the network
                        state['levels'][f'{lev}']['weight'] = p.real.clone(memory_format=torch.preserve_format).detach()   
                        state['levels'][f'{lev}']['weight_smth'] = p.real.clone(memory_format=torch.preserve_format).detach()     
    
                        # - lr used.          
                        state['levels'][f'{lev}']["lr_m_save"] = group['lr_init']*torch.ones_like(p, memory_format=torch.preserve_format, device=p.device) # torch.ones((1,), device=p.device)
                    
                state['step'] += 1
                steps.append(state['step'])
                
                # Level Lists for this parameter
                weight_llist, weight_smth_llist = [], []
                grad_llist, grad_smth_llist, grad_in_llist = [],[],[]
                lrm_save_llist = []        
                
                # -  for all levels
                for lev in range(1,group['p']+1):
                  grad_smth_llist.append(state['levels'][f'{lev}']['grad_smth'])
                  grad_in_llist.append(state['levels'][f'{lev}']['grad_in'])
                  grad_llist.append(p.grad)
                    
                  weight_llist.append(state['levels'][f'{lev}']['weight'])
                  weight_smth_llist.append(state['levels'][f'{lev}']['weight_smth'])      
                         
                  # - (history stores, mean and second moment for alpha_hat_t.)
                  lrm_save_llist.append(state['levels'][f'{lev}']['lr_m_save'])       

                # List of Level Lists for each 
                # parameter with a gradient in the ANN.
                weight_list.append(weight_llist)
                weight_smth_list.append(weight_smth_llist)
                #
                grad_list.append(grad_llist)
                grad_in_list.append(grad_in_llist)
                grad_smth_list.append(grad_smth_llist)
                  
                lrm_save_list.append(lrm_save_llist)
        
        pplists = [params_with_grad_list, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_in_list, lrm_save_list]
        
        return com_sets, has_sparse_grad, pplists, steps
        
    @torch.no_grad()
    def step(self, rank=0, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (Callable, optional): A closure taht reevaluates the model and returns the loss
            
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        # lr_lists = []
        for group in self.param_groups:
                        
            com_sets, has_sparse_grad, \
            pplists, steps = self._init_group(group)
            
            sgm(com_sets, steps, pplists,
                has_sparse_grad = has_sparse_grad,
                foreach=group['foreach'], 
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None), rank=rank)
            
            # update state
            pass
        
        return loss
    
def sgm(com_sets:CommonSets, steps:List[Tensor], 
        pplists:List[List[List[Optional[Tensor]]]],*,
        has_sparse_grad:bool=None,
        foreach:Optional[bool]=None,
        grad_scale:Optional[Tensor]=None,
        found_inf:Optional[Tensor]=None, rank=0):
    
    r""" Functional API performing the SGM algorithm computation
    
    See :class:`~torch.optim.SGM` for details.
    
    """
    # PyTorch's JIT scripting issues (Conditionals, Optionals)
    # logic to use multi_tensor: foreach, fused or single_tensor
    if foreach is None: foreach = False
            
    if foreach and torch.jit.is_scripting():
        raise RuntimeError('torch.jit.script not supported with foreach ops.')

    func = _single_tensor_sgm

    func(com_sets, steps, pplists,
        has_sparse_grad=has_sparse_grad,
        grad_scale=grad_scale,
        found_inf=found_inf, rank=rank)

def _single_tensor_sgm(com_sets:CommonSets, steps: List[Tensor], 
        pplists:List[List[List[Optional[Tensor]]]],*,has_sparse_grad:bool,       
        grad_scale:Optional[Tensor],
        found_inf:Optional[Tensor], rank=0):
    
    ''' Typical for loop implementation
    '''
    
    assert grad_scale is None and found_inf is None
    
    params, weight_list, weight_smth_list, grad_list, grad_smth_list, grad_in_list, lrm_save_list = pplists
    
    dtype = params[0].dtype
    device= params[0].device
    
    com_sets.grp_devdt(device,dtype)
    levels = com_sets.p
    a_t = com_sets.rcf_cmp(steps[0], com_sets.rcf_cfg)    
        
    # LOG.
    if rank==0 and steps[0] == 1: com_sets.log_stats(params)    
                
    for i, param in enumerate(params):
        step = steps[i]

        w_t = weight_list[i]
        w_smth = weight_smth_list[i]

        grad = grad_list[i]
        grad_smth = grad_smth_list[i]
        grad_in_t = grad_in_list[i]
       
        lrm = lrm_save_list[i]
        
        # handle if complex parameters
        if torch.is_complex(param):
            param = torch.view_as_real(param)
        
        pl, rpl = levels-1, levels-1

        # - trace gradient
        smthval, val = com_sets.grader(step, pl, grad, grad_smth, grad_in_t)
        m_t, g_t = [smthval], [val]       
        #::end trace

        # back trace for next iteration
        com_sets.back_grade(rpl, grad_in_t, g_t) 

        if com_sets.fwd:
            # compute lr.
            alpha_hat_t = com_sets.lr_compute(rpl, step, m_t, g_t)               
            
            # integrate: state update
            com_sets.integrator(w_t, m_t, rpl, alpha_hat_t, a_t) 
        else:
            # integrate: state update
            alpha_hat_t = lrm[rpl]
            com_sets.integrator(w_t, m_t, rpl, alpha_hat_t, a_t)  

            # compute lr.
            alpha_hat_t = com_sets.lr_compute(rpl, step, m_t, g_t)                 
            
        # smooth projection and pass back to mdl.
        com_sets.project(rpl, step, param, w_t, w_smth)
               
        # log lr
        com_sets.logginglr(rpl, lrm, alpha_hat_t)

        #::end flow

   

AutoSGM.__doc__ = ""

