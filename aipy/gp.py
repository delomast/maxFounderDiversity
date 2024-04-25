import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

from rmbin.utility import *


from aipy.mdls import ILinear
from aipy.qautosgm import AutoSGM

def gp1(args, svlists=None):
    zero_hits = 0

    # mdl: linear model
    mdl = ILinear(dim=args.n, eps=args.err_opt_acc)
    if args.use_cuda: mdl.cuda()
    mdl.data_matrix(args.A, args.use_corr)
    # opt:asgmq
    opt = AutoSGM([mdl.param_u], mdl,
                  lr_mode=(True, 1), beta_cfg=(True, 0.1, 0.1, 1),)

    # init eval: obj
    for t in range(1, args.max_steps+1):
        # learn: optimize params.
        # - backprop.
        opt.zero_grad(set_to_none=True)
        cost_u = mdl.quad_cost_unc(u=mdl.param_u.relu()) # opt.wsmth
        cost_u.backward()
        # - one SGM step and projection
        opt.step()
        
        # eval: obj unconstrained
        dcost_u = mdl.delta_cost_u(cost_u.item())
        # eval: obj constrained
        cost_c = mdl.quad_cost_sum1()
        # eval: metric
        coan_val = mdl.coan_metric()

        if svlists:
            svlists.t.append(t)
            svlists.clmb_t.append(0.5*mdl.lmda.item())  
            svlists.cost_u_t.append(cost_u.item())      
            svlists.dcost_u_t.append(dcost_u.item())
            svlists.cost_c_t.append(mdl.quad_cost_sum1_tf().item())  
            svlists.he_t.append(mdl.coan_metric_tf().item())
            # svlists.lr_t.append(lrt.item())        
            # svlists.bt_t.append(btt.item())

        # stop rule
        if stopping(args, t, dcost_u, zero_hits):
            break

    return mdl, coan_val.item(), (cost_u.item(), dcost_u.item(), cost_c.item())

def stopping(args, t, dcost_u, zero_hits):
    '''
    stopping condition: returns True or False
    '''
    boolval = ((args.no_maxsteps) and (t > 1)) and \
        (dcost_u < args.err_opt_acc) or \
        (zero_hits > 0)

    if boolval and args.debug: 
        print(f"curr_step:{t}")
    
    return boolval


# comparisons
# cvx
# slsqp