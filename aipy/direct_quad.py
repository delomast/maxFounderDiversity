import os,sys, shutil, random, copy, json
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn

from aipy.asgm_quadl import AutoSGMQuad
from aipy.lmdl import LNet


def unc_sel(A, USE_CUDA=False, USE_CORR=False, MAX_STEPS=100, NO_MAXSTEPS=False, ERR_OPT_ACC=1e-15, debug=True):
  n = A.shape[0]

  # USE_CORR=True
  # USE_CUDA=False
  # NO_MAXSTEPS=False
  # MAX_STEPS=100
  # ERR_OPT_ACC=1e-15
  # instantiate model
  mdl = LNet(dim=n)
  if USE_CUDA: mdl = mdl.cuda()  
  mdl.data_matrix(A, use_corr=USE_CORR)
  mdl.set_learner(AutoSGMQuad(mdl.parameters(), usecuda=USE_CUDA) )

  zerocnt = 0
  for k_id in range(MAX_STEPS):
    # forward pass:
      
    # output head
    u = 1*mdl.weight
    # u.relu_()
    # cost eval.
    cost = mdl.uncquadcost(u.relu())
    delta_cost = mdl.delta_cost(cost.item())
    
    # backward pass: 
    # zero and backpropagate: compute current gradient
    mdl.learner.zero_grad(set_to_none=True)
    cost.backward()
    # opt. step, learn weight matrix
    step, lrks, betain_ks = mdl.learner.step(mdl.A,mdl.b)
        
    if (NO_MAXSTEPS and (k_id > 1) and (delta_cost < ERR_OPT_ACC)) or (zerocnt > 4): 
      if debug: print(f"Steps:{k_id+1}")
      break
  
  bt_u = (mdl.ones_vec.T.mm(mdl.M.mm(u.relu())))
  
  lmda_opt = 1/bt_u
  y_opt_uc = mdl.M.mm(u.relu()) # no constraint
  y_opt = mdl.M.mm(u.relu())*lmda_opt # with sum to 1 constraint
  print(sum(y_opt))
  return y_opt_uc.detach().clone(), y_opt.detach().clone(), lmda_opt.detach().clone(), cost.detach().clone().item()


def edingetal_sel(A, PLOT_PATH):
  if A.abs().min() < 1e-3: A.abs_()
  Ainv = torch.linalg.inv(A)
  n = A.shape[0]
  ovn=torch.ones((n,1), dtype=torch.float)
  
  cn = Ainv.mm(ovn)
  cd = ovn.T.mm(cn)
  eding_f=1/cd
  eding_c=cn/cd
  ecs, ecid = eding_c.sort(dim=0, descending=True)
  
  secs = torch.clamp_min(ecs,torch.tensor(0,dtype=float))
  zid = torch.where(secs==0)
  try:
    if len(zid) != 0 or zid[0].numel() != 0:
      ssecs = secs[0:zid[0][0]]
      secid = ecid[0:zid[0][0]]
    else:
      ssecs = secs
      secid = ecid
  except:
    ssecs = secs
    secid = ecid

  # ecs.softmax(dim=0)
  # print(ecs.sum())
  

  ecs, ecid, eding_f, secs, secid = ecs.flatten(), ecid.flatten(), eding_f.flatten(), ssecs.flatten(), secid.flatten()


  edresult = {
  'k':len(secs),
  'sel_c_star':secs.tolist(),
  'sel_pop_sort_idxs': secid.tolist(),
  'c_star':ecs.tolist(),
  'pop_sort_idxs': ecid.tolist(),
  }
  
  resultpath = f"{PLOT_PATH}/edingr.json"
  def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()   
      
  with open(resultpath, 'w', encoding='utf-8', errors='ignore') as svfile:
    json.dump(edresult, svfile, default=np_encoder, ensure_ascii=False, indent=4) 
  
  print(f"{('////')*20}")
  print("-- cf.: Eding et.al.--")
  print('Pick k =',len(secs))
  print(f"{('////')*20}")
  # print('selected pop. ids\n', secid.tolist())
  # print('selected pop. ctrbs.')
  # psecs = [ float(f"{el:2.4f}") for el in secs.numpy().tolist()]
  # print(f"{psecs}")
  # print(f"{('////')*20}")