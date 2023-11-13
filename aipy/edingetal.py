import os,sys, shutil, random, copy, json
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn

from aipy.asgm_quadl import AutoSGMQuad
from aipy.lmdl import LNet



def dir_sel(A):
  n = A.shape[0]

  USE_CORR=True
  USE_CUDA=False
  NO_MAXSTEPS=False
  MAX_STEPS=100
  ERR_OPT_ACC=1e-15
  # instantiate model
  mdl = LNet(dim=n)
  if USE_CUDA: mdl = mdl.cuda()
  mdl.set_learner(AutoSGMQuad(mdl.parameters(), usecuda=USE_CUDA) )
  mdl.data_matrix(A, use_corr=USE_CORR)
  
  zerocnt = 0
  for k_id in range(MAX_STEPS):
    # forward pass:
      
    # output head
    cn = 1*mdl.weight
    # cost eval.
    cost = mdl.quadcost(cn)
    delta_cost = mdl.delta_cost(cost.item())
    
    # backward pass: 
    # zero and backpropagate: compute current gradient
    mdl.learner.zero_grad(set_to_none=True)
    cost.backward()
    # opt. step, learn weight matrix
    step, lrks, betain_ks = mdl.learner.step(mdl.A,mdl.b)
        
    if (NO_MAXSTEPS and (k_id > 1) and (delta_cost < ERR_OPT_ACC)) or (zerocnt > 4): 
      print(f"Steps:{k_id+1}")
      break
  
  cd = (mdl.ones_vec.T.mm(cn))
  dir_c = cn/(cd)
  dir_f = 0.5*1/(cd)
  
  c_sstar, pop_id = dir_c.sort(dim=0, descending=True)
  
  secs = torch.clamp_min(c_sstar,torch.tensor(0,dtype=float))
  zid = torch.where(secs==0)
  try:
    if len(zid) != 0 or zid[0].numel() != 0:
      ssecs = secs[0:zid[0][0]]
      secid = pop_id[0:zid[0][0]]
    else:
      ssecs = secs
      secid = pop_id
  except:
    ssecs = secs
    secid = pop_id

  # c_sstar.softmax(dim=0)
  # print(c_sstar.sum())

  return c_sstar.flatten(), pop_id.flatten(), dir_f.flatten(), ssecs.flatten(), secid.flatten()

def eding_sel(A):
  Ainv = torch.linalg.pinv(A)
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

  return ecs.flatten(), ecid.flatten(), eding_f.flatten(), ssecs.flatten(), secid.flatten()


def save_eding(PLOT_PATH, ecs, ecid, eding_f, secs, secid):
    edresult = {
    'k':len(secs),
    'avg.co-ancestry':eding_f.item(),
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
    print('avg. kinship =', eding_f.item())
    print('Pick k =',len(secs))
    print(f"{('////')*20}")
    # print('selected pop. ids\n', secid.tolist())
    # print('selected pop. ctrbs.')
    # psecs = [ float(f"{el:2.4f}") for el in secs.numpy().tolist()]
    # print(f"{psecs}")
    # print(f"{('////')*20}")