import os,sys, shutil, random, copy, json
from pathlib import Path

import math
import numpy as np

import torch
import torch.nn as nn


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