import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import json
import collections

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from ssldefs import *

# print(Path.cwd())

# set `noPlots` back to `True` for production tests
cfgs = {}
cfgs["MAX_BATCHSIZE"] = 512
cfgs["USE_CORR"] = True
cfgs["NO_MAXSTEPS"] = True
cfgs["MAXSTEPS"] = 1000
cfgs["noPlots"] = True
cfgs["debug"] = False

cfgs["S_PLOT_PATH"] = f"cmdlogs"
os.makedirs(cfgs["S_PLOT_PATH"], exist_ok=True)
shutil.rmtree(cfgs["S_PLOT_PATH"])

SCRATCH_FOLDERS = ["alle_frq_dirs/test_af",  "alle_frq_dirs/sthd_af"]
for dir in SCRATCH_FOLDERS:
  # cfgs["S_PLOT_PATH"] = f"static/svdirs/dev-session/{dir}"
  cfgs["S_PLOT_PATH"] = f"cmdlogs/{dir}"
  k_rec, summ = rdim_opt(cfgs, SCRATCH=dir)
  pass

  

  

  
  
  
  
  
