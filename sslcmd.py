import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import argparse

import secrets

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from aipy.lssloc import QSSLNet
from aipy.asgm_quad import AutoSGMQuad
from aipy.asgm import AutoSGM
from datapy.popdataloader import PopDatasetStreamerLoader
from utility import trainmdl, get_optimal_sets, render_results, web_render_results

import matplotlib
matplotlib.use("Agg")

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError("Input should be False or True")
    return s == 'True'
  
parser = argparse.ArgumentParser(description="SSL CLI Tool!")

parser.add_argument("-b", "--batchsize", help='batch-size (int)', 
                    type=int, default=1)

parser.add_argument("-s", "--scaler", help='normalize data (bool)', 
                    type=str2bool, default=True)

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--files", 
                    help='list of source files', 
                    type=argparse.FileType('r'), nargs='+')
group.add_argument("--source_dir", 
                    help='directory path to source files (on disk)', 
                    type=Path)
group.add_argument("--coan_matrix", 
                    help='conacestry matrix path (on disk)', 
                    type=Path)

args = parser.parse_args()


secret_key = secrets.token_hex()
cfgs = {}
cfgs["MAX_BATCHSIZE"] = args.batchsize
# cfgs["STREAM_LEARN"] = args.streamer
cfgs["USE_CORR"] = args.scaler
# if args.learner == "quad":
# cfgs["QUAD_OBJ_CHOICE"] = True
print(args)

# SERVER_ROOT = Path(__file__).parents[0]
# DATA_PATH = (SERVER_ROOT / f"static/cmdsession/{secret_key}/alle_frq_dirs/test_af" ).resolve()
# os.makedirs(DATA_PATH, exist_ok=True)
# cfgs["DATA_PATH"] = str(DATA_PATH)

ismatrix = False # logic to know if a pre-computed co-ancestry matrix is used.

ext = str(args.coan_matrix).split('.')

if args.coan_matrix:
  if ext[-1] in ['txt', 'csv']:
    POP_FILES = np.loadtxt(args.coan_matrix)
  if ext[-1] == 'npy':
    POP_FILES = np.load(args.coan_matrix)
  if ext[-1] == 'npz':
    POP_FILES = np.load(args.coan_matrix)['arr_0']
    
  ismatrix = True
elif args.files:
  inpfiles = []
  for file in args.files:
    inpfiles.append(file.name)
  POP_FILES = inpfiles
  print(POP_FILES)
else:
  cfgs["DATA_PATH"] = args.source_dir
  DATA_ROOT = cfgs["DATA_PATH"]
  POP_FILES = glob.glob(f"{DATA_ROOT}/*")
  print(DATA_ROOT)
  
cfgs["S_PLOT_PATH"] = f"static/cmdsession/{secret_key}/trainplts"
# print(ismatrix)
  


def run_cmd_ssl(cfgs, POP_FILES, ismatrix=False):
  
  SERVER_ROOT = Path(__file__).parents[0]
  PLOT_PATH = (SERVER_ROOT / cfgs["S_PLOT_PATH"] ).resolve()
  os.makedirs(PLOT_PATH, exist_ok=True)
  
  if not ismatrix:
    N_EFF = len(POP_FILES)
  else:
    N_EFF = POP_FILES.shape[0]
  USE_CUDA = False
  USE_CORR = cfgs["USE_CORR"]
  MAX_BATCHSIZE = int(cfgs["MAX_BATCHSIZE"])
  MAX_EPOCHS = 1
  ERR_OPT_ACC = 1E-15 # 1E-5, 1E-8, 1E-10
  QUAD_OBJ_CHOICE = True
  MAX_STEPS = (N_EFF**2)

  # batch_size: selected data size per batch
  if not ismatrix:
    data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)
    n = data_ldr.neff
  else:
    n = POP_FILES.shape[0]

  # instantiate self-supervised model
  mdl = QSSLNet(in_dim=n,out_dim=n)
  if USE_CUDA: 
    mdl = mdl.cuda()
  mdl.set_learner(
      AutoSGMQuad(mdl.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA) 
      )

  b_idx = 0
  for epoch in range(MAX_EPOCHS):
    ''' EPOCH BEGIN: A single pass through the data'''
    LOSS_LIST, FC_LOSS_LIST = [],[]
    W_T, G_T = [],[]
    C_T, Y_T, LR_T, BT_T = [],[],[],[]
    
    W_T.append(1*mdl.weight_W.detach().numpy(force=True).flatten())
    print("Epoch: " + str(epoch+1)) 
    walltime = time.time()
    ''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''

    if not ismatrix:
      # load dataset, and dataloader
      for (b_idx, batch) in enumerate(data_ldr):
        A = batch[0] #homozygozity
      
      #.. learn after full pass over data
      loss, c_t, y, alphas, betas = trainmdl(mdl, A,
              USE_CUDA, USE_CORR, MAX_STEPS, ERR_OPT_ACC, 
              QUAD_OBJ_CHOICE, LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
              C_T, Y_T, LR_T, BT_T)  
      
    else:
      b_idx = 0
      A = torch.tensor(POP_FILES,dtype=torch.float)
      if USE_CUDA: A = A.cuda()
      #.. learn after full pass over data
      loss, c_t, y, alphas, betas = trainmdl(mdl, A,
              USE_CUDA, USE_CORR, MAX_STEPS, ERR_OPT_ACC, 
              QUAD_OBJ_CHOICE, LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
              C_T, Y_T, LR_T, BT_T)  
      

    ''' EPOCH END.'''
    walltime = (time.time() - walltime)/60 
    print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
    if not ismatrix:
      data_ldr.close()
      data_ldr.batches = b_idx+1
    print("End epoch.")

    ''' PART 2: CHOOSE POPULATIONS. '''
    results = get_optimal_sets(POP_FILES, n, c_t, ismatrix)

    ''' PLOTS. ''' 
    web_render_results(PLOT_PATH, n, results,
            LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
            C_T, Y_T, LR_T, BT_T)
    
  return PLOT_PATH
  

# Run!
PLOT_PATH = run_cmd_ssl(cfgs,POP_FILES, ismatrix)

print('Done!')
dir_list = os.listdir(PLOT_PATH)
print("Saved Decision Plots to\n'",PLOT_PATH,"'\n")
for dir_file in dir_list: print(dir_file) 
