import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import argparse

import secrets

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from aipy.lssloc import LSSLNet
from aipy.asgmlin import AutoSGMLin
from aipy.asgm import AutoSGM
from datapy.popdataloader import PopDatasetStreamerLoader
from utility import trainer, choose_pops, render_results, web_render_results

import matplotlib
matplotlib.use("Agg")

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")



parser = argparse.ArgumentParser(description="SSL CLI Tool!")

parser.add_argument("-b", "--batchsize", help='batch-size', 
                    type=int, default=1)
parser.add_argument("-s", "--streamer", help='streaming data?', 
                    type=bool, default=False)
parser.add_argument("-l", "--learner", help='set learning algorithm', 
                    type=str, choices=["linear","generic"],default="linear")

group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("--files", 
                    help='list of source files', 
                    type=argparse.FileType('r'), nargs='+')
group.add_argument("--source_dir", 
                    help='directory path to source files (on disk)', 
                    type=Path)

args = parser.parse_args()


secret_key = secrets.token_hex()
cfgs = {}
cfgs["MAX_BATCHSIZE"] = args.batchsize
cfgs["STREAM_LEARN"] = args.streamer
if args.learner == "linear":
  cfgs["LIN_OPTIM_CHOICE"] = True
else:
  cfgs["LIN_OPTIM_CHOICE"] = False

print(args)

# SERVER_ROOT = Path(__file__).parents[0]
# DATA_PATH = (SERVER_ROOT / f"static/cmdsession/{secret_key}/scratch" ).resolve()
# os.makedirs(DATA_PATH, exist_ok=True)
# cfgs["DATA_PATH"] = str(DATA_PATH)

if args.files:
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

  


def run_cmd_ssl(cfgs, POP_FILES):
  
  SERVER_ROOT = Path(__file__).parents[0]
  PLOT_PATH = (SERVER_ROOT / cfgs["S_PLOT_PATH"] ).resolve()
  os.makedirs(PLOT_PATH, exist_ok=True)

  N_EFF = len(POP_FILES)
  USE_CUDA = False
  MAX_BATCHSIZE = int(cfgs["MAX_BATCHSIZE"])
  MAX_EPOCHS = 1
  ERR_OPT_ACC = 1E-8 # 1E-5, 1E-8, 1E-10
  LIN_OPTIM_CHOICE = cfgs["LIN_OPTIM_CHOICE"]  # True: LIN | False: GEN 
    
  
  STREAM_LEARN = cfgs["STREAM_LEARN"]  

  if STREAM_LEARN == "True":
    STREAM_LEARN = True
  else:
    STREAM_LEARN = False

  if STREAM_LEARN:
    MAX_STEPS = MAX_BATCHSIZE
    # LIN_OPTIM_CHOICE = True
  else:
    if LIN_OPTIM_CHOICE:
      MAX_STEPS = 2*(N_EFF**2)
    else: MAX_STEPS = 2500


  # batch_size: selected data size per batch
  data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)
  n = data_ldr.neff


  # instantiate self-supervised model
  mdl = LSSLNet(in_dim=n,out_dim=n)
  if USE_CUDA: 
    mdl = mdl.cuda()
  # mdl.train()  
  if LIN_OPTIM_CHOICE:
    mdl.set_learner(
      AutoSGMLin(mdl.parameters(), usecuda=USE_CUDA)
      )
  else: 
    mdl.set_learner(
        AutoSGM(mdl.parameters(),auto=True, lr_init=1e-1, beta_in_smooth=0.9, usecuda=USE_CUDA)
        )


  for epoch in range(MAX_EPOCHS):
    ''' EPOCH BEGIN: A single pass through the data'''
    LOSS_LIST, FC_LOSS_LIST = [],[]
    print("Epoch: " + str(epoch+1)) 
    walltime = time.time()
    ''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''

    # load dataset, and dataloader
    for (b_idx, batch) in enumerate(data_ldr):
      # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
      # print(batch[0])  # homozygozity
      # print(batch[1])  # heterozygozity
      # print()
      A =  (1 - batch[1] + batch[0])/(3)
  
      if STREAM_LEARN:
      # -* learn for each batch stream of data
        p = trainer(A,mdl,
          USE_CUDA, MAX_STEPS, ERR_OPT_ACC, 
          LIN_OPTIM_CHOICE, LOSS_LIST, FC_LOSS_LIST)
      # -*

    # -* learn after full pass over data
    if not STREAM_LEARN or (STREAM_LEARN and epoch>0):
      p = trainer(A,mdl,
        USE_CUDA, MAX_STEPS, ERR_OPT_ACC, 
        LIN_OPTIM_CHOICE, LOSS_LIST, FC_LOSS_LIST)
    # -*
          
    ''' EPOCH END.'''
    walltime = (time.time() - walltime)/60 
    print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
    data_ldr.close()
    data_ldr.batches = b_idx+1
    print("End epoch.")

    ''' PART 2: CHOOSE POPULATIONS. '''
    p = mdl.Di.mm(mdl.weight.mm(mdl.x)).softmax(dim=0)
    phat,pop_sortidxs, z,dz, klow,kupp, \
    df_relctrbs,df_poploptcombs = choose_pops(POP_FILES, n, p)

    ''' PLOTS. ''' 
    web_render_results(PLOT_PATH,n,phat,pop_sortidxs,
                      z,dz,klow,kupp,LOSS_LIST,FC_LOSS_LIST)
    
  return klow,kupp,df_relctrbs,df_poploptcombs, PLOT_PATH
  
  
  
# Run!
klow, kupp, df_relctrbs, df_poploptcombs, PLOT_PATH = run_cmd_ssl(cfgs,POP_FILES)

print('Done!')
dir_list = os.listdir(PLOT_PATH)
print("Saved Decision Plots to\n'",PLOT_PATH,"'\n")
for dir_file in dir_list: print(dir_file) 
