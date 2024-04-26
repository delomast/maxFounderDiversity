import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import argparse

import secrets

import torch
import torch.nn as nn


from ssldefs import *
  
override = False # production behaviour.
# override = True # uncomment for testing/debugging.

if not override:
  parser = argparse.ArgumentParser(description="SSL CLI Tool!")

  parser.add_argument("-b", "--batchsize", help='batch-size (int)', 
                      type=int, default=1)
  parser.add_argument("-s", "--scaler", help='normalize data (bool)', 
                      type=str2bool, default=True)
  parser.add_argument("-c", "--MAXSTEPS", help="upper limit on the total number of learning iterations (int)", type=int, default=100)
  parser.add_argument("-m", "--NO_MAXSTEPS", help="don't max-out the total number of learning iterations (bool)", type=str2bool, default=True)
  parser.add_argument("--noPlots", help="return text output insted of plots and a .json file (saves time)", action=argparse.BooleanOptionalAction) # requires python 3.9+

  group = parser.add_mutually_exclusive_group(required=True)
  group.add_argument("--files", 
                      help='list of source files', 
                      type=argparse.FileType('r'), nargs='+')
  group.add_argument("--source_dir", 
                      help='directory path to source files (on disk)', 
                      type=Path)
  group.add_argument("--coan_matrix", 
                      help='co-ancestry matrix path (on disk)', 
                      type=Path)

  args = parser.parse_args()
  coan_matrix = args.coan_matrix
  files = args.files
  source_dir = args.source_dir

  cfgs = {}
  cfgs["MAX_BATCHSIZE"] = args.batchsize
  cfgs["USE_CORR"] = args.scaler
  cfgs["NO_MAXSTEPS"] = args.NO_MAXSTEPS
  cfgs["MAXSTEPS"] = args.MAXSTEPS
  cfgs["noPlots"] = args.noPlots
  cfgs["debug"] = False

  print(args)
else: 
  ''' 
  Test => override cmdline, note: comment out from first 
  'parser = ...' line to 'print(args)' line
  '''
  cfgs = {}
  cfgs["MAX_BATCHSIZE"] = 1
  cfgs["USE_CORR"] = False
  cfgs["NO_MAXSTEPS"] = True
  cfgs["MAXSTEPS"] = 1000
  cfgs["noPlots"] = False
  cfgs["debug"] = True
  # coan_matrix = "coan_matrix_files/co_mat.txt"
  # coan_matrix = "coan_matrix_files/co_mat_sthd.txt"
  coan_matrix = "coan_matrix_files/plink_mat.rel"
  files = None
  source_dir = None


secret_key = secrets.token_hex(nbytes=7)

ismatrix = False # logic to know if a pre-computed co-ancestry matrix is used.

ext = str(coan_matrix).split('.')
if coan_matrix:
  if ext[-1] in ['txt', 'csv', 'rel']:
    POP_FILES = np.loadtxt(coan_matrix)
  if ext[-1] == 'npy':
    POP_FILES = np.load(coan_matrix)
  if ext[-1] == 'npz':
    POP_FILES = np.load(coan_matrix)['arr_0']
  # else try loading as txt, if this fails , throw error
    
  ispd = np.all(np.linalg.eigvals(POP_FILES) > 0)
  print('p.d matrix',ispd)
  if not ispd: raise ValueError('Data matrix not positive-definite!')
  ismatrix = True

elif files:
  inpfiles = []
  for file in files:
    inpfiles.append(file.name)
  POP_FILES = inpfiles
  print(POP_FILES)
else:
  cfgs["DATA_PATH"] = source_dir
  DATA_ROOT = cfgs["DATA_PATH"]
  POP_FILES = glob.glob(f"{DATA_ROOT}/*")
  print(DATA_ROOT)
  
# cfgs["S_PLOT_PATH"] = f"cmdlogs/{secret_key}/"
cfgs["S_PLOT_PATH"] = f"cmdlogs"
os.makedirs(cfgs["S_PLOT_PATH"], exist_ok=True)
shutil.rmtree(cfgs["S_PLOT_PATH"])
 
# Run!
k_rec, _ = rdim_opt(cfgs, POP_FILES=POP_FILES, ismatrix=ismatrix)

  

  

  
  
  
  
  
