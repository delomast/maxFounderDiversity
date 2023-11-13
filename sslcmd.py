import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import argparse

import secrets

import torch
import torch.nn as nn


from utility import *

import matplotlib
matplotlib.use("Agg")


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
  EPS = 1E-15
  QUAD_OBJ_CHOICE = True
  MAX_STEPS = cfgs["MAXSTEPS"]
  NO_MAXSTEPS = cfgs["NO_MAXSTEPS"]
  

  # batch_size: selected data size per batch
  if not ismatrix:
    data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)
    n = data_ldr.neff
  else:
    n = POP_FILES.shape[0]

  # instantiate self-supervised model
  mdl = QSSLNet(in_dim=n,out_dim=n, eps=EPS)
  if USE_CUDA: mdl = mdl.cuda()
  mdl.set_learner(AutoSGMQuad(mdl.Lin_W.parameters(), eps=EPS, usecuda=USE_CUDA) )

  b_idx = 0
  ''' EPOCH BEGIN: A single pass through the data'''
  SVLISTS = dict()
  SVLISTS['cost'] = []
  SVLISTS['dfcost'] = []
  SVLISTS['wt'] = []
  SVLISTS['gt'] = []
  SVLISTS['ct'] = []
  SVLISTS['yt'] = []
  SVLISTS['sst'] = []
  SVLISTS['btt'] = []
  SVLISTS['wt'].append(1*mdl.weight_W.detach().numpy(force=True).flatten())
  
  # print("Epoch: " + str(epoch+1)) 
  print(f"{('~~~~')*20}")
  walltime = time.time()
  ''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''

  if not ismatrix:
    # load dataset, and dataloader
    for (b_idx, batch) in enumerate(data_ldr): pass
    A = batch[0] #homozygozity
  else:
    b_idx = 0
    batch = None
    A = torch.tensor(POP_FILES, dtype=torch.float)
    
  #.. learn after full pass over data
  cost, c_t, y, alphas, betas = trainmdl(mdl, A,
        USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, ERR_OPT_ACC, 
        QUAD_OBJ_CHOICE, SVLISTS)  
  results = get_optimal_sets(POP_FILES, n, c_t, A, ismatrix)
  
  ''' EPOCH END.'''
  walltime = (time.time() - walltime)/60 
  # print(f"\nTotal batches: {b_idx+1}")
  print(f"time elapsed: {walltime:.2f}-mins") 
  if not ismatrix:
    data_ldr.close()
    data_ldr.batches = b_idx+1
  # Print out ...
  print(f"avg. kinship = {results['avg-kinship'].item()}")
  
  ''' PLOTS. ''' 
  with open(f"{PLOT_PATH}/contributions.txt", "w") as out_file:
    out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
    [ out_file.write("\t".join([str(results["pop_sort_idxs"][i]), str(results["c_star"][i])]) + "\n")  
      for i in range(0,len(results["c_star"])) 
    ]     
  if not cfgs['noPlots']:
    web_render_results(PLOT_PATH, n, results, SVLISTS, skip_heavy_plots=False)
  
  #
  ecs, ecid, eding_f, secs, secid = dir_sel(mdl.A)
  save_eding(PLOT_PATH, ecs, ecid, eding_f, secs, secid)
  # print('Done!')
  print(f"{('~~~~')*20}")
  
  ''' PART 2: CHOOSE POPULATIONS. '''
  k_rec = recurse_dim_returns(A, n, results, PLOT_PATH, ERR_OPT_ACC, USE_CORR, USE_CUDA)
  return k_rec
  

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError("Input should be False or True")
    return s == 'True'
  
override = False # production behaviour.
override = True # uncomment for testing/debugging.

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
  cfgs["noPlots"] = args.noPlots

  print(args)
else: 
  ''' 
  Test- override cmdline, note: comment out from first 
  'parser = ...' line to 'print(args)' line
  '''
  cfgs = {}
  cfgs["MAX_BATCHSIZE"] = 1
  cfgs["USE_CORR"] = False
  cfgs["NO_MAXSTEPS"] = True
  cfgs["MAXSTEPS"] = 100
  cfgs["noPlots"] = False
  # coan_matrix = "coan_matrix_files/co_mat.txt"
  # coan_matrix = "coan_matrix_files/co_mat_sthd.txt"
  coan_matrix = "coan_matrix_files/plink_mat.rel"
  files =None
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
  
cfgs["S_PLOT_PATH"] = f"static/svdirs/cmdsession/{secret_key}/"
  
# Run!
k_rec = run_cmd_ssl(cfgs,POP_FILES, ismatrix)
