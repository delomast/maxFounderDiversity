import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path


import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from aipy.lssloc import LSSLNet, QuadObj
from aipy.asgmlin import AutoSGMLin
from aipy.asgm import AutoSGM
from datapy.popdataloader import PopDatasetStreamerLoader




# GLOBAL configs
# print(Path.cwd())
SERVER_ROOT = Path(__file__).parents[0]
SCRATCH_FOLDER = "scratch"
DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
# print(server_root)
# print(data_root)

# search+select .frq files in scratch
POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")

N_EFF = len(POP_FILES)
# N_EFF = 4
USE_CUDA = False
MAX_BATCHSIZE = 100
MAX_EPOCHS = 1
ERR_OPT_ACC = 1E-8 # 1E-5, 1E-8, 1E-10
LIN_OPTIM_CHOICE = True # 1: LIN | 0: GEN 
STREAM_LEARN = False

if STREAM_LEARN:
  MAX_STEPS = MAX_BATCHSIZE
  # LIN_OPTIM_CHOICE = True
else:
  if LIN_OPTIM_CHOICE:
    MAX_STEPS =  2*(N_EFF**2)
  else: MAX_STEPS = 2500

# buffer_size: data size < actual data size to load into mem. 
# batch_size: selected data size per batch <= buffer_size
# onesample_size: size of one data-point >= 1, < buffer_size
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


def trainer(A, mdl, USE_CUDA, MAX_STEPS, ERR_OPT_ACC, LIN_OPTIM_CHOICE, LOSS_LIST, FC_LOSS_LIST):
  # 0. get data matrix
  if USE_CUDA: 
    A = A.to('cuda',dtype=torch.float32) 
    
  # accept data
  mdl.data_matrix(A)
  # print(mdl.A)
  print()
  
  for k_id in range(MAX_STEPS):
    # 1. inference: forward pass: linear layer
    y = mdl()    
    # output head
    p = mdl.csoftmax(y)
        
    # gradin = (mdl.A.mm(y) - mdl.b).mm(mdl.x.T)   
    
    loss = mdl.criterion(y,mdl.A,mdl.b)
    fcloss = mdl.fcloss(loss.item())
    
    LOSS_LIST.append(loss.item())
    FC_LOSS_LIST.append(fcloss.item())

    # 2. learning: backward pass
    # - zero gradient
    # - backpropagate: compute gradient
    mdl.learner.zero_grad(set_to_none=True) # mdl.weight.grad = None 
    loss.backward()
    # opt. step : learn weight matrix
    if LIN_OPTIM_CHOICE:
      step, lrks, betain_ks = mdl.learner.step(1*mdl.x,1*mdl.A)
    else:
      _, step = mdl.learner.step()
      
    mdl.x.copy_(p.detach())
    
    # print(f"loss:{loss.item()}")
    # print(mdl)  
    # print(f"latent output vector: {y.T}")
    # print()
    # print(f"output belief vector: {p.T}")
    print(f"Step: {int(step)}")
    
    # 3. stopping criterion.
    # gradnorm = torch.norm(mdl.weight.grad)
    # print(f"gradient norm: {gradnorm}")
    print(f"fractional loss change: {fcloss}")    
    if k_id > 1 and (fcloss < ERR_OPT_ACC): break
  return p


for epoch in range(MAX_EPOCHS):
  ''' EPOCH BEGIN.'''
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
    # test: # A = torch.tensor([[2.5409, -0.0113],[-0.0113, 0.5287]])
    # A = batch[0] # A = (1-(batch[1]))
    A =  (1 - batch[1] + batch[0])/(3)
    
    if STREAM_LEARN:
      #.. learn for each batch stream of data
      p = trainer(A,mdl,
            USE_CUDA, MAX_STEPS, ERR_OPT_ACC, 
            LIN_OPTIM_CHOICE, LOSS_LIST, FC_LOSS_LIST)
      #..

  #.. learn after full pass over data
  if not STREAM_LEARN or (STREAM_LEARN and epoch>0):
    p = trainer(A,mdl,
            USE_CUDA, MAX_STEPS, ERR_OPT_ACC, 
            LIN_OPTIM_CHOICE, LOSS_LIST, FC_LOSS_LIST)
  #..
            
  ''' EPOCH END.'''
  walltime = (time.time() - walltime)/60 
  print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
  data_ldr.close()
  data_ldr.batches = b_idx+1
  print("End epoch.")
  
  # dwp
  # p = mdl.Di.mm(mdl.weight.mm(mdl.x)).softmax(dim=0)
  ''' PART 2: CHOOSE POPULATIONS. '''
  # sort contributions 
  p_numpy = p.detach().numpy(force=True).flatten()
  phat, pop_sortidxs = p.detach().sort(dim=0,descending=True)
  # cummulative sum of sorted contributions and its first derivative.
  z = (phat.cumsum(dim=0)).numpy(force=True).flatten()
  dz = z*(1-z)

  pop_sortidxs = pop_sortidxs.numpy(force=True).flatten()
  phat = phat.numpy(force=True).flatten()
  df1 = pd.DataFrame({"idx": pop_sortidxs, "p":phat})
  # get population ids according to sorted contribs.
  # combspop = []
  # for p_idx in range(n): combspop.append(pop_sortidxs[:p_idx+1])
  combspop = [pop_sortidxs[:p_idx+1] for p_idx in range(n)]
  df2 = pd.DataFrame({"combs_idx": combspop, "z":z, "dz": dz})
  
  # choose: find inflection points (logistic curve approx.)
  klow = np.where((z>=1/2))[0][0]
  kupp = np.where((z>=1/2)&(dz<1/6))[0][0]
  
  # get population names according to sorted contribs.
  pop_names = [POP_FILES[p_id].split("\\")[-1].split("_af_")[0] for p_id in pop_sortidxs]
  combspop_nms = [pop_names[:p_idx+1] for p_idx in range(n)]
  
  
  print(); print(df1); print(df2)
  print(f"Estimated optimal foundation set: Choose combination in k = {klow+1} -> {kupp+1}.")
  print(f"k={klow+1}, {combspop[klow]} => {combspop_nms[klow]}")
  print(f"k={kupp+1}, {combspop[kupp]} => {combspop_nms[kupp]}")
  pass

  ''' PLOTS. ''' 
  # plot of pop_comb_idx, k=1:n vs z and dz, draw shaded lines on kl-kh boders, fill between
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  import matplotlib.animation as animation
  from matplotlib import gridspec
  from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize

  cmd_exists = lambda x: shutil.which(x) is not None
  if cmd_exists('latex'):
      plt.rcParams.update({
          "text.usetex": True,
          "font.family": "sans-serif", # serif
          "font.sans-serif": "Helvetica", # Times, Palatino, Computer Modern Roman
      })
  else:
      plt.rcParams.update({
          "text.usetex": False,
          "mathtext.fontset": "cm",
      })
  plt.rcParams["animation.html"] = "jshtml"
  plt.rcParams['figure.dpi'] = 300  
  
  x = np.arange(n)
  # ctrb_xticks = [r"$\mathrm{\mathbf{P}_{"+f"{p_idx}"+r"}}$" for p_idx in range(n)]    
  ctrb_xticks = [r"$\mathrm{\mathbf{P}_{"+f"{p_idx}"+r"}}$" for p_idx in pop_sortidxs]  
  pop_xticks = [r"$\mathcal{P}_{"+f"{p_idx+1}"+r"}$" for p_idx in range(n)]  
  colors2 = [
    'green', 
    # 'white',
    'pink',
    'red',
    'white',
    'orange',
    'green',
    'white',    
    'yellow',
    'red',
    'blue',
    'purple',
    'black',    
  ]
  colors = [
    'red', 
    # 'white',
    'pink',
    'green',
    'white',
    'orange',
    'red',
    'white',    
    'yellow',
    'green',
    'blue',
    'purple',
    'black',    
  ]
  # cmap = ListedColormap(colors)
  cmap = LinearSegmentedColormap.from_list("cbcmap", colors)
  # cmapstr = "hsv"
  # cmap = mpl.colormaps[cmapstr]
  # cmap = mpl.colormaps[cmapstr].reversed() #.resampled(50)
  figsz = (max(3,n/3), 2)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].bar(ctrb_xticks,phat, color=cmap(phat))
  plt.xticks(x, ctrb_xticks, rotation=60)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{Populations}},~\mathrm{\mathbf{P}}_i$")
  ax[0].set_ylabel(r"$\mathrm{\mathsf{optimum~relative}}$"+"\n"+"$\mathrm{\mathsf{contributions}},~p_i$")
  # plt.legend()
  figpath = str((SERVER_ROOT / f"static/trainplts/relctrbs_sslplot.png").resolve())
  plt.savefig(figpath, dpi=300)
  plt.close(fig)
  
  figsz = (max(3,n/3), 3)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,z,label=r"$\mathsf{z}_k$",marker='o',markersize=2.4)
  ax[0].plot(x,dz,label=r"$d\mathsf{z}_k$",marker='o',markersize=2.4)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{optimal~population~sets/combinations}},\mathcal{P}_k$")
  ax[0].set_ylabel(r"$\mathrm{\mathsf{relative~genetic~variation}}, z_k$")
  ax[0].axvspan(klow,kupp, edgecolor='k', facecolor='tab:green', alpha=0.2, linewidth=1)
  ax[0].annotate(r'$\mathcal{P}^\star$',xy=(klow,z[klow]),xytext=(klow+(kupp-klow)/2,z[klow]))
  plt.xticks(x, pop_xticks, rotation=60)
  plt.legend()
  figpath = str((SERVER_ROOT / f"static/trainplts/popchoice_sslplot.png").resolve())
  plt.savefig(figpath, dpi=300)
  # plt.show()
  plt.close(fig)

  x = np.arange(len(LOSS_LIST))
  lloss = np.array(LOSS_LIST)
  lfcloss = np.array(FC_LOSS_LIST)
  figsz = (4, 3)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,lloss,label=r"$\mathsf{l}_t$",marker='o',markersize=0.4)
  ax[0].plot(x,lfcloss,label=r"$\Delta\mathsf{l}_t$",marker='o',markersize=0.4)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$")
  ax[0].set_ylabel(r"$\mathrm{\mathsf{loss}}$")
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  
  plt.legend()
  figpath = str((SERVER_ROOT / f"static/trainplts/loss_plot.png").resolve())
  plt.savefig(figpath, dpi=300)
  plt.close(fig)
  # plt.show()