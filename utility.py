import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path


import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from aipy.lssloc import LSSLNet
from aipy.asgmlin import AutoSGMLin
from aipy.asgm import AutoSGM
from datapy.popdataloader import PopDatasetStreamerLoader

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


# GLOBAL configs
# print(Path.cwd())
SERVER_ROOT = Path(__file__).parents[0]
SCRATCH_FOLDER = "scratch"
DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
# print(server_root)
# print(data_root)


''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
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



''' PART 2: CHOOSE POPULATIONS. '''
def choose_pops(POP_FILES, n, p):
  '''
  Choose Populations

  Args:
      POP_FILES (object): opened list of population files
      n (int): number of populations
      p (vector): relative contribution of population in [0,1]

  Returns:
      phat: sorted p in descending order, from highest to lowest p value, 
      pop_sortidxs: actual ids of the populations, sorted from highest to lowest p value, 
      z: cummulative sum of phat, 
      dz: first derivative of z, 
      klow: lower bound estimate of k <= n, 
      kupp: upper bound estimate of k <= n
      df_relctrbs: table of relative contributions (sorted), 
      df_poploptcombs: table of optimum combinations of populations from k=1 to n.
  '''
  # sort contributions 
  # p_numpy = p.detach().numpy(force=True).flatten()
  phat, pop_sortidxs = p.detach().sort(dim=0,descending=True)
  # cummulative sum of sorted contributions and its first derivative.
  z = (phat.cumsum(dim=0)).numpy(force=True).flatten()
  dz = z*(1-z)

  pop_sortidxs = pop_sortidxs.numpy(force=True).flatten()
  phat = phat.numpy(force=True).flatten()
  df_relctrbs = pd.DataFrame({"idx": pop_sortidxs, "p":phat})
  # get population ids according to sorted contribs.
  combspop = [pop_sortidxs[:p_idx+1] for p_idx in range(n)]
  df_poploptcombs = pd.DataFrame({"combs_idx": combspop, "z":z, "dz": dz})

  # choose: find inflection points (logistic curve approx.)
  klow = np.where((z>=1/2))[0][0]
  kupp = np.where((z>=1/2)&(dz<1/6))[0][0]

  # get population names according to sorted contribs.
  pop_names = [POP_FILES[p_id].split("\\")[-1].split("_af_")[0] for p_id in pop_sortidxs]
  combspop_nms = [pop_names[:p_idx+1] for p_idx in range(n)]


  print(); print(df_relctrbs); print(df_poploptcombs)
  print(f"Estimated optimal foundation set: Choose combination in k = {klow+1} -> {kupp+1}.")
  print(f"k={klow+1}, {combspop[klow]} => {combspop_nms[klow]}")
  print(f"k={kupp+1}, {combspop[kupp]} => {combspop_nms[kupp]}")
  
  
  return phat,pop_sortidxs, z,dz, klow,kupp, df_relctrbs,df_poploptcombs

''' PLOTS. ''' 
def render_results(n,phat,pop_sortidxs,z,dz,klow,kupp,LOSS_LIST,FC_LOSS_LIST):
  
  '''
  Plots results of this Learning process.
  '''
  
  # plot of pop_comb_idx, k=1:n vs z and dz, draw shaded lines on kl-kh boders, fill between
  x = np.arange(n)  
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
  figpath = (SERVER_ROOT / f"static/trainplts/relctrbs_sslplot.png").resolve()._str
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
  figpath = (SERVER_ROOT / f"static/trainplts/popchoice_sslplot.png").resolve()._str
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
  ax[0].plot(x,lloss,label=r"$\mathsf{l}_t$",marker='o',markersize=2.4)
  ax[0].plot(x,lfcloss,label=r"$\Delta\mathsf{l}_t$",marker='o',markersize=2.4)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$")
  ax[0].set_ylabel(r"$\mathrm{\mathsf{loss}}$")
  plt.legend()
  figpath = (SERVER_ROOT / f"static/trainplts/loss_plot.png").resolve()._str
  plt.savefig(figpath, dpi=300)
  plt.close(fig)
  # plt.show()
  
  

''' PLOTS. ''' 
def web_render_results(PLOT_PATH,n,phat,pop_sortidxs,z,dz,klow,kupp,LOSS_LIST,FC_LOSS_LIST):
  
  '''
  Plots results of this Learning process.
  '''
  
  # plot of pop_comb_idx, k=1:n vs z and dz, draw shaded lines on kl-kh boders, fill between
  x = np.arange(n)  
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
  figpath = f"{PLOT_PATH}/relctrbs_sslplot.png"
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
  figpath = f"{PLOT_PATH}/popchoice_sslplot.png"
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
  figpath = f"{PLOT_PATH}/loss_plot.png"
  plt.savefig(figpath, dpi=300)
  plt.close(fig)
  # plt.show()
  
  






