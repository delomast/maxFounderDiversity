import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path


import torch
import torch.nn as nn

import numpy as np
import pandas as pd

from aipy.lssloc import QSSLNet
from aipy.asgm_quad import AutoSGMQuad
from aipy.asgm import AutoSGM as AutoSGMGen
from datapy.popdataloader import PopDatasetStreamerLoader




# GLOBAL configs
# print(Path.cwd())
SERVER_ROOT = Path(__file__).parents[0]
# SCRATCH_FOLDER = "alle_frq_dirs/test_af"
SCRATCH_FOLDER = "alle_frq_dirs/sthd_af"
DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
# print(server_root)
# print(data_root)

# search+select .frq files in alle_frq_dirs/test_af
POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")

N_EFF = len(POP_FILES)
# N_EFF = 5
USE_CUDA = False
MAX_BATCHSIZE = 512
MAX_EPOCHS = 1
ERR_OPT_ACC = 1E-15 
QUAD_OBJ_CHOICE = True # 1: ~LIN | 0: GEN 
USE_CORR = True

if QUAD_OBJ_CHOICE:
  MAX_STEPS = (N_EFF**2)
else: MAX_STEPS = 2500

# buffer_size: data size < actual data size to load into mem. 
# batch_size: selected data size per batch <= buffer_size
# onesample_size: size of one data-point >= 1, < buffer_size
data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)

n = data_ldr.neff
# instantiate self-supervised model
mdl = QSSLNet(in_dim=n,out_dim=n, eps=ERR_OPT_ACC)
if USE_CUDA: 
  mdl = mdl.cuda() 

if QUAD_OBJ_CHOICE:
  mdl.set_learner(
    AutoSGMQuad(mdl.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA) 
    )
else: 
  # create plots for -2,-3,-4,-5,-6,-7,-8 comparing step-size for iteration dep and constant with respect to performance (c_t converging/y_t converging/loss).
  mdl.set_learner(
      AutoSGMGen(mdl.Lin_W.parameters(),auto=True, lr_init=1e-2, beta_in_smooth=0.9, beta_den=0.99, beta_num=0.99, usecuda=USE_CUDA),
      )


def trainmdl(mdl:QSSLNet, A, USE_CUDA, USE_CORR, MAX_STEPS, 
            ERR_OPT_ACC, QUAD_OBJ_CHOICE, LOSS_LIST, FC_LOSS_LIST,
            W_T,G_T,C_T, Y_T, LR_T,BT_T):
  
  # accept data matrix
  if USE_CUDA: 
    A = A.to('cuda',dtype=torch.float32) 
    
  mdl.data_matrix(A, use_corr=USE_CORR)
  
  for k_id in range(MAX_STEPS):
    # forward pass:
    
    # linear fcn.
    y = mdl.act()   
    # output head
    c_t = mdl.csoftmax(y)
    # cost eval.
    costW = mdl.quadcost(y)
    delta_costW = mdl.delta_cost(costW.item())
    
    # backward pass: 
    # zero and backpropagate: compute current gradient
    mdl.learnerW.zero_grad(set_to_none=True)
    costW.backward()
    # opt. step, learn weight matrix
    if QUAD_OBJ_CHOICE:
      step, lrks, betain_ks = mdl.learnerW.step(mdl.A,mdl.b,mdl.x,c_t)
    else:
      step, lrks, betain_ks = mdl.learnerW.step()
      
    mdl.x = (c_t.detach())
        
    print(f"Step: {int(step)}")
    # 3. stopping criterion.
    W_T.append(1*mdl.weight_W.detach().numpy(force=True).flatten())
    LOSS_LIST.append(costW.item())
    FC_LOSS_LIST.append(delta_costW.item())
    G_T.append(1*mdl.weight_W.grad.detach().numpy(force=True).flatten())
    C_T.append(1*c_t.detach().numpy(force=True).flatten())
    Y_T.append(1*y.detach().numpy(force=True).flatten())
    LR_T.append(1*lrks.numpy(force=True).flatten())
    BT_T.append(1*betain_ks.numpy(force=True).flatten())
    print(f"fractional loss change: {delta_costW}")  
      
    if k_id > 1 and (delta_costW < ERR_OPT_ACC): break
    
  return costW.item(), 1*c_t, 1*y, 1*lrks, 1*betain_ks

def get_optimal_sets(POP_FILES, n, c_t):
  
  c_star, pop_sortidxs = c_t.detach().sort(dim=0,descending=True)
  # cummulative sum of sorted contributions and its first derivative.
  z = (c_star.cumsum(dim=0)).numpy(force=True).flatten()
  dz = z*(1-z)

  pop_sortidxs = pop_sortidxs.numpy(force=True).flatten()
  c_star = c_star.numpy(force=True).flatten()
  df1 = pd.DataFrame({"idx": pop_sortidxs, "c":c_star})
  
  # get population ids according to sorted contribs.
  combspop = [pop_sortidxs[:p_idx+1] for p_idx in range(n)]
  df2 = pd.DataFrame({"combs_idx": combspop, "z":z, "dz": dz})

  # choose: find inflection points (logistic curve approx.)
  klow_id = np.where((z>=1/2))[0][0]
  kupp_id = np.where((z>=1/2)&(dz<1/6))[0][0]
  
  # get population names according to sorted contribs.
  pop_names = [POP_FILES[p_id].split("\\")[-1].split("_af_")[0] for p_id in pop_sortidxs]
  combspop_nms = [pop_names[:p_idx+1] for p_idx in range(n)]
  
  result = {
    'c_star':c_star,
    'pop_sort_idxs': pop_sortidxs,
    'z':z,'dz':dz,
    'k_low':klow_id+1,'k_upp':kupp_id+1,
    'id_k_low':klow_id,'id_k_upp':kupp_id,
    'dataframe_1':df1,'dataframe_2':df2,
    'id_pop_combs':combspop,
    'names_pop_combs': combspop_nms
  }
  return result



for epoch in range(MAX_EPOCHS):
  ''' 1 EPOCH BEGIN.'''
  # per iteration list
  LOSS_LIST, FC_LOSS_LIST = [],[]
  W_T, G_T = [],[]
  C_T, Y_T, LR_T, BT_T = [],[],[],[]
  
  W_T.append(1*mdl.weight_W.detach().numpy(force=True).flatten())
  print("Epoch: " + str(epoch+1)) 
  walltime = time.time()
  
  ''' LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
  # load dataset, and dataloader
  for (b_idx, batch) in enumerate(data_ldr):
    # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
    # test: # A = torch.tensor([[2.5409, -0.0113],[-0.0113, 0.5287]])
    A = batch[0] #xz homozygozity
    
  #.. learn after full pass over data
  loss, c_t, y, alphas, betas = trainmdl(mdl, A,
          USE_CUDA, USE_CORR, MAX_STEPS, ERR_OPT_ACC, 
          QUAD_OBJ_CHOICE, LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
          C_T, Y_T, LR_T, BT_T)
  
  walltime = (time.time() - walltime)/60 
  print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
  data_ldr.close()
  data_ldr.batches = b_idx+1
  print("End epoch.")
  ''' 1 EPOCH END.'''
  
  ''' CHOOSE POPULATIONS. '''
  # sort contributions     
  result = get_optimal_sets(POP_FILES, n, c_t)
  
  # Print out ...
  print(f"loss:{loss}")   
  print(); print(result['dataframe_1']); print(result['dataframe_2'])
  print(f"Estimated optimal foundation set: Choose combination in k = {result['k_low']} -> {result['k_upp']}.")
  for  k in range(result['k_low'],result['k_upp']+1):
    print(f"k={k}, {result['id_pop_combs'][k-1]} =>\n{result['names_pop_combs'][k-1]}")


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
  ctrb_xticks = [r"$\mathrm{\mathit{s}_{"+f"{p_idx}"+r"}}$" for p_idx in result['pop_sort_idxs']]  
  pop_xticks = [r"$\mathcal{H}_{"+f"{p_idx+1}"+r"}$" for p_idx in range(n)]  
  colors2 = [
    'green', 
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
  
  wx_ratio = n/2
  
  svpath = (SERVER_ROOT/f"static/trainplts/{SCRATCH_FOLDER}").resolve()._str
  print(svpath)
  os.makedirs(svpath, exist_ok = True)
  
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.13*wx_ratio, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].bar(ctrb_xticks,result['c_star'], color=cmap(result['c_star']))
  plt.xticks(x, ctrb_xticks, rotation=0)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{Ordered~populations}}$", fontsize=2, labelpad=1.5)
  ax[0].set_ylabel(r"$\mathbf{c}^\star$",  fontsize=2, labelpad=1.5)  
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  plt.tight_layout(pad=0.25)

  figpath = f"{svpath}/relctrbs_sslplot.png"
  plt.savefig(figpath, bbox_inches='tight', dpi=1200)
  plt.close(fig)
  
  # grouping
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.16*wx_ratio, 0.7)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,result['z'],label=r"$\mathrm{\mathbf{z}}$",marker='o',markersize=0.4,lw=0.25)
  ax[0].plot(x,result['dz'],label=r"$d\mathrm{\mathbf{z}}$",marker='o',markersize=0.4,lw=0.25)
  ax[0].axvspan(result['k_low']-1,result['k_upp']-1, facecolor='tab:green', alpha=0.25, linewidth=1)
  
  ax[0].annotate(r'$\mathcal{H}^\star$',xy=(result['k_low'],result['z'][result['k_low']]),xytext=((result['k_low']-1)+(result['k_upp']-result['k_low'])/2,result['z'][result['k_low']-1]), fontsize=4)
  
  plt.xticks(x, pop_xticks, rotation=60)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{optimal~population~subsets/combinations}},\mathcal{H}_j$", fontsize=2.5, labelpad=1.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{cummulative~value}, \mathbf{z}}$",  fontsize=2.5, labelpad=1.5)
  
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=2.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=3, fancybox=False, edgecolor='black', frameon=False)
  plt.tight_layout(pad=0.25)

  figpath = f"{svpath}/popchoice_sslplot.png"
  plt.savefig(figpath,  bbox_inches='tight', dpi=1200)
  # plt.show()
  plt.close(fig)

  # COST
  # plt.rcParams.update(plt.rcParamsDefault)
  x = np.arange(1, len(LOSS_LIST)+1)
  lloss = np.array(LOSS_LIST)
  lfcloss = np.array(FC_LOSS_LIST)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.5)
  fig = plt.figure(figsize=figsz,tight_layout=True, dpi=1200)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]

  ax[0].plot(x,lloss,label=r"$f_t$",linewidth=0.3, marker=',', markersize=0.1, markevery=0.2)
  ax[0].plot(x,lfcloss,label=r"${\delta^f_t}$",linewidth=0.3, marker=',', markersize=0.1, markevery=0.2)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{cost}}$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=2, fancybox=False, edgecolor='black', frameon=False)
  
  plt.tight_layout(pad=0.5)
  figpath = f"{svpath}/cost_plot.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  # plt.show()
  
  
  # P_T 
  x = np.arange(1, len(LOSS_LIST)+1)
  y = np.array(C_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.15)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{c}}_t$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/ctrbrel_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  
  # Y_T 
  x = np.arange(1, len(LOSS_LIST)+1)
  y = np.array(Y_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.15)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{y}}_t$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/lin_y_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  
  # LR_T 
  x = np.arange(1, len(LOSS_LIST)+1)
  y = np.array(LR_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.2,marker='+', markersize=0.05)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{\alpha}}_t$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  # plt.legend()
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/lrt_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  # BTi_T 
  x = np.arange(1, len(LOSS_LIST)+1)
  y = np.array(BT_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.2, marker='+', markersize=0.05)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{{\beta}}_{i,t}$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/betai_t_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  
  # G_T 
  x = np.arange(1, len(LOSS_LIST)+1)
  y = np.array(G_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.2)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{G}}_t$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/gradw_t_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  
  # W_T 
  x = np.arange(0, len(LOSS_LIST)+1)
  y = np.array(W_T)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y,linewidth=0.2)
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\mathrm{\mathsf{W}}_t$", fontsize=3, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)

  plt.tight_layout(pad=0.25)
  figpath = f"{svpath}/w_t_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)