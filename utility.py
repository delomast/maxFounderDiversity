import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import json
import collections

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import operator

from aipy.lssloc import QSSLNet
from aipy.asgm_quad import AutoSGMQuad
from aipy.asgm import AutoSGM as AutoSGMGen
from datapy.popdataloader import PopDatasetStreamerLoader
from aipy.edingetal import eding_sel, save_eding, dir_sel

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

from matplotlib.ticker import FixedLocator, FixedFormatter
 
    
   
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


# GLOBAL configs
# print(Path.cwd())
SERVER_ROOT = Path(__file__).parents[0]
SCRATCH_FOLDER = "alle_frq_dirs/test_af"
DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
# print(server_root)
# print(data_root)


''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
def trainmdl(mdl:QSSLNet, A, USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, 
            ERR_OPT_ACC, QUAD_OBJ_CHOICE=True, SVLISTS=None):
  
  # accept data matrix
  if USE_CUDA: 
    A = A.to('cuda',dtype=torch.float32) 
    
  mdl.data_matrix(A, use_corr=USE_CORR)
  zerocnt = 0
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
    step, lrks, betain_ks = mdl.learnerW.step(mdl.A,mdl.b,mdl.x,c_t)
      
    mdl.x = (c_t.detach())
        
    # print(f"Step: {int(step)}")
    # 3. stopping criterion.
    if SVLISTS is not None:
      SVLISTS['wt'].append(1*mdl.weight_W.detach().numpy(force=True).flatten())
      SVLISTS['cost'].append(costW.item())
      SVLISTS['dfcost'].append(delta_costW.item())
      SVLISTS['gt'].append(1*mdl.weight_W.grad.detach().numpy(force=True).flatten())
      SVLISTS['ct'].append(1*c_t.detach().numpy(force=True).flatten())
      SVLISTS['yt'].append(1*y.detach().numpy(force=True).flatten())
      SVLISTS['sst'].append(1*lrks.numpy(force=True).flatten())
      SVLISTS['btt'].append(1*betain_ks.numpy(force=True).flatten())
    # print(f"fractional loss change: {delta_costW}")  
    if delta_costW == 0: zerocnt+=1
    
    if (NO_MAXSTEPS and (k_id > 1) and (delta_costW < ERR_OPT_ACC)) or (zerocnt > 4): 
      print(f"Steps:{k_id+1}")
      break
    
    
  return costW.item(), 1*c_t, 1*y, 1*lrks, 1*betain_ks

''' PART 2: CHOOSE POPULATIONS. '''
def get_optimal_sets(POP_FILES, n, c_t, Amat=None, ismatrix= False):
  '''
  Choose Populations

  Args:
      POP_FILES (object): opened list of population files
      n (int): number of populations
      c_t (vector): relative contribution of population in [0,1]

  Returns:
      result dictionary:
      c_star: sorted p in descending order, from highest to lowest p value, 
      pop_sortidxs: actual ids of the populations, sorted from highest to lowest p value, 
      z: cummulative sum of phat, 
      dz: first derivative of z, 
      klow: lower bound estimate of k <= n, 
      kupp: upper bound estimate of k <= n
      dataframe_1: table of relative contributions (sorted), 
      dataframe_2: table of optimum combinations of populations from k=1 to n.
  '''  
  avgkin = 0.5*(c_t.T.mm(Amat.mm(c_t)))
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
  if not ismatrix:
    pop_names = [POP_FILES[p_id].split("\\")[-1].split("_af_")[0] for p_id in pop_sortidxs]
  else:
    pop_names = [f"POP_{p_id}" for p_id in pop_sortidxs]
  combspop_nms = [pop_names[:p_idx+1] for p_idx in range(n)]
  
  result = {
    'avg-kinship':avgkin,
    'c_star':c_star.tolist(),
    'pop_sort_idxs': pop_sortidxs.tolist(),
    'z':z,'dz':dz,
    'k_low':klow_id+1,'k_upp':kupp_id+1,
    'id_k_low':klow_id,'id_k_upp':kupp_id,
    'dataframe_1':df1,'dataframe_2':df2,
    'id_pop_combs':combspop,
    'pop_sort_nms': pop_names,
    'names_pop_combs': combspop_nms
  }
  
  # Print out ...
  print(); print(result['dataframe_1']); 
  # print(result['dataframe_2'])
  # print(f"Estimated optimal foundation set: Choose pop. combination in k = {result['k_low']} -> {result['k_upp']}.")
  # for  k in range(result['k_low'],result['k_upp']+1):
  #   # print(f"k={k}, {result['id_pop_combs'][k-1]} =>\n{result['names_pop_combs'][k-1]}")
  #   print(f"k={k}, {result['pop_sort_idxs'].tolist()[0:k]} =>\n{result['names_pop_combs'][k-1]}")
  # print(f"lower-bound k={result['k_low']}, {result['pop_sort_idxs'][0:result['k_low']]} =>\n{result['names_pop_combs'][result['k_low']-1]}")
    
  return result


''' PLOTS. ''' 
def web_render_results(PLOT_PATH,n,results,SVLISTS, skip_heavy_plots=True):
  
  c_star_sorted = results['c_star']
  pop_idxs_sorted = results['pop_sort_idxs']
  z = results['z']
  dz = results['dz']
  klow = results['k_low']
  kupp = results['k_upp']
  
  '''
  Plots results of this Learning process.
  '''

  x = np.arange(n) 
  ctrb_xticks = [r"$\mathrm{\mathit{s}_{"+f"{p_idx}"+r"}}$" for p_idx in pop_idxs_sorted]  
  pop_xticks = [r"$\mathcal{H}_{"+f"{p_idx+1}"+r"}$" for p_idx in range(n)] 
  
  fctrb_xticks = [ctrb_xticks[0], ctrb_xticks[kupp-1]]
  fxtick_loc2 = [0, n-1]
  fpop_xticks = [pop_xticks[0], pop_xticks[klow-1], pop_xticks[kupp-1],pop_xticks[-1]]
  fxtick_loc = [0, klow-1, kupp-1, n-1]
  
  sorted_name_ctrb = dict(zip(ctrb_xticks, c_star_sorted))
  
  # cmap = ListedColormap(colors)
  cmap = LinearSegmentedColormap.from_list("cbcmap", colors)
  # cmapstr = "hsv"
  # cmap = mpl.colormaps[cmapstr]
  # cmap = mpl.colormaps[cmapstr].reversed() #.resampled(50)
  
  jsresult = {
    'cost': SVLISTS['cost'][-1],
    'avg.co-ancestry': results['avg-kinship'][0].item(),
    'k_low':results['k_low'],'k_upp':results['k_upp'],
    'id_k_low':results['id_k_low'],'id_k_upp':results['id_k_upp'],
    'c_sort_star':results['c_star'],
    'low_pop_combs':results['pop_sort_idxs'][0:results['k_low']],
    'upp_pop_combs':results['pop_sort_idxs'][0:results['k_upp']],
    'low_pop_combs_names':results['pop_sort_nms'][0:results['k_low']],
    'upp_pop_combs_names':results['pop_sort_nms'][0:results['k_upp']], 'pop_sort_idxs': results['pop_sort_idxs'],
    'pop_sort_names': results['pop_sort_nms'],
    'ranking': sorted_name_ctrb,
    
  }
  
  resultpath = f"{PLOT_PATH}/results.json"
  def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()
      
  with open(resultpath, 'w', encoding='utf-8', errors='ignore') as svfile:
    json.dump(jsresult, svfile, default=np_encoder, ensure_ascii=False, indent=4)
  
  # wx_ratio = n/2
  wx_ratio = 12/2
  
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.13*wx_ratio, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  
  # categorizing sets by name, since name is unique compared to number. 
  xsetl = list(sorted(collections.Counter(ctrb_xticks[0:klow]), key=ctrb_xticks.index))
  setl = [sorted_name_ctrb[popnm_key] for popnm_key in xsetl]
  xseth = list(sorted(collections.Counter(ctrb_xticks[0:kupp]) - collections.Counter(ctrb_xticks[0:klow]), key=ctrb_xticks.index))
  seth = [sorted_name_ctrb[popnm_key] for popnm_key in xseth]
  xsetd = list(sorted(collections.Counter(ctrb_xticks) - collections.Counter(ctrb_xticks[0:kupp]), key=ctrb_xticks.index))
  setd = [sorted_name_ctrb[popnm_key] for popnm_key in xsetd]


  fctrb_xticks = [ctrb_xticks[0], ctrb_xticks[klow-1], ctrb_xticks[kupp-1], ctrb_xticks[n-1]]
  fxtick_loc2 = [0, klow-1, kupp-1, n-1]

  ax[0].bar(xsetl,setl, color='green', width=0.25)
  ax[0].bar(xseth,seth, color='purple', width=0.25)
  ax[0].bar(xsetd,setd, color='pink', width=0.25)
  plt.xticks(fxtick_loc2, fctrb_xticks, rotation=0)
  # plt.xticks(x, ctrb_xticks, rotation=0)
  # x_formatter = FixedFormatter(fctrb_xticks)
  # x_locator = FixedLocator(fxtick_loc2)
  # ax[0].xaxis.set_major_formatter(x_formatter)
  # ax[0].xaxis.set_major_locator(x_locator)
  
  ax[0].set_xlabel(r"$\mathrm{\mathsf{Ordered~populations}}$", fontsize=2, labelpad=1.5)
  ax[0].set_ylabel(r"$\mathbf{c}^\star$",  fontsize=2, labelpad=1.5)  
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  plt.tight_layout(pad=0.25)

  figpath = f"{PLOT_PATH}/relctrbs_sslplot.png"
  plt.savefig(figpath,  bbox_inches='tight', dpi=1200)
  plt.close(fig)
  
  # grouping
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.16*wx_ratio, 0.7)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  
  ax[0].plot(x,results['z'],label=r"$\mathsf{values},\,\mathrm{\mathbf{z}}$",marker='o',markersize=0.4,lw=0.25, markevery=fxtick_loc)
  ax[0].plot(x,results['dz'],label=r"$\mathsf{returns},\,d\mathrm{\mathbf{z}}$",marker='o',markersize=0.4,lw=0.25, markevery=fxtick_loc)
  
  ax[0].axvspan(results['k_low']-1,results['k_upp']-1, facecolor='tab:green', alpha=0.25, linewidth=1)
  
  ax[0].annotate(r'$\mathcal{H}^\star$',xy=(results['k_low'],results['z'][results['k_low']]),xytext=((results['k_low']-1)+(results['k_upp']-results['k_low'])/2,results['z'][results['k_low']-1]), fontsize=4)
  
  # plt.xticks(x, pop_xticks, rotation=60)
  plt.xticks(fxtick_loc, fpop_xticks, rotation=60)
  
  x_formatter = FixedFormatter(fpop_xticks)
  x_locator = FixedLocator(fxtick_loc)
  ax[0].xaxis.set_major_formatter(x_formatter)
  ax[0].xaxis.set_major_locator(x_locator)
  
  
  ax[0].set_xlabel(r"$\mathrm{\mathsf{optimal~population~subsets/combinations}},\mathcal{H}_j$", fontsize=2.5, labelpad=1.5)
  ax[0].set_ylabel(r"$\mathsf{returns},\,d\mathrm{\mathbf{z}}$",  fontsize=2.5, labelpad=1.5)
  
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=2,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=2.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=2.5, fancybox=False, edgecolor='black', frameon=False)
  plt.tight_layout(pad=0.25)

  figpath = f"{PLOT_PATH}/popchoice_sslplot.png"
  plt.savefig(figpath, bbox_inches='tight', dpi=1200)
  # plt.show()
  plt.close(fig)

  # COST
  # plt.rcParams.update(plt.rcParamsDefault)
  x = np.arange(1, len(SVLISTS['cost'])+1)
  lloss = np.array(SVLISTS['cost'])
  lfcloss = np.array(SVLISTS['dfcost'])
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
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
  ax[0].xaxis.set_tick_params(labelsize=1.25,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.25,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=2, fancybox=False, edgecolor='black', frameon=False)
  
  plt.tight_layout(pad=0.25)
  figpath = f"{PLOT_PATH}/cost_plot.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  # plt.show()
  
  
  # P_T 
  x = np.arange(1, len(SVLISTS['cost'])+1)
  y = np.array(SVLISTS['ct'])
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
  figpath = f"{PLOT_PATH}/ctrbrel_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  
  # SVLISTS['yt'] 
  x = np.arange(1, len(SVLISTS['cost'])+1)
  y = np.array(SVLISTS['yt'])
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
  figpath = f"{PLOT_PATH}/lin_y_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  
  if not skip_heavy_plots:
  
    # SVLISTS['sst'] 
    x = np.arange(1, len(SVLISTS['cost'])+1)
    y = np.array(SVLISTS['sst'])
    ym = y.mean(axis=1)
    yse = y.std(axis=1)/np.sqrt(y.shape[1])
    plt.rcParams['axes.linewidth'] = 0.35
    figsz = (0.7, 0.4)
    fig = plt.figure(figsize=figsz,tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    ax = [plt.subplot(gsi) for gsi in gs]
    
    # ax[0].plot(x,y,linewidth=0.2, marker='+', markersize=0.05)
    ax[0].errorbar(x, ym, yse, alpha=0.99,linewidth=0.2, marker='+', markersize=0.05, ecolor='black', elinewidth=0.05, color=colors[0])
    ax[0].fill_between(x, (ym-yse), (ym+yse), alpha=0.08, facecolor=colors[0],linewidth=0)
    
    ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
    ax[0].set_ylabel(r"$\mathrm{\mathsf{\alpha}}_t$", fontsize=3, labelpad=1.5)
    ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    # plt.legend()
    csts = {'LW':0.25}
    #
    ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].margins(y=0.25, tight=True)

    plt.tight_layout(pad=0.25)
    figpath = f"{PLOT_PATH}/lrt_curve.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(fig)
  
    # BTi_T 
    x = np.arange(1, len(SVLISTS['cost'])+1)
    y = np.array(SVLISTS['btt'])
    ym = y.mean(axis=1)
    yse = y.std(axis=1)/np.sqrt(y.shape[1])
    plt.rcParams['axes.linewidth'] = 0.35
    figsz = (0.7, 0.4)
    fig = plt.figure(figsize=figsz,tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    ax = [plt.subplot(gsi) for gsi in gs]
    
    # ax[0].plot(x,y,linewidth=0.2, marker='+', markersize=0.05)
    ax[0].errorbar(x, ym, yse, alpha=0.99,linewidth=0.2, marker='+', markersize=0.05, ecolor='black', elinewidth=0.05, color=colors[0])
    ax[0].fill_between(x, (ym-yse), (ym+yse), alpha=0.08, facecolor=colors[0],linewidth=0)
    
    ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
    ax[0].set_ylabel(r"$\mathrm{{\beta}}_{i,t}$", fontsize=3, labelpad=1.5)
    ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    csts = {'LW':0.25}
    #
    ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].margins(y=0.25, tight=True)

    plt.tight_layout(pad=0.25)
    figpath = f"{PLOT_PATH}/betai_t_curve.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(fig)
      
  
    # SVLISTS['gt'] 
    x = np.arange(1, len(SVLISTS['cost'])+1)
    y = np.array(SVLISTS['gt'])
    ym = y.mean(axis=1)
    yse = y.std(axis=1)/np.sqrt(y.shape[1])
    plt.rcParams['axes.linewidth'] = 0.35
    figsz = (0.7, 0.4)
    fig = plt.figure(figsize=figsz,tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    ax = [plt.subplot(gsi) for gsi in gs]
    
    # ax[0].plot(x,y,linewidth=0.2, marker='+', markersize=0.05)
    ax[0].errorbar(x, ym, yse, alpha=0.99,linewidth=0.2, marker='+', markersize=0.05, ecolor='black', elinewidth=0.05, color=colors[0])
    ax[0].fill_between(x, (ym-yse), (ym+yse), alpha=0.08, facecolor=colors[0],linewidth=0)
    
    ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
    ax[0].set_ylabel(r"$\mathrm{\mathsf{G}}_t$", fontsize=3, labelpad=1.5)
    ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    csts = {'LW':0.25}
    #
    ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].margins(y=0.25, tight=True)

    plt.tight_layout(pad=0.25)
    figpath = f"{PLOT_PATH}/gradw_t_curve.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(fig)
  
  
    # SVLISTS['wt'] 
    x = np.arange(0, len(SVLISTS['cost'])+1)
    y = np.array(SVLISTS['wt'])
    ym = y.mean(axis=1)
    yse = y.std(axis=1)/np.sqrt(y.shape[1])
    plt.rcParams['axes.linewidth'] = 0.35
    figsz = (0.7, 0.4)
    fig = plt.figure(figsize=figsz,tight_layout=True)
    gs = gridspec.GridSpec(1,1)
    ax = [plt.subplot(gsi) for gsi in gs]
    
    # ax[0].plot(x,y,linewidth=0.2, marker='+', markersize=0.05)
    ax[0].errorbar(x, ym, yse, alpha=0.99,linewidth=0.2, marker='+', markersize=0.05, ecolor='black', elinewidth=0.05, color=colors[0])
    ax[0].fill_between(x, (ym-yse), (ym+yse), alpha=0.08, facecolor=colors[0],linewidth=0)
    
    ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
    ax[0].set_ylabel(r"$\mathrm{\mathsf{W}}_t$", fontsize=3, labelpad=1.5)
    ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
    csts = {'LW':0.25}
    #
    ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
    ax[0].margins(y=0.25, tight=True)

    plt.tight_layout(pad=0.25)
    figpath = f"{PLOT_PATH}/w_t_curve.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(fig)
    
  
  # norm
  x = np.arange(1, len(SVLISTS['cost'])+1)
  y1 = np.array([ np.linalg.norm(cvt - SVLISTS['ct'][-1]) for cvt in SVLISTS['ct'] ])
  y2 = np.array([ np.linalg.norm(cvt - SVLISTS['ct'][-1], float(np.inf)) for cvt in SVLISTS['ct'] ])
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.7, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  ax[0].plot(x,y1,linewidth=0.2,label=r"$\Vert\mathrm{\mathsf{c}}_t-\mathrm{\mathsf{c}}^\star\Vert_2$")
  ax[0].plot(x,y2,linewidth=0.2,label=r"$\Vert\mathrm{\mathsf{c}}_t-\mathrm{\mathsf{c}}^\star\Vert_1$")
  ax[0].set_xlabel(r"$\mathrm{\mathsf{iterations}},t$", fontsize=3, labelpad=0.5)
  ax[0].set_ylabel(r"$\Vert\mathrm{\mathsf{c}}_t-\mathrm{\mathsf{c}}^\star\Vert$", fontsize=2, labelpad=1.5)
  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.5,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.05, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=1.5, fancybox=False, edgecolor='black', frameon=False)
  
  plt.tight_layout(pad=0.25)
  figpath = f"{PLOT_PATH}/normctrbrel_curve.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)  
  

def plot_coans(CMS, CMU, PLOT_PATH, krec):
  # COAN-PLOT
  # plt.rcParams.update(plt.rcParamsDefault)
  x = np.arange(1, len(CMS)+1)
  y1 = 1-np.array(CMS)
  y2 = 1-np.array(CMU)
  plt.rcParams['axes.linewidth'] = 0.35
  figsz = (0.6, 0.4)
  fig = plt.figure(figsize=figsz,tight_layout=True, dpi=1200)
  gs = gridspec.GridSpec(1,1)
  ax = [plt.subplot(gsi) for gsi in gs]
  
  
  xs = krec
  y3 = [el for el in y1[xs[0]-1:xs[-1]]] #todo

  # ax[0].plot(x,y1,label=r"$\mathsf{learnt}$",linewidth=0.2, marker=',', markersize=0.1, markevery=0.2)
  # ax[0].plot(x,y2,label=r"$\mathsf{uniform}$",linewidth=0.1, marker=',', markersize=0.1, markevery=0.2, linestyle="dashed")
  ax[0].plot(x,y1,linewidth=0.2, marker=',', markersize=0.1, markevery=0.2)
  ax[0].plot(xs,y3,label=r"$\mathcal{H}^\star$",linewidth=0.1, marker='x',  markersize=0.1)
  
  ax[0].axvspan(xs[0],xs[-1], facecolor='tab:green', alpha=0.25, linewidth=1)

  ax[0].xaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))
  
  ax[0].set_xlabel(r"$\mathrm{\mathsf{k}}$", fontsize=1.5, labelpad=1.5)
  ax[0].set_ylabel(r"$\mathsf{expected~diversity}$", fontsize=1.5, labelpad=1.5)
  
  csts = {'LW':0.25}
  #
  ax[0].xaxis.set_tick_params(labelsize=1.25,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].yaxis.set_tick_params(labelsize=1.25,length=1.5, width=csts['LW'],pad=0.5)
  ax[0].margins(y=0.1, tight=True)
  ax[0].legend( loc='best', ncols=1, borderaxespad=0.,fontsize=1.25, fancybox=False, edgecolor='black', frameon=False)
  plt.tight_layout(pad=0.25)
  figpath = f"{PLOT_PATH}/coan_plot.png"
  plt.savefig(figpath, dpi=1200)
  plt.close(fig)
  # plt.show()

def np_encoder(object):
  if isinstance(object, np.generic): return object.item()
  
  
def save_k_rctrbs(c_t, PLOT_PATH, sel_ids, k_n, avgkin):
    c_t = c_t.numpy().flatten().tolist()
    # jsresult = {
    # 'avg.co-ancestry': avgkin.item(),
    # 'c_star':c_t.detach().numpy(force=True).flatten().tolist(),
    # # 'pop_idxs': sel_ids,
    # }
    
    kresultpath = f"{PLOT_PATH}/ks"
    os.makedirs(kresultpath, exist_ok=True)
    # resultpath = kresultpath+f"/result_k={k_n}.json"
    # with open(resultpath, 'w', encoding='utf-8', errors='ignore') as svfile:
    #   json.dump(jsresult, svfile, default=np_encoder, ensure_ascii=False, indent=4)
      
    with open(f"{kresultpath}/contribs-k={k_n}.txt", "w") as out_file:
      out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
      [ out_file.write("\t".join([str(sel_ids[i]), str(c_t[i])]) + "\n")  
      for i in range(0,len(c_t)) 
      ]     

def recurse_dim_returns(A, n, result, 
                        PLOT_PATH, ERR_OPT_ACC, USE_CORR, USE_CUDA=False):
    
    NO_MAXSTEPS = True
    MAX_STEPS = 300
    sel_ids = result['pop_sort_idxs'][:1]
    a_opt = A[sel_ids,:][:,sel_ids]
    # list for learnt rel. ctrbs.
    CMS = [0.5*a_opt.item()]
    # uniform rel. ctrbs.
    CMU = [0.5*a_opt.item()]
    
    # if n < 1:
    for k_n in range(2,n,1):
      print("k:",k_n)
      sel_ids = result['pop_sort_idxs'][:k_n]
      # re-arrange and load
      A_E = A[sel_ids,:][:,sel_ids]
      # re-arrange data
      # POP_FILES_E = operator.itemgetter(*sel_ids)(POP_FILES)
      # data_ldre = PopDatasetStreamerLoader(POP_FILES=POP_FILES_E, neff=k_n,max_batch_size=MAX_BATCHSIZE, avgmode=3)    
      # # load dataset, and dataloader
      # for (b_idx, batch) in enumerate(data_ldre): pass
      # A_E = 1*batch[0] #xz homozygozity
      # data_ldre.close()      
      if k_n < 500:
        # instantiate self-supervised model
        mdle = QSSLNet(in_dim=k_n,out_dim=k_n, eps=ERR_OPT_ACC)  
        mdle.set_learner(
        AutoSGMQuad(mdle.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA
        ) )
        # print(f"{('~~~~')*20}")
        # walltime = time.time()
        #.. learn after full pass over data
        cost, c_t, y, alphas, betas = trainmdl(mdle, A_E,
              USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, ERR_OPT_ACC)
        # walltime = (time.time() - walltime)/60 
        # Print out ...
        # print(f"cost:{cost}")
        # print(f"time elapsed: {walltime:.2f}-mins") 
      
        # calculate avg. kinship     
        CMS.append((0.5*(c_t.T.mm(A_E.mm(c_t)))).item())
        CMU.append(0.5*A_E.mean().item())  
        # # save c_t_s for each k.
        # save_k_rctrbs(c_t, PLOT_PATH, sel_ids, k_n, avgkin)
      else:
      # for k_n in range(2,n,1):
      #   print("k:",k_n)
        # sel_ids = result['pop_sort_idxs'][:k_n]
        # re-arrange and load
        # A_E = A[sel_ids,:][:,sel_ids]
        CMS.append(0.5*A_E.mean().item())
        CMU.append(0.5*A_E.mean().item())  
    
    CMS.append(result['avg-kinship'].item())
    CMU.append(0.5*A.mean().item())
        
    # est. heterozygozity
    q=1-np.array(CMS)
    klo_id = np.argmax(q)
    s=np.diff(np.diff(q[2:]))
    kupp_id = np.where(s>0)
    if len(kupp_id) > 0: kupp_id = klo_id + kupp_id[0].flatten()[0]
    else: kupp_id = klo_id + 0
    krec = np.arange(klo_id+1, kupp_id+2).flatten()
    # save c_t_s for each recomended k.
    for k_n in krec:  
      print("k_star:",k_n)
      sel_ids = result['pop_sort_idxs'][:k_n]
      # re-arrange and load
      A_E = A[sel_ids,:][:,sel_ids]

      # instantiate self-supervised model
      mdle = QSSLNet(in_dim=k_n,out_dim=k_n, eps=ERR_OPT_ACC)  
      mdle.set_learner(
      AutoSGMQuad(mdle.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA
      ) )
      #.. learn after full pass over data
      cost, c_t, y, alphas, betas = trainmdl(mdle, A_E,
            USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, ERR_OPT_ACC)
    
      # calculate avg. kinship     
      avgkin = 0.5*(c_t.T.mm(A_E.mm(c_t)))      
      save_k_rctrbs(c_t, PLOT_PATH, sel_ids[:k_n], k_n, CMS[k_n])
    
    plot_coans(CMS, CMU, PLOT_PATH, krec)
    print("Recommended Parent Set: Pick pop. combination 'k' in", krec)
    print("See",f"{PLOT_PATH}\ks",)
    return krec