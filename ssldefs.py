import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import argparse

import secrets

import torch
import torch.nn as nn


import json
import collections

import numpy as np
import pandas as pd

import operator
from datapy.popdataloader import PopDatasetStreamerLoader
from aipy.eding import edingetal_sel

from aipy.gp import gp1
from aipy.ext import extsolv

import plotutils as pla
from plotutils import *

import matplotlib
matplotlib.use("Agg")

class Props():
    def __init__(self, **kwds):
        self.__dict__.update(kwds)

def trunc(values, decs=0):
    return np.trunc(values*10**decs)/(10**decs)
  
def check_fix_psd(A):
    '''
    # check and fix approximately 0 entries 
    # => s.p.d to p.d A matrix
    '''
    if A.abs().min() < 1e-3: 
       return A.abs()
    else: 
       return 1*A
    
    
def summarydict(args, c_t, ismatrix= False):
  '''
  Choose Populations

  Args:
      POP_FILES (object): opened list of population files
      n (int): number of populations
      c_t (vector): relative contribution of population in [0,1]

  Returns:
      result dictionary:
      c_star: sorted p. in descending order, from highest to lowest p. value, 
      pop_sortidxs: actual ids of the populations, sorted from highest to lowest p. value, 
      df: table of relative contributions (sorted), 
  '''  
  avgkin = 0.5*(c_t.T.mm(args.A.mm(c_t)))
  c_star, pop_sortidxs = c_t.detach().sort(dim=0,descending=True)
  pop_sortidxs = pop_sortidxs.numpy(force=True).flatten()
  c_star = c_star.numpy(force=True).flatten()
  df = pd.DataFrame({"idx": pop_sortidxs, "c":c_star})
  
  c_sel = c_star[c_star > 1e-3]
  k_o = len(c_sel)
  dfsel = pd.DataFrame({"idx": pop_sortidxs[:k_o], "c":c_sel})

  # get population ids according to sorted contribs.
  combspop = [pop_sortidxs[:p_idx+1] for p_idx in range(args.n)]
  
  # get population names according to sorted contribs.
  if not ismatrix:
    pop_names = [args.pop_files[p_id].split("\\")[-1].split("_af_")[0] for p_id in pop_sortidxs]
  else:
    pop_names = [f"P_{p_id}" for p_id in pop_sortidxs]
  
  result = {
    'coan':avgkin,
    'c_star':c_star.tolist(),
    'csel_vals': c_sel.tolist(),
    'pop_sort_idxs': pop_sortidxs.tolist(),
    'csel_ids': pop_sortidxs[:k_o].tolist(),
    'pop_sort_nms': pop_names,
    'dftable':df,
    'dfsel':dfsel,
    'ko': k_o,
    'id_pop_combs':combspop,
  }
  
  if args.debug:
    # Print out ...
    print()
    print(result['dftable'])
    print(result['dfsel'])
    
  return result

def writetxt_dim(summ1, k_rec, kresultpath):
    os.makedirs(kresultpath, exist_ok=True)
    with open(f"{kresultpath}/dim_ctrbs-k={k_rec}.txt", "w") as out_file:
      out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
      [out_file.write(
       "\t".join([str(summ1['csel_ids'][i]), 
                  str(summ1['csel_vals'][i])]) + "\n")  
      for i in range(0, k_rec) ]

def writetxt_opt(summ1, kresultpath):
    os.makedirs(kresultpath, exist_ok=True)
    with open(f"{kresultpath}/opt_ctrbs-k={summ1['ko']}.txt", "w") as out_file:
      out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
      [out_file.write(
       "\t".join([str(summ1['csel_ids'][i]), 
                  str(summ1['csel_vals'][i])]) + "\n")  
      for i in range(0, summ1['ko']) ]

def he_plt(args, PLOT_PATH, helist, k_rec, th):
    # if not args.noplts:
        # plt.rcParams.update({'text.usetex': plt.rcParamsDefault['text.usetex']})

    plt.rcParams['axes.linewidth'] = 0.1
    csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
    figsz = (0.4, 0.25)
    dpi = 1900
    figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
    ax = plt.gca()
    ax.plot(th, helist, alpha=0.9, linewidth=csts['LW'])
    ax.plot(th[k_rec-1], helist[k_rec-1], marker='x', linewidth=csts['LW'], label=r'${k}^\star$', markersize=0.1)
    
    lims = ax.get_ylim()
    uc = 1-0.5*args.A.mean().item()
    if lims[0] < uc:  
      ax.set_ylim(bottom=uc)      
      if lims[1] > 1:
        plt.ylim(top=helist[k_rec-1]+0.0003)
    
    pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/rdim2_plt", r'size, $k$', r'expected heterozygosity', int=True, dpi=dpi)

def ho_plt(args, PLOT_PATH, cm_list, k_rec, th):
    # if not args.noplts:
    #     plt.rcParams.update({'text.usetex': plt.rcParamsDefault['text.usetex']})
       
    plt.rcParams['axes.linewidth'] = 0.1
    csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
    figsz = (0.4, 0.25)
    dpi = 1900
    figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
    ax = plt.gca()
    ax.plot(th, cm_list, alpha=0.9, linewidth=csts['LW'])
    ax.plot(th[k_rec-1], cm_list[k_rec-1], marker='x', linewidth=csts['LW'], label=r'${k}^\star$', markersize=0.1)

    lims = ax.get_ylim()
    uc = 0.5*args.A.mean().item()
    if lims[1] > uc:
      plt.ylim(top=uc)
      if lims[0] < 0:
        plt.ylim(bottom=cm_list[k_rec-1]-0.0003)

    pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/rdim1_plt", r'size, $k$', r'expected homozygosity', int=True, dpi=dpi)

def ana_rdim(PLOT_PATH, args, summ1, ismatrix):
    
    atids = summ1['csel_ids'][:1]
    Ak = 0.5*args.A[atids,:][:,atids]
    cm_list = [Ak.item()]
    cts = {}
    for kn in range(2, summ1['ko']):
      argsn = copy.deepcopy(args)
      argsn.debug = False
      atids = summ1['csel_ids'][:kn]
      argsn.A = args.A[atids,:][:,atids]
      argsn.n = kn

      mdln, coan_val, costs_uc = gp1(argsn)  
      summn = summarydict(argsn, mdln.tf(), ismatrix)

      cm_list.append(summn['coan'].item())
      cts[kn] = summn['csel_vals']

    cm_list.append(summ1['coan'].item())
    cts[summ1['ko']] = summ1['csel_vals']

    # choose dimnishing return pop. size.
    he = 1-np.array(cm_list)
    helist = list(he)
    k_rec = np.argmax(he)+1
    th = np.arange(1, len(cm_list)+1)
  
    # assert k_rec == summ1['ko']

    # write summary of:
    kresultpath = f"{PLOT_PATH}/ks"
    # opt. ctrbs. to .txt
    writetxt_opt(summ1, kresultpath)     
    # # dim. ctrbs. to .txt
    # writetxt_dim(summ1, k_rec, kresultpath)    

    # plot
    if not args.noplts:
      ho_plt(args, PLOT_PATH, cm_list, k_rec, th)
      he_plt(args,PLOT_PATH, helist, k_rec, th)

    return k_rec

def cmp_costs(args, PLOT_PATH, 
              svlists, allf, cvxf, tcvx, tsqp):
    
    ucost = 0.5*args.A.mean().item()
    if args.debug:
       print('uniform cost', ucost)

    plt.rcParams['axes.linewidth'] = 0.1
    csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
    figsz = (0.4, 0.25)
    dpi = 1900
    figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
    ax = plt.gca()
    # ax.plot(svlists.t, svlists.clmb_t, alpha=0.3, linewidth=csts['LW'], label=r'$0.5*\lambda$')
    ax.plot([0,]+tcvx, [ucost,]+cvxf, alpha=0.3, linewidth=csts['LW'], label='cvx')
    ax.plot([0,]+tsqp, [ucost,]+allf['slsqp'], alpha=0.3, linewidth=csts['LW'], label='slsqp')
    ax.plot([0,]+svlists.t, [ucost,]+svlists.cost_c_t, alpha=0.9, linewidth=csts['LW'], label='ours')
    pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/coan_t", r'iterations, $t$',r'cost', int=True, dpi=dpi)
    return dpi

def ctrbs_bar(PLOT_PATH, summ1, dpi):
    csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':2, 'Fx':2, 'figsvdir':'','fignm':''}
    fig, ax = plt.subplots(figsize=(0.8, 0.5), dpi=dpi)
    ax.barh(summ1['csel_ids'], summ1['csel_vals'], align='center')

    labels = ax.get_xticklabels()
    plt.setp(labels, rotation=0, horizontalalignment='left')
    ax.set_yticks(summ1['csel_ids'], labels=summ1['csel_ids'])

    ylabel='Population index' 
    xlabel='optimum contributions'
    ax.xaxis.set_tick_params(labelsize=csts['Fx']-0.5,length=csts['AL'], width=csts['LW'],pad=0.5)
    ax.yaxis.set_tick_params(labelsize=csts['Fy']-0.5,length=csts['AL'], width=csts['LW'],pad=0.5)
    ax.set_xlabel(xlabel, fontsize=csts['Fx'], labelpad=0.5)
    ax.set_ylabel(ylabel, fontsize=csts['Fy'], labelpad=1.5)
    # ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(integer=True))

    # Label with specially formatted floats
    # ax.bar_label(barhdl, fmt='%.2f')
    ax.margins(y=0.05, tight=True)
    plt.tight_layout(pad=0.1)
    figpath = f"{PLOT_PATH}/ctrbs_bar.png"
    plt.savefig(figpath, dpi=dpi)
    plt.close(fig)


# main
def rdim_opt(cfgs, SCRATCH=None, POP_FILES=None, ismatrix=False):
    
  SERVER_ROOT = Path(__file__).parents[0]
  PLOT_PATH = (SERVER_ROOT / cfgs["S_PLOT_PATH"]).resolve()
  os.makedirs(PLOT_PATH, exist_ok=True)

  if POP_FILES is None and SCRATCH is None:
     DATA_ROOT = cfgs["DATA_PATH"]
     POP_FILES = glob.glob(f"{DATA_ROOT}/*")
  
  elif POP_FILES is None and SCRATCH is not None:
     DATA_ROOT = (SERVER_ROOT / SCRATCH ).resolve()
     POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")
     

  if not ismatrix:
    n_ = len(POP_FILES)
  else:
    n_ = POP_FILES.shape[0]

  args = Props(pop_files=POP_FILES, n=n_, noplts=cfgs["noPlots"],
              use_cuda=False, use_corr=cfgs["USE_CORR"], max_steps=cfgs["MAXSTEPS"], no_maxsteps=cfgs["NO_MAXSTEPS"],
              max_batchsize=int(cfgs["MAX_BATCHSIZE"]), err_opt_acc=1e-6, debug=cfgs["debug"])
  
  svlists = Props(t=[],he_t=[], clmb_t=[], cost_u_t=[], dcost_u_t=[], cost_c_t=[], dcost_c_t=[], lr_t=[], bt_t=[])

  # buffer_size: data size < actual data size to load into mem. 
  # batch_size: selected data size per batch <= buffer_size
  # onesample_size: size of one data-point >= 1, < buffer_size
  if not ismatrix:
    data_ldr = PopDatasetStreamerLoader(args.pop_files,args.n,args.max_batchsize)
    assert args.n == data_ldr.neff
    
  # print("Epoch: " + str(epoch+1)) 
  print(f"{('~~~~')*20}")
  walltime = time.time()
  
  ''' LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
  if not ismatrix:
    # load dataset, and dataloader, full pass over data
    for (b_idx, batch) in enumerate(data_ldr): pass
    # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
    # homozygozity
    args.A = check_fix_psd(batch[0])
  else:
    b_idx = 0
    batch = None
    args.A = check_fix_psd(torch.tensor(POP_FILES, dtype=torch.float))

  if args.debug:
    # check: sqp
    sol, answr, allx, allf, cvxf = extsolv(args, PLOT_PATH)
    tcvx = list(np.arange(1, len(cvxf)+1))
    tsqp = list(np.arange(1, len(allf['slsqp'])+1))

    # check: inversion (unconstr. bounds)
    edingetal_sel(args.A, PLOT_PATH)

  # print('\n**GP1**')
  mdl, coan_val, costs_uc = gp1(args, svlists)  
  summ1 = summarydict(args, mdl.tf(), ismatrix)
  
  if args.debug:
    print(mdl)
    print('metric', coan_val, 'lambda', mdl.lmda.item())
    print('costs: [cost_u, dcost_u, cost_c]',  costs_uc)
    print(mdl.param_c.sum().item(), 
          'sum', mdl.tf().sum().item(), 
          'metric', mdl.coan_metric_tf().item(), 
          'cost',mdl.quad_cost_sum1_tf().item()
        )
    
    print(f"avg. kinship: {summ1['coan'].item()}")

  # detach().numpy(force=True).flatten())
  dpi = 1900
  if args.debug and not args.noplts:
    dpi = cmp_costs(args, PLOT_PATH, svlists, allf, cvxf, tcvx, tsqp)
    ctrbs_bar(PLOT_PATH, summ1, dpi)

  # Optimum Contribution Lists
  if args.debug:
    print('gp1')
    print(trunc(mdl.tf().detach().numpy(),5).T.tolist()[0])
    # print('mdlo.y')
    # print(trunc(y_opt.detach().numpy(), 5).T.tolist()[0])
    print('cvx')
    print(trunc(np.where(np.array(sol['x']) < 1e-5, 0, np.array(sol['x'])),5).T.tolist()[0])
    print('slsqp')
    print(trunc(answr['slsqp'].x, 5).T.tolist())

  walltime = (time.time() - walltime)/60 
  # print(f"\nTotal batches: {b_idx+1}")
  print(f"time elapsed: {walltime:.2f}-mins", ', optimum population size:', summ1['ko']) 

  if not ismatrix:
    data_ldr.close()
    data_ldr.batches = b_idx+1
  # print("End epoch.")
  ''' 1 EPOCH END.'''

  # analyze diminishing returns
  walltime = time.time()
  k_rec = ana_rdim(PLOT_PATH, args, summ1, ismatrix)
  walltime = (time.time() - walltime)/60 
  # print(f"\nTotal batches: {b_idx+1}")
  print(f"time elapsed: {walltime:.2f}-mins", ', recommended population size: <=', k_rec) 

  print('Done!')
  print(f"{('~~~~')*20}")

  return k_rec, summ1


# GLOBAL configs

def dir_path(path):
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"readable_dir:{path} is not a valid path")


def str2bool(s):
    if s not in {'False', 'True'}:
        raise ValueError("Input should be False or True")
    return s == 'True'

  

  

  
  
  
  
  
