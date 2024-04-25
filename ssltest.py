import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

import json
import collections

import torch
import torch.nn as nn

import numpy as np
import pandas as pd

import operator
from datapy.popdataloader import PopDatasetStreamerLoader
from aipy.direct_quad import edingetal_sel, unc_sel

from aipy.gp import gp1
from aipy.ext import extsolv

import plotutils as pla
from plotutils import *

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
  
  # Print out ...
  print()
  print(result['dftable'])
  print(result['dfsel'])
    
  return result

def rdim_opt(SCRATCH_FOLDER):
    
  SERVER_ROOT = Path(__file__).parents[0]
  DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
  
  PLOT_PATH = str(SERVER_ROOT/f"static/svdirs/dev-session/{SCRATCH_FOLDER}")
  os.makedirs(PLOT_PATH, exist_ok=True)
  # search+select .frq files in alle_frq_dirs/test_af
  POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")

  args = Props(pop_files=POP_FILES, n=len(POP_FILES), 
              use_cuda=False, use_corr=True, max_steps=1000, no_maxsteps=True,
              max_batchsize=512, err_opt_acc=1e-15, debug=True)
  
  svlists = Props(t=[],he_t=[], clmb_t=[], cost_u_t=[], dcost_u_t=[], cost_c_t=[], dcost_c_t=[], lr_t=[], bt_t=[])

  # buffer_size: data size < actual data size to load into mem. 
  # batch_size: selected data size per batch <= buffer_size
  # onesample_size: size of one data-point >= 1, < buffer_size
  data_ldr = PopDatasetStreamerLoader(args.pop_files,args.n,args.max_batchsize, avgmode=3)
  assert args.n == data_ldr.neff


  # print("Epoch: " + str(epoch+1)) 
  print(f"{('~~~~')*20}")
  walltime = time.time()
  
  ''' LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
  # load dataset, and dataloader, full pass over data
  for (b_idx, batch) in enumerate(data_ldr): pass
  # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
  # test: 
  # args.A = torch.tensor([[2.5409, -0.0113],[-0.0113, 0.5287]])
  # homozygozity
  args.A = check_fix_psd(batch[0])
  
  # check: sqp
  sol, answr, allx, allf, cvxf = extsolv(args, PLOT_PATH)
  tcvx = list(np.arange(1, len(cvxf)+1))
  tsqp = list(np.arange(1, len(allf['slsqp'])+1))

  # check: inversion (unconstr. bounds)
  edingetal_sel(args.A, PLOT_PATH)

  print('\n**GP1**')
  mdl_gp1, coan_val, costs_uc = gp1(args, svlists)  
  print(mdl_gp1)
  print('metric', coan_val, 'lambda', mdl_gp1.lmda.item())
  print('costs: [cost_u, dcost_u, cost_c]',  costs_uc)
  print(mdl_gp1.param_c.sum().item(), 
        'sum', mdl_gp1.tf().sum().item(), 
        'metric', mdl_gp1.coan_metric_tf().item(), 
        'cost',mdl_gp1.quad_cost_sum1_tf().item()
      )
  summ1 = summarydict(args, mdl_gp1.tf())
  print(f"avg. kinship: {summ1['coan'].item()}")

  # detach().numpy(force=True).flatten())
  
  plt.rcParams['axes.linewidth'] = 0.1
  csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
  figsz = (0.4, 0.25)
  dpi = 1900
  figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
  ax = plt.gca()
  # ax.plot(svlists.t, svlists.clmb_t, alpha=0.3, linewidth=csts['LW'], label=r'$0.5*\lambda$')
  ax.plot(tcvx, cvxf, alpha=0.3, linewidth=csts['LW'], label='cvx')
  ax.plot(tsqp, allf['slsqp'], alpha=0.3, linewidth=csts['LW'], label='slsqp')
  ax.plot(svlists.t, svlists.cost_c_t, alpha=0.9, linewidth=csts['LW'], label='ours')
  pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/coan_t", r'iterations, $t$',r'cost', int=True, dpi=dpi)


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

  print('\n**PRE**')
  y_opt_uc, y_opt, lmda_opt, cf = unc_sel(args.A, args.use_cuda, args.use_corr, args.max_steps, args.no_maxsteps, args.err_opt_acc)
  print(lmda_opt.item())
  summ2=summarydict(args, y_opt)
  print(f"loss:{cf}")
  print(f"avg. kinship: {summ2['coan'].item()}")


  # Optimum Contribution Lists
  print('gp1')
  print(trunc(mdl_gp1.tf().detach().numpy(),5).T.tolist()[0])
  print('mdlo.y')
  print(trunc(y_opt.detach().numpy(), 5).T.tolist()[0])
  print('cvx')
  print(trunc(np.where(np.array(sol['x']) < 1e-5, 0, np.array(sol['x'])),5).T.tolist()[0])
  print('sl')
  print(trunc(answr['slsqp'].x, 5).T.tolist())

  walltime = (time.time() - walltime)/60 
  print(f"Total batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
  data_ldr.close()
  data_ldr.batches = b_idx+1
  # print("End epoch.")
  ''' 1 EPOCH END.'''

  # view diminishing returns

  atids = summ1['csel_ids'][:1]
  Ak = 0.5*args.A[atids,:][:,atids]
  cm_list = [Ak.item()]
  cts = {}
  for kn in range(2, summ1['ko']):
    argsn = copy.deepcopy(args)
    atids = summ1['csel_ids'][:kn]
    Ak = 0.5*argsn.A[atids,:][:,atids]

    mdln, coan_val, costs_uc = gp1(argsn)  
    summn = summarydict(argsn, mdln.tf())

    cm_list.append(summn['coan'].item())
    cts[kn] = summn['csel_vals']

  cm_list.append(summ1['coan'].item())
  cts[summ1['ko']] = summ1['csel_vals']

  # choose dimnishing return pop. size.
  ho = 1-np.array(cm_list)
  holist = list(ho)
  k_rec = np.argmax(ho)+1
  th = np.arange(1, len(cm_list)+1)

  # write summary of opt. ctrbs. to .txt
  kresultpath = f"{PLOT_PATH}/ks"
  os.makedirs(kresultpath, exist_ok=True)
  with open(f"{kresultpath}/opt_ctrbs-k={summ1['ko']}.txt", "w") as out_file:

    out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
    [out_file.write(
       "\t".join([str(summ1['csel_ids'][i]), 
                  str(summ1['csel_vals'][i])]) + "\n")  
    for i in range(0, summ1['ko']) ]     

  # write summary of dim. ctrbs. to .txt
  kresultpath = f"{PLOT_PATH}/ks"
  os.makedirs(kresultpath, exist_ok=True)
  with open(f"{kresultpath}/dim_ctrbs-k={k_rec}.txt", "w") as out_file:

    out_file.write("\t".join(["Pop_ID", "Proportion"]) + "\n")
    [out_file.write(
       "\t".join([str(summ1['csel_ids'][i]), 
                  str(summ1['csel_vals'][i])]) + "\n")  
    for i in range(0, k_rec) ]    

  plt.rcParams['axes.linewidth'] = 0.1
  csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
  figsz = (0.4, 0.25)
  dpi = 1900
  figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
  ax = plt.gca()
  ax.plot(th, cm_list, alpha=0.9, linewidth=csts['LW'], label=r'$\mathrm{H_e^\star}$')
  ax.plot(th[k_rec-1], cm_list[k_rec-1], marker='x', linewidth=csts['LW'], label=r'$\bar{k}$', markersize=0.1)
  pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/rdim1_plt", r'size, $k$', r'metric', int=True, dpi=dpi)

  plt.rcParams['axes.linewidth'] = 0.1
  csts = {'BM':0.5,'LW':0.1, 'AL':1, 'BW':0.15, 'TL':0.92, 'Fy':1, 'Fx':1, 'figsvdir':'','fignm':''}
  figsz = (0.4, 0.25)
  dpi = 1900
  figh = plt.figure(figsize=figsz,
          tight_layout=True, dpi=dpi,clear=True)
  ax = plt.gca()
  ax.plot(th, holist, alpha=0.9, linewidth=csts['LW'], label=r'$\mathrm{H_o^\star}$')
  ax.plot(th[k_rec-1], holist[k_rec-1], marker='x', linewidth=csts['LW'], label=r'$\bar{k}$', markersize=0.1)
  pla.nicefmt3(figh, ax, csts, f"{PLOT_PATH}/rdim2_plt", r'size, $k$', r'metric', int=True, dpi=dpi)

  print('Done!')
  print(f"{('~~~~')*20}")

  return k_rec


# GLOBAL configs
# print(Path.cwd())
SCRATCH_FOLDER = "alle_frq_dirs/test_af"
# SCRATCH_FOLDER = "alle_frq_dirs/sthd_af"
k_rec = rdim_opt(SCRATCH_FOLDER)

  

  

  
  
  
  
  
