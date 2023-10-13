import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path


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

from flask import (
    Flask, Response, redirect, render_template, request, 
    session, flash, jsonify, send_file, url_for
)
from werkzeug.utils import secure_filename

import time
import secrets


# GLOBAL configs
  

app = Flask(__name__)
# app.debug = True
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True
# app.config['TESTING'] = True
# print(app.config)

# from werkzeug.middleware.proxy_fix import ProxyFix
# app.wsgi_app = ProxyFix(app.wsgi_app,x_host=1,x_prefix=1)

app.secret_key = secrets.token_hex()

@app.route("/", methods=("POST", "GET"))
def index():
  return render_template('ssl_index.html')


@app.route('/setup', methods=['GET', 'POST'])
def setup():
  
  app.secret_key = secrets.token_hex()
  session["MAX_BATCHSIZE"] = request.form["batchsize"]
  # session["STREAM_LEARN"] = request.form["streamer"]
  # session["QUAD_OBJ_CHOICE"] = request.form["learner"]
  session["USE_CORR"] = request.form["scaler"]
  inpfiles = request.files.getlist("files")
  print(inpfiles)
  
  SERVER_ROOT = Path(__file__).parents[0]
  DATA_PATH = (SERVER_ROOT / f"static/session/{app.secret_key}/alle_frq_dirs/test_af" ).resolve()
  os.makedirs(DATA_PATH, exist_ok=True)
  # print(DATA_PATH)
  session["DATA_PATH"] = str(DATA_PATH)
  session["S_PLOT_PATH"] = f"static/session/{app.secret_key}/trainplts"
  
  for file in inpfiles:
      filename = secure_filename(file.filename)
      file.save(os.path.join(DATA_PATH, filename))
  
  # print(session["DATA_PATH"])
  # print(session["QUAD_OBJ_CHOICE"])
  
  return Response(status=204) # 


@app.route("/view", methods=("POST", "GET"))
def view():
  
  print(session)
  print(app.secret_key)
  
  results = run_web_ssl(cfgs=session)
  klow = results['k_low'] 
  kupp = results['k_upp']
  df_relctrbs = results['dataframe_1']
  df_poploptcombs = results['dataframe_2']
   
  print(list(df_relctrbs.values.tolist()))
  
  return render_template('ssl_view_embed.html', 
          relctrbs_cols=df_relctrbs.columns.values, 
          relctrbs_idxs=df_relctrbs.index, 
          relctrbs_rowdata=list(df_relctrbs.values.tolist()), 
          optcombs_cols=df_poploptcombs.columns.values, 
          poploptcombs_idxs=df_poploptcombs.index, 
          optcombs_rowdata=list(df_poploptcombs.values.tolist()),
          relctrbsfig=f"{session['S_PLOT_PATH']}/relctrbs_sslplot.png",
          popcombsfig=f"{session['S_PLOT_PATH']}/popchoice_sslplot.png",
          trlossfig=f"{session['S_PLOT_PATH']}/loss_plot.png",
          klow=klow, kupp=kupp,
          zip=zip, int=int)
  

def run_web_ssl(cfgs):
  
  SERVER_ROOT = Path(__file__).parents[0]
  PLOT_PATH = (SERVER_ROOT / cfgs["S_PLOT_PATH"] ).resolve()
  os.makedirs(PLOT_PATH, exist_ok=True)
  DATA_ROOT = cfgs["DATA_PATH"]

  POP_FILES = glob.glob(f"{DATA_ROOT}/*")
  # print(POP_FILES)
  N_EFF = len(POP_FILES)
  USE_CUDA = False
  USE_CORR = cfgs["USE_CORR"]
  if USE_CORR == "True":
    USE_CORR = True
  else:
    USE_CORR = False
  MAX_BATCHSIZE = int(cfgs["MAX_BATCHSIZE"])
  MAX_EPOCHS = 1
  ERR_OPT_ACC = 1E-15 # 1E-5, 1E-8, 1E-10
  QUAD_OBJ_CHOICE = True
  MAX_STEPS = (N_EFF**2)

  # batch_size: selected data size per batch
  data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)
  n = data_ldr.neff

  # instantiate self-supervised model
  mdl = QSSLNet(in_dim=n,out_dim=n)
  if USE_CUDA: 
    mdl = mdl.cuda()
  mdl.set_learner(
      AutoSGMQuad(mdl.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA) 
      )

  for epoch in range(MAX_EPOCHS):
    ''' EPOCH BEGIN: A single pass through the data'''
    LOSS_LIST, FC_LOSS_LIST = [],[]
    W_T, G_T = [],[]
    C_T, Y_T, LR_T, BT_T = [],[],[],[]
    
    W_T.append(1*mdl.weight_W.detach().numpy(force=True).flatten())
    print("Epoch: " + str(epoch+1)) 
    walltime = time.time()
    ''' PART 1: LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''

    # load dataset, and dataloader
    for (b_idx, batch) in enumerate(data_ldr):
      A = batch[0] #homozygozity
    
    #.. learn after full pass over data
    loss, c_t, y, alphas, betas = trainmdl(mdl, A,
            USE_CUDA, USE_CORR, MAX_STEPS, ERR_OPT_ACC, 
            QUAD_OBJ_CHOICE, LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
            C_T, Y_T, LR_T, BT_T)  
          
    ''' EPOCH END.'''
    walltime = (time.time() - walltime)/60 
    print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
    data_ldr.close()
    data_ldr.batches = b_idx+1
    print("End epoch.")

    ''' PART 2: CHOOSE POPULATIONS. '''
    # sort contributions     
    results = get_optimal_sets(POP_FILES, n, c_t)

    ''' PLOTS. ''' 
    # n,phat,pop_sortidxs,z,dz,klow,kupp
    web_render_results(PLOT_PATH, n, results,
            LOSS_LIST, FC_LOSS_LIST, W_T, G_T,
            C_T, Y_T, LR_T, BT_T)
    
  return results
  
  
  
  