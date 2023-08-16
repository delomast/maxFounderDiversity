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
from utility import trainer, choose_pops, render_results, web_render_results

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
app.debug = True

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app,x_host=1,x_prefix=1)

app.secret_key = secrets.token_hex()

@app.route("/", methods=("POST", "GET"))
def index():
  return render_template('ssl_index.html')


@app.route('/setup', methods=['GET', 'POST'])
def setup():
  
  app.secret_key = secrets.token_hex()
  session["MAX_BATCHSIZE"] = request.form["batchsize"]
  session["STREAM_LEARN"] = request.form["streamer"]
  session["LIN_OPTIM_CHOICE"] = request.form["learner"]
  inpfiles = request.files.getlist("files")
  print(inpfiles)
  
  SERVER_ROOT = Path(__file__).parents[0]
  DATA_PATH = (SERVER_ROOT / f"static/session/{app.secret_key}/scratch" ).resolve()
  os.makedirs(DATA_PATH, exist_ok=True)
  # print(DATA_PATH)
  session["DATA_PATH"] = DATA_PATH._str
  session["S_PLOT_PATH"] = f"static/session/{app.secret_key}/trainplts"
  
  for file in inpfiles:
      filename = secure_filename(file.filename)
      file.save(os.path.join(DATA_PATH, filename))
  
  # print(session["DATA_PATH"])
  # print(session["LIN_OPTIM_CHOICE"])
  
  return Response(status=204) # 


@app.route("/view", methods=("POST", "GET"))
def view():
  
  print(session)
  print(app.secret_key)
  
  klow, kupp, df_relctrbs, df_poploptcombs = run_web_ssl(cfgs=session)
  
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
  MAX_BATCHSIZE = int(cfgs["MAX_BATCHSIZE"])
  MAX_EPOCHS = 1
  ERR_OPT_ACC = 1E-8 # 1E-5, 1E-8, 1E-10
  LIN_OPTIM_CHOICE = cfgs["LIN_OPTIM_CHOICE"]  # True: LIN | False: GEN 
  if LIN_OPTIM_CHOICE == "True":
    LIN_OPTIM_CHOICE = True
  else:
    LIN_OPTIM_CHOICE = False
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
      # A =  (1 - batch[1] + batch[0])/(3)
      A = batch[0]
  
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
    
  return klow,kupp,df_relctrbs,df_poploptcombs 
  
  
  
  