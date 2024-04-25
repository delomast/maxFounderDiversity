import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path


import torch
import torch.nn as nn

from ssldefs import *

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

# run: flask --app sslview --debug run   

app = Flask(__name__)
# app.debug = True
# app.config['ENV'] = 'development'
# app.config['DEBUG'] = True
# app.config['TESTING'] = True
# print(app.config)

from werkzeug.middleware.proxy_fix import ProxyFix
app.wsgi_app = ProxyFix(app.wsgi_app,x_host=1,x_prefix=1)

app.secret_key = secrets.token_hex()

@app.route("/", methods=("POST", "GET"))
def index():
  return render_template('ssl_index.html')


@app.route('/setup', methods=['GET', 'POST'])
def setup():
  
  #todo: add more options like in sslcmd.py
  app.secret_key = secrets.token_hex()
  session["MAX_BATCHSIZE"] = request.form["batchsize"]
  # session["STREAM_LEARN"] = request.form["streamer"]
  session["NO_MAXSTEPS"] = True
  session["MAXSTEPS"] = 1000
  session["USE_CORR"] = request.form["scaler"]
  inpfiles = request.files.getlist("files")
  print(inpfiles)
  
  SERVER_ROOT = Path(__file__).parents[0]
  DATA_PATH = str((SERVER_ROOT / f"static/session/{app.secret_key}/alle_frq_dirs/test_af" ).resolve())
  os.makedirs(DATA_PATH, exist_ok=True)
  # print(DATA_PATH)
  session["DATA_PATH"] = DATA_PATH
  session["S_PLOT_PATH"] = f"static/session/{app.secret_key}/trainplts"
  session["noPlots"] = False
  session["debug"] = True

  for file in inpfiles:
      filename = secure_filename(file.filename)
      file.save(os.path.join(DATA_PATH, filename))
  
  # print(session["DATA_PATH"])
    
  return Response(status=204) # 


@app.route("/view", methods=("POST", "GET"))
def view():
  
  print(session)
  print(app.secret_key)
  
  klow, results = rdim_opt(cfgs=session)
  kupp = results['ko']
  df_relctrbs = results['dfsel']
  df_poploptcombs = results['dftable'].iloc[:klow]
   
  print(list(df_relctrbs.values.tolist()))
  
  return render_template('ssl_view_embed.html', 
          relctrbs_cols=df_relctrbs.columns.values, 
          relctrbs_idxs=df_relctrbs.index, 
          relctrbs_rowdata=list(df_relctrbs.values.tolist()), 
          optcombs_cols=df_poploptcombs.columns.values, 
          poploptcombs_idxs=df_poploptcombs.index, 
          optcombs_rowdata=list(df_poploptcombs.values.tolist()),
          relctrbsfig=f"{session['S_PLOT_PATH']}/ctrbs_bar.png",
          popcombsfig=f"{session['S_PLOT_PATH']}/rdim1_plt.png",
          trlossfig=f"{session['S_PLOT_PATH']}/coan_t.png",
          klow=klow, kupp=kupp,
          zip=zip, int=int)
  
  
  
  
  