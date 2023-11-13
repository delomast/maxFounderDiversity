import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

from utility import *



def sel_k_dimreturns(SCRATCH_FOLDER):
    
    SERVER_ROOT = Path(__file__).parents[0]
    DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()

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

    # MAX_STEPS = (N_EFF**2)
    MAX_STEPS = 100
    NO_MAXSTEPS = True


    POP_FILES_T = POP_FILES

    # buffer_size: data size < actual data size to load into mem. 
    # batch_size: selected data size per batch <= buffer_size
    # onesample_size: size of one data-point >= 1, < buffer_size
    data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=N_EFF,max_batch_size=MAX_BATCHSIZE, avgmode=3)
    n = data_ldr.neff


    # instantiate self-supervised model
    mdl = QSSLNet(in_dim=n,out_dim=n, eps=ERR_OPT_ACC)
    if USE_CUDA: mdl = mdl.cuda() 

    if QUAD_OBJ_CHOICE:
      mdl.set_learner(
    AutoSGMQuad(mdl.Lin_W.parameters(), eps=ERR_OPT_ACC, usecuda=USE_CUDA, maximize=False) 
    )
    else: 
      # create plots for -2,-3,-4,-5,-6,-7,-8 comparing step-size for iteration dep   and constant with respect to performance (c_t converging/y_t converging/loss).
      mdl.set_learner(
      AutoSGMGen(mdl.Lin_W.parameters(),auto=True, lr_init=1e-2, beta_in_smooth=0.9, beta_den=0.99, beta_num=0.99, usecuda=USE_CUDA),
      )


    for epoch in range(MAX_EPOCHS):
      ''' 1 EPOCH BEGIN.'''
      # per iteration list
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

      ''' LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
      # load dataset, and dataloader

      for (b_idx, batch) in enumerate(data_ldr): pass
      # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
      # test: # A = torch.tensor([[2.5409, -0.0113],[-0.0113, 0.5287]])
      A = 1*batch[0] #xz homozygozity
  
      #.. learn after full pass over data
      cost, c_t, y, alphas, betas = trainmdl(mdl, A,
          USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, ERR_OPT_ACC, 
          QUAD_OBJ_CHOICE, SVLISTS)

      ''' CHOOSE POPULATIONS. '''
      # sort contributions     
      result = get_optimal_sets(POP_FILES, n, c_t, A)  

      walltime = (time.time() - walltime)/60 
      print(f"Total batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
      data_ldr.close()
      data_ldr.batches = b_idx+1
      # Print out ...
      print(f"avg. kinship:{result['avg-kinship'].item()}")
      # print("End epoch.")   
      ''' 1 EPOCH END.'''

      PLOT_PATH = str(SERVER_ROOT/f"static/svdirs/dev-session/{SCRATCH_FOLDER}")
      os.makedirs(PLOT_PATH, exist_ok=True)
      web_render_results(PLOT_PATH,n,result,SVLISTS, skip_heavy_plots=False)
      print('Done!')
      print(f"{('~~~~')*20}")

      # ecs, ecid, eding_f, secs, secid = eding_sel(A)
      ecs, ecid, eding_f, secs, secid = dir_sel(A)
      save_eding(PLOT_PATH, ecs, ecid, eding_f, secs, secid)

      # dir_list = os.listdir(PLOT_PATH)
      # print("Saved Decision Plots to\n'",PLOT_PATH,"'\n")
      # for dir_file in dir_list: print(dir_file) 

      # ---
      k_rec = recurse_dim_returns(A, n, result, PLOT_PATH, ERR_OPT_ACC, USE_CORR, USE_CUDA)
      
      return k_rec


# GLOBAL configs
# print(Path.cwd())
SCRATCH_FOLDER = "alle_frq_dirs/test_af"
SCRATCH_FOLDER = "alle_frq_dirs/sthd_af"
k_rec = sel_k_dimreturns(SCRATCH_FOLDER)

  
  

  
  
  
  
  
  
  
  
