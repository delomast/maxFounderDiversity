import os,sys, shutil,glob,time,random,copy,math

from pathlib import Path

from utility import *



# GLOBAL configs
# print(Path.cwd())
SCRATCH_FOLDER = "alle_frq_dirs/test_af"
# SCRATCH_FOLDER = "alle_frq_dirs/sthd_af"
SERVER_ROOT = Path(__file__).parents[0]
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

# MAX_STEPS = (N_EFF**2)
MAX_STEPS = 2500
NO_MAXSTEPS = True

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
  print("Epoch: " + str(epoch+1)) 
  walltime = time.time()
  
  ''' LEARN RELATIVE CONTRIBUTIONS OF EACH POPULATION. '''
  # load dataset, and dataloader
  for (b_idx, batch) in enumerate(data_ldr):
    # print("epoch: " + str(epoch) + "   batch: " + str(b_idx)) 
    # test: # A = torch.tensor([[2.5409, -0.0113],[-0.0113, 0.5287]])
    A = 1*batch[0] #xz homozygozity
    
  #.. learn after full pass over data
  loss, c_t, y, alphas, betas = trainmdl(mdl, A,
          USE_CUDA, USE_CORR, MAX_STEPS, NO_MAXSTEPS, ERR_OPT_ACC, 
          QUAD_OBJ_CHOICE, SVLISTS)
  
  ''' CHOOSE POPULATIONS. '''
  # sort contributions     
  result = get_optimal_sets(POP_FILES, n, c_t)  
  
  walltime = (time.time() - walltime)/60 
  print(f"\nTotal batches: {b_idx+1}, time elapsed: {walltime:.2f}-mins") 
  data_ldr.close()
  data_ldr.batches = b_idx+1
  print("End epoch.")
  # Print out ...
  print(f"loss:{loss}")   
  ''' 1 EPOCH END.'''
  

  PLOT_PATH = str(SERVER_ROOT/f"static/trainplts/{SCRATCH_FOLDER}")
  os.makedirs(PLOT_PATH, exist_ok=True)
  web_render_results(PLOT_PATH,n,result,SVLISTS, skip_heavy_plots=False)
  