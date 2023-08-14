#!/usr/bin/env python3
# this will read in allele frequencies for multiple 
# populations from the files 
# created by (or in the style of) the output of vcftools --freq
# 
# these files are tab delimited and contain a header row
# 
# the first four columns are: chromosome name, position, 
# number of alleles, and number of observations used to make the estimate
# of these only the first two are used. The other two columns are requried to be there
# but the values are ignored by this script
# the following columns (fifth and thereafter) contain an allele and its frequency
# in the format AlleleName:###
# 
# each population is represented by one file and all files must have 
# _the same loci in the same order_
# 
import os,sys,copy,time,glob,itertools,shutil,re,json
from pathlib import Path

# import more_itertools as mit
#
import math, random
import numpy as np

#
import torch
torch.cuda.empty_cache()
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda

import pandas as pd


from aipy.asgmlin import LPF


class PopDatasetStreamerLoader(): 
    """
    Population Dataset Streaming+Loading  Class
    
    Iterates over a batch of loci: 1 (one line) <= B <= L (all lines) 
    from a set of input Population files.
    
    Computes the Average Coancenstry and Heterozygozity matrix for the given batch
    
    Expects all input Population files (POP_FILES) have the same locus
    
    Useful when working with potentially large genomic datasets (millions of loci). By parsing locus by locus rather than reading the whole dataset (N*L loci) into memory, it reduces memory demand.   
    
    Arguments:
        POP_FILES
        
        neff
        
        max_batch_size
        
        avgmode
        
        device


    """
    @torch.no_grad()
    def __init__(self, POP_FILES,
        neff=0, max_batch_size=1, avgmode=3, freq=1, 
        shuffle=False, seed=0, debug=False, device='cpu'):
        
        n = len(POP_FILES)
        if n < 2:
            raise Exception(f"Expected Two or more populations, \
                            but got {n} populations!")
        if neff > n:
            raise Exception(f"Expected neff <= {n}, but got neff = {neff}!")
        if neff == 0: neff = n
        
        self.n = n    
        self.neff = neff
        self.POP_SET = tuple(POP_FILES[0:neff]) 
            
        self.Ho_mat = torch.zeros((neff,neff),dtype=torch.float32)    
        self.He_mat = torch.zeros((neff,neff),dtype=torch.float32)

        self.OPEN_POP_SET = []
        # open pop_files for reading and append to list    
        for pop_id in range(neff):
            
            pop_fread = open(self.POP_SET[pop_id], 'r')
            
            # in first population,
            # peek at any demo_line pattern of the populations data
            if pop_id == 0:
                pos = pop_fread.tell() # pointer to current position in file
                self.demo_line = re.split("\t|:",  
                            (next(itertools.islice(pop_fread,1,None), None)).rstrip())
                pop_fread.seek(pos) # point back to current position in file
                
            # skip header row
            self.header_row = next(pop_fread)
            self.OPEN_POP_SET.append(pop_fread)

            
        self.NALLELES = int(self.demo_line[2])
        self.START_COL = 5
        self.TOT_COLS = self.START_COL + (2*self.NALLELES) - 1 # len(demo_line)
        
        self.freq = freq
        self.MAX_BUFFER_SIZE = max_batch_size
        self.MAX_BATCH_SIZE = max_batch_size

        self.stable_avgfcn = LPF()
        self.BETA = torch.tensor(0.9999999,dtype=torch.float) 
        self.avgmode = avgmode       
        
        self.debug = debug
        self.response = 0
        self.device = device
        self.shuffle = shuffle
        self.seed = seed
        self.ptr = 0
        self.sizeinbuffer = 0
        self.line_cnt = torch.tensor(0,dtype=torch.int) 
        
    @torch.no_grad()
    def get_buffer(self,):
        '''
        get_buffer:     
        Computes the Average Coancenstry and Heterozygozity matrix over the max_batch_size from the set of opened input Population files.


        Returns:
            response (int): 0 (success)  
            SYS_MAT (tuple): Homozygozity and Heterozygozity matrices
        '''        
        
        # 1. READ LINES
        freqcnt = 0; 
        buf_cnt = 0; 
        POP_EOF = False
        while freqcnt < 1 and not POP_EOF:
            Ho_buffer = []
            He_buffer = [] 
            # track number of lines read into current batch/buffer
            buf_cnt = 0 
            # tot. number of lines read so far.
            # self.line_cnt
            
            while buf_cnt < self.MAX_BUFFER_SIZE and not POP_EOF:
                '''
                Line-by-Line computes the allele frequency similarity matrix for all opened population files
                
                Averaged allele frequency similarity matrix forms the 
                Heterozygozity and Homozygozity matrix.
                
                '''
                try:
                    locusline = np.array(
                        [re.split("\t|:", self.OPEN_POP_SET[pop_id].readline().strip()) for pop_id in range(self.neff)]) # .strip() => remove trailing newline
                    POP_EOF = False
                    
                    #TODO
                    
                    # allele frequency matrix for k alleles
                    ppmat_l = torch.from_numpy(np.asfarray(locusline[:,np.arange(self.START_COL,self.TOT_COLS,self.NALLELES)]))
                    
                    # homozygosity and heterozygozity matrix at each locus
                    Ho_Smat_l = ppmat_l.mm(ppmat_l.T)
                    He_Smat_l = 0.5*(1-Ho_Smat_l)
                    
                    self.line_cnt += 1    
                    _, self.Ho_mat, _ = self.stable_avgfcn.torch_ew_compute(
                        in_k=Ho_Smat_l, x=self.Ho_mat, beta=self.BETA, 
                        step=self.line_cnt, mode=self.avgmode
                    )
                    _, self.He_mat, _ = self.stable_avgfcn.torch_ew_compute(
                        in_k=He_Smat_l, x=self.He_mat, beta=self.BETA, 
                        step=self.line_cnt, mode=self.avgmode
                    )
                            
                    # Ho_buffer.append(Ho_Smat_l.clone())  
                    # He_buffer.append(He_Smat_l.clone()) 
                    buf_cnt += 1
                except:
                    locusline = None
                    POP_EOF = True
                    break
                    
            freqcnt +=1
            
        if buf_cnt == 0:
            response = -1
            self.sizeinbuffer = buf_cnt #len(Ho_buffer) # buf_cnt
            SYS_MAT = (self.Ho_mat, self.He_mat)
        else:
            response = 0
            #TODO: shuffle
            self.sizeinbuffer = buf_cnt #len(Ho_buffer) # buf_cnt
            SYS_MAT = (self.Ho_mat, self.He_mat)
        
        self.ptr = 0
        return response, SYS_MAT


    def __iter__(self):
        return self

    def __next__(self):  
        ''' Iterator on get_buffer(): 
        returns next batch as a tuple 
        
        Batch-by-Batch, the iterator moves from beginning to end of file 
        for all population files at once. 
        '''
        
        # init
        self.response = 0 

        # reload
        if self.ptr==0 or self.ptr >= self.sizeinbuffer:
            if self.debug: print(" ** getting buffer/batch ** \n")
            
            self.response, SYS_MAT = self.get_buffer() 
            # 0 = success, -1 = hit eof + empty buffer 
            # 1 = load next file [useful for loading from many files]

        if self.response == 0:
            self.ptr = self.ptr + self.MAX_BATCH_SIZE
            return SYS_MAT
        elif self.response < 0:      
        # reached EOF,
        # prepare for next epoch
            raise StopIteration

    def __len__(self):
        ''' Returns the length of locus lines in the current batch/buffer'''
        return self.Ho_buffer.shape[0] # size in buffer # len(self.input)

    def close(self):
        ''' Close all the opened population files '''
        for pop_id in range(self.neff):
            self.OPEN_POP_SET[pop_id].close()


# # print(Path.cwd())
# SERVER_ROOT = Path(__file__).parent 
# SCRATCH_FOLDER = "scratch"
# DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
# # print(server_root)
# # print(data_root)

# # search+select .frq files in scratch
# POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")

# data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,neff=0,max_batch_size=100, avgmode=3)
# print(data_ldr.neff)

#     # now we have a coancestry matrix that
#     # is also a matrix whose values represent the 
#     # (mean) expected HOMOzygosity of an offspring produced
#     # by crossing individuals from two pops
#     # NOTE: this can be considered the "M" matrix described in section 2.1 of Eding et al 2002
#     # where they use the term "kinships" instead of coancestries
#     print(coanMatrix)

#     # an example of calculating average expected HOMOzygosity
#     # (which is 1 - expected HETEROzygosity) for a population
#     # created by selecting broodstock from the populations
#     # with proportions given by c
#     # NOTE: this is application of equation (1) from Eding et al 2002
#     # for this example we are sampling broodstock in equal proportions from all populations
#     # making this a two-dimensional array (matrix) here just to make the math explicitly match
#     # Eding et al 2002 and related literature for ease of interpretation
#     c = np.full(coanMatrix.shape[0], 1 / coanMatrix.shape[0])[:,None]
#     meanOffspCoan = c.T @ coanMatrix @ c
#     print(meanOffspCoan[0,0])

#     # so the goal is to find c that MINIMIZES meanOffspCoan (which thereby 
#     # MAXIMIZES 1 - meanOffspCoan, or expected heterozygosity) with a 
#     # restricted number of values in c > 0
#     # c is the proportion of broodstock sampled from each population,
#     # so naturally the values of c are constrained to be in the range [0,1]
#     # and sum(c) = 1



if __name__ == "__main__":
    
    # print(Path.cwd())
    SERVER_ROOT = Path(__file__).parents[1] 
    SCRATCH_FOLDER = "scratch"
    DATA_ROOT = (SERVER_ROOT / SCRATCH_FOLDER ).resolve()
    # print(server_root)
    # print(data_root)

    # search+select .frq files in scratch
    POP_FILES = glob.glob(f"{DATA_ROOT}/*.frq")

    data_ldr = PopDatasetStreamerLoader(POP_FILES=POP_FILES,max_batch_size=100, avgmode=3)
    print(data_ldr.neff)