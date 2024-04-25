import sys, os
from pathlib import Path
import copy, time, glob, itertools,shutil, re, json

import collections
import dataclasses

from typing import Optional, Tuple, TypeVar, List, Dict, Protocol
from collections import defaultdict, deque
from queue import Queue, PriorityQueue, LifoQueue

import math,random, cmath

import numpy as np

import pandas as pd

import torch, torchvision
torchvision.disable_beta_transforms_warning()

import torch.nn as nn
import torch.nn.functional as tf
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor, Lambda, v2

#
import matplotlib.pyplot as plt 
import matplotlib.ticker as ticker

    
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
from mpl_toolkits import mplot3d
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize


prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
# cmap = ListedColormap(colors)
cmap = LinearSegmentedColormap.from_list("cbcmap", colors)

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath} \usepackage{amssymb}')

plt.rcParams['axes.linewidth'] = 0.25
# sans: Helvetica, 'Computer Modern Serif'
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    # "font.sans-serif": "Helvetica", 
    "font.sans-serif": ["Computer Modern Serif"],
    })
# # for Palatino and other serif fonts use: 'New Century Schoolbook', 'Bookman', 'Times', 'Palatino', 'Charter', 'Computer Modern Roman'
plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "serif",
    "font.serif": ["Times"],
})

torch.random.manual_seed(0)

def reformat_large_tick_values(tick_val, pos=0):
    """
    https://dfrieds.com/data-visualizations/how-format-large-tick-values.html
    Turns large tick values (in the billions, millions and thousands) such as 4500 into 4.5K and also appropriately turns 4000 into 4K (no zero after the decimal).
    """
    if tick_val >= 1000000000:
        val = round(tick_val/1000000000, 1)
        new_tick_format = '{:}B'.format(val)
    elif tick_val >= 1000000:
        val = round(tick_val/1000000, 1)
        new_tick_format = '{:}M'.format(val)
    elif tick_val >= 1000:
        val = round(tick_val/1000, 1)
        new_tick_format = '{:}K'.format(val)
    elif tick_val < 1000:
        new_tick_format = round(tick_val, 1)
    else:
        new_tick_format = tick_val

    # make new_tick_format into a string value
    new_tick_format = str(new_tick_format)
    
    # code below will keep 4.5M as is but change values such as 4.0M to 4M since that zero after the decimal isn't needed
    index_of_decimal = new_tick_format.find(".")
    
    if index_of_decimal != -1:
        value_after_decimal = new_tick_format[index_of_decimal+1]
        if value_after_decimal == "0":
            # remove the 0 after the decimal point since it's not needed
            new_tick_format = new_tick_format[0:index_of_decimal] + new_tick_format[index_of_decimal+2:]
            
    return new_tick_format
  
def reformat_small_tick_values(tick_val, pos=0):
    """
    Formats small tick values
    """
    negsign = False
    sgn = str(tick_val)[0]
    if sgn == '-':
        negsign = True
    
    # fix for values greater than 0.01
    if tick_val < 10:
        
        estr = 'e-'
        ise = estr in str(tick_val)
        if not ise:
            estr = 'E-'
            ise = estr in str(tick_val)
            
        if ise:
            val, sigits = str(tick_val).split(estr)
            val = round(float(val),1)
            rdigits = str(val).split('.')
            if rdigits[-1] == '0':
                val = rdigits[0]
            if float(sigits) == 0:
                new_tick_format = f"{val}"
            else:
                new_tick_format = f"{val}E-{int(float(sigits))}"            
        else:
            tick_val = float(tick_val)/(10**2)
            
            estr = 'e-'
            ise = estr in str(tick_val)
            if not ise:
                estr = 'E-'
                ise = estr in str(tick_val)
                
            if ise:
                val, sigits = str(tick_val).split(estr)
                val = round(float(val),1)
                rdigits = str(val).split('.')
                if rdigits[-1] == '0':
                    val = rdigits[0]
                if int(float(sigits)-2) == 0:
                    new_tick_format = f"{val}"
                else:
                    new_tick_format = f"{val}E-{int(float(sigits)-2)}"
                    
            else:  
                sigits = len(str(tick_val))-2
                val = tick_val
                if sigits > 0:
                    sv = str(tick_val)
                    cnt = -1
                    for chr in sv.split('0'):
                        cnt+=1
                        if chr not in ['','.','-']:
                            val = chr
                            break
                    sigits = cnt
                    try:
                        digs = len(val)-1
                        val = round(float(val)/(10**digs),1)
                        rdigits = str(val).split('.')
                        if rdigits[-1] == '0':
                            val = rdigits[0]
                        if sigits-2 <= 0:
                            new_tick_format = f"{val}"
                        else:
                            new_tick_format = f"{val}E-{sigits-2}"
                    except:
                        new_tick_format = tick_val
                else:
                    new_tick_format = tick_val
    else:
        new_tick_format = reformat_large_tick_values(tick_val, pos)
            
    if negsign: 
        new_tick_format = '-' + new_tick_format
        
    return new_tick_format


# print(reformat_small_tick_values(1.6055063e-5))
# print(reformat_small_tick_values(0.000016055))
# print(reformat_small_tick_values(0.001))
# print(reformat_small_tick_values(-0.05))


def plotrc(imgs, keys=None, row_title=None):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False, figsize=(12,1.5))
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.plot(img, label= keys[col_idx], c=colors[col_idx])
            ax.legend()
            # ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
    # num_rows = len(imgs[0])
    # num_cols = len(imgs)
    # _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    # for col_idx, row in enumerate(imgs):
    #     for row_idx, img in enumerate(row):
    #         ax = axs[row_idx, col_idx]
    #         ax.plot(img)
    #         ax.legend()
    #         ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])
            
    
    plt.tight_layout()
    
    

def plotim(imgs, row_title=None, **imshow_kwargs):
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])
    _, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(img, **imshow_kwargs)
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    if row_title is not None:
        for row_idx in range(num_rows):
            axs[row_idx, 0].set(ylabel=row_title[row_idx])

    plt.tight_layout()
    

def nicefmt(figh5, ax, csts,name,xlabel,ylabel):
    ax.xaxis.set_tick_params(labelsize=csts['Fx']-0.5,length=1.5, width=csts['LW'],pad=0.5)
    ax.yaxis.set_tick_params(labelsize=csts['Fy']-0.5,length=1.5, width=csts['LW'],pad=0.5)
    ax.margins(y=0.01, tight=True)
    ax.set_xlabel(xlabel, fontsize=csts['Fx'], labelpad=0.5)
    ax.set_ylabel(ylabel, fontsize=csts['Fy'], labelpad=1.5)

    plt.grid(lw=0.05, axis='both')
    # plt.legend(loc='best', ncol=1, mode="shrink", shadow=False, fancybox=False,frameon=False, borderaxespad=0.,prop={'size':1.5})
    # plt.show()
    plt.tight_layout(pad=0.1)
    figpath = f"{name}.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(figh5)

def nicefmt2(figh5, ax, csts,name,xlabel,ylabel,legtitle,legcol=1):
    ax.xaxis.set_tick_params(labelsize=2.5,length=1.5, width=csts['LW'],pad=0.5)
    ax.yaxis.set_tick_params(labelsize=2.5,length=1.5, width=csts['LW'],pad=0.5)
    ax.margins(y=0.01, tight=True)
    ax.set_xlabel(xlabel, fontsize=2.5, labelpad=0.5)
    ax.set_ylabel(ylabel, fontsize=2.5, labelpad=1.5)

    plt.grid(lw=0.05, axis='both')
    plt.legend(loc='best', ncol=legcol, mode="shrink", shadow=False, fancybox=False,frameon=False, borderaxespad=0.,prop={'size':2}, title=legtitle, title_fontsize=2)
    # plt.show()
    plt.tight_layout(pad=0.1)
    figpath = f"{name}.png"
    plt.savefig(figpath, dpi=1200)
    plt.close(figh5)



def plotrc2(imgs, keys=None, enable_lbl=True):
    
    if not isinstance(imgs[0], list):
        # Make a 2d grid even if there's just 1 row
        imgs = [imgs]

    num_rows = len(imgs)
    num_cols = len(imgs[0])

    fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, 
                            squeeze=False, tight_layout=True,
                            figsize=((5/3)*num_cols, (1.25)*num_rows))
  
    cnt = 0
    for row_idx, row in enumerate(imgs):
        for col_idx, img in enumerate(row):
            
            ax = axs[row_idx, col_idx]
            xlbl = r"$t, \mathrm{\mathsf{iterations}}$"
            lbl = r'$'+f'{keys[row_idx][col_idx]}'+r'_t$'
            if enable_lbl:
                ax.plot(img, label=lbl, alpha=0.8, linewidth=0.5, c=colors[cnt])
            else:
                ax.plot(img, alpha=0.8, linewidth=0.5, c=colors[cnt])
            # ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
            
            ax.margins(y=0.1, tight=True)

            ax.xaxis.set_tick_params(labelsize=4)
            ax.yaxis.set_tick_params(labelsize=4)

            ax.set_xlabel(xlbl, fontsize=4)
            ax.set_ylabel(lbl, fontsize=4)
            ax.xaxis.set_major_formatter(
                ticker.FuncFormatter(reformat_large_tick_values))
            ax.yaxis.set_major_formatter(
                ticker.FuncFormatter(reformat_small_tick_values))

            ax.legend(loc='best', ncol=1, mode="shrink", shadow=False, fancybox=False,frameon=False, borderaxespad=0.,prop={'size':4})
            cnt+=1
                
            
    return fig

def nicefmt3(figh5, ax, csts,name,xlabel=None,ylabel=None, int=False, dpi=1200):
    ax.xaxis.set_tick_params(labelsize=csts['Fx']-0.5,length=csts['AL'], width=csts['LW'],pad=0.5)
    ax.yaxis.set_tick_params(labelsize=csts['Fy']-0.5,length=csts['AL'], width=csts['LW'],pad=0.5)
    
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=csts['Fx'], labelpad=0.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=csts['Fy'], labelpad=1.5)

    if int:
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    plt.legend(loc='best', ncol=1, mode="shrink", shadow=False, fancybox=False,frameon=False, borderaxespad=0.,prop={'size':1})
    
    ax.margins(y=0.05, tight=True)
    plt.tight_layout(pad=0.1)
    # plt.show()
    figpath = f"{name}.png"
    plt.savefig(figpath, dpi=dpi)
    plt.close(figh5)

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    