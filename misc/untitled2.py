# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 00:25:59 2017

@author: SrivatsanPC
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
sns.set(style='white', palette='Set1')
FONT_SIZE = 20
font = {'weight' : 'bold',
        'size'   : FONT_SIZE}
matplotlib.rc('font', **font)

SMALL_SIZE = 8
MEDIUM_SIZE=12
BIGGER_SIZE=14

data_2 = pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\pi_loss_1.csv")
data_1 = pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\q_loss_1.csv")
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

def moving_avg(arr, window):
    return [np.mean(arr[i:i+window]) for i in range(len(arr) - window + 1)]

yy = moving_avg((data_1['Value']/10).tolist(), 50)
xx = data_1['Step'].tolist()[:len(yy)]

xx1 = data_2['Step'].tolist()[:len(yy)]
max_p = max(data_2['Value'])
min_p = min(data_2['Value'])
pi = max_p - (max_p - data_2['Value'] ) * 1.5 / (max_p - min_p)
#import pdb; pdb.set_trace()
yy1 = moving_avg(pi[:len(yy)+49],50)

xx1 = range(0,15000,200)
xx = xx1
#import pdb; pdb.set_trace()
fig, ax1 = plt.subplots(figsize=(10, 10))
ax1.plot(xx, yy, 'b', label='Pi')
ax1.set_xlabel('Number of Iterations', fontsize=FONT_SIZE)
ax1.set_ylabel('MSE Loss of Q Network', fontsize=FONT_SIZE)
ax1.yaxis.label.set_color('b')
ax1.tick_params(labelsize=FONT_SIZE)
#ax1.legend(loc=3)

ax2 = ax1.twinx()
ax2.plot(xx1, yy1, 'r', label='Q')
ax2.tick_params(labelsize=FONT_SIZE)
ax2.set_ylabel('Cross Entropy Loss of Pi Network', fontsize=FONT_SIZE)
ax2.yaxis.label.set_color('r')
#ax2.legend(loc=3)
#fig.legend(handles=[ax1,ax2], labels = ['Pi', 'Q'], loc=3, fontsize=FONT_SIZE)
plt.show()