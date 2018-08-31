# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 01:01:50 2017

@author: SrivatsanPC
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data_1 = pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\p1_reward.csv")
data_2 = pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\p2_reward.csv")
#data_2 = pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\NFSP_v_DQN.csv")
#data_2= pd.read_csv("E:\\CS281AdvancedML\\cs281-final-project\\Reports\\num.csv")
def moving_avg(x, window=100):
    return [np.mean(x[k:k+window]) for k in range(len(x)-window)]
mvg_avg_1 =  moving_avg(data_1["Value"], window=35)
mvg_avg_2 =  moving_avg(data_2["Value"], window=35)
#mvg_avg_2 = moving_avg(data_2["Value"]*-1 + 0.7, window=150)
leng = len(mvg_avg_1)
plt.plot(range(0,leng * 75,75), mvg_avg_1[:leng], label = 'Player 1(NFSP)')
plt.plot(range(0,leng * 75,75), mvg_avg_2[:leng], label = 'Player 2(NFSP)')
plt.ylabel("Avg Rewards/ Episode")
plt.xlabel("Number of Games")
#plt.title("Simulation Results")
plt.legend(loc=2)
SMALL_SIZE = 8
MEDIUM_SIZE=12
BIGGER_SIZE=12

plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.show()