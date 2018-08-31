# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 17:43:30 2017

@author: SrivatsanPC
"""

import holdem_calc as hc
import holdem_functions as hf
import time

#Example for calculating hand strength for a pair of cards without any board. For instance,
#let us calculate it for 3s 4h.
#start_time = time.time()
#card_1 = hf.Card('Ks')
#card_2 = hf.Card('Kh')
#combo = (card_1,card_2)
#out = hc.run((tuple(combo),),int(1e4),False,None,None,False)
#print(out)
#print("Time diff is", end_time-start_time)

#Example for calculating hand strength including the board cards.

card_1 = hf.Card('7h')
card_2 = hf.Card('Qd')
combo = (card_1,card_2)
#board=None
board = [hf.Card('6h'), hf.Card('6d'), hf.Card('7d'),hf.Card('Qs'),hf.Card('Qc')]
start_time = time.time()
out = hc.run((tuple(combo),),int(1000),True, board,None,False)
end_time = time.time()
print(out)
#print("Time diff is", end_time-start_time)


#Example to run with opponent and board
#start_time = time.time()
#card_e_1 = hf.Card('5c')
#card_e_2 = hf.Card('Ah')
#card_1 = hf.Card('3s')
#card_2 = hf.Card('4h')
#combo = (card_1,card_2)
#combo_e = (card_e_1,card_e_2)
#board = [hf.Card('Ac'), hf.Card('Tc'), hf.Card('Qd')]
#out = hc.run((tuple([combo,combo_e])),int(1e5),False, board,None,False,pad_opp=False)
#end_time = time.time()
#print(out)
#print("Time diff is", end_time-start_time)
