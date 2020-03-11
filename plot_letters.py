#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean, stdev
import re
from data_sort import *

f1 = 'data/alphabet_02_19'
f2 = 'data/alphabet_02_19_logkeys'

# # accData entries look like [String time, String x, String y, String z]
# acc_entry_list = []
# with open(f1) as f:  # accelerametor data
    # content = f.read()
    # accData = content.splitlines()[:-1]
    # for i in accData:
        # acc_entry_list.append(AccEntry(i))

# # lkData entries look like [String time, '>', String key]
# lk_entry_list = []
# with open(f2) as f:  # logkey data
    # content = f.read()
    # pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
    # lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
    # for line in lkData:
        # line = line.split()
        # lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))

acc_entry_list = make_AccEntry_List(f1)
lk_entry_list = make_LKEntry_List(f2)


checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

window_dict = make_window_dict(checkLs,acc_entry_list,lk_entry_list)
add_non_keypress(window_dict,f1)
# print(window_dict['none'])
yeet = {}
yeeteronie = {}
# for window in window_dict['none']:
    # window.plot_window()
for letter in checkLs:
    win_mags = []
    yeeteronie[letter] = []
    for window in window_dict[letter]:
        window.plot_window(window_dict)
        yeeteronie[letter].append(window.get_magnitudes())
        win_mags.append(window.get_teddy_mags())
    print('character: ' + letter + ' mags')
    # for win_mag in win_mags:
        # print(' mag = {}'.format(win_mag))
    # yeet[letter] = win_mags
# for letter in yeet:
    # plt.hist(yeet[letter], bins=100, range=(60, 350))
    # plt.title(letter)
    # plt.show()
# for letter in yeeteronie:
    # plt.hist(yeeteronie[letter], bins=100, )
    # plt.show()
