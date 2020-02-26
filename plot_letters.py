#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean, stdev
import re
from data_sort import *

f1 = 'data/alphabet_02_19'
f2 = 'data/alphabet_02_19_logkeys'

# accData entries look like [String time, String x, String y, String z]
acc_entry_list = []
with open(f1) as f:  # accelerametor data
    content = f.read()
    accData = content.splitlines()[:-1]
    for i in accData:
        acc_entry_list.append(AccEntry(i))

# lkData entries look like [String time, '>', String key]
lk_entry_list = []
with open(f2) as f:  # logkey data
    content = f.read()
    pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
    lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
    for line in lkData:
        line = line.split()
        lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))


checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

yeet = {}
yeeteronie = {}
for letter in checkLs:
    window_list = []
    win_mags = []
    yeeteronie[letter] = []
    key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
    for k in key_presses:
        window_list.append(Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)))
    for window in window_list:
        window.plot_window()
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
