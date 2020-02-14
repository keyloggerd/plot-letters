#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean
import re

f1 = 'alphabet_11_12'
f2 = 'alphabet_11_12_logkeys'


class AccEntry(object):
    def __init__(self, lst):
        lst = lst.split()
        self.time = float(lst[0])
        self.acceleration = (float(lst[1]), float(lst[2]), float(lst[3]))

    def get_x(self):
        return self.acceleration[0]

    def get_y(self):
        return self.acceleration[1]

    def get_z(self):
        return self.acceleration[2]

    def get_acceleration(self):
        return self.acceleration

    def get_maggy(self):
        return np.linalg.norm(self.acceleration)

    def get_magnitude(self):
        return self.get_maggy()


class LKEntry(object):
    def __init__(self, time, key):
        self.time = float(time) - 60
        self.key = key


class Window(object):
    def __init__(self, letter, acc_entry_list, center_idx, rng=20):
        self.letter = letter
        self.window = self.get_range(acc_entry_list, center_idx, rng)

    def get_range(self, vectors, idx, rng=20):
        # return vectors[max(0, idx-int(rng/2)):min(len(vectors), idx+int(rng/2))]
        return vectors[idx-int(rng/2):idx+int(rng/2)]

    def get_magnitudes(self):
        return [a.get_maggy() for a in self.window]


def get_index_of_matching_time(acc_entry_list, time):
    for i in range(len(acc_entry_list)-1):
        if float(acc_entry_list[i].time) <= time and float(acc_entry_list[i+1].time) >= time:
            return i


def plotLetter(currentChar, line, ind, l, rng=20):
    if ind - rng/2 < 0:
        lInd = 0
    else:
        lInd = ind - int(rng/2)

    if ind + rng/2 > l:
        rInd = l
    else:
        rInd = ind + int(rng/2)

    t = []
    x = []
    y = []
    z = []

    for i in range(lInd, rInd):
        # t.append(float(line[i][0]))
        x.append(float(line[i][1]))
        y.append(float(line[i][2]))
        z.append(float(line[i][3]))

    plt.rcParams.update({'font.size': 32})
    fig, axs = plt.subplots(3, 1, constrained_layout=True)
    fig.suptitle('Accelerometer data for ' + currentChar)

    # plot x
    axs[0].plot(x)
    axs[0].set_title('Acceleration in x direction')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('G')
    axs[0].grid(True)

    # plot y
    axs[1].plot(y)
    axs[1].set_title('Acceleration in y direction')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('G')
    axs[1].grid(True)

    # plot z
    axs[2].plot(z)
    axs[2].set_title('Acceleration in z direction')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('G')
    axs[2].grid(True)

    plt.show()


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



vectors = []

checkLs = ['y', 'z']
windows = {'y': [], 'z': []}

for letter in checkLs:
    avg_window = []
    window_list = []
    key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
    for k in key_presses:
        window_list.append(Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)))
    for i in range(len(window_list[0].window)):
        maggies = []
        for j in range(len(window_list)):
            maggies.append(window_list[j].window[i].get_maggy())
        avg_window.append(mean(maggies))
    plt.plot(avg_window)
    plt.show()
