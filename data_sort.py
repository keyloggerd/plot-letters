#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean, stdev
import re

class AccEntry(object):
    def __init__(self, lst):
        lst = lst.split()
        self.time = float(lst[0])
        self.acceleration = (float(lst[1]), float(lst[2]), float(lst[3]))

    def get_time(self):
        return self.time

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

def make_AccEntry_List(acc_file):
    # accData entries look like [String time, String x, String y, String z]
    acc_entry_list = []
    with open(acc_file) as f:  # accelerametor data
        content = f.read()
        accData = content.splitlines()[:-1]
        for i in accData:
            acc_entry_list.append(AccEntry(i))
    return acc_entry_list

class LKEntry(object):
    def __init__(self, time, key):
        self.time = float(time)
        self.key = key

def make_LKEntry_List(lk_file):
    # lkData entries look like [String time, '>', String key]
    lk_entry_list = []
    with open(lk_file) as f: # logkey data
        content = f.read()
        pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
        lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
        for line in lkData:
            line = line.split()
            lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))
    return lk_entry_list


class Window(object):
    def __init__(self, letter, acc_entry_list, center_idx, rng=20):
        self.letter = letter
        self.window = self.get_range(acc_entry_list, center_idx, rng)

    def get_range(self, vectors, idx, rng=20):
        # return vectors[max(0, idx-int(rng/2)):min(len(vectors), idx+int(rng/2))]
        return vectors[idx-int(rng/2):idx+int(rng/2)]

    def get_magnitudes(self):
        return [a.get_maggy() for a in self.window]

    def get_teddy_mags(self):
        win_mag = self.get_magnitudes()
        return 10000 * stdev(win_mag) * np.sqrt(max(win_mag)**2 + min(win_mag)**2)

    def plot_window(self):
        print(self.window)
        plt.rcParams.update({'font.size': 32})
        fig, axs = plt.subplots(3, 1, constrained_layout=True)
        fig.suptitle('Accelerometer data for ' + self.letter)

        x = []
        y = []
        z = []
        t = []

        for acc_ent in self.window:
            x.append(acc_ent.get_x())
            y.append(acc_ent.get_y())
            z.append(acc_ent.get_z())
            t.append(float('%.3f'%(acc_ent.get_time()%1000)))

        # plot x
        axs[0].plot(t,x)
        axs[0].set_title('Acceleration in x direction')
        axs[0].set_xlabel('time')
        axs[0].set_ylabel('G')
        axs[0].grid(True)

        # plot y
        axs[1].plot(t,y)
        axs[1].set_title('Acceleration in y direction')
        axs[1].set_xlabel('time')
        axs[1].set_ylabel('G')
        axs[1].grid(True)

        # plot z
        axs[2].plot(t,z)
        axs[2].set_title('Acceleration in z direction')
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('G')
        axs[2].grid(True)

        plt.show()

def make_window_dict(checkLs,acc_entry_list=[],lk_entry_list=[]):
    window_dict = {}
    if len(acc_entry_list) == 0 or len(lk_entry_list) == 0:
        print("nah dude")
        return
    else:
        for letter in checkLs:
            window_dict[letter] = []
            key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
            for k in key_presses:
                window_dict[letter].append(Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)))
    return window_dict

def add_non_keypress(window_dict,acc_file):
    times = []
    window_dict['none'] = []
    for letter in window_dict:
        for window in window_dict[letter]:
            for acc_entry in window.window:
                times.append(acc_entry.get_time())
    with open(acc_file) as f:
        content = f.read()
        accData = content.splitlines()[:-1]
        unique_accs = []
        for lst in accData:
            t = float(lst.split()[0])
            if t not in times:
                unique_accs.append(AccEntry(lst))
            elif len(unique_accs) > 0:
                window_dict['none'].append(Window('none',unique_accs,int(len(unique_accs)/2),len(unique_accs)))
                unique_accs = []

def get_index_of_matching_time(acc_entry_list, time):
    for i in range(len(acc_entry_list)-1):
        if float(acc_entry_list[i].time) <= time and float(acc_entry_list[i+1].time) >= time:
            return i

