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


class LKEntry(object):
    def __init__(self, time, key):
        self.time = float(time)
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

def get_index_of_matching_time(acc_entry_list, time):
    for i in range(len(acc_entry_list)-1):
        if float(acc_entry_list[i].time) <= time and float(acc_entry_list[i+1].time) >= time:
            return i

