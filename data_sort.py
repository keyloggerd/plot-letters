#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean, stdev
import re
import pandas as pd

import time

default_range = 20

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

def make_AccEntry_List(acc_files):
    # accData entries look like [String time, String x, String y, String z]
    if not isinstance(acc_files,list): acc_files = [acc_files]
    acc_entry_list = []
    for acc_file in acc_files:
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

def make_LKEntry_List(lk_files):
    # lkData entries look like [String time, '>', String key]
    if not isinstance(lk_files,list): lk_files = [lk_files]
    lk_entry_list = []
    for lk_file in lk_files:
        with open(lk_file) as f: # logkey data
            content = f.read()
            pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > \w")
            lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
            for line in lkData:
                line = line.split()
                lk_entry_list.append(LKEntry(line[2]+"."+line[4], line[6]))
    return lk_entry_list


class Window(object):
    def __init__(self, letter, acc_entry_list, center_idx, rng=default_range):
        self.letter = letter
        self.window = self.get_range(acc_entry_list, center_idx, rng)

    def get_range(self, vectors, idx, rng=default_range):
        # return vectors[max(0, idx-int(rng/2)):min(len(vectors), idx+int(rng/2))]
        return vectors[idx-int(rng/2):idx+int(rng/2)]

    def get_magnitudes(self):
        return [a.get_maggy() for a in self.window]

    def get_teddy_mags(self):
        win_mag = self.get_magnitudes()
        return 10000 * stdev(win_mag) * np.sqrt(max(win_mag)**2 + min(win_mag)**2)

    def plot_window(self,window_dict,norm=True):
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

        if norm:
            times = get_window_times(window_dict)
            norm_fac =float('%.3f'%(min(times)%1000))
            t = [x - norm_fac for x in t]

        acc = get_window_acc(window_dict)
        x_all = [a[0] for a in acc]
        y_all = [a[1] for a in acc]
        z_all = [a[2] for a in acc]

        # plot x
        axs[0].plot(t,x)
        axs[0].set_title('Acceleration in x direction')
        # axs[0].set_xlabel('time')
        axs[0].set_ylabel('G')
        axs[0].set_ylim(min(x_all),max(x_all))
        axs[0].grid(True)

        # plot y
        axs[1].plot(t,y)
        axs[1].set_title('Acceleration in y direction')
        # axs[1].set_xlabel('time')
        axs[1].set_ylabel('G')
        axs[1].set_ylim(min(y_all),max(y_all))
        axs[1].grid(True)

        # plot z
        axs[2].plot(t,z)
        axs[2].set_title('Acceleration in z direction')
        axs[2].set_xlabel('time')
        axs[2].set_ylabel('G')
        axs[2].set_ylim(min(z_all),max(z_all))
        axs[2].grid(True)

        plt.show()

def make_window_dict(checkLs,acc_entry_list=[],lk_entry_list=[]):
    if len(acc_entry_list) == 0 or len(lk_entry_list) == 0:
        print("nah dude")
        return
    else:
        window_dict = {}
        for letter in checkLs:
            window_dict[letter] = []
            key_presses = list(filter(lambda x: x.key == letter, lk_entry_list))
            for k in key_presses:
                if(get_index_of_matching_time(acc_entry_list, k.time) != None):
                    window_dict[letter].append(Window(letter, acc_entry_list, get_index_of_matching_time(acc_entry_list, k.time)))
    return window_dict

def make_dataframe(checkLs,acc_entry_list=[],lk_entry_list=[],rng=default_range):
    if len(acc_entry_list) == 0 or len(lk_entry_list) == 0:
        print("nah dude")
        return
    else:
        # df = pd.DataFrame()
        df_list = []
        for lk_entry in lk_entry_list:
            if lk_entry.key.isalpha():
                idx = get_index_of_matching_time(acc_entry_list,lk_entry.time)
                acc_sub = acc_entry_list[idx-int(rng/2):idx+int(rng/2)]
                # print(acc_sub)
                # acc_sub_list = [(a.get_time,a.get_acceleration) for a in acc_sub]
                for acc in acc_sub:
                    df_list.append([lk_entry.key,acc.get_time(),acc.get_x(),acc.get_y(),acc.get_z()])
                # for acc in acc_sub:
                    # df_entry = pd.DataFrame([[lk_entry.key,acc.get_time(),acc.get_x(),acc.get_y(),acc.get_z()]],columns=['letter','time','x','y','z'])
                    # df = df.append(df_entry)
        df = pd.DataFrame(df_list,columns=['letter','time','x','y','z'])
        print('sorting')
        df.sort_values('time')
        print('done sorting')
        print(df)
        return df

def get_window_times(window_dict):
    if isinstance(window_dict,dict):
        times = []
        for letter in window_dict:
            for window in window_dict[letter]:
                for acc_entry in window.window:
                    times.append(acc_entry.get_time())
    elif isinstance(window_dict,pd.DataFrame):
        times = window_dict['time'].values.tolist()
    return times

def get_window_acc(window_dict):
    acc = []
    for letter in window_dict:
        for window in window_dict[letter]:
            for acc_entry in window.window:
                acc.append(acc_entry.get_acceleration())
    return acc

def add_non_keypress(data,acc_files,split=False,rng=default_range):
    if not isinstance(acc_files,list): acc_files = [acc_files]
    times = get_window_times(data)
    if isinstance(data,dict):
        data['none'] = []
    for acc_file in acc_files:
        with open(acc_file) as f:
            content = f.read()
            accData = content.splitlines()[:-1]
            unique_accs = []
            for lst in accData:
                t = float(lst.split()[0])
                if t not in times:
                    unique_accs.append(AccEntry(lst))
                elif len(unique_accs) > 0:
                    if split:
                        for i in range(0,len(unique_accs),rng):
                            if i+rng < len(unique_accs):
                                if isinstance(data,dict):
                                    data['none'].append(Window('none',unique_accs,i+int(rng/2),rng))
                                elif isinstance(data,pd.DataFrame):
                                    data_sub = []
                                    for j in range(i,i+rng):
                                        data_sub.append(['none',unique_accs[j].get_time(),unique_accs[j].get_x(),unique_accs[j].get_y(),unique_accs[j].get_z()])
                                    data = data.append(pd.DataFrame(data_sub,columns=['letter','time','x','y','z']),ignore_index=True)
                    else:
                        if isinstance(data,dict):
                            data['none'].append(Window('none',unique_accs,int(len(unique_accs)/2),len(unique_accs)))
                        elif isinstance(data,pd.DataFrame):
                            data_sub = []
                            for j in range(len(unique_accs)):
                                data_sub.append(['none',unique_accs[j].get_time(),unique_accs[j].get_x(),unique_accs[j].get_y(),unique_accs[j].get_z()])
                            data.append(pd.DataFrame(data_sub),ignore_index=True)
                    unique_accs = []
    return data

def get_index_of_matching_time(acc_entry_list, time):
    for i in range(len(acc_entry_list)-1):
        if float(acc_entry_list[i].time) <= time and float(acc_entry_list[i+1].time) >= time:
            return i

#def get_finger(window_dict):
#    pointer={'r', 't', 'y', 'u', 'f', 'g', 'h', 'j', 'v', 'b', 'n', 'm'}
#    middle={'e', 'd', 'c', 'i', 'k'}
#    ring = {'w', 's', 'x', 'i', 'k'}
#    pinky = {'q','a','z','p'}
#    for letter in window_dict:
#        if letter in pointer:
#            for window in window_dict[letter]:
                
def test_stuff():

    f1 = 'data/alphabet_04_10'
    f2 = 'data/alphabet_04_10_logkeys'
    f3 = 'data/alphabet_02_19'
    f4 = 'data/alphabet_02_19_logkeys'

    # letters to look for
    checkLs = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    acc_entry_list = make_AccEntry_List(f1)
    lk_entry_list = make_LKEntry_List(f2)
    
    # df = make_dataframe(checkLs,acc_entry_list,lk_entry_list)
    start = time.time()
    df = make_dataframe(checkLs,acc_entry_list,lk_entry_list)
    end = time.time()
    print('first function: ' + str(end-start))
    start = end
    window_dict = make_window_dict(checkLs,acc_entry_list,lk_entry_list)
    end = time.time()
    print('second function: ' + str(end-start))

    start = time.time()
    df = add_non_keypress(df,f1,split=True)
    end = time.time()
    print('third function: ' + str(end-start))
    start = time.time()
    window_dict = add_non_keypress(window_dict,f1,split=True)
    end = time.time()
    print('fourth function: ' + str(end-start))
    print(df)

    return df
    

df = test_stuff()
