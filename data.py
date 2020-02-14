import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft
from statistics import mean
import re

f1 = 'alphabet_11_12'
f2 = 'alphabet_11_12_logkeys'

def plotVectors(currentChar,vectors,ind,l,rng=20):
    if ind - rng/2 < 0:
        lInd = 0
    else:
        lInd = ind - int(rng/2)

    if ind + rng/2 > l:
        rInd = l
    else:
        rInd = ind + int(rng/2)
    
    t = []

    v = vectors[lInd:rInd]
    maggies = [np.linalg.norm(a) for a in v]

    # plt.rcParams.update({'font.size': 32})
    # plt.title('Accelerometer data for ' + currentChar)
    # plt.plot(maggies)

    return mean(maggies)

    # # plot x
    # axs[0].plot(x)
    # axs[0].set_title('Acceleration in x direction')
    # axs[0].set_xlabel('time')
    # axs[0].set_ylabel('G')
    # axs[0].grid(True)

    # # plot y
    # y = v[1]
    # axs[1].plot(y)
    # axs[1].set_title('Acceleration in y direction')
    # axs[1].set_xlabel('time')
    # axs[1].set_ylabel('G')
    # axs[1].grid(True)

    # #plot z
    # z = v[3]
    # axs[2].plot(z)
    # axs[2].set_title('Acceleration in z direction')
    # axs[2].set_xlabel('time')
    # axs[2].set_ylabel('G')
    # axs[2].grid(True)

    plt.show()

def plotLetter(currentChar,line,ind,l,rng=20):
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

    for i in range(lInd,rInd):
        # t.append(float(line[i][0]))
        x.append(float(line[i][1]))
        y.append(float(line[i][2]))
        z.append(float(line[i][3]))

    plt.rcParams.update({'font.size': 32})
    fig, axs = plt.subplots(3,1,constrained_layout=True)
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

    #plot z
    axs[2].plot(z)
    axs[2].set_title('Acceleration in z direction')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('G')
    axs[2].grid(True)

    plt.show()


with open(f1) as f: #accelerametor data
    content = f.read()
    accData = content.splitlines()[:-1]
    for i in range(len(accData)):
        accData[i] = accData[i].split()

with open(f2) as f: #logkey data
    content = f.read()
    pattern = re.compile("^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2} \d+ - \d+ > ")
    lkData = [a for a in content.splitlines() if pattern.match(a) is not None]
    for i in range(len(lkData)):
        lkData[i] = lkData[i][20:30] + '.' + lkData[i][33:]
        lkData[i] = lkData[i].split()
        # print(lkData)
        # input()


vectors = []

for l in accData:
    # print(l)
    v = []
    for i in range(1,len(l)):
        v.append(float(l[i]))
    vectors.append(v)

checkLs = ['y','z']
lMags = {}
for letter in checkLs:
    lMags[letter]=[0]

for i in range(len(accData)-1):
    for j in range(len(lkData)):
        if float(accData[i][0]) <= float(lkData[j][0]) and float(accData[i+1][0]) >= float(lkData[j][0]):
            for letter in checkLs:
                if lkData[j][-1] == letter:
                    # print(lkData[j][-1])
                    # plotLetter(c,accData,i,len(accData))
                    curMag = plotVectors(letter,vectors,i,len(accData))
                    lMags[letter].append(curMag)
for letter in checkLs:
    print(letter + " average magnitude: " + str(mean(lMags[letter])))
