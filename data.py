import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import fft

f1 = 'gibberish_11_12'
f2 = 'gibberish_11_12_logkeys'

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

    # fig = plt.figure()
    # plt.tight_layout()
    plt.rcParams.update({'font.size': 32})
    # plt.rcParams['figure.constrained_layout.use'] = True
    fig, axs = plt.subplots(3,1,constrained_layout=True)
    fig.suptitle('Accelerometer data for ' + currentChar)

    # plot x
    axs[0].plot(x)
    axs[0].set_title('Acceleration in x direction')
    axs[0].set_xlabel('time')
    axs[0].set_ylabel('G')
    axs[0].grid(True)
    # ax1 = fig.add_subplot(311)
    # ax1.set_ylabel('G-Force (Gs)')
    # ax1.set_xlabel('Time')
    # ax1.set_title('Acceleration in x direction')
    # ax1.plot(x)

    # plot y
    axs[1].plot(y)
    axs[1].set_title('Acceleration in y direction')
    axs[1].set_xlabel('time')
    axs[1].set_ylabel('G')
    axs[1].grid(True)
    # ax2 = fig.add_subplot(312)
    # ax2.set_ylabel('G-Force (Gs)')
    # ax2.set_xlabel('Time')
    # ax2.set_title('Acceleration in y direction')
    # ax2.plot(y)

    #plot z
    axs[2].plot(z)
    axs[2].set_title('Acceleration in z direction')
    axs[2].set_xlabel('time')
    axs[2].set_ylabel('G')
    axs[2].grid(True)
    # ax3 = fig.add_subplot(313)
    # ax3.set_ylabel('G-Force (Gs)')
    # ax3.set_xlabel('Time')
    # ax3.set_title('Acceleration in z direction')
    # ax3.plot(z)

    plt.show()


with open(f1) as f:
    content = f.read()
    lines1 = content.splitlines()[:-1]
    for i in range(len(lines1)):
        lines1[i] = lines1[i].split()

with open(f2) as f:
    content = f.read()
    lines2 = content.splitlines()[14:453]
    for i in range(len(lines2)):
        lines2[i] = lines2[i][20:30] + '.' + lines2[i][33:]
        lines2[i] = lines2[i].split()

# print(lines1[0])
# print(lines2[-1])

x = []

for l in lines1:
    x.append(float(l[3]))

# fs = 331
# X = fft(x,fs)
# X = abs(X[:len(X)//2])
# fmax = int(np.argmax(abs(X))*(fs/(X.size*2)))
# print(fmax)
# plt.plot(X)
# plt.show()
# # plt.plot(x)
# # plt.show()
    
checkLs = ['l']
for i in range(len(lines1)-1):
    for j in range(len(lines2)):
        if float(lines1[i][0]) <= float(lines2[j][0]) and float(lines1[i+1][0]) >= float(lines2[j][0]):
            # print(lines1[i])
            # print(lines2[j])
            # print(lines1[i+1])
            # print()
            # print(lines2[j][-1])

            for c in checkLs:
                if lines2[j][-1] == c:
                    print(lines2[j][-1])
                    plotLetter(c,lines1,i,len(lines1))

