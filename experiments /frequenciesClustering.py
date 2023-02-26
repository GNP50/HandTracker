import os
import random
import matplotlib.pyplot as plt

import numpy as np
import scan

from pipeline import *
import cv2
import tqdm
import random
def randomColor():
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    rgb = (b,g,r)
    return rgb#(255,0,0)


path = '../data/images'
file_list = os.listdir(path)
random.shuffle(file_list)
data = np.array([])
data = data.reshape(0,3)

data = np.array([])
data = data.reshape(0,2)

dataset = np.array([])
granularity = 2
N = (2 * 10 ** (granularity)) ** 2

dataset = dataset.reshape(0,N,2)

m = None
M = None
s_ = None


for f in tqdm.tqdm(file_list[:100]):
    file = f
    file = cv2.imread(f'{path}/{file}')

    p = Pipeline(file)
    p.pushStage(ResizeFactorStage,{'w':0.4,'h':0.4})

    p.pushStage(DbScan, {'tx':100, 'ty':100,'color':255})
    p.pushStage(BWStage)
    p.calculate()
    out = p.calculate()
    fr = np.where(out > 0)

    fr = fr[0] - fr[0].min(),fr[1] - fr[1].min()
    fr = 2*fr[0]/fr[0].max(),2*fr[1]/fr[1].max()
    fr = fr[0]-1,fr[1]-1
    fr = fr[0].round(decimals=2),fr[1].round(decimals=2)


    fr = np.array([
        complex(fr[0][i],fr[1][i])
    for i in range(0,fr[0].shape[0])])
    fr = scan.prepareFourierGrid(fr, granularity)

    toRet = np.fft.fftfreq(N, 10**(-granularity))

    fCoeff = np.fft.fft(fr)
    idx = np.arange(0, fCoeff.shape[0])
    rapp = int(N/fr.shape[0])
    sampled = np.fft.fftfreq(fr.shape[0],10**(-granularity))


    fixed_coeff = np.zeros((N,2),dtype=complex)
    fixed_coeff[idx*rapp] = fCoeff[idx]
    fixed_coeff = fixed_coeff.reshape(1, fixed_coeff.shape[0], fixed_coeff.shape[1])
    dataset = np.append(dataset,fixed_coeff,axis=0)

    del fixed_coeff
    del sampled
    del fr
    del out
    del file
    del p

    #rv = np.fft.ifft(fixed_coeff)

    #plt.scatter(rv[:,0],rv[:,1])

    #plt.show()


print(f'Max {M} hz and min {m} hz sampling {s_}')
np.save('frequenciesBorder',dataset)



