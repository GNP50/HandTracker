import os
import random
import matplotlib.pyplot as plt

import numpy as np

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

    fr = np.array([
        complex(fr[0][i],fr[1][i])
    for i in range(0,fr[0].shape[0])])
    fr = (fr-np.mean(fr))/np.std(fr)

    freq = np.fft.fftfreq(fr.shape[0])
    fr = (fr - np.mean(fr)) / np.std(fr)

    #plt.plot(freq, fr.real ** 2 + fr.imag ** 2)
    cv2.imshow('image2',out)

    #cv2.imshow('image',file)

    clusters = p.stage_instances[-1]
    toConc = np.array([[freq[i], fr.real[i] ** 2 + fr.imag[i] ** 2] for i in range(0, freq.shape[0])])
    data = np.unique(np.concatenate((data,toConc),0),axis=0)

    cv2.waitKey(1)


model = DBSCAN(eps=0.1, min_samples=24)
res = model.fit(data)

m = res.labels_.max()

info = {i: data[res.labels_ == i].shape[0] for i in range(0, m)}
avg = sum([info[k] for k in info]) / len(info)
info = {k: info[k] for k in info if info[k] >= avg}

dim = int(np.sqrt(len(info)))

for i in info:
    a = data[np.where(res.labels_ == i)]
    plt.scatter(a[:,0],a[:,1])
plt.show()
np.save('freq.np',data)



