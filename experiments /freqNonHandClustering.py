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


dataset = np.array([])
granularity = 2
N = (2 * 10 ** (granularity)) ** 2
dataset = dataset.reshape(0,N,2)

p = Pipeline()
p.pushStage(ResizeFactorStage, {'w': 0.1, 'h': 0.1})
p.pushStage(FourierRegionTransform, {'drawBox':True,'granularity':granularity,'tx': 100, 'ty': 100})


for f in tqdm.tqdm(file_list[:1000]):
    file = f
    file = cv2.imread(f'{path}/{file}')
    p.input = file

    out = p.calculate()

    regions = p.stage_instances[-1].regions
    cv2.imshow('image',out)
    cv2.waitKey(1000)

print(f'Max {M} hz and min {m} hz sampling {s_}')
np.save('nonHands',dataset)



