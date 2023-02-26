import os

from pipeline import *
import cv2
import tqdm
import random

import subprocess

task = 'images'
path = f'../data/{task}'

toSaveDir = f'extraction_task_{task}'
subprocess.run(f'rm -rf {toSaveDir}', cwd=os.getcwd(),shell=True)
subprocess.run(f'rm -rf {toSaveDir}.zip', cwd=os.getcwd(),shell=True)

os.mkdir(toSaveDir)

file_list = os.listdir(path)
random.shuffle(file_list)


dataset = np.array([])
granularity = 1
N = (2 * 10 ** (granularity)) ** 2
dataset = dataset.reshape(0,N,2)

p = Pipeline()
p.pushStage(ResizeFactorStage, {'w': 0.4, 'h': 0.4})
p.pushStage(FourierRegionTransform, {'drawBox':False,'granularity':granularity,'tx': 100, 'ty': 100})

steps =10
step =0

for off,f in enumerate(tqdm.tqdm(file_list)):
    file = f
    file = cv2.imread(f'{path}/{file}')
    p.input = file

    out = p.calculate()

    regions = p.stage_instances[-1].regions
    for r in regions:
        fixed_coeff = r.coeff
        fixed_coeff = fixed_coeff.reshape(1, fixed_coeff.shape[0], fixed_coeff.shape[1])
        dataset = np.append(dataset, fixed_coeff, axis=0)

    if (off+1)%steps == 0:
        np.save(f'{toSaveDir}/step_{step}', dataset)
        del dataset
        dataset = np.array([])
        dataset = dataset.reshape(0, N, 2)
        step += 1

subprocess.run(f'zip -r  {toSaveDir}.zip {toSaveDir}', cwd=os.getcwd(),shell=True)
subprocess.run(f'rm -rf {toSaveDir}', cwd=os.getcwd(),shell=True)


