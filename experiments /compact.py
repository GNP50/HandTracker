import os

from pipeline import *
import tqdm
import random


task = 'images'

toSaveDir = f'extraction_task_{task}'

file_list = os.listdir(toSaveDir)
random.shuffle(file_list)


dataset = np.array([])
granularity = 1
N = (2 * 10 ** (granularity)) ** 2
dataset = dataset.reshape(0,N,2)

for off,f in enumerate(tqdm.tqdm(file_list)):
    fixed_coeff = np.load(f'{toSaveDir}/{f}')
    dataset = np.append(dataset, fixed_coeff, axis=0)
np.save(f'merged_{task}',dataset)

