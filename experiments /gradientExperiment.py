import cv2
import numpy as np
from pipeline import *

p = Pipeline(None)
p.pushStage(ResizeFactorStage, {'w': 0.2, 'h': 0.2})
p.pushStage(
    HSVThr,{
        "H_max":20.0,"H_min":0.0,
        "V_max":254.0,"V_min":41.0,
        "S_max":229.0,"S_Min":100.0
    }
)
p.pushStage(DeepRegionFilter, {'drawBox':False,'granularity':1,'tx': 100, 'ty': 100})

#p.pushStage(DbScan)

def convertVideo(filename):
    #cap = cv2.VideoCapture(f'../data/{filename}.mp4')
    cap = cv2.VideoCapture(0)

    out = None
    old = None
    new = None

    # Read until video is completed
    thr = 245

    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            if type(old) == type(None):
                old = frame
                height, width, layers = frame.shape
                size = (width, height)
                out = cv2.VideoWriter(f'{filename}.avi', cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)
                continue

            new = frame
            toDisp = new - old

            p.input = new
            # Display the resulting frame
            out = p.calculate()


            #new[0:out.shape[0],0:out.shape[1]] = out#cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

            cv2.imshow('Lines2',out)

            old = new
            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        else:
            break
    out.release()
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()


import os
files = os.listdir('../data/')
for i in files:
    if '.mp4' in i:
        name = i.replace('.mp4','')
        convertVideo(name)