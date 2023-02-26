import cv2
import numpy as np
from  deepModels import *
import torch
from sklearn.cluster import DBSCAN
import random
import copy
import scan

def getFrequenciesFromPoints(fr):
    fr = np.array([
        complex(fr[0][i], fr[1][i])
        for i in range(0, fr[0].shape[0])])
    fr = (fr - np.mean(fr)) / np.std(fr)

    freq = np.fft.fftfreq(fr.shape[0])
    fr = (fr - np.mean(fr)) / np.std(fr)
    return freq,fr


def getDbScanInfo(model,X):
    res = model.fit(X)
    m = res.labels_.max()
    info = {i: X[res.labels_ == i].shape[0] for i in range(0, m)}
    if len(info) == 0:
        return {},res
    avg = sum([info[k] for k in info]) / len(info)
    info = {i: info[i] for i in info if info[i] > avg}
    return info,res

def randomColor():
    r = random.randint(100, 255)
    g = random.randint(100, 255)
    b = random.randint(100, 255)
    rgb = (b,g,r)
    return rgb#(255,0,0)

class PipelineStage:
    def __init__(self,src):
        self.src = src
        self.out = src

    def getOut(self):
        self.out

class Pipeline:
    def __init__(self,input=None):
        self.stages = []
        self.input = input
        self.params = []
        self.stage_instances = []

    def pushStage(self,stage,params={}):
        self.stages.append(stage)
        self.params.append(params)

    def calculate(self):
        in_ = self.input

        for off,i in enumerate(self.stages):
            a = i(in_)
            self.stage_instances.append(a)
            for k in self.params[off]:
                a.__dict__[k] = self.params[off][k]

            in_ = a.getOut()
        return in_


class HSVThr(PipelineStage):
    def __init__(self,src,H_max=255,H_min=0,S_max=255,S_min=0,V_max=255,V_min=0):
        PipelineStage.__init__(self,src)
        self.H_max = H_max
        self.H_min = H_min
        self.S_max = S_max
        self.S_min = S_min
        self.V_max = V_max
        self.V_min = V_min

    def getOut(self):
        self.out = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)
        idx = self.out[...,0] > self.H_max
        idx = np.logical_or(idx,self.out[...,1] > self.S_max)
        idx = np.logical_or(idx,self.out[...,2] > self.V_max)

        idx = np.logical_or(idx, self.out[..., 0] < self.H_min)
        idx = np.logical_or(idx, self.out[..., 1] < self.S_min)
        idx = np.logical_or(idx, self.out[..., 2] < self.V_min)

        self.out[idx] = (0,0,0)

        return  self.out


class BWStage(PipelineStage):
    def __init__(self,src):
        PipelineStage.__init__(self,src)
    def getOut(self):
        self.out = cv2.cvtColor(self.src, cv2.COLOR_BGR2GRAY)
        return  self.out

class ResizeFactorStage(PipelineStage):
    def __init__(self,src,w=0.5,h=0.5):
        PipelineStage.__init__(self,src)
        self.w = w
        self.h = h

    def getOut(self):
        width = int(self.src.shape[1] * self.w)
        height = int(self.src.shape[0] * self.h)
        dim = (width, height)

        self.out = cv2.resize(self.src, dim, interpolation=cv2.INTER_AREA)
        return  self.out


class DiffStage(PipelineStage):
    def __init__(self,src):
        PipelineStage.__init__(self,src)
        self.new =  None
    def getOut(self):
        self.out = self.src-self.new
        return  self.out

class ThreasholdStage(PipelineStage):
    def __init__(self,src,threshold,min=255,max=0):
        PipelineStage.__init__(self,src)
        self.threshold = threshold
        self.min = min
        self.max = max

    def getOut(self,new):
        self.out = self.src
        self.out[self.out > self.threshold] = self.max
        self.out[self.out <= self.threshold] = self.min
        return  self.out



class BlurStage(PipelineStage):
    def __init__(self, src,w=10,h=10):
        PipelineStage.__init__(self, src)
        self.w = w
        self.h = h

    def getOut(self, new):
        self.out = cv2.blur(self.src,(self.w,self.h))
        return self.out

class CannyStage(PipelineStage):
    def __init__(self, src,tx=10,ty=10,color=(255,0,0)):
        PipelineStage.__init__(self, src)
        self.tx = tx
        self.ty = ty
        self.color = color
        self.edges = None

    def getOut(self):
        img = self.src
        self.edges = cv2.Canny(img, self.tx,self.ty)
        cdst = img

        cdst[self.edges == 255] = self.color
        self.out = cdst
        return self.out

class ClusterStage(PipelineStage):
    def __init__(self,src,clusterName=None):
        PipelineStage.__init__(self, src)
        self.clusterName = clusterName
        self.len = 8

    def getOut(self):
        self.cluster = np.load(self.clusterName)
        self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)

        self.out = scan.filterWithArray(self.src.astype(np.uint8),self.cluster.astype(np.uint8),20)
        #self.out = cv2.split(self.out)[2]
        return self.out

class ClusterFreqStage(PipelineStage):
    def __init__(self,src,clusterName=None):
        PipelineStage.__init__(self, src)
        self.clusterName = clusterName
        self.len = 8

    def getOut(self):
        self.cluster = np.load(self.clusterName)
        self.src = cv2.cvtColor(self.src, cv2.COLOR_BGR2HSV)

        self.out = scan.filterWithArray(self.src.astype(np.uint8),self.cluster.astype(np.uint8),20)
        #self.out = cv2.split(self.out)[2]
        return self.out


class FrequenciesRegions(PipelineStage):
    def __init__(self,src,tx=10,ty=10,clusterName=None,drawBox=False):
        PipelineStage.__init__(self, src)
        self.clusterName = clusterName
        self.len = 8

        self.tx = tx
        self.ty = ty
        self.regions = {}
        self.drawBox = drawBox

    def getOut(self):
        self.canny = CannyStage(copy.deepcopy(self.src), self.tx, self.ty)
        self.canny.getOut()

        edges = self.canny.edges

        cdst = self.src
        x,y = np.where(edges > 200)
        if len(x) == 0:
            return self.src

        color = self.src[x, y]
        color = color.reshape(-1, color.shape[0], 3)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        color = color.reshape(color.shape[1], 3)
        X = np.array([[x[off], y[off], color[off][0], color[off][1], color[off][2]] for off in range(0, y.shape[0])])
        model = DBSCAN(eps=15, min_samples=3)

        info,res =getDbScanInfo(model,X)

        edges[:] = 0

        for i in info:
            toChange = X[res.labels_ == i]
            self.regions[i] = toChange[:,0],toChange[:,1]
            if self.drawBox:
                edges[ toChange[:,0],toChange[:,1]] = 255

        self.out =  self.src
        return edges

class Region:
    def __init__(self,region,granularity=2):
        self.upper_right = ( region[0].max(),region[1].max())
        self.bottom_left = ( region[0].min(),region[1].min())

        self.granularity = granularity

        # scale region to stay in -1 and 1
        self.region = region
        self.region = 2 * self.region[0] / self.region[0].max(), 2 * self.region[1] / self.region[1].max()
        self.region = self.region[0] - 1, self.region[1] - 1

        #convert to complex and reduce dimensionality
        self.region = np.array([
            complex(self.region[0][i], self.region[1][i])
            for i in range(0, self.region[0].shape[0])])
        self.region = scan.prepareFourierGrid(self.region, self.granularity)

        #convert to a fixed frequencies fourier coeff
        N = (2 * 10 ** (self.granularity)) ** 2

        fCoeff = np.fft.fft(self.region)
        idx = np.arange(0, fCoeff.shape[0])
        rapp = int(N / self.region.shape[0])

        fixed_coeff = np.zeros((N, 2), dtype=complex)
        fixed_coeff[idx * rapp] = fCoeff[idx]

        self.coeff = fixed_coeff

        del fCoeff
        del idx
        del rapp
class DeepRegionFilter(PipelineStage):
    def __init__(self,src,granularity=2,tx=10,ty=10,drawBox=False,model='firstDeep'):
        PipelineStage.__init__(self, src)
        self.granularity = granularity

        self.tx = tx
        self.ty = ty

        self.regions = []
        self.drawBox = drawBox
        self.model =torch.load(model)
        self.model.eval()

        self.device = 'cpu'

    def getOut(self):
        self.freq = FourierRegionTransform(self.src, tx=self.tx, ty=self.ty,drawBox=self.drawBox,granularity=self.granularity)
        self.freq.getOut()
        regions = self.freq.regions
        doPred = np.array([])
        doPred  = doPred.reshape(0,self.freq.N,2)

        for r in regions:
            fixed_coeff = r.coeff
            fixed_coeff = fixed_coeff.reshape(1, fixed_coeff.shape[0], fixed_coeff.shape[1])
            doPred = np.append(doPred, fixed_coeff, axis=0)
        doPred = np.abs(doPred)
        doPred = doPred.reshape(doPred.shape[0], doPred.shape[1] * doPred.shape[2])
        tensor_x = torch.Tensor(doPred).to(self.device)
        pred = self.model(tensor_x)
        pred = pred.argmax(dim=1)
        founded = torch.where(pred==0)[0].tolist()

        for i in founded:
            self.src = cv2.rectangle(self.src,
                                     regions[i].upper_right,
                                     regions[i].bottom_left,
                                     randomColor(), 2)
        return self.src


class FourierRegionTransform(PipelineStage):
    def __init__(self,src,granularity=2,tx=10,ty=10,drawBox=False):
        PipelineStage.__init__(self, src)
        self.granularity = granularity

        self.tx = tx
        self.ty = ty

        self.regions = []
        self.drawBox = drawBox
        self.N = (2 * 10 ** (self.granularity)) ** 2

    def getOut(self):
        self.freq = FrequenciesRegions(copy.deepcopy(self.src), self.tx, self.ty,drawBox=self.drawBox)
        self.freq.getOut()
        regions = self.freq.regions

        self.src[:] = (0,0,0)

        for r in regions:
            fr = regions[r]
            if fr[0].max() == 0 or  fr[1].max() == 0:
                continue

            self.regions.append(Region(fr,granularity=self.granularity))
            if self.drawBox:
                self.src = cv2.rectangle(self.src,
                                         self.regions[-1].upper_right,
                                         self.regions[-1].bottom_left,
                                          randomColor(), 2)
            else:
                self.src[fr[0],fr[1]] = randomColor()

        self.out = self.src
        return self.src


class DbScan(PipelineStage):
    def __init__(self, src,w=10,h=10,tx=10,ty=10,color=(255,0,0),angle_res=np.pi / 180,votes=60,distance_res=1):
        PipelineStage.__init__(self, src)
        self.w = w
        self.h = h
        self.tx = tx
        self.ty = ty
        self.color = color
        self.canny = CannyStage(src, tx, ty, color)
        self.angle_resolution = angle_res
        self.votes = votes
        self.distance_res = distance_res
        self.class_colors = {}

    def getOut(self):
        self.canny = CannyStage(copy.deepcopy(self.src), self.tx, self.ty, self.color)

        self.canny.getOut()
        edges = self.canny.edges

        cdst = self.src
        x,y = np.where(edges > 0)
        if len(x) == 0:
            return self.src



        color = self.src[x,y]
        color = color.reshape(-1, color.shape[0], 3)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
        color = color.reshape(color.shape[1], 3)
        X = np.array([[x[off],y[off],color[off][0],color[off][1],color[off][2]] for off in range(0,y.shape[0])])
        model = DBSCAN(eps=10, min_samples=2)
        res = model.fit(X)
        cdst[:,:] = 0
        m = res.labels_.max()
        info = {i: X[res.labels_ == i].shape[0] for i in range(0, m)}
        if len(info) == 0:
            return cdst

        avg = sum([info[k] for k in info]) / len(info)
        info = {k: info[k] for k in info if info[k] >= avg}

        for i in info:
            toChange = X[res.labels_ == i]
            colors =  np.array([[int(i[2]),int(i[3]),int(i[4])] for i in toChange])

            toChange = (np.array([int(i[0]) for i in toChange]), np.array([int(i[1]) for i in toChange]))
            self.class_colors[i] = colors

            toSet = np.mean(colors,axis=0)
            toSet = toSet.reshape(1,1,3)
            toSet = toSet.astype('uint8')
            toSet = cv2.cvtColor(toSet, cv2.COLOR_HSV2BGR)[0][0]
            toSet = (toSet[0],toSet[1],toSet[2])

            cdst[toChange[0], toChange[1]] = toSet

        self.out =  cdst
        return self.out






