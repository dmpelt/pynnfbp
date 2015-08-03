#-----------------------------------------------------------------------
#Copyright 2015 Daniel M. Pelt
#
#Contact: D.M.Pelt@cwi.nl
#Website: http://www.dmpelt.com
#
#
#This file is part of the PyNN-FBP, a Python implementation of the
#NN-FBP tomographic reconstruction method.
#
#PyNN-FBP is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#PyNN-FBP is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyNN-FBP. If not, see <http://www.gnu.org/licenses/>.
#
#-----------------------------------------------------------------------
import astra
import numpy as np
from six.moves import range
from scipy.signal import fftconvolve
import scipy.ndimage.filters as snf
import numpy.linalg as na
import glob
import tifffile
import scipy.io as sio
import six

import os, errno

import random

from .TrainingData import MATTrainingData

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc: # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else: raise


class plugin_prepare(astra.plugin.ReconstructionAlgorithm2D):
    """Prepares a training set for the NN-FBP method [1].
    
    Options:
    
    'hqrecfiles': HQ reconstruction files (in TIFF format), e.g. '/path/to/rec*.tiff'.
    'traindir': folder where output training files should be stored
    'z_id': z-index of current slice
    'npick' (optional): number of random pixels to pick (per slice)
    'nlinear' (optional): number of linear steps in exponential binning
    
    [1] Pelt, D. M., & Batenburg, K. J. (2013). Fast tomographic reconstruction 
        from limited data using artificial neural networks. Image Processing, 
        IEEE Transactions on, 22(12), 5238-5251.
    """
    
    astra_name="NN-FBP-prepare"
    
    def customFBP(self, f, s):
        sf = np.zeros_like(s)
        padded = np.zeros(s.shape[1]*2)
        l = int(s.shape[1]/2.)
        r = l+s.shape[1]
        bl = f.shape[0]/len(s)
        for i in range(sf.shape[0]):
            padded[l:r] = s[i]
            padded[:l] = padded[l]
            padded[r:] = padded[r-1]
            sf[i] = fftconvolve(padded,f,'same')[l:r]
        return (self.W.T*sf).reshape(self.v.shape)
    
    def initialize(self, cfg, hqrecfiles, z_id, traindir, nlinear=2, npick=100):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.npick = npick
        self.td = traindir
        mkdir_p(traindir)
        self.rec = tifffile.imread(sorted(glob.glob(hqrecfiles))[z_id])
        self.bck = (self.W.T*np.ones_like(self.s)<self.s.shape[0]-0.5).reshape(self.v.shape)
        self.outfn = traindir + os.sep + "train_{:05d}.mat".format(z_id)
        self.z_id = z_id
        fs = self.s.shape[1]
        if fs%2==0:
            fs += 1
        mf = int(fs/2)
        w=1
        c=mf
        bas = np.zeros(fs,dtype=np.float32)
        self.basis = []
        count=0
        while c<fs:
            bas[:]=0
            l = c
            r = c+w
            if r>fs: r=fs
            bas[l:r]=1
            if l!=0:
                l = fs-c-w
                r = l+w
                if l<0: l=0
                bas[l:r]=1
            self.basis.append(bas.copy())
            c += w
            count += 1
            if count>nlinear:
                w=2*w
        self.nf = len(self.basis)

    def run(self, iterations):
        out = np.zeros((self.npick,self.nf+1))
        pl = np.random.random((self.npick,2))
        pl[:,0]*=self.v.shape[0]
        pl[:,1]*=self.v.shape[1]
        pl = pl.astype(np.int)
        for i in range(self.npick):
            while self.bck[pl[i,0],pl[i,1]]:
                pl[i,0] = np.random.random(1)*self.v.shape[0]
                pl[i,1] = np.random.random(1)*self.v.shape[1]
        out[:,-1] = self.rec[pl[:,0],pl[:,1]]
        for i, bas in enumerate(self.basis):
            img = self.customFBP(bas, self.s)
            out[:,i] = img[pl[:,0],pl[:,1]]
        sio.savemat(self.outfn,{'mat':out},do_compression=True)
        six.print_('Slice {} done...'.format(self.z_id))
        
import numexpr
import time
import scipy.sparse as ss
import scipy.linalg as la
try:
    import scipy.linalg.fblas as fblas
    hasfblas=True
except:
    hasfblas=False

def sigmoid(x):
    '''Sigmoid function'''
    return numexpr.evaluate("1./(1.+exp(-x))")

class Network(object):
    '''
    The neural network object that performs all training and reconstruction.
    
    :param nHiddenNodes: The number of hidden nodes in the network.
    :type nHiddenNodes: :class:`int`
    :param projector: The projector to use.
    :type projector: A ``Projector`` object (see, for example: :mod:`nnfbp.SimpleCPUProjector`)
    :param trainData: The training data set.
    :type trainData: A ``DataSet`` object (see: :mod:`nnfbp.DataSet`)
    :param valData: The validation data set.
    :type valData: A ``DataSet`` object (see: :mod:`nnfbp.DataSet`)
    :param reductor: Optional reductor to use.
    :type reductor: A ``Reductor`` object (see: :mod:`nnfbp.Reductors`, default:``LogSymReductor``)
    :param nTrain: Number of pixels to pick out of training set.
    :type nTrain: :class:`int`
    :param nVal: Number of pixels to pick out of validation set.
    :type nVal: :class:`int`
    :param tmpDir: Optional temporary directory to use.
    :type tmpDir: :class:`string`
    :param createEmptyClass: Used internally when loading from disk, to create an empty object. Do not use directly.
    :type createEmptyClass: :class:`boolean`
    '''
    
    def __init__(self, nHiddenNodes, trainData, valData):
        self.tTD = trainData
        self.vTD = valData
        self.nHid = nHiddenNodes
        self.nIn = self.tTD.getDataBlock(0).shape[1]-1
        self.jacDiff = np.zeros((self.nHid) * (self.nIn+1) + self.nHid + 1);
        self.jac2 = np.zeros(((self.nHid) * (self.nIn+1) + self.nHid + 1, (self.nHid) * (self.nIn+1) + self.nHid + 1))
    def __inittrain(self):
        '''Initialize training parameters, create actual training and validation
        sets by picking random pixels from the datasets'''
        self.l1 = 2 * np.random.rand(self.nIn+1, self.nHid) - 1
        beta = 0.7 * self.nHid ** (1. / (self.nIn))
        l1norm = np.linalg.norm(self.l1)
        self.l1 *= beta / l1norm
        self.l2 = 2 * np.random.rand(self.nHid + 1) - 1
        self.l2 /= np.linalg.norm(self.l2)
        self.minl1 = self.l1.copy()
        self.minl2 = self.l2.copy()
        self.minmax = self.tTD.getMinMax()
        self.tTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.vTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.ident = np.eye((self.nHid) * (self.nIn+1) + self.nHid + 1)
        
        
    def __processDataBlock(self,data):
        ''' Returns output values (``vals``), 'correct' output values (``valOut``) and
        hidden node output values (``hiddenOut``) from a block of data.'''
        valOut = data[:, -1].copy()
        data[:, -1] = -np.ones(data.shape[0])
        hiddenOut = np.empty((data.shape[0],self.l1.shape[1]+1))
        hiddenOut[:,0:self.l1.shape[1]] = sigmoid(np.dot(data, self.l1))
        hiddenOut[:,-1] = -1
        rawVals = np.dot(hiddenOut, self.l2)
        vals = sigmoid(rawVals)
        return vals,valOut,hiddenOut
    
    
    
    def __getTSE(self, dat):
        '''Returns the total squared error of a data block'''
        tse = 0.
        for i in xrange(dat.nBlocks):
            data = dat.getDataBlock(i)
            vals,valOut,hiddenOut = self.__processDataBlock(data)
            tse += numexpr.evaluate('sum((vals - valOut)**2)')
        return tse
    
    def __setJac2(self):
        '''Calculates :math:`J^T J` and :math:`J^T e` for the training data.
        Used for Levenberg-Marquardt method.''' 
        self.jac2.fill(0)
        self.jacDiff.fill(0)
        for i in xrange(self.tTD.nBlocks):
            data = self.tTD.getDataBlock(i)
            vals,valOut,hiddenOut = self.__processDataBlock(data)
            diffs = numexpr.evaluate('valOut - vals')
            jac = np.empty((data.shape[0], (self.nHid) * (self.nIn+1) + self.nHid + 1))
            d0 = numexpr.evaluate('-vals * (1 - vals)')
            ot = (np.outer(d0, self.l2))
            dj = numexpr.evaluate('hiddenOut * (1 - hiddenOut) * ot') 
            I = np.tile(np.arange(data.shape[0]), (self.nHid + 1, 1)).flatten('F')
            J = np.arange(data.shape[0] * (self.nHid + 1))
            Q = ss.csc_matrix((dj.flatten(), np.vstack((J, I))), (data.shape[0] * (self.nHid + 1), data.shape[0]))
            jac[:, 0:self.nHid + 1] = ss.spdiags(d0, 0, data.shape[0], data.shape[0]).dot(hiddenOut)
            Q2 = np.reshape(Q.dot(data), (data.shape[0],(self.nIn+1) * (self.nHid + 1)))
            jac[:, self.nHid + 1:jac.shape[1]] = Q2[:, 0:Q2.shape[1] - (self.nIn+1)]
            if hasfblas:
                self.jac2 += fblas.dgemm(1.0,a=jac.T,b=jac.T,trans_b=True)
                self.jacDiff += fblas.dgemv(1.0,a=jac.T,x=diffs)
            else:
                self.jac2 += np.dot(jac.T,jac)
                self.jacDiff += np.dot(jac.T,diffs)
    
    def train(self):
        '''Train the network using the Levenberg-Marquardt method.'''
        self.__inittrain()
        mu = 100000.;
        muUpdate = 10;
        prevValError = np.Inf
        bestCounter = 0
        tse = self.__getTSE(self.tTD)
        curTime = time.time()
        for i in xrange(1000000):
            self.__setJac2()
            dw = -la.cho_solve(la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
            done = -1
            while done <= 0:
                self.l2 += dw[0:self.nHid + 1]
                for k in xrange(self.nHid):
                    start = self.nHid + 1 + k * (self.nIn+1);
                    self.l1[:, k] += dw[start:start + self.nIn+1]
                newtse = self.__getTSE(self.tTD)
                if newtse < tse:
                    if done == -1:
                        mu /= muUpdate
                    if mu <= 1e-100:
                        mu = 1e-99
                    done = 1;
                else:
                    done = 0;
                    mu *= muUpdate
                    if mu >= 1e20:
                        done = 2
                        break;
                    self.l2 -= dw[0:self.nHid + 1]
                    for k in xrange(self.nHid):
                        start = self.nHid + 1 + k * (self.nIn+1);
                        self.l1[:, k] -= dw[start:start + self.nIn+1]
                    dw = -la.cho_solve(la.cho_factor(self.jac2 + mu * self.ident), self.jacDiff)
            gradSize = np.linalg.norm(self.jacDiff)
            if done == 2:
                break 
            curValErr = self.__getTSE(self.vTD)
            if curValErr > prevValError:
                bestCounter += 1
            else:
                prevValError = curValErr
                self.minl1 = self.l1.copy()
                self.minl2 = self.l2.copy()
                if (newtse / tse < 0.999):
                    bestCounter = 0
                else:
                    bestCounter +=1
            if bestCounter == 50:
                break
            if(gradSize < 1e-8):
                break
            tse = newtse
            six.print_('Validation set error:', prevValError)
        self.l1 = self.minl1
        self.l2 = self.minl2
        self.valErr = prevValError
        
    
    
    def saveToDisk(self,fn):
        '''Save a trained network to disk, so that it can be used later
        without retraining.
        
        :param fn: Filename to save it to.
        :type fn: :class:`string`
        '''
        sio.savemat(fn,{'l1':self.l1,'l2':self.l2,'minmax':self.minmax},do_compression=True)


def plugin_train(traindir, nhid, filter_file, val_rat=0.5):
    fls = glob.glob(traindir + os.sep + '*.mat')
    random.shuffle(fls)
    nval = int(val_rat*len(fls))
    val = MATTrainingData(fls[:nval])
    trn = MATTrainingData(fls[nval:])
    n = Network(nhid, trn, val)
    n.train()
    n.saveToDisk(filter_file)

class plugin_rec(astra.plugin.ReconstructionAlgorithm2D):
    """Reconstructs using the NN-FBP method [1].
    
    Options:
    
    'filter_file': file with trained filters to use
    'nlinear' (optional): number of linear steps in exponential binning
    
    [1] Pelt, D. M., & Batenburg, K. J. (2013). Fast tomographic reconstruction 
        from limited data using artificial neural networks. Image Processing, 
        IEEE Transactions on, 22(12), 5238-5251.
    """
    
    astra_name="NN-FBP"
    
    def customFBP(self, f, s):
        sf = np.zeros_like(s)
        padded = np.zeros(s.shape[1]*2)
        l = int(s.shape[1]/2.)
        r = l+s.shape[1]
        bl = f.shape[0]/len(s)
        for i in range(sf.shape[0]):
            padded[l:r] = s[i]
            padded[:l] = padded[l]
            padded[r:] = padded[r-1]
            sf[i] = fftconvolve(padded,f,'same')[l:r]
        return (self.W.T*sf).reshape(self.v.shape)
    
    def initialize(self, cfg, filter_file, z_id, nlinear=2):
        self.W = astra.OpTomo(cfg['ProjectorId'])
        self.z_id = z_id
        fs = self.s.shape[1]
        if fs%2==0:
            fs += 1
        mf = int(fs/2)
        w=1
        c=mf
        bas = np.zeros(fs,dtype=np.float32)
        self.basis = []
        count=0
        while c<fs:
            bas[:]=0
            l = c
            r = c+w
            if r>fs: r=fs
            bas[l:r]=1
            if l!=0:
                l = fs-c-w
                r = l+w
                if l<0: l=0
                bas[l:r]=1
            self.basis.append(bas.copy())
            c += w
            count += 1
            if count>nlinear:
                w=2*w
        self.nf = len(self.basis)
        fl = sio.loadmat(filter_file)
        self.l1 = fl['l1']
        self.l2 = fl['l2'].transpose()
        minmax = fl['minmax'][0]
        minL = minmax[0]
        maxL = minmax[1]
        self.minIn = minmax[2]
        self.maxIn = minmax[3]
        mindivmax = minL/(maxL-minL)
        mindivmax[np.isnan(mindivmax)]=0
        mindivmax[np.isinf(mindivmax)]=0
        divmaxmin = 1./(maxL-minL)
        divmaxmin[np.isnan(divmaxmin)]=0
        divmaxmin[np.isinf(divmaxmin)]=0
        nHid = self.l1.shape[1]
        self.filters = np.empty((nHid,self.basis[0].shape[0]))
        self.offsets = np.empty(nHid)
        for i in range(nHid):
            wv = (2*self.l1[0:self.l1.shape[0]-1,i]*divmaxmin).transpose()
            self.filters[i] = np.zeros_like(self.basis[0])
            for t, bas in enumerate(self.basis):
                self.filters[i] += wv[t]*bas
            self.offsets[i] = 2*np.dot(self.l1[0:self.l1.shape[0]-1,i],mindivmax.transpose()) + np.sum(self.l1[:,i])

    def run(self, iterations):
        self.v[:]=0
        for i in range(self.l2.shape[0]-1):
            mult = float(self.l2[i])
            offs = float(self.offsets[i])
            back = self.customFBP(self.filters[i],self.s)
            self.v[:] += numexpr.evaluate('mult/(1.+exp(-(back-offs)))')
        self.v[:] = sigmoid(self.v-self.l2[-1])
        self.v[:] = (self.v-0.25)*2*(self.maxIn-self.minIn) + self.minIn
