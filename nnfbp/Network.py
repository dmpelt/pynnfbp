#-----------------------------------------------------------------------
#Copyright 2013 Centrum Wiskunde & Informatica, Amsterdam
#
#Author: Daniel M. Pelt
#Contact: D.M.Pelt@cwi.nl
#Website: http://dmpelt.github.io/pynnfbp/
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

from TrainingData import *
import numpy as np

import tables as ts
import sys
import os
import Reductors 
import numexpr
import scipy.sparse as ss
import scipy.linalg as la
try:
    import scipy.linalg.fblas as fblas
    hasfblas=True
except:
    hasfblas=False
import time

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
    
    def __init__(self, nHiddenNodes, projector, trainData, valData, reductor=None,nTrain=1000000,nVal = 1000000,tmpDir=None,createEmptyClass=False):       
        self.proj = projector
        self.__setOutCircle()
        if createEmptyClass==True:
            return
        self.tD = trainData
        self.vD = valData
        if not reductor==None:
            self.red = reductor
        else:
            self.red = Reductors.LogSymReductor(projector.filterSize)
        self.tmpDir = tmpDir
        self.nHid = nHiddenNodes
        self.nIn = self.red.outSize
        self.nT = nTrain
        self.nV = nVal
        
        
        
        self.jacDiff = np.zeros((self.nHid) * (self.nIn+1) + self.nHid + 1);
        self.jac2 = np.zeros(((self.nHid) * (self.nIn+1) + self.nHid + 1, (self.nHid) * (self.nIn+1) + self.nHid + 1))
    
    def __setOutCircle(self):
        '''Creates a :class:`numpy.ndarray` mask of a circle'''
        xx, yy = np.mgrid[:self.proj.recSize, :self.proj.recSize]
        mid = (self.proj.recSize-1.)/2.
        circle = (xx - mid) ** 2 + (yy - mid) ** 2
        bnd = self.proj.recSize**2/4.
        self.outCircle=circle>bnd
    
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
        self.tTD = HDF5TrainingData(self.tD,self.nT,self)
        self.vTD = HDF5TrainingData(self.vD,self.nV,self)
        self.minmax = self.tTD.getMinMax()
        self.tTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.vTD.normalizeData(self.minmax[0], self.minmax[1], self.minmax[2], self.minmax[3])
        self.ident = np.eye((self.nHid) * (self.nIn+1) + self.nHid + 1)
        
    def __sigmoid(self,x):
        '''Sigmoid function'''
        return numexpr.evaluate("1./(1.+exp(-x))")
    
    def __createFilters(self):
        '''After training, creates the actual filters and offsets by undoing the scaling.'''
        self.minL = self.minmax[0]
        self.maxL = self.minmax[1]
        self.minIn = self.minmax[2]
        self.maxIn = self.minmax[3]
        mindivmax = self.minL/(self.maxL-self.minL)
        mindivmax[np.isnan(mindivmax)]=0
        mindivmax[np.isinf(mindivmax)]=0
        divmaxmin = 1./(self.maxL-self.minL)
        divmaxmin[np.isnan(divmaxmin)]=0
        divmaxmin[np.isinf(divmaxmin)]=0
        self.filters = np.empty((self.nHid,self.red.filters.shape[0]))
        self.offsets = np.empty(self.nHid)
        for i in xrange(self.nHid):
            wv = 2*self.l1[0:self.l1.shape[0]-1,i]*divmaxmin
            self.filters[i] = self.red.getFilter(wv)
            self.offsets[i] = 2*np.dot(self.l1[0:self.l1.shape[0]-1,i],mindivmax) + np.sum(self.l1[:,i])
    
        
    def __processDataBlock(self,data):
        ''' Returns output values (``vals``), 'correct' output values (``valOut``) and
        hidden node output values (``hiddenOut``) from a block of data.'''
        valOut = data[:, -1].copy()
        data[:, -1] = -np.ones(data.shape[0])
        hiddenOut = np.empty((data.shape[0],self.l1.shape[1]+1))
        hiddenOut[:,0:self.l1.shape[1]] = self.__sigmoid(np.dot(data, self.l1))
        hiddenOut[:,-1] = -1
        rawVals = np.dot(hiddenOut, self.l2)
        vals = self.__sigmoid(rawVals)
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
            print 'Validation set error:', prevValError
        self.l1 = self.minl1
        self.l2 = self.minl2
        self.valErr = prevValError
        self.tTD.close()
        self.vTD.close()
        self.__createFilters()
    
    def reconstruct(self,sinogram):
        '''Reconstruct an image from a sinogram, after training.
        
        :param sinogram: The sinogram to reconstruct.
        :type sinogram: :class:`numpy.ndarray`
        '''
        reco = np.zeros((self.proj.recSize,self.proj.recSize))
        for i in xrange(self.nHid):
            mult = float(self.l2[i])
            offs = float(self.offsets[i])
            back = self.proj.reconstructWithFilter(sinogram,self.filters[i])
            reco += numexpr.evaluate('mult/(1.+exp(-(back-offs)))')
        reco = self.__sigmoid(reco-self.l2[-1])
        #scipy.weave.inline(self.inplaceScaleCode,['minIn','maxIn','reco','size'],compiler = 'gcc',verbose=1,extra_compile_args=['-march=native','-fopenmp'],extra_link_args=['-lgomp'])
        reco = (reco-0.25)*2*(self.maxIn-self.minIn) + self.minIn
        reco[self.outCircle]=0
        return reco
    
    def saveToDisk(self,fn):
        '''Save a trained network to disk, so that it can be used later
        without retraining.
        
        :param fn: Filename to save it to.
        :type fn: :class:`string`
        '''
        
        f = ts.openFile(fn,mode='w')
        atom = ts.Atom.from_dtype(self.filters.dtype)
        ds = f.createCArray(f.root, "filters", atom,self.filters.shape)
        ds[:] = self.filters
        atom = ts.Atom.from_dtype(self.offsets.dtype)
        ds = f.createCArray(f.root, "offsets", atom,self.offsets.shape)
        ds[:] = self.offsets
        atom = ts.Atom.from_dtype(self.l2.dtype)
        ds = f.createCArray(f.root, "l2", atom,self.l2.shape)
        ds[:] = self.l2
        f.createArray('/', 'minmax', [self.minIn,self.maxIn], "")
        f.close()
    
def readFromDisk(fn,projector):
    '''Read a saved network from disk. The specified projector should be
    similar to the one used during training.
    
    :param fn: Filename to load network from.
    :type fn: :class:`string`
    :param projector: Projector to use when reconstructing.
    :type projector: A ``Projector`` object (see, for example: :mod:`nnfbp.SimpleCPUProjector`)
    :returns: A :class:`nnfbp.Network.Network` instance, ready to reconstruct with.
    '''
    n = Network(None,projector,None,None,createEmptyClass=True)
    f = ts.openFile(fn,mode='r')
    n.filters = f.getNode('/','filters').read()
    n.offsets = f.getNode('/','offsets').read()
    n.l2 = f.getNode('/','l2').read()
    arr = f.getNode('/','minmax').read()
    n.minIn = arr[0]
    n.maxIn = arr[1]
    f.close()
    n.proj = projector
    n.nHid = n.filters.shape[0]
    return n
        

        
