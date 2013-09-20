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

import math
import numpy as np

class Reductor(object):
    '''Base object of a ``Reductor``, that takes input data and reduces it.
    
    Implementing objects should define `outSize`, the number of elements after
    reduction, and a ``filters`` :class:`numpy.ndarray` of size ``(inSize,outSize)``, where
    each row is a basis vector.
    
    :param inSize: Input size of vectors.
    :type inSize: :class:`int`
    '''
    def __init__(self,inSize):
        self.size = inSize
        self.inSize = self.size
    def getFilter(self,weights):
        '''Returns actual FBP filters, given the resulting weights of a trained neural network.'''
        return np.dot(self.filters,weights)
 
class IdentityReductor(Reductor):
    '''An implementation of a ``Reductor`` that performs no reduction at all.'''
    def __init__(self,size):
        Reductor.__init__(self,size)
        self.filters = np.zeros((self.size,self.size))
        self.name="Identity"
        for i in xrange(self.size):
            self.filters[i,i] = 1
        self.outSize = self.size
        
class LogSymReductor(Reductor):
    '''An implementation of a ``Reductor`` with exponentially growing bin widths, and symmetric bins.
    
    :param nLinear: Number of bins of width 1 before starting exponential growth.'
    :type nLinear: :class:`int`
    '''
    def __init__(self,size,nLinear=2):
        Reductor.__init__(self,size)
        self.name="LogSym"
        self.indices = np.array(np.floor(np.log2(np.abs(np.arange(self.size)-(self.size-1)/2)+1)),dtype=np.int32)
        self.indices = self.indices+nLinear
        mid = (self.size-1)/2
        self.indices[mid]=0
        for q in xrange(nLinear):
            self.indices = np.insert(self.indices, (mid,mid+1), nLinear-q)
            self.indices = np.delete(self.indices, (0,self.indices.shape[0]-1))
        nFilt = np.max(self.indices)+1
        self.filters = np.zeros((self.size,nFilt))
        for i in xrange(nFilt):
            self.filters[:,i][self.indices==i] = 1
        self.outSize = nFilt
