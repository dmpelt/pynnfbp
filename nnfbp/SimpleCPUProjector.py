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

import numpy as np
import scipy.weave


class ProjectorTranspose():
    """Implements the ``proj.T`` functionality.
    
    Do not use directly, since it can be accessed as member ``.T`` of
    an :class:`Projector` object.
    
    """
    def __init__(self, parentProj):
        self.parentProj = parentProj

    def __mul__(self, data):
        return self.parentProj.backProject(data)


class Projector(object):
    
    """Implementation of the projector interface using the CPU.

    A projector needs to implement:
    
    * Forward projecting
    * Back projecting
    * Creating a FBP reconstruction with a custom filter

    You can use this class as an abstracted weight matrix :math:`W`: multiplying an instance
    ``proj`` of this class by an image results in a forward projection of the image, and multiplying
    ``proj.T`` by a sinogram results in a backprojection of the sinogram::
    
        proj = Projector(...)
        fp = proj*image
        bp = proj.T*sinogram
    
    :param recSize: Width/height of the reconstruction volume. (only 2D squares are supported)
    :type recSize: :class:`int` 
    :param angles: Array with angles in radians.
    :type angles: :class:`numpy.ndarray` 
    """
    
    __backProjectCode = """
    #pragma omp parallel for firstprivate(sinA,cosA,size,nAngles)
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
          float detMid = (size-1.)/2.;
          img(i,j)=0;
          //if((i-detMid)*(i-detMid)+(j-detMid)*(j-detMid)>=bnd) return;
          float x = (j - detMid);
          float y = -(i-detMid);
          float projPos;
          int lr,rr;
          float df;
          for(int k=0;k<nAngles;k++){
            projPos = x*cosA(k)+y*sinA(k) + detMid;
            lr = (int)projPos;
            rr = lr+1;
            df = projPos-lr;
            if(lr>=0&&lr<size) img(i,j) += (1-df)*sino(k,lr);
            if(rr>=0&&rr<size) img(i,j) += df*sino(k,rr);
          }
        }
    }
    """
    
    __forwProjectCode = """
    #pragma omp parallel for firstprivate(sinA,cosA,size,nAngles)
    for(int i=0;i<size;i++){
        for(int j=0;j<nAngles;j++){
          float detMid = (size-1.)/2.;
          float sA = sinA(j);
          float cA = cosA(j);
          float df,df2;
          float xpos,ypos;
          float corr;
          int yps,xps;
          int lr,rr;
          sino(j,i)=0;
          if(fabs(cA)>fabs(sA)){
              df = sA/cA;
              corr = abs(1./cA);
              yps = 0;
              xpos = detMid+((i-detMid)-(yps-detMid)*sA)/cA;
              for(int k=0;k<size;k++){
                  lr = (int)xpos;
                  rr = lr+1;
                  df2 = xpos-lr;
                  if(lr>=0&&lr<size) sino(j,i) += (1-df2)*img(size-yps-1,lr);
                  if(rr>=0&&rr<size) sino(j,i) += df2*img(size-yps-1,rr);
                  yps++;
                  xpos-=df;
              }
          }else{
              df = cA/sA;
              corr = abs(1./sA);
              xps = 0;
              ypos = detMid+((i-detMid)-(xps-detMid)*cA)/sA;
              for(int k=0;k<size;k++){
                  lr = (int)ypos;
                  rr = lr+1;
                  df2 = ypos-lr;
                  if(lr>=0&&lr<size) sino(j,i) += (1-df2)*img(size-lr-1,xps);
                  if(rr>=0&&rr<size) sino(j,i) += df2*img(size-rr-1,xps);
                  xps++;
                  ypos-=df;
              }
          }
          sino(j,i)*=corr;
        }
    }
    
     """

    def __mul__(self, data):
        return self.forwProject(data)

    def filterSinogram(self, filt, sino):
        '''Filter a sinogram ``sino`` with filter ``filt``. Used internally.'''
        ff = np.fft.rfft(filt, n=self.fSnpo2)
        sf = np.fft.rfft(sino, n=self.fSnpo2, axis=1)
        cv = np.fft.irfft(sf*ff, axis=1)
        return cv[:, self.nDet/2:3*self.nDet/2]
    
    def reconstructWithFilter(self,sinogram,filt):
        """Create a FBP reconstruction of the sinogram with a custom filter
        
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :param filt: 1D custom filter
        :type filt: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The reconstruction.
        
        """
        fSino = self.filterSinogram(filt,sinogram)
        return self.backProject(fSino)
        

    def forwProject(self, image):
        """Forward project an image.
        
        :param image: The image data.
        :type image: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The forward projection.
        
        """
        sino = np.zeros((self.nProj, self.nDet))
        scipy.weave.inline(self.__forwProjectCode, ['img', 'sino', 'size', 'nAngles', 'sinA', 'cosA'], local_dict={'img': image, 'sino': sino, 'size':
                            self.recSize, 'nAngles': self.nProj, 'sinA': self.sinA, 'cosA': self.cosA},type_converters=scipy.weave.converters.blitz,extra_compile_args=["-march=native","-fopenmp"],libraries=['gomp'])
        return sino
    def backProject(self, sino):
        """Backproject a sinogram.
        
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The backprojection.
        
        """
        image = np.zeros((self.recSize, self.recSize))
        scipy.weave.inline(self.__backProjectCode, ['img', 'sino', 'size', 'nAngles', 'sinA', 'cosA'], local_dict={'img': image, 'sino': sino, 'size':
                            self.recSize, 'nAngles': self.nProj, 'sinA': self.sinA, 'cosA': self.cosA},type_converters=scipy.weave.converters.blitz,extra_compile_args=["-march=native","-fopenmp"],libraries=['gomp'])
        return image

    def calcNextPowerOfTwo(self, val):
        '''Calculated the power of two larger than ``val``. Used internally.'''
        ival = int(val)
        ival -= 1
        ival = (ival >> 1) | ival
        ival = (ival >> 2) | ival
        ival = (ival >> 4) | ival
        ival = (ival >> 8) | ival
        ival = (ival >> 16) | ival
        ival += 1
        return ival

    def __init__(self, recSize, angles):
        self.recSize = recSize
        self.angles = angles
        self.nProj = len(angles)
        self.nDet = recSize
        self.cosA = np.cos(angles)
        self.sinA = np.sin(angles)
        self.filterSize = self.nDet+1
        self.fSnpo2 = self.calcNextPowerOfTwo(2*self.filterSize)
        self.backProjImgSize = recSize
        self.forwProjSize = recSize*angles.shape[0]
        self.angles = angles
        self.T = ProjectorTranspose(self)
