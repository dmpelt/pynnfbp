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

import pycuda.driver as cuda
import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pyfft.cuda import Plan
from pycuda.elementwise import ElementwiseKernel
import numpy as np
import numpy.linalg as na
import Reductors

class SimplePyCUDAProjectorTranspose():
    """Implements the ``proj.T`` functionality.
    
    Do not use directly, since it can be accessed as member ``.T`` of
    an :class:`SimplePyCUDAProjector` object.
    
    """
    def __init__(self,parentProj):
        self.parentProj = parentProj
    
    def __mul__(self,data):
        return self.parentProj.backProject(data)

class SimplePyCUDAProjector(object):
    
    """Implementation of the projector interface using PyCUDA and pyfft.

    A projector needs to implement:
    
    * Forward projecting
    * Back projecting
    * Creating a FBP reconstruction with a custom filter

    You can use this class as an abstracted weight matrix :math:`W`: multiplying an instance
    ``proj`` of this class by an image results in a forward projection of the image, and multiplying
    ``proj.T`` by a sinogram results in a backprojection of the sinogram::
    
        proj = SimplePyCUDAProjector(...)
        fp = proj*image
        bp = proj.T*sinogram
    
    :param recSize: Width/height of the reconstruction volume. (only 2D squares are supported)
    :type recSize: :class:`int` 
    :param angles: Array with angles in radians.
    :type angles: :class:`numpy.ndarray` 
    """
    
    __cudaCode = """
        #include <stdio.h>

        __global__ void backproject(float *img, float *sino,int size, float *sinA, float *cosA,int bnd, int startK,int endK)
        {
          const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
          const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
          if(i>=size || j>=size) return;
          const int idx = i+j*size;
          float detMid = (size-1.)/2.;
          img[idx]=0;
          if((i-detMid)*(i-detMid)+(j-detMid)*(j-detMid)>=bnd) return;
          float x = -(j - detMid);
          float y = (i-detMid);
          float projPos;
          int lr,rr;
          float df;
          for(int k=startK;k<endK;k++){
            projPos = x*cosA[k]+y*sinA[k] + detMid;
            lr = (int)projPos;
            rr = lr+1;
            df = projPos-lr;
            if(lr>=0&&lr<size) img[idx] += (1-df)*sino[k*size+lr];
            if(rr>=0&&rr<size) img[idx] += df*sino[k*size+rr];
          }
        }
        
        
        __global__ void forwproject(float *sino, float *img, int size, int nAngles,float *sinA, float *cosA)
        {
          const uint i = (blockIdx.x * blockDim.x) + threadIdx.x;
          const uint j = (blockIdx.y * blockDim.y) + threadIdx.y;
          if(i>=size || j>=nAngles) return;
          const int idx = i+j*size;
          float detMid = (size-1.)/2.;
          float sA = sinA[j];
          float cA = cosA[j];
          float df,df2;
          float xpos,ypos;
          float corr;
          int yps,xps;
          int lr,rr;
          sino[idx]=0;
          if(fabs(cA)>fabs(sA)){
              df = sA/cA;
              corr = abs(1./cA);
              yps = 0;
              xpos = detMid+((i-detMid)-(yps-detMid)*sA)/cA;
              for(int k=0;k<size;k++){
                  lr = (int)xpos;
                  rr = lr+1;
                  df2 = xpos-lr;
                  if(lr>=0&&lr<size) sino[idx] += (1-df2)*img[yps+(size-lr-1)*size];
                  if(rr>=0&&rr<size) sino[idx] += df2*img[yps+(size-rr-1)*size];
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
                  if(lr>=0&&lr<size) sino[idx] += (1-df2)*img[lr+(size-xps-1)*size];
                  if(rr>=0&&rr<size) sino[idx] += df2*img[rr+(size-xps-1)*size];
                  xps++;
                  ypos-=df;
              }
          }
          sino[idx]*=corr;
        }
        
        
        """
        
    def filterSinogram(self,filt,sinogram):
        '''Filter a sinogram ``sino`` with filter ``filt``. Used internally.'''
        self.filtPF[0,0:self.filterSize] = filt
        self.filtF.set(self.filtPF)
          
        self.plan.execute(self.filtF)
        
        if not sinogram==None:
            self.sinoPF[0:self.nProj,0:self.nDet]=sinogram
            self.sinoF.set(self.sinoPF)
            self.plan.execute(self.sinoF)
        
        self.multKern(self.sinoF,self.filtF,self.fftComb)
        self.plan.execute(self.fftComb,inverse=True)
        cOut = self.fftComb.get().real
        return cOut[0:self.nProj,self.nDet/2:1.5*self.nDet]
        
    
    def reconstructWithFilter(self,sinogram,filt,clipCircle=False,returnResult=True):
        """Create a FBP reconstruction of the sinogram with a custom filter
        
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :param filt: 1D custom filter
        :type filt: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The reconstruction.
        
        """
        convM = self.filterSinogram(filt,sinogram)
        return self.backProject(convM,clipCircle=clipCircle,returnResult=returnResult)
        
    def backProject(self,sinogram=None,returnResult=True,clipCircle=False,singleAngle=None):
        """Backproject a sinogram.
        
        :param sinogram: The sinogram data. If ``None``, use existing data on the GPU.
        :type sinogram: :class:`numpy.ndarray`
        :param returnResult: Whether to return the result as a :class:`numpy.ndarray`, or simply keep in on the GPU.
        :type returnResult: :class:`bool`
        :param clipCircle: Whether to only reconstruct the unit disk.
        :type clipCircle: :class:`bool`
        :param singleAngle: Only reconstruct using this angle. ``None`` if you want to reconstruct normally, using all angles.
        :type singleAngle: :class:`int`
        :returns: :class:`numpy.ndarray` -- The backprojection.
        
        """
        size = self.backProjImgSize
        nAngles = self.nAngles
        if clipCircle:
            bnd = np.int32(size*size/4)
        else:
            bnd = np.int32(size*size)
        if not sinogram==None:
            self.sino.set(sinogram.astype(np.float32))
        if singleAngle==None:
            startK=0
            endK =self.nAngles
        else:
            startK=singleAngle
            endK = singleAngle+1
        self.backProjFunc(self.img.gpudata,self.sino.gpudata,np.int32(size),self.cosA.gpudata,self.sinA.gpudata,bnd,np.int32(startK),np.int32(endK),block=self.BPblock,grid=self.BPgrid)
        if returnResult: return self.img.get()
    
    def forwProject(self,image=None,returnResult=True):
        """Forward project an image.
        
        :param image: The image data. If ``None``, use existing data on the GPU.
        :type image: :class:`numpy.ndarray`
        :param returnResult: Whether to return the result as a :class:`numpy.ndarray`, or simply keep in on the GPU.
        :type returnResult: :class:`bool`
        :returns: :class:`numpy.ndarray` -- The forward projection.
        
        """
        size = self.backProjImgSize
        nAngles = self.nAngles
        if not image==None:
            self.img.set(image.astype(np.float32))
        self.forwProjFunc(self.sino.gpudata,self.img.gpudata,np.int32(size),np.int32(nAngles),self.cosA.gpudata,self.sinA.gpudata,block=self.FPblock,grid=self.FPgrid)
        if returnResult: return self.sino.get()
        
    def calcNextPowerOfTwo(self,val):
        '''Calculated the power of two larger than ``val``. Used internally.'''
        ival = int(val)
        ival-=1
        ival = (ival >> 1) | ival
        ival = (ival >> 2) | ival
        ival = (ival >> 4) | ival
        ival = (ival >> 8) | ival
        ival = (ival >> 16) | ival
        ival+=1
        return ival
    
    def __mul__(self,data):
        return self.forwProject(data)
  
    def __init__(self,recSize,angles):
        self.recSize = recSize
        self.angles = angles
        self.nProj = len(angles)
        self.nDet = recSize
        self.cosA = np.cos(angles)
        self.sinA = np.sin(angles)
        self.filterSize = self.nDet+1
        self.backProjImgSize = recSize
        self.forwProjSize = recSize*angles.shape[0]
        self.angles = angles
        self.cosA = gpuarray.to_gpu(np.cos(angles).astype(np.float32))
        self.sinA = gpuarray.to_gpu(np.sin(angles).astype(np.float32))
        self.nAngles = angles.shape[0]
        self.img = gpuarray.zeros((self.backProjImgSize,self.backProjImgSize),dtype=np.float32)
        self.sino = gpuarray.zeros((self.nAngles,self.backProjImgSize),dtype=np.float32)

        mod = SourceModule(self.__cudaCode)
        self.backProjFunc = mod.get_function("backproject")
        self.forwProjFunc = mod.get_function("forwproject")
        self.multKern = ElementwiseKernel("pycuda::complex<float> *x, pycuda::complex<float> *y, pycuda::complex<float> *z","z[i] = x[i]*y[i]","mult_kernel")
        fftsize = (self.calcNextPowerOfTwo(2*self.nProj),self.calcNextPowerOfTwo(2*self.filterSize))
        self.fftsize = fftsize
        self.plan = Plan(fftsize)
        self.sinoF = gpuarray.zeros(fftsize,dtype=np.complex64)
        self.filtF = gpuarray.zeros(fftsize,dtype=np.complex64)
        self.fftComb = gpuarray.zeros(fftsize,dtype=np.complex64)
        self.sinoPF = np.zeros(fftsize,dtype=np.complex64)
        self.filtPF = np.zeros(fftsize,dtype=np.complex64)
        
        if self.nAngles<32:
            bs3 = self.nAngles
        else:
            bs3 = 32
        self.BPblock = (32,32,1)
        self.BPgrid = (self.backProjImgSize/32 + (self.backProjImgSize%32>0),self.backProjImgSize/32  + (self.backProjImgSize%32>0),1)
        self.FPblock = (32,bs3,1)
        self.FPgrid = (self.nDet/32  + (self.nDet%32>0),self.nAngles/bs3 + (self.nAngles%bs3>0),1)
        
        self.T = SimplePyCUDAProjectorTranspose(self)
