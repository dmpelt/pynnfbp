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
import astra as at
import math
import numpy as np

class ASTRAProjector2DTranspose():
    """Implements the ``proj.T`` functionality.
    
    Do not use directly, since it can be accessed as member ``.T`` of
    an :class:`ASTRAProjector2D` object.
    
    """
    def __init__(self,parentProj):
        self.parentProj = parentProj
    
    def __mul__(self,data):
        return self.parentProj.backProject(data)
    
class ASTRAProjector2D(object):
    
    """Implementation of the projector interface using the ASTRA toolbox with CUDA.

    A projector needs to implement:
    
    * Forward projecting
    * Back projecting
    * Creating a FBP reconstruction with a custom filter

    You can use this class as an abstracted weight matrix :math:`W`: multiplying an instance
    ``proj`` of this class by an image results in a forward projection of the image, and multiplying
    ``proj.T`` by a sinogram results in a backprojection of the sinogram::
    
        proj = ASTRAProjector2D(...)
        fp = proj*image
        bp = proj.T*sinogram
    
    :param proj_geom: The projection geometry.
    :type proj_geom: :class:`dict` 
    :param vol_geom: The volume geometry.
    :type vol_geom: :class:`dict`
    :param offsets: Optional offsets for the detectors
    :type offsets: :class:`numpy.ndarray`
    """
        
    def __init__(self,proj_geom,vol_geom,offsets=None):
        self.vol_geom = vol_geom
        self.recSize = vol_geom['GridColCount']
        self.angles = proj_geom['ProjectionAngles']
        self.nDet = proj_geom['DetectorCount']
        nexpow = int(pow(2, math.ceil(math.log(2*self.nDet, 2))))
        self.filterSize = nexpow/2 +1
        self.nProj = self.angles.shape[0]
        self.proj_geom = proj_geom
        filt_proj_geom = at.create_proj_geom('parallel',1.0,self.filterSize,self.angles)
        
        if not offsets==None:
            self.proj_geom['option'] = {}
            self.proj_geom['option']['ExtraDetectorOffset'] = offsets
            filt_proj_geom['option'] = {}
            filt_proj_geom['option']['ExtraDetectorOffset'] = offsets
        
        self.filt_id = at.data2d.create('-sino', filt_proj_geom, 0)
        self.sino_id = at.data2d.create('-sino', self.proj_geom, 0)
        self.vol_id = at.data2d.create('-vol',self.vol_geom,0)
        forwProjString = 'FP_CUDA'
        backProjString = 'BP_CUDA'
        cfg = at.astra_dict(backProjString)
        cfg['ProjectionDataId'] = self.sino_id
        cfg['ReconstructionDataId'] = self.vol_id
        self.backProjAlgorithm = at.algorithm.create(cfg)
        
        cfg = at.astra_dict(forwProjString)
        cfg['ProjectionDataId'] = self.sino_id
        cfg['VolumeDataId'] = self.vol_id
        self.forwProjAlgorithm = at.algorithm.create(cfg)
        
        self.T = ASTRAProjector2DTranspose(self)
        
    
    def backProject(self,sinogram):
        """Backproject a sinogram.
        
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The backprojection.
        
        """
        at.data2d.store(self.sino_id,sinogram)
        at.algorithm.run(self.backProjAlgorithm)
        return at.data2d.get(self.vol_id)
    
    def forwProject(self,image):
        """Forward project an image.
        
        :param image: The image data.
        :type image: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The forward projection.
        
        """
        at.data2d.store(self.vol_id,image)
        at.algorithm.run(self.forwProjAlgorithm)
        return at.data2d.get(self.sino_id)
    
    def reconstructWithFilter(self,sinogram,filt):
        """Create a FBP reconstruction of the sinogram with a custom filter
        
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :param filt: 1D custom filter
        :type filt: :class:`numpy.ndarray`
        :returns: :class:`numpy.ndarray` -- The reconstruction.
        
        """
        at.data2d.store(self.sino_id,sinogram)
        at.data2d.store(self.filt_id,np.tile(filt,(self.nProj,1)))
        cfg = at.astra_dict('FBP_CUDA')
        cfg['ProjectionDataId'] = self.sino_id
        cfg['ReconstructionDataId'] = self.vol_id
        cfg['FilterType'] = 'rsinogram'
        cfg['FilterSinogramId'] = self.filt_id
        recoAlg = at.algorithm.create(cfg)
        at.algorithm.run(recoAlg,1)
        at.algorithm.delete(recoAlg)
        return at.data2d.get(self.vol_id)
    
    def reconstruct(self,method,sinogram,nIters=1,fbpfilter=None):
        """Helper function to reconstruct a sinogram using the ASTRA toolbox.
        
        This function does not have to be implemented by other projectors, as
        it is not used by PyNN-FBP.
        
        :param method: Name of the reconstruction algorithm.
        :type method: :class:`string`
        :param sinogram: The sinogram data
        :type sinogram: :class:`numpy.ndarray`
        :param nIters: Number of iterations to run.
        :type nIters: :class:`int`
        :param fbpfilter: Optional string to specify FBP filter (hamming, hann, etc)
        :type fbpfilter: :class:`string`
        :returns: :class:`numpy.ndarray` -- The reconstruction.
        
        """
        cfg = at.astra_dict(method)
        if not 'CUDA' in method: raise Exception('Use CUDA algorithms only')
        cfg['ProjectionDataId'] = self.sino_id
        cfg['ReconstructionDataId'] = self.vol_id
        if not fbpfilter==None:
            cfg['FilterType']=fbpfilter
        recoAlg = at.algorithm.create(cfg)
        at.data2d.store(self.sino_id,sinogram)
        at.data2d.store(self.vol_id,0)
        at.algorithm.run(recoAlg,nIters)
        at.algorithm.delete(recoAlg)
        return at.data2d.get(self.vol_id)
    
    def __mul__(self,data):
        return self.forwProject(data)
    
