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

'''This module has some implementations of the DataSet interface that is 
used in PyNN-FBP. A dataset should be an iterable object, that returns
a :class:`tuple` of :class:`numpy.ndarray` (image, sinogram, angles). A
dataset should also define the :class:`int` `nImages`, that gives the number
of images in the set. A user can write other implementations suitable to
their datasets.
'''

hastables=True
try:
	import tables as ts
except ImportError:
	hastables=False

hasfabio=True
try:
	import fabio
except ImportError:
	hasfabio=False
	
import numpy as np
import scipy.weave
import scipy.io


class PhantomSet(object):
    '''
    Data set of phantom simulation data.
    
    :param proj: A projector.
    :type proj: :class:`Projector` 
    :param phantom: A phantom (see :mod:`nnfbp.Phantoms`).
    :type phantom: :class:`Phantom`
    :param nImages: Number of images in the set.
    :type nImages: :class:`int`
    :param reduceFactor: Factor to reduce the number of detectors by.
    :type reduceFactor: :class:`int`
    :param I0: Number of virtual photon counts during Poisson noise addition.
    :type I0: :class:`int`
    :param recP: Projector to use for reconstructing the sinogram.
    :type recP: :class:`Projector`
    :param fwP: Projector to use for forward projecting the sinogram.
    :type fwP: :class:`Projector`
    '''
    
    __meanImageShrinkCode="""
        #pragma omp parallel for firstprivate(fac,sz)
        for(int x=0;x<sz;x++){
            for(int y=0;y<sz;y++){
                image(x,y)=0.;
                for(int x2=fac*x;x2<fac*(x+1);x2++){
                    for(int y2=fac*y;y2<fac*(y+1);y2++){
                        image(x,y)+=imageIn(x2,y2);
                    }
                }
                image(x,y)/=fac*fac;
            }
        }
        """

    def addnoisetosino(self,sinogramRaw):
        max_sinogramRaw = sinogramRaw.max()
        sinogramRawScaled = sinogramRaw / max_sinogramRaw
        sinogramCT = self.I0 * np.exp(-sinogramRawScaled)
        sinogramCT_C = np.zeros_like(sinogramCT)
        for i in xrange(sinogramCT_C.shape[0]):
            for j in xrange(sinogramCT_C.shape[1]):
                sinogramCT_C[i, j] = np.random.poisson(sinogramCT[i, j])
        sinogramCT_D = sinogramCT_C / self.I0
        return -max_sinogramRaw * np.log(sinogramCT_D)
    
    def __init__(self,proj,phantom,nImages,reduceFactor=None,I0=None,recP=None,fwP=None):
        self.proj = proj
        self.ph = phantom
        self.nImages = nImages
        self.rF = reduceFactor
        self.I0 = I0
        self.recP = recP
        self.fwP = fwP
        
    def __len__(self):
        return self.nImages
        
    def __getitem__(self,i):
        '''
        Get item ``i`` from the set.
        
        :returns: :class:`tuple` of :class:`numpy.ndarray` -- (image, sinogram, angles)
        '''
        if i<0 or i>=self.nImages:
            raise IndexError()
        imageIn = self.ph.get()
        if not self.fwP==None:
            prj = self.fwP
        else:
            prj = self.proj
        sinoIn = prj*imageIn
        if not self.I0==None:
            sinoIn = self.addnoisetosino(sinoIn)
        if not self.rF==None:
            fac = self.rF
            sz = imageIn.shape[0]/fac
            sino = np.empty((sinoIn.shape[0],sinoIn.shape[1]/fac))
            image = np.zeros((sz,sz))
            for q in xrange(sino.shape[1]):
                sino[:,q] = sinoIn[:,range(fac*q,fac*(q+1))].sum(1)
            sino/=fac*fac
            scipy.weave.inline(self.__meanImageShrinkCode,['sz','image','imageIn','fac'],compiler = 'gcc',verbose=1,extra_link_args=['-lgomp'],extra_compile_args=['-march=native','-fopenmp'],type_converters = scipy.weave.converters.blitz)
        else:
            sino = sinoIn
            image = imageIn
        if not self.recP==None:
            image = self.recP.reconstruct('FBP_CUDA',sino)
        if not self.fwP==None:
            fca = self.fwP.nProj/self.proj.nProj
            sino = sino[0:sino.shape[0]:fca,:]
        return (image,sino,self.proj.angles) 
        
class EDFSet(object):
    def __init__(self,imgfiles,sinofiles,angles,nproj=None):
        if not hasfabio:
            raise Exception("FabIO has to be installed to use EDFSet")
        self.nImages = len(files)
        self.angles = angles
        self.nproj=nproj
        self.imgfiles = imgfiles
        self.sinofiles = sinofiles
        
    def __len__(self):
        return self.nImages
    
    def __getitem__(self,i):
        '''
        Get image ``i`` from the set.
        
        :returns: :class:`tuple` of :class:`numpy.ndarray` -- (image, sinogram, angles)
        '''
        if i<0 or i>=self.nImages:
            raise IndexError()
        fl = fabio.open(self.imgfiles[i])
        image = fl.data
        fl = fabio.open(self.sinofiles[i])
        sino = fl.data
        angles = self.angles.copy()
        if not self.nproj==None:
            picked = np.array(np.round(np.linspace(0,sino.shape[0],self.nproj,False)),dtype=np.int)
            sino = sino[picked,:]
            angles = angles[picked]
        return (image,sino,angles)

class DMPSet(object):
    def __init__(self,sinofiles,angles,recfiles=None,nproj=None,sinoSize=None,center=None,proj=None,padType='constant'):
        self.nImages = len(sinofiles)
        self.angles = angles
        self.nproj=nproj
        self.sfiles = sinofiles
        self.rfiles = recfiles
        self.sinoSize = sinoSize
        self.center = center
        self.p = proj
        self.padType = padType
        
    def __len__(self):
        return self.nImages
    
    def readImageDmp(self, filename ):
        fd = open( filename , 'rb' )
        datatype = 'h'
        numberOfHeaderValues = 3
        headerData = np.zeros(numberOfHeaderValues)
        headerData = np.fromfile(fd, datatype, numberOfHeaderValues)
        imageShape = (headerData[1], headerData[0])
        imageData = np.fromfile(fd, np.float32, -1)
        imageData = imageData.reshape(imageShape)
        fd.close()
        return imageData.astype(np.float32)
    
    def padAndCenter(self,sinoIn,sinoSize,center,padType='constant'):
        sino = np.zeros((sinoIn.shape[0],sinoSize))
        shft = center-sinoIn.shape[1]/2
        lr = sino.shape[1]/2-sinoIn.shape[1]/2+shft
        rr = sino.shape[1]/2+sinoIn.shape[1]/2+shft
        sino[:,lr:rr]=sinoIn
        if padType=='constant':
            lw = np.tile(np.ones(lr),(sinoIn.shape[0],1))
            rw = np.tile(np.ones(sino.shape[1]-rr),(sinoIn.shape[0],1))
        elif padType=='sqrt':
            lw = np.tile(np.sqrt(np.linspace(0,1,lr)),(sinoIn.shape[0],1))
            rw = np.tile(np.sqrt(np.linspace(1,0,sino.shape[1]-rr)),(sinoIn.shape[0],1))
        sino[:,0:lr] = lw*(np.tile(sino[:,lr],(lr,1)).transpose())
        sino[:,rr:sino.shape[1]] = rw*(np.tile(sino[:,rr-1],(sino.shape[1]-rr,1)).transpose())
        return sino
        
    
    def __getitem__(self,i):
        '''
        Get image ``i`` from the set.
        
        :returns: :class:`tuple` of :class:`numpy.ndarray` -- (image, sinogram, angles)
        '''
        if i<0 or i>=self.nImages:
            raise IndexError()
        sinoIn = self.readImageDmp(self.sfiles[i])
        if self.sinoSize==None:
            sino = sinoIn
        else:
            sino = self.padAndCenter(sinoIn,self.sinoSize,self.center,padType=self.padType)
        angles = self.angles.copy()
        if self.rfiles==None:
            image = self.p.reconstruct('FBP_CUDA',sino)
        else:
            image = self.readImageDmp(self.rfiles[i])
        if not self.nproj==None:
            picked = np.array(np.round(np.linspace(0,sino.shape[0],self.nproj,False)),dtype=np.int)
            sino = sino[picked,:]
            angles = angles[picked]
        return (image,sino,angles)  

class MATSet(object):
    def __init__(self,files,angles,nproj=None,sinoname='sino',recname='rec'):
        self.nImages = len(files)
        self.angles = angles
        self.nproj=nproj
        self.files = files
        self.sinoname=sinoname
        self.recname=recname
        
    def __len__(self):
        return self.nImages
    
    def __getitem__(self,i):
        '''
        Get image ``i`` from the set.
        
        :returns: :class:`tuple` of :class:`numpy.ndarray` -- (image, sinogram, angles)
        '''
        if i<0 or i>=self.nImages:
            raise IndexError()
        fl = scipy.io.loadmat(self.files[i])
        image = fl[self.recname]
        sino = fl[self.sinoname]
        angles = self.angles.copy()
        if not self.nproj==None:
            picked = np.array(np.round(np.linspace(0,sino.shape[0],self.nproj,False)),dtype=np.int)
            sino = sino[picked,:]
            angles = angles[picked]
        if 'mask' in fl:
            return (image,sino,angles,fl['mask'])  
        else:
            return (image,sino,angles)  

        
		
    
class HDF5Set(object):
    '''Dataset defined by a HDF5 file. The HDF5 file should contain an array with the name 
    ``nameArray`` (default:``'names'``), which has a row for each image. The first column
    gives the name of the array with the image data, the second column the name of the array
    with sinogram data, and the optional third column should give the name of the array with angle data.
    One can also define an array with the name ``globalAngles`` (default:``'angles'``), that gives the angle data
    for all images.
    
    Example HDF5 file:
    
    - ``/names``: ``[ ['img01','sino01'],['img02','sino02']]``
    - ``/img01``: :class:`numpy.ndarray` with an image
    - ``/sino01``: :class:`numpy.ndarray` with a sinogram
    - ``/img02``: :class:`numpy.ndarray` with an image
    - ``/sino02``: :class:`numpy.ndarray` with a sinogram
    - ``/angles``: :class:`numpy.ndarray` with angle data
    
    :param fn: Filename of the HDF5 file
    :type fn: :class:`string`
    :param nproj: Optional number of angles to downsample to
    :type nproj: :class:`int`
    :param nameArray: Array name of the nameArray (default:``'names'``)
    :type nameArray: :class:`string`
    :param globalAngles: Array name of the globalAngles (default:``'angles'``)
    :type globalAngles: :class:`string`
    :param normalize: Whether to normalize images to `(0,1)` range (default:``True``)
    :type normalize: :class:`boolean`
    '''
    def __init__(self,fn,nproj=None,nameArray='names',globalAngles='angles',normalize=True):
        if not hastables:
            raise Exception("PyTables has to be installed to use HDF5Set")
        self.fn = fn
        h5file = ts.openFile(self.fn, mode='r')
        self.names = h5file.getNode(h5file.root, nameArray).read()
        self.nImages = len(self.names)
        self.globalAngles = globalAngles
        h5file.close()
        self.norm = normalize
        self.nproj = nproj
    
    def __len__(self):
        return self.nImages
    
    def __getitem__(self,i):
        '''
        Get image ``i`` from the set.
        
        :returns: :class:`tuple` of :class:`numpy.ndarray` -- (image, sinogram, angles)
        '''
        if i<0 or i>=self.nImages:
            raise IndexError()
        h5file = ts.openFile(self.fn, mode='r')
        image = h5file.getNode(h5file.root, self.names[i][0]).read()
        sino = h5file.getNode(h5file.root, self.names[i][1]).read()
        if len(self.names[i])<3:
            angles = h5file.getNode(h5file.root, self.globalAngles).read()
        else:
            angles = h5file.getNode(h5file.root, self.names[i][2]).read()
        h5file.close()
        if self.norm:
            maxI = image.max()
            image /= maxI
            sino /= maxI
        if not self.nproj==None:
            origNproj = sino.shape[0]
            sino = sino[np.array(np.round(np.linspace(0,origNproj,self.nproj,False)),dtype=np.int),:]
            angles = angles[np.array(np.round(np.linspace(0,origNproj,self.nproj,False)),dtype=np.int)]
        return (image,sino,angles)
