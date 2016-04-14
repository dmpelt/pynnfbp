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

import nnfbp
import astra
import numpy as np
import scipy.io as sio
import os
import tifffile

# Geometry details: number of projections, size of dataset, and amount of noise
na = 64
nd = 256
ns = 128
i0 = 10**3

# Create ASTRA geometries
vol_geom = astra.create_vol_geom(nd,nd)
proj_geom = astra.create_proj_geom('parallel',1.0,nd,np.linspace(0,np.pi,na,False))

# Create the ASTRA projector
pid = astra.create_projector('linear',proj_geom,vol_geom) # change 'linear' to 'cuda' to use GPU
p = astra.OpTomo(pid)

# Create simulated HQ dataset for training
hq = np.zeros((ns,nd,nd))
hq[:,nd//4:3*nd//4,nd//4:3*nd//4]=1
hq[:,3*nd//8:5*nd//8,3*nd//8:5*nd//8]=0

try:
    os.mkdir('hqrecs/')
except OSError:
    pass
projections = np.zeros((ns, na, nd))
for i in range(ns):
    projections[i] = astra.add_noise_to_sino((p*hq[i]).reshape(p.sshape),i0)
    tifffile.imsave('hqrecs/{:04d}.tiff'.format(i),hq[i])

# Prepare training files
astra.plugin.register(nnfbp.plugin_prepare)
try:
    os.mkdir('trainfiles/')
except OSError:
    pass
for i in range(ns):
    p.reconstruct('NN-FBP-prepare',projections[i],extraOptions={'hqrecfiles':'hqrecs/*.tiff','z_id':i,'traindir':'trainfiles/','npick':1000})

# Train filters and weights
nnfbp.plugin_train('trainfiles/', 4, 'filters.mat')

# Generate test sinogram
testSino = astra.add_noise_to_sino((p*hq[ns//2]).reshape(p.sshape),i0)

# Reconstruct the image using NN-FBP, FBP, and SIRT.
astra.plugin.register(nnfbp.plugin_rec)
nnRec = p.reconstruct('NN-FBP',testSino,extraOptions={'filter_file':'filters.mat'})
if astra.projector.is_cuda(pid):
    fbpRec = p.reconstruct('FBP_CUDA',testSino)
    sirtRec = p.reconstruct('SIRT_CUDA',testSino,1000)
else:
    fbpRec = p.reconstruct('FBP',testSino)
    sirtRec = p.reconstruct('SIRT',testSino,1000)


# Show the different reconstructions on screen
import pylab
pylab.gray()
pylab.subplot(141)
pylab.axis('off')
pylab.title('Phantom')
pylab.imshow(hq[ns//2],vmin=0,vmax=1)
pylab.subplot(142)
pylab.axis('off')
pylab.title('NNFBP')
pylab.imshow(nnRec,vmin=0,vmax=1)
pylab.subplot(143)
pylab.axis('off')
pylab.title('FBP')
pylab.imshow(fbpRec,vmin=0,vmax=1)
pylab.subplot(144)
pylab.axis('off')
pylab.title('SIRT-1000')
pylab.imshow(sirtRec,vmin=0,vmax=1)
pylab.tight_layout()
pylab.show()
