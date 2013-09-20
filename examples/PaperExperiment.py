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

# This example reproduces results from the ThreeShape experiment of [1].
#
# [1] Fast tomographic reconstruction from limited data using artificial
#     neural networks, Daniel M. Pelt and Kees Joost Batenburg,
#     submitted for publication, 2013


import nnfbp
from nnfbp.ASTRAProjector import ASTRAProjector2D
import astra
import numpy as np

# Create ASTRA geometries
vol_geom_l = astra.create_vol_geom(4096,4096)
proj_geom_l = astra.create_proj_geom('parallel',1.0,4096,np.linspace(0,np.pi,8,False))
vol_geom = astra.create_vol_geom(1024,1024)
proj_geom = astra.create_proj_geom('parallel',1.0,1024,np.linspace(0,np.pi,8,False))

# Create the ASTRA projectors
p = ASTRAProjector2D(proj_geom,vol_geom)
p_l = ASTRAProjector2D(proj_geom_l,vol_geom_l)


# Define the training and validation set
# Note that we use simulation phantoms as dataset
# If you have existing data, use HDF5Set or define
# a custom DataSet class. The phantoms are defined
# on a 4096x4096 pixel grid, but the phantoms
# and its projections are downsampled to 1024x1024
# pixels and 1024 detectors.
phantom = nnfbp.Phantoms.ThreeShape(4096)
trainingSet = nnfbp.PhantomSet(p,phantom,1000,reduceFactor=4,fwP=p_l)
validationSet = nnfbp.PhantomSet(p,phantom,1000,reduceFactor=4,fwP=p_l)

# Define a NN-FBP Network with 8 hidden nodes.
network = nnfbp.Network(8,p,trainingSet,validationSet,nTrain=1000000,nVal=1000000)

# Train the network on the given data
network.train()

# Create a test set of phantoms
testSet = nnfbp.PhantomSet(p,phantom,100,reduceFactor=4,fwP=p_l)

# Calculate mean absolute errors for test set
fbpErr=0.
sirtErr=0.
nnfbpErr=0.
nPixelsInDisk = (network.outCircle==0).sum()
for i in xrange(len(testSet)):
    d = testSet[i]
    im = d[0]
    sino = d[1]
    nnRec = network.reconstruct(sino)
    fbpRec = p.reconstruct('FBP_CUDA',sino)
    sirtRec = p.reconstruct('SIRT_CUDA',sino,200)
    # Zero central disk first
    nnRec[network.outCircle]=0
    fbpRec[network.outCircle]=0
    sirtRec[network.outCircle]=0
    # Calculate mean absolute error for central disk
    fbpErr += np.abs(im-fbpRec).sum()/nPixelsInDisk
    sirtErr += np.abs(im-sirtRec).sum()/nPixelsInDisk
    nnfbpErr += np.abs(im-nnRec).sum()/nPixelsInDisk
    # Print errors to screen
    print 'Errors after', i+1, 'images: FBP:', fbpErr/(i+1), 'SIRT:', sirtErr/(i+1), 'NNFBP:', nnfbpErr/(i+1)
