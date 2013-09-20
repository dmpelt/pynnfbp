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
from nnfbp.ASTRAProjector import ASTRAProjector2D
import astra
import numpy as np

# Create ASTRA geometries
vol_geom = astra.create_vol_geom(256,256)
proj_geom = astra.create_proj_geom('parallel',1.0,256,np.linspace(0,np.pi,64,False))

# Create the ASTRA projector
p = ASTRAProjector2D(proj_geom,vol_geom)


# Define the training and validation set
# Note that we use simulation phantoms as dataset
# If you have existing data, use HDF5Set or define
# a custom DataSet class.
phantom = nnfbp.Phantoms.ThreeShape(256)
trainingSet = nnfbp.PhantomSet(p,phantom,100)
validationSet = nnfbp.PhantomSet(p,phantom,100)

# Define a NN-FBP Network with 8 hidden nodes.
# Note that we only use 1000 pixels for the training set and validation set
# In practice, more pixels should be used for better results
network = nnfbp.Network(8,p,trainingSet,validationSet,nTrain=1000,nVal=1000)

# Train the network on the given data
network.train()

# Create a test phantom and calculate its forward projection
testPhantom = phantom.get()
testSino = p*testPhantom

# Reconstruct the image using the trained network, FBP, and SIRT.
nnRec = network.reconstruct(testSino)
fbpRec = p.reconstruct('FBP_CUDA',testSino)
sirtRec = p.reconstruct('SIRT_CUDA',testSino,1000)

# Show the different reconstructions on screen
import pylab
pylab.gray()
pylab.subplot(141)
pylab.axis('off')
pylab.title('Phantom')
pylab.imshow(testPhantom,vmin=0,vmax=1)
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

# Save the network to disk, so it can be used later.
network.saveToDisk('exampleNetworkASTRA.h5')

# Load the saved network from disk.
# Note that a similar projector as the one it was trained with has to be
# given. After loading a network, it can be used to reconstruct images
# without retraining.
networkLoaded = nnfbp.readFromDisk('exampleNetworkASTRA.h5',p)
nnRec2 = networkLoaded.reconstruct(testSino)
