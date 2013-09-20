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
from nnfbp.SimpleCPUProjector import Projector
import numpy as np

# Create the CPU projector
p = Projector(256,np.linspace(0,np.pi,64,False))


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

# Reconstruct the image using the trained network
nnRec = network.reconstruct(testSino)

# Show the reconstruction on screen
import pylab
pylab.gray()
pylab.subplot(121)
pylab.axis('off')
pylab.title('Phantom')
pylab.imshow(testPhantom,vmin=0,vmax=1)
pylab.subplot(122)
pylab.axis('off')
pylab.title('NNFBP')
pylab.imshow(nnRec,vmin=0,vmax=1)
pylab.tight_layout()
pylab.show()

# Save the network to disk, so it can be used later.
network.saveToDisk('exampleNetworkCPU.h5')

# Load the saved network from disk.
# Note that a similar projector as the one it was trained with has to be
# given. After loading a network, it can be used to reconstruct images
# without retraining.
networkLoaded = nnfbp.readFromDisk('exampleNetworkCPU.h5',p)
nnRec2 = networkLoaded.reconstruct(testSino)
