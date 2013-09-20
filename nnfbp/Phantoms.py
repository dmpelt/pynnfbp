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
from random import Random
random = Random()
import math

addEllipseCode="""
    double center = (size-1.)/2.;
    unsigned int max = size*size/4;
    double a2 = a*a;
    double b2 = b*b;
    double diffx,diffy,val1,val2;
    #pragma omp parallel for firstprivate(center,max,a2,b2,v) private(diffx,diffy,val1,val2) if (size>64)
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            diffx = (i+1)-xc;
            diffy = (j+1)-yc;
            val1 = (diffx*cosa+diffy*sina);
            val2 = (diffx*sina - diffy*cosa);
            if(val1*val1/a2+val2*val2/b2<=1){
                img[size*j+i] = v;
            }
        }
    }
"""

addGaussianCode="""
    #pragma omp parallel for firstprivate(a,xc,xs2,yc,ys2,cs2) if (size>64)
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            img[size*j+i] += a*exp(-((i-xc)*(i-xc)*xs2 + (j-yc)*(j-yc)*ys2 + 2*(i-xc)*(j-yc)*cs2));
        }
    }
"""

addRectangleCode="""
    double xrot,yrot;
    #pragma omp parallel for firstprivate(a,xc,ct,yc,st,w2,h2) private(xrot,yrot) if (size>64)
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            xrot = ct*(i-xc)-st*(j-yc);
            yrot = st*(i-xc)+ct*(j-yc);
            if(xrot>=-w2&&xrot<=w2&&yrot>=-h2&&yrot<=h2) img[size*j+i]=a;
        }
    }
"""

addSiemensCode="""
    double dx,dy,rpos,tpos;
    #pragma omp parallel for firstprivate(a,xc,theta,spokeRange,r2) private(dx,dy,rpos,tpos) if (size>64)
    for(int i=0;i<size;i++){
        for(int j=0;j<size;j++){
            dx = (i-xc);
            dy = (j-yc);
            rpos = dx*dx+dy*dy;
            tpos = atan2(dy,dx)-theta;
            while(tpos<0) tpos+=M_PI;
            if(((int)(tpos/spokeRange))%2==0){
                if(rpos<=r2) img[size*j+i]+=a;
            }            
        }
    }
"""

def _addSiemensStar(img,xc,yc,a,r,npoints,theta):
    size = img.shape[0]
    spokeRange = 2*math.pi/npoints
    r2 =r**2
    scipy.weave.inline(addSiemensCode,['img','size','xc','yc','a','spokeRange','theta','r2'],compiler = 'gcc',verbose=1,extra_compile_args=['-march=native','-fopenmp'],extra_link_args=['-lgomp'])
    

def _addEllipse(img,xc,yc,angle,a,b,v):
    size = img.shape[0]
    cosa = math.cos(angle)
    sina = math.sin(angle)
    scipy.weave.inline(addEllipseCode,['img','size','xc','yc','cosa','sina','a','b','v'],compiler = 'gcc',verbose=1,extra_compile_args=['-march=native','-fopenmp'],extra_link_args=['-lgomp'])

def _addGaussianBlob(img,xc,yc,a,xs,ys,theta):
    size = img.shape[0]
    xs2 = math.cos(theta)**2/(2*xs**2)+math.sin(theta)**2/(2*ys**2)
    cs2 = math.sin(2*theta)/(4*ys**2)-math.sin(2*theta)/(4*xs**2)
    ys2 = math.sin(theta)**2/(2*xs**2)+math.cos(theta)**2/(2*ys**2)
    scipy.weave.inline(addGaussianCode,['img','size','xc','yc','a','xs2','ys2','cs2'],compiler = 'gcc',verbose=1,extra_compile_args=['-march=native','-fopenmp'],extra_link_args=['-lgomp'])

def _addRectangle(img,xc,yc,a,width,height,theta):
    size = img.shape[0]
    ct = math.cos(theta)
    st = math.sin(theta)
    w2 = width/2
    h2 = height/2
    scipy.weave.inline(addRectangleCode,['img','size','xc','yc','a','ct','st','w2','h2'],compiler = 'gcc',verbose=1,extra_compile_args=['-march=native','-fopenmp'],extra_link_args=['-lgomp'])
    

class Phantom(object):
    ''' A base Phantom object to be used with :class:`nnfbp.DataSet.PhantomSet`. Implementing
    objects should define a ``createPhantom(self,img)`` method, that creates a (random) phantom in
    ``img``, a :class:`numpy.ndarray`.
    
    :param size: Number of rows/columns of the phantom. Only square phantoms are supported at the moment.
    :type size: :class:`int`
    :param rSeed: Optional random seed to use.
    :type rSeed: :class:`int`
    '''
    def __init__(self,size,rSeed=None):
        self.size=size
        random.seed(rSeed)
        self.__setOutCircle()
    
    def setRandomSeed(self,seed):
        random.seed(seed)
    
    def __setOutCircle(self):
        '''Creates a :class:`numpy.ndarray` mask of a circle'''
        xx, yy = np.mgrid[:self.size, :self.size]
        mid = (self.size-1.)/2.
        circle = (xx - mid) ** 2 + (yy - mid) ** 2
        bnd = self.size**2/4.
        self.outCircle=circle>bnd
    
    def clipCircle(self,img):
        '''Zeroes elements outside circle.'''
        img[self.outCircle]=0
    
    def get(self):
        '''Return a random phantom image.'''
        img = np.zeros((self.size,self.size))
        self.createPhantom(img)
        return img

class EightEllipses(Phantom):
    '''Object that creates ``8Ellipses`` phantoms.
    
    :param size: Number of rows/columns of the phantom. Only square phantoms are supported at the moment.
    :type size: :class:`int`
    :param rSeed: Optional random seed to use.
    :type rSeed: :class:`int`
    '''
    def createPhantom(self,img):
        size = self.size
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,1)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.875)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.75)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.625)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.5)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.375)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.25)
        _addEllipse(img,(0.25+0.5*random.random())*size,(0.25+0.5*random.random())*size,2*np.pi*random.random(),(0.25+random.random())*size/6,(0.25+random.random())*size/6,0.125)
        self.clipCircle(img)


class ThreeShape(Phantom):
    '''Object that creates ``ThreeShape`` phantoms.
    
    :param size: Number of rows/columns of the phantom. Only square phantoms are supported at the moment.
    :type size: :class:`int`
    :param rSeed: Optional random seed to use.
    :type rSeed: :class:`int`
    '''
    def createPhantom(self,img):
        size = self.size
        _addGaussianBlob(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, -0.25+1.25*random.random(), (0.05+random.random()*0.1)*size, (0.05+random.random()*0.1)*size, random.random()*2*np.pi)
        _addGaussianBlob(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, -0.25+1.25*random.random(), (0.05+random.random()*0.1)*size, (0.05+random.random()*0.1)*size, random.random()*2*np.pi)
        _addGaussianBlob(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, -0.25+1.25*random.random(), (0.05+random.random()*0.1)*size, (0.05+random.random()*0.1)*size, random.random()*2*np.pi)
        img[img<0]=0
        _addRectangle(img,(0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.25+0.5*random.random())*size,(0.0125+0.0375*random.random())*size, random.random()*2*np.pi)
        _addRectangle(img,(0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.25+0.5*random.random())*size,(0.0125+0.0375*random.random())*size, random.random()*2*np.pi)
        _addRectangle(img,(0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.25+0.5*random.random())*size,(0.0125+0.0375*random.random())*size, random.random()*2*np.pi)
        _addSiemensStar(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.025+0.1*random.random())*size, 16, 2*np.pi*random.random())
        _addSiemensStar(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.025+0.1*random.random())*size, 16, 2*np.pi*random.random())
        _addSiemensStar(img, (0.25+0.5*random.random())*size, (0.25+0.5*random.random())*size, random.random(),(0.025+0.1*random.random())*size, 16, 2*np.pi*random.random())
        self.clipCircle(img)
        img/=img.max()
        
