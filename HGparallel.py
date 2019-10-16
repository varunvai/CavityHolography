# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 11:48:20 2016

@author: varun
"""

import numpy
from HG import HG
import time
import glob
from joblib import Parallel, delayed

def decomp(Folder):
    """
    This function is used to parallelize mode decomposition on the clusters.
    It assumes that inside each folder, there is a averaged andor image saved
    as a numpy file 'MeanImg.npy'. This image is loaded, cropped to a specified
    ROI and then fit. Change this function as needed to suit what you are doing.
    
    - VDV
    """
    print(Folder)
    """ 
    Load image and set ROI.
    """
    img = numpy.load(Folder+'/MeanImg.npy');
    ymin=227; xmin=245; size = 75
    z = img[ymin:ymin+size+1,xmin:xmin+size+1];
    x,y = numpy.meshgrid(numpy.linspace(0,75,76),numpy.linspace(0,75,76))
    
    " Set up HG parameters "
    Nmax = 20;
    hg = HG();
    hg.X0=37.5; hg.Y0=39.0;
    hg.WX=8.5; hg.WY=8.5;
    
    " Create initial guesses "
    A = numpy.ones((Nmax/2+1)**2);
    phi = numpy.zeros((Nmax/2+1)**2)*2*numpy.pi;
    guess = numpy.concatenate(([Nmax],A,phi));
#    guess = numpy.load(Folder+'/pout.npy')
    
    " Fix Nmax and phase of the 00 mode "
    vary = numpy.ones(len(guess));
    vary[0] = 0; vary[(Nmax/2+1)**2+1] = 0;
    
    " Fit the image "
    pout,fitres = hg.evens_fit(z,(x,y),guess,vary)
    zfit = hg.evens((x,y),pout)
    
    " Save the fit parameters 'pout', the original image 'z' and the reconstructed fit 'zfit'"
    numpy.save(Folder+'/pout.npy',pout)    
    numpy.save(Folder+'/zfit.npy',zfit)
    numpy.save(Folder+'/z.npy',z)

if __name__=='__main__':
    FolderList = glob.glob('*/');
    # This creates a thread of 'decomp' for each of the 'n_jobs' cores requested
    # on the cluster
    Parallel(n_jobs=8)(delayed(decomp)(folder) for folder in FolderList)	
