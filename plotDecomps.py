# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 10:43:34 2016

@author: varun
"""

import numpy
import glob
import matplotlib.pyplot as plt
import tarfile
import scipy

def getWeights(p):
    """
    Given a set of fit parameters from an even mode decomposition, this function pulls 
    out the normalized mode weights and total weight of all modes in a given family.
    
    - VDV
    """

    # Get Nmax
    Nmax = int(p[0]);

    # Calculate normalized weight of each mode
    amps = p[1:(Nmax/2+1)**2+1]**2;
    weights = amps**2/(numpy.sum(amps**2));
    
    # Calculate total weight of all modes in each family
    family_list = range(0,Nmax+2,2);
    weights_fam = numpy.zeros(len(family_list));
    iStart=0
    for n in family_list:
        weights_fam[family_list.index(n)] = numpy.sum(weights[iStart:iStart+n+1])
        iStart = iStart+n+1
        
    return (weights,weights_fam,family_list)


        
FolderList = glob.glob('*/');
ymin=230; xmin=194; size = 70	
detuning = numpy.zeros(len(FolderList))
plt.ioff()
for folder in FolderList:
    if folder != 'Results\\':
        print(folder)
        z = numpy.load(folder+'z.npy');
        zfit = numpy.load(folder+'zfit.npy');
        params = numpy.load(folder+'pout.npy');
        
        (w,wfam,fams) = getWeights(params);
        numpy.save(folder+'w.npy',w);
        numpy.save(folder+'wfam.npy',wfam);
        numpy.save(folder+'fams',fams);
        
#        filename = glob.glob(folder+'*.tgz')[0]    
#        datafile = tarfile.open(filename)
#        datafile.extract('RunData.mat')
#        RunData = scipy.io.loadmat('RunData.mat')['runData']
#        datafile.close()
#        LSBFreq = RunData['LSBFreq'][0][0][0][0]
#        SRScenter = RunData['SRScenter'][0][0][0][0]
#        
#        ind = FolderList.index(folder)
#        detuning[ind] = -2.0*(LSBFreq*1.0-SRScenter*1.0)/1e6
                    
        ax1 = plt.subplot2grid((2,2),(0,0));
        ax1.imshow(-z,vmin = -numpy.max(z), vmax=0, cmap='jet')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
        ax2 = plt.subplot2grid((2,2),(0,1));
        ax2.imshow(-zfit,vmin=-numpy.max(z), vmax=0,cmap='jet')
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])
        
        ax3 = plt.subplot2grid((2,2),(1,0),colspan=2);
        ax3.bar(fams,wfam,width=1.8,align='center');
        ax3.set_xlim((-1,21))
        ax3.set_ylim((1e-3,1))
        ax3.set_yscale('log')
        ax3.set_xlabel('l+m')
        ax3.set_ylabel('Total weight')
        
        plt.gcf().suptitle(folder[:-1])
        plt.savefig('Results/'+folder[:-1]+'.png')

plt.ion()
plt.close('all')