# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:46:57 2016

@author: varun
"""
import scipy
import scipy.optimize
from scipy.interpolate import interp1d
from scipy.special import eval_hermite as hermite
from math import factorial,sqrt
import types
import cmath
from scipy.constants import pi

class HG:
    """
    This is a class used to evaluate various kinds of Hermite-Gaussians used by
    cQED. It uses a high resolution lookup table of HG functions upto 100th 
    order. The table extends from -10*waist to +10*waist with a resolution of 
    0.001*waist.
    
    Methods:
        - HG.HG1D - 1D Hermite-Gaussian function
        - HG.HG2D - 2D Hermite-Gaussian function
        
        - HG.family - Intensity from a superposition of HGs in a given l+m family in 2D
        - HG.familyrand - Intensity from a random superposition of HGs in a given family
        - HG.family_fit - Fit to a superposition of HGs in a given family
        
        - HG.evens - Intensity from a superposition of even HGs
        - HG.evensrand - Intensity from a random superposition of even HGs
        - HG.evens_fit - Fit to a superposition of even HGs
        
    -VDV
    """
    
    def __init__(self):
        """
        Instantiating the HG class loads the lookup table, and makes interpolation
        functions for every HG-order in the lookup table. The interpolation functions
        are stored in a the list self.HGinterp. E.g. self.HGinterp[10] is the 
        interpolation function for the 10th order HG function.
                
        - VDV
        """
       
        # Load lookup table
        HGtable = scipy.load('HGhash_0_100.npy')
        self.HGtable = HGtable;
        self.x0 = HGtable[-1,0];
        self.dx = HGtable[-1,1]-HGtable[-1,0];       
        
        # Generate interpolation functions
        self.HGinterp = [];
        for m in range(0,101):
            self.HGinterp.append(interp1d(HGtable[-1],HGtable[m,:]))
            
        # Waist and center for mode decomposition. These are only for the 
        # 'family', 'familyrand', 'evens' and 'evensrand' methods. Change these after 
        # instantiation to whatever the fitted values of w and x0 are. 
        self.WX = 1; self.WY = 1;
        self.X0 = 0; self.Y0 = 0;

                  
    def HG1D(self,x,p=[0,0,1]):
        """
        1D Hermite-Gaussian
            x = co-ordinate
            
            p = HG parameters
                p[0] = order of HG
                p[1] = center position
                p[2] = waist
                
        - VDV
        """
        
        " Using scipy hermite polynomial function "
#        gauss = scipy.exp(-((x-p[1])/p[2])**2)
#        herm = hermite(int(p[0]),sqrt(2)*(x-p[1])/p[2])
#        return sqrt(sqrt(2/pi))*herm*gauss*sqrt(1./(factorial(p[0])*2^(p[0])))                             
        
        " Interpolate on lookup table "
#        return self.HGinterp[int(p[0])]((x-p[1])/p[2])
        
        " Lookup without interpolation "
        xprime = (x-p[1])/p[2];
        ind = scipy.rint((xprime-self.x0)/self.dx)
        return self.HGtable[int(p[0]),ind.astype(int)]
      
                
    def HG2D(self,X,P=[0,0,1,0,0,1]):
        """
        2D Hermite-Gaussian
            X = co-ordinates
                X[0] = x-coordinate
                X[1] = y-coordinate
                
            P = HG parameters
                P[0] = x-order of HG
                P[1] = x-center position
                P[2] = x-waist
                P[3] = y-order of HG
                P[4] = y-center position
                P[5] = y-waist
             
        -VDV
        """

        " Old, dumb method of evaluating at every point on the grid "        
#        return self.HG1D(X[0],[P[0],P[1],P[2]])*self.HG1D(X[1],[P[3],P[4],P[5]])
        
        " The smarter way assumes a square grid and uses seperability of 2D HGs "
        x = X[0][0,:]; y = X[1][:,0]
        HGx = self.HG1D(x,[P[0],P[1],P[2]]);
        HGy = self.HG1D(y,[P[3],P[4],P[5]]);
        HGmesh = scipy.meshgrid(HGx,HGy);
        return HGmesh[0]*HGmesh[1]

                        
    def family(self,X,P):
        """
        Evaluates the intensity of a superposition of HGs from a given l+m
        family. The values of the waist and position are held fixed to the
        relevant class variables (self.X0, self.WX etc.). These can be changed 
        as needed in a given instance of the class.
            
            X = co-ordinates
                X[0] = x-coordinate
                X[1] = y-coordinate
            
            P = Parameters
                P[0] = family number (l+m)
                Amplitudes:
                    P[1] = amplitude of HG_N0
                    P[2] = amplitude of HG_(N-1)1 etc...
                    P[N+1] = amplitude of HG_0N
                Phases:
                    P[N+1+1] = phase of HG_N0
                    P[N+1+2] = phase of HG_(N-1)1 etc...
                    P[2*(N+1)] = phase of HG_0N
                    
        - VDV
        """
        # get HG family number
        N = int(P[0]);
        
        # pull out positions and waists        
        X0 = self.X0; Y0 = self.Y0;
        WX = self.WX; WY = self.WY;
        field = scipy.zeros(X[0].shape,dtype=scipy.complex128);
        
        # calculate field of the superposition
        for n in range(0,N+1):
            field += cmath.rect(P[n+1],P[(N+1)+(n+1)])*self.HG2D(X,[N-n,X0,WX,n,Y0,WY]);
    
        # return intensity
        return abs(field)**2
    
    def __family_field(self,X,P):
        """
        Identical to 'family' but returns the field instead of the intensity
                    
        - VDV
        """
        # get HG family number
        N = int(P[0]);
        
        # pull out positions and waists        
        X0 = self.X0; Y0 = self.Y0;
        WX = self.WX; WY = self.WY;
        field = scipy.zeros(X[0].shape,dtype=scipy.complex128);
        
        # calculate field of the superposition
        for n in range(0,N+1):
            field += cmath.rect(P[n+1],P[(N+1)+(n+1)])*self.HG2D(X,[N-n,X0,WX,n,Y0,WY]);
    
        # return field
        return field
        
        
    def evens(self,X,P):
        """
        Evaluates the intensity from a superposition of all even HGs UPTO a 
        maximum order l+m=N given by P[0].
    
            X = point to evaluate
                X[0] = x-coordinate
                X[1] = y-coordinate
            
            P = Parameters
                P[0] = N
                Amplitudes:
                    P[1] = amplitude of HG_00
                    P[2] = amplitude of HG_20 
                    P[3] = amplitude of HG_11
                    P[4] = amplitude of HG_02 etc...
                    P[(N/2+1)^2] = amplitude of HG_0N
                Phases:
                    P[(N/2+1)^2+1] = phase of HG_00 etc ...
                    P[2*(N/2+1)^2] = phase of HG_0N
                    
        - VDV        
        """
        # get maximum family number and calculate total number of modes
        Nmax = int(P[0]);
        
        field = scipy.zeros(X[0].shape,dtype=scipy.complex128);
        family_list = scipy.arange(0,Nmax+2,2,dtype=int);
        
        " Iterate through each family "
        for n in family_list:
            
            " Pull out parameters for each family n "
            mEnd = int((n/2+1)**2);
            mStart = mEnd - n;            
            amps = P[mStart:mEnd+1];
            phases = P[mStart+(n/2+1)**2:mEnd+(n/2+1)**2+1]
            params = scipy.concatenate(([n],amps,phases))
            
            " Evaluate the total field from family n and add it on "
            field += self.__family_field(X,params)
        
        return abs(field)**2
              
    
    def familyrand(self,X,N):
        """
        Makes a random superposition of Hermite-Gaussians in the Nth family.
            X = point to evaluate
                X[0] = x-coordinate
                X[1] = y-coordinate
            
            N = family number
        
        - VDV
        """

        X0 = self.X0; Y0 = self.Y0;
        WX = self.WX; WY = self.WY;
        field = 0; j = 0;
        
        A = scipy.random.rand(N+1);
        phi = scipy.random.rand(N+1)*2*scipy.pi;
        
        P = scipy.concatenate(([N],A,phi));
        
        return self.family(X,P)
    
    
    def evensrand(self,X,Nmax):
        """
        Makes a random superposition of even Hermite-Gaussians upto the Nth
        family.

            X = co-ordinates
                X[0] = x-coordinate
                X[1] = y-coordinate
            
            N = family number
            
        - VDV
        """
        
        A = scipy.random.rand((Nmax/2+1)**2);
        phi = scipy.random.rand((Nmax/2+1)**2)*2*pi;
        P = scipy.concatenate(([Nmax],A,phi));
        
        return self.evens(X,P)

    
    def family_fit(self,data,X,P0,vary):
        """
        Fits image in 'data' to a family of HG modes.
            
            data = Image that needs to be fit
            X = meshgrid of image co-ordinates
            P0 = Initial guess of fit parameters. This should have the same 
                 format as the parameters of 'family'
            vary = parameters to vary in the fit
            
        - VDV
        """
        
        " Figure out what parameters to keep fixed "
        c = P0[scipy.where(vary == 0)];
        p0 = P0[scipy.where(vary == 1)];
        
        " Fit the image "
        fitres = scipy.optimize.leastsq(lambda P:(data.ravel()-self.__fixPars(self.family,X,c,P,vary)),p0)       

        " Reorganize the output to have the same format as the input "
        pout = scipy.zeros(len(P0));
        pout[scipy.where(vary == 0)] = c;
        pout[scipy.where(vary == 1)] = fitres[0];
        return pout


    def evens_fit(self,data,X,P0,vary):
        """
        Fits image in 'data' to a superposition of even HG families.
            
            data = Image that needs to be fit
            X = meshgrid of image co-ordinates
            P0 = Initial guess of fit parameters. This should have the same 
                 format as the parameters of 'family'
            vary = parameters to vary in the fit
        
        - VDV
        """
 
        " Figure out what parameters to keep fixed "
        c = P0[scipy.where(vary == 0)];
        p0 = P0[scipy.where(vary == 1)];
        
        " Fit the image "
        fitres = scipy.optimize.leastsq(lambda P:(data.ravel()-self.__fixPars(self.evens,X,c,P,vary)),p0,maxfev=2000*(len(p0)+1),full_output=1)  
        
        " Reorganize the output to have the same format as the input "
        pout = scipy.zeros(len(P0));
        pout[scipy.where(vary == 0)] = c;
        pout[scipy.where(vary == 1)] = fitres[0];
        return (pout,fitres)

        
    def __fixPars(self,fitfunc,X,consts,free,vary):
        """
        This lets you call any of the above HG functions with fixed
        fit parameters. Almost identical to the levlab.fit version.
        - VDV        
        """
        params = scipy.zeros(len(vary));
        params[scipy.where(vary==0)] = consts;
        params[scipy.where(vary==1)] = free;
        return fitfunc(X,params).ravel()
        
            
    " Do not use anything below this line. - VDV "
#==============================================================================
#     def familydecomp(self,X,Int,P0):
#         P = scipy.optimize.minimize(self.costfunc,P0,args=(X,Int));
#         return P
#     
#     def costfunc(self,P,X,I):
#         Ifit = self.family(X,P);
#         return 1-scipy.sum(Ifit*I)/(scipy.sqrt(scipy.sum(I*I)*scipy.sum(Ifit*Ifit)));
#==============================================================================
