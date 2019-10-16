# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 11:22:14 2018

@author: varun
"""
import scipy
import scipy.optimize
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
from sifreader_v3 import SifFile
import cmath
from scipy.constants import pi
import colorcet as cc
import numpy as np

class hologram:
    """
    Loads a cavity hologram and extracts phase given fringe calibration data.
    Corrects for local oscillator variations intensity if LO image is provided.
    
    Required Args:
        - holog - Hologram image. Can either be an array, .sif or .tgz file.
        - k_bw - Demodulation bandwidth in units of k-space pixels.
        - k_fringe - Fringe wavector in the format [kx0, ky0] in units of k-space pixels
        - cal_img - Path to TEM00 calibration image. (Avoid using this)
    
    Optional Args:
        - LO - Local oscillator image.
        - inten - Path to intensity mask.
    
    Vars:
        - holo_file - SifFile object of hologram.
        - LO_file - SifFile object of local oscillator image.
        - inten_file - SifFile object of intensity mask.
        - kx0, ky0 - Fringe wavector components in k-space pixels.
        - amp - Calculated E-field.
        - phase - Calculated phase.
    
    Methods:
        - fringe_calib - Calibrate fringe wavector using reference 00 hologram.
        - extract_field - Extract amplitude and phase of hologram.
        - phase_rgba - Convert phase into RGB array using cyclic colormap.
        - plot_fft - Plot fft of hologram. Useful for determining demod bandwidth.
                  
    -VDV
    """
    def __init__(self, holog, k_bw = 150, **kwargs):
        " Load holgram data "
        try:
            self.holo_file = SifFile(holog);
        except:
            self.holo_img = holog;
        else:
            self.holo_img = self.holo_file.data;
            if self.holo_file.properties['OutputAmplifier'] == '1': 
                self.holo_img = np.fliplr(self.holo_img);
                                         
        (self.Ny, self.Nx) = self.holo_img.shape;
        
        " Load cavity intensity mask if provided "
        if 'inten' in kwargs.keys():
            try:
                self.inten_file = SifFile(kwargs['inten']);
            except:
                self.inten_img = kwargs['inten'];
            else:
                self.inten_img = self.inten_file.data;
                if self.inten_file.properties['OutputAmplifier'] == '1':
                    self.inten_img = np.fliplr(self.inten_img);
                        
        " Load LO intensity if provided "
        if 'LO' in kwargs.keys():
            try:
                self.LO_file = SifFile(kwargs['LO']);
            except:
                self.LO_img = kwargs['LO'];
            else:
                self.LO_img = self.LO_file.data;
                if self.LO_file.properties['OutputAmplifier'] == '1':
                    self.LO_img = np.fliplr(self.LO_img);
                
        " Set fringe wavevector "
        (self.kx, self.ky) = np.meshgrid(np.arange(self.Nx),np.arange(self.Ny));
        self.kx = self.kx - (self.Nx - 1.)/2; 
        self.ky = self.ky - (self.Ny - 1.)/2;
        try:
            (self.kx0, self.ky0) = kwargs['k_fringe'];
        except KeyError:
            try:
                self.fringe_calib(kwargs['cal_img'], cal_cutoff = 100);
            except KeyError:
                print('ERROR: No fringe calibration data provided');
            else:
                pass
        else:
            if 'LO' in kwargs.keys():
                (self.amp, self.phase) = self.extract_field(self.holo_img, \
                                        LO_img = self.LO_img, k_cutoff = k_bw);
            else:
                (self.amp, self.phase) = self.extract_field(self.holo_img, \
                                                            k_cutoff = k_bw);
               
    
    def fringe_calib(self, calib_image, calib_cutoff = 50):
        """
        Uses fringe calibration image to calculate and set fringe wavevector.
        
        Args:
            - calib_image - Fringe calibration image
            - calib_cutoff - momentum width of DC component to be blanked out
        
        Output:
            - (kx0, ky0) - Fringe wavevector in units of inverse pixels.
            
        - VDV
        """
        " Load fringe calibration image "
        fringeCal = SifFile(calib_image).data;
        
        " Perform FFT and blank out DC component "
        (self.kx,self.ky) = np.meshgrid(range(1024),range(1024));
        self.kx = self.kx-511.5; self.ky = self.ky-511.5;
        fft00 = np.fft.fftshift(np.fft.fft2(fringeCal))
        fft00 = fft00*((np.abs(self.kx) > calib_cutoff) | (np.abs(self.ky) > calib_cutoff))
        fft00_abs = np.abs(fft00);
        
        " Fit TEM00 hologram to find k-vector of fringes "
        peakFit = lambda p,kx,ky: p[0]*(np.exp((-(kx-p[1])**2-(ky-p[2])**2)/p[3]**2) + \
                        np.exp((-(kx+p[1])**2-(ky+p[2])**2)/p[3]**2))

        kx_guess = self.kx[np.where(fft00_abs == np.max(fft00_abs))]
        ky_guess = self.ky[np.where(fft00_abs == np.max(fft00_abs))]
        
        p0 = [4e6,kx_guess,ky_guess,15]
        (pfit,cov,info,mesg,ier) = scipy.optimize.leastsq(lambda P:(fft00_abs.ravel() - \
                                peakFit(P,self.kx,self.ky).ravel()),p0,full_output=1)
        
        self.kx0 = pfit[1]; self.ky0 = pfit[2];
        return (self.kx0, self.ky0)
    
    
    def extract_field(self, holo_img, k_cutoff = 150, **kwargs):
        """
        Extract field amplitude and phase by demodulation at fringe wavelength.
        Can be used on arrays independently of the hologram object if needed, 
        as long as fringe calibration is still valid.
        
        Required Args:
            - holo_img - Hologram of image. Must be a 2D array.
            - k_cutoff - Demodulation bandwidth for a gaussian mask.
        
        Optional Args:
            - LO_img - Local oscillator image. Must be 2D array of same size as holo_img
            - window - Custom window function to override gaussian mask. Must be 2D array of same size as holo_img.
        
        Outputs:
            - (amp, phase) - Extracted amplitude and phase.
        
        - VDV   
        """
        " Subtract LO and normalize hologram if LO image is provided "
        if 'LO_img' in kwargs.keys():
            LO_img = kwargs['LO_img']
            self.holo_corr = (holo_img - LO_img)/np.sqrt(LO_img);
            self.holo_corr[np.where(np.invert(np.isfinite(self.holo_corr)))] = 0
        else:
            self.holo_corr = holo_img;
        
        " Demodulate normalized hologram by mixing with exp^(i*k_fringe*r) "
        (x,y) = np.meshgrid(np.arange(self.Nx), np.arange(self.Ny));
        shift_transform = np.exp(1j*2*np.pi*(self.kx0*x/self.Nx + self.ky0*y/self.Ny));
        holo_demod = self.holo_corr*shift_transform;
               
        " Perform FFT and low-pass filter with provided window function "
        holo_fft = np.fft.fftshift(np.fft.fft2(holo_demod));
        if 'window' in kwargs.keys():
            LP_filt = kwargs['window'];
        else:
            # Circular window
#            LP_filt = (self.kx**2 + self.ky**2) < k_cutoff**2;
            # Gaussian window
#            LP_filt = np.exp(-(self.kx**2 + self.ky**2)/k_cutoff**2);
            # Flat Top window
            LP_filt = np.exp(-((self.kx**2 + self.ky**2)/k_cutoff**2)**(6));
        holo_fft = holo_fft*LP_filt; self.holo_fft = holo_fft
        
        " Inverse FFT to extract field amplitude and phase "
        holo_demod = np.fft.ifft2(np.fft.ifftshift(holo_fft)); 
        phase = np.angle(holo_demod);
        amp = np.abs(holo_demod);
        return (amp, phase);
        
    
    def phase_rgba(self, phase = None, inten = None, gamma = 0.5):
        """ 
        Convert phase into RGBA map for ease of plotting with imshow or pcolor.
        RGB values are calculated from a cyclic color map. 
        Alpha (transparency) is controlled by inten^gamma. 
        gamma = 0 is an unscaled image, 0.5 is E-field scaling, 1 is intensity etc. 
        Intensity-masked output looks best when plotted on a black background.
        
            Args:
                - phase - M x N array of phases between -pi and pi
                - inten - M x N array with intensity of cavity light
                - gamma - alpha value is controlled by inten^gamma.
                
            Outputs:
                - phase_plot - M x N x 4 array of RGBA values.
                
        - VDV
        """
        
#        col_map = cc.cm['colorwheel']
#        col_map = cc.cm['cyclic_mrybm_35_75_c68']
        conv = {'red':   ((0.0,  20.0/255, 20.0/255),
                   (0.25, 0, 0),
                   (0.5,  1.0, 1.0),
                   (0.75, 158.0/255, 158.0/255),  
                   (1.0, 20.0/255, 20.0/255)),

         'green': ((0.0,  80.0/255, 80.0/255),
                   (0.25, 98.0/255, 98.0/255),
                   (0.5, 128.0/255, 128.0/255),
                   (0.75, 21.0/255, 21.0/255), 
                   (1.0,  80.0/255, 80.0/255)),

         'blue':  ((0.0,  135.0/255, 135.0/255),
                   (0.25, 90.0/255, 90.0/255),
                   (0.5,  0, 0),
                   (0.75, 25.0/255, 25.0/255),
                   (1.0, 135.0/255, 135.0/255))}
                   
        col_map = LinearSegmentedColormap('Converging', conv)
        
        if phase.all() == None:
            phase = self.phase;
        
        if inten.all() == None:
            inten = np.ones(phase.shape);
            
        phase_plot = np.zeros(phase.shape);
        phase_plot = phase_plot[:,:,np.newaxis].repeat(4,2);
            
        phase_plot[:,:,0] = col_map((phase + np.pi)/(2*np.pi))[:,:,0]
        phase_plot[:,:,1] = col_map((phase + np.pi)/(2*np.pi))[:,:,1]
        phase_plot[:,:,2] = col_map((phase + np.pi)/(2*np.pi))[:,:,2]
        phase_plot[:,:,3] = (np.abs(inten)/np.max(np.abs(inten)))**gamma; 
            
        return phase_plot
    
    
    def plot_fft(self):
        self.FFT = np.fft.fftshift(np.fft.fft2(self.holo_img));
        plt.imshow(np.log(np.abs(self.FFT)), cmap = 'jet_r', vmax = 14);
        plt.plot([self.kx0+511.5],[self.ky0+511.5],'.r')