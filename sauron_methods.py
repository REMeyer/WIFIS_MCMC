import pandas as pd
from scipy.interpolate import interp1d, interp2d
from scipy import stats
import numpy as np
from astropy.io import fits
from sys import exit

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.gridspec as gridspec

from glob import glob
import os

from astropy.visualization import (PercentileInterval, LinearStretch,
                                    ImageNormalize, ZScaleInterval)
from astropy import wcs
from astropy import units as u
from astropy.modeling.models import Ellipse2D
from astropy.coordinates import Angle

import WIFISSpectrum as WS

bluelow =  [4827.875, 4946.500, 5142.625]
bluehigh = [4847.875, 4977.750, 5161.375]
linelow =  [4847.875, 4977.750, 5160.125]
linehigh = [4876.625, 5054.000, 5192.625]
redlow =   [4876.625, 5054.000, 5191.375]
redhigh =  [4891.625, 5065.250, 5206.375]

c = 299792.458

def ellipse_region_sauron(cube, center, a, ecc, theta, annulus = False):
    ''' Function that returns a mask that defines the elements that lie
        within an ellipsoidal region. The ellipse can also be annular.
        
        Inputs:
            Cube:    The data array
            center:  Tuple of the central coordinates (x_0, y_0) (spaxels)
            a:       The semi-major axis length (spaxels).
            ecc:     The eccentricity of the ellipse (0<ecc<1)
            theta:   The rotation angle of the ellipse from vertical (degrees)
                     rotating clockwise. 
            
        Optional Inputs:
            annulus: False if just simple ellipse, otherwise is the INNER
                     annular radius (spaxels)
        
    '''
    
    # Define angle, eccentricity term, and semi-minor axis
    an = Angle(theta, 'deg')
    e = np.sqrt(1 - (ecc**2))
    b = a * e
    
    # Create outer ellipse
    ell_region = Ellipse2D(amplitude=10, x_0 = center[0], y_0 = center[1],\
                a=a, b=b, theta=an.radian)
    x,y = np.mgrid[0:cube.shape[0], 0:cube.shape[1]]
    ell_good = ell_region(x,y)
    
    if annulus:
        # Define inner ellipse parameters and then create inner mask
        a2 = annulus
        b2 = a2 * e
        print(a2, b2)
        ell_region_inner = Ellipse2D(amplitude=10, x_0 = center[0],\
                y_0 = center[1], a=a2, b=b2, theta=an.radian)
        
        # Set region of outer mask and within inner ellipse to zero, leaving
        # an annular elliptical region
        ell_inner_good = ell_region_inner(x,y)
        ell_good[ell_inner_good > 0] = 0
        
    #fig, ax = mpl.subplots(figsize = (12,7))
    #mpl.imshow(np.nansum(cube, axis = 0), origin = 'lower', interpolation = None)
    #mpl.imshow(ell_good, alpha=0.2, origin = 'lower', interpolation = None)
    #mpl.show()
    
    # Return the mask
    return np.array(ell_good, dtype = bool)

def prepare_sauron(fl, ellipse, write=False, plot=False):

    ff = fits.open(fl)
    data = ff[0].data
    head = ff[0].header
    image = ff[1].data
    table = ff[2].data
    wl = np.arange(data.shape[1])*head['CDELT1'] + head['CRVAL1']

    x = table['A']
    y = table['D']
    SNflat = table['SN']

    xx, yy = np.meshgrid(np.sort(list(set(x))), np.sort(list(set(y))))
    zz = np.zeros(xx.shape)
    ss = np.zeros((xx.shape[0],xx.shape[1],data.shape[1]))
    SN = np.zeros(xx.shape) + 0.0001
    im = np.mean(data, axis = 1)
    #print(xx[0,:].shape, yy[:,0].shape,zz.shape,xx.shape, yy.shape,\
        #len(np.sort(list(set(x)))), len(np.sort(list(set(y)))))
    for i, val in enumerate(zip(x,y)):
        xi = np.where(xx[0,:] == val[0])
        yi = np.where(yy[:,0] == val[1])
        try:
            zz[yi,xi] = im[i]
            SN[yi,xi] = SNflat[i]
            ss[yi,xi,:] = data[i,:]
        except:
            print(xi,yi,i)
        
    SNspec = np.zeros(ss.shape)
    for i in range(ss.shape[-1]):
        SNspec[:,:,i] = ss[:,:,i] / SN * np.sqrt(ss.shape[-1])
    
    #ellipse = (cx[0][0],cx[1][0],-28.,5,0.759,False)
    cx = np.where(np.logical_and(xx == 0, yy == 0))
    whgood = ellipse_region_sauron(zz, (cx[0][0], cx[1][0]),\
                    ellipse[3], ellipse[4], ellipse[2], \
                    annulus = ellipse[5])
    whgoodflat = whgood.flatten()
    
    ssnew = np.moveaxis(ss, 2, 0)
    ssflat = ssnew.reshape(ssnew.shape[0], -1)
    specslice = ssflat[:,whgoodflat]
    
    specmeanorig = np.nanmean(specslice, axis = 1)
    goodi = np.ones(specslice.shape[1], dtype=bool)
    for i in range(specslice.shape[1]):
        if np.median(specslice[:,i]) < 0.05:
            goodi[i] = False
    specslice = specslice[:,goodi]
    
    specmean = np.nanmean(specslice, axis = 1)
    specsum = np.nansum(specslice, axis = 1)
    
    SNnew = np.moveaxis(SNspec, 2, 0)
    SNflat = SNnew.reshape(SNnew.shape[0], -1)
    noiseslice = SNflat[:,whgoodflat]
    noiseslice = noiseslice[:,goodi]
    noisemean = np.sqrt(np.sum(noiseslice**2.0, axis = 1)) / specslice.shape[1]
    noisesum = np.sqrt(np.sum(noiseslice**2.0, axis = 1))
    
    if plot:
        gg = np.zeros(zz.shape)
        gg[whgood] = 1.0
        fig, ax = plt.subplots(figsize=(8,8))
        #ax.scatter(x,y,c = np.mean(data, axis = 1), s=50)
        ax.imshow(zz, origin='lower', extent = (np.min(x)-0.4,np.max(x)+0.4,\
            np.min(y)-0.4, np.max(y)+0.4), interpolation=None)
        
        if ellipse[5]:
            e= np.sqrt(1 - (ellipse[4]**2.0))
            an = Angle(ellipse[2],'deg')
            b = ellipse[3]*e
            el = patches.Ellipse((0,0), 2*ellipse[3],2*b, angle=-an.degree+90,
                        linewidth=2, facecolor='none', edgecolor='m')
            ax.add_patch(el)
            
            b = ellipse[5]*e
            el2 = patches.Ellipse((0,0), 2*ellipse[5],2*b, angle=-an.degree+90,
                        linewidth=2, facecolor='none', edgecolor='r')
            ax.add_patch(el2)
        
        else:
            e= np.sqrt(1 - (ellipse[4]**2.0))
            an = Angle(ellipse[2],'deg')
            b = ellipse[3]*e
            el = patches.Ellipse((0,0), 2*ellipse[3],2*b, angle=-an.degree+90,
                        linewidth=2, facecolor='none', edgecolor='r')
            ax.add_patch(el)
        
        plt.show()
        
        fig, ax = plt.subplots(figsize=(10,6))
        
        ax.plot(wl, specsum, label='CorrMean')
        #ax.plot(wl, specmeanorig, label='Mean')
        ax.fill_between(wl, specsum - noisesum, y2 = specsum + noisesum,\
                 alpha = 0.3, color='#333333')
        ax.legend()
        plt.show()
        
        fig, ax = plt.subplots(figsize=(14,10))
        
        for i in range(specslice.shape[1]):
            ax.plot(wl, specslice[:,i]/specslice[20,i])
        plt.show()

    if write:
        print("Writing extracted spectrum....")
        hdu = fits.PrimaryHDU(specsum)
        hdu2 = fits.ImageHDU(wl, name = 'WL')
        hdu3 = fits.ImageHDU(noisesum, name = 'ERR')
        hdul = fits.HDUList([hdu,hdu2,hdu3])

        hdul.writeto(write, overwrite=True)
        print("Wrote to: ", write)
    
    return wl, specsum, noisesum