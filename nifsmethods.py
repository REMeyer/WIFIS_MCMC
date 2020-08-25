from __future__ import print_function

import numpy as np
from astropy.io import fits
from glob import glob
from sys import exit
from scipy import stats
import matplotlib.pyplot as mpl

###############################################
# A number of routines for nifs reduction.
# 
# make_flattened
# flatten_region
# get_extensions
# vis_slices
# modify_flat
# genwlarr
# sigmaclipspec
#
################################################

def flatten_cube(data, region = 'None', method = 'mean'):
    '''Flattens array by averaging. Assumes its in the form (Z,X,Y) where the X and Y axes are the ones to 
    flatten. Can flatten a specific region bounded by X1Y1X2Y2. Can flatten in the spatial regions via mean, median
    or a 'meanclip' where each wavelength is masked for values above 0 and then sigmaclipped.'''
   
    if region != 'None':
        dataregion = data[:, region[0]:region[1], region[2]:region[3]]
    else:
        dataregion = data
    
    if method == 'mean':
        arr1 = np.mean(dataregion,axis=1)
        flattened = np.mean(arr1,axis=1)
    elif method == 'median':
        arr1 = np.median(dataregion,axis=1)
        flattened = np.median(dataregion,axis=1)
    
    elif method == 'meanclip':
        flattened = []
        for i in range(len(dataregion[:,0,0])):
            wlslice = dataregion[i,4:58,:]
            #sl = dataregion[i,4:58,:].flatten()
            sl = wlslice.flatten()            
            mask = sl > 0
            
            #if np.mean(sl[mask]) > 0:
            #    n, bins, patches = mpl.hist(sl[mask], 50)
            #    mpl.title(str(i))
            #    mpl.show()
            #    if np.argmax(n) == 0:
            #        mpl.imshow(dataregion[i,4:58,:], interpolation = 'none', origin = 'lower')
            #        mpl.show()

            clipped = stats.sigmaclip(sl[mask], low = 5, high = 5)[0] ##NUMPY METHOD
            flattened.append(np.mean(clipped))
        flattened = np.array(flattened)
        
    elif method == 'meanclipzeros':
        flattened = []
        for i in range(len(dataregion[:,0,0])):
            wlslice = dataregion[i,4:58,:]
            sl = wlslice.flatten()            
            clipped = stats.sigmaclip(sl, low = 5, high = 5)[0] ##NUMPY METHOD
            flattened.append(np.mean(clipped))
        flattened = np.array(flattened)
    
    return flattened

def get_extensions(hdu, name):
    '''Get extensions from hdu with name 'name'.'''

    sciext = []
    for n,ext in enumerate(hdu):
        if ext.name == name:
            sciext.append([n,ext])

    return np.array(sciext)

def vis_slices(fl, writepath):
    '''Un-cuts a nfprepare'd cut and processed 2d-spectral image'''

    ffl = fits.open(fl)
    
    sciext = get_extensions(ffl, 'SCI')

    sampledata = sciext[0].data
    newimg = np.zeros((2,sampledata.shape[1]))
    
    for sci in sciext.values():
        sciarr = sci.data
        for row in sciarr:
            newimg = np.append(newimg, [row], axis = 0)
        newimg = np.append(newimg, [np.zeros(scidata.shape[2])], axis = 0)
        newimg = np.append(newimg, [np.zeros(scidata.shape[2])], axis = 0)

    #for sciarr in scidata:
    #    newimg = np.append(s
    hdu = fits.PrimaryHDU(newimg)
    hdu.writeto(writepath, clobber=True)

    return newimg

def modify_flat(inputfl, outputfl):
    '''Changes the SCI and Data Quality extentions to be completely
    in order to reduce the impact of flat fielding on the rest of the
    reduction'''

    f = fits.open(inputfl)

    sciext = get_extensions(f, 'SCI')
    dqext = get_extensions(f, 'DQ')

    for ext in sciext:
        f[ext[0]].data = f[ext[0]].data*0.0 + 1.0
    for ext in dqext:
        f[ext[0]].data = f[ext[0]].data*0.0

    f.writeto(outputfl, clobber=True)

def genwlarr(hdu, speclen = 2040, resample = 1., verbose=False):
    '''Generates a wavelength array from as supplied by keywords in the fitsheader'''

    for i in range(1,4):
        if hdu.header['WAT'+str(i)+'_001'] == 'wtype=linear axtype=wave':
            head = str(i)

    wstart_head = 'CRVAL'+str(head)
    dw_head = 'CD'+str(head)+'_'+str(head)

    w1 = hdu.header[wstart_head]
    dw = hdu.header[dw_head] / resample

    speclen = int(speclen*resample)

    wlarr = []
    for i in range(speclen):
        wlarr.append(w1 + dw*i)
    wlarr = np.array(wlarr)

    if verbose:
        print("Creating wl_arr with start %f and step %f. Length of wlarray is %i." % (w1, dw, len(wlarr)))
        
    return wlarr

def sigmaclipspec(spec, halfwidth = 10, sigma = 3):

    print("Starting sigma clipping...")
    print("Sigma = %i, halfwidth = %i" % (sigma, halfwidth))
    print("## ##")

    for i in range(halfwidth, len(spec)-halfwidth):
        value = spec[i]
        stdrange = spec[i-halfwidth:i+halfwidth]
        stdrange = np.delete(stdrange, halfwidth)
        stddev = np.std(stdrange)
        rangemean = np.mean(stdrange)
        
        upper = rangemean + sigma*stddev
        lower = rangemean - sigma*stddev
        if value > upper or value < lower:
            print("Correcting value %f at %i" % (value, i))
            spec[i] = rangemean
    print()
    return spec
        
def normalizespec(spec, method = 'median'):

    if method == 'median':
        return spec/np.median(spec)
    if method == 'mean':
        return spec/np.mean(spec)
    if method == 'meansub':
        return (spec - np.mean(spec))/np.median(spec)

def blackbody(wl, T):

    wl = np.array(wl)/1.0e10
    kb = 1.38064852e-23
    c = 299792458
    h = 6.62607e-34

    firstterm = 2*h*(c**2.0)/ wl**5.0
    secondterm = 1.0 / (np.exp((h*c)/(wl*kb*T)) - 1.0)

    return firstterm * secondterm

def broaden(lam, sigma):

    c = 299792.458 

    return lam/(1 - sigma/c) - lam

def reduce_resolution(wlarr, data, v=2):

    newdata = []
    newwlarr = []

    i = 0
    while i < len(data):
        if i + v - 1 < len(data):
            newwlarr.append(np.mean(wlarr[i:i+v-1]))
            newdata.append(np.mean(data[i:i+v-1]))
        else:
            newwlarr.append(wlarr[i])
            newdata.append(data[i])            
        i += v
    #print len(newwlarr), len(newdata) 
    return np.array(newwlarr), np.array(newdata)

def skymask(wlarr, data, galaxy, band):

    if galaxy == 'M87':
        skystart = [10374, 10329, 11390, 11701, 11777, 11788]
        skyend =   [10379, 10333, 11398, 11705, 11785, 11795]

    elif galaxy == 'M85':
        skystart = [10346,11411]
        skyend =   [10353,11415]

    wls = []
    conts = []
    #print wlarr
    for i in range(len(skystart)):
        #blueband = np.where((wlarr >= skystart[i] - 4) & (wlarr <= skystart[i]))[0]
        #redband = np.where((wlarr >= skyend[i]) & (wlarr <= skyend[i] + 4))[0]
        
        #if len(blueband) == 0:
        #    continue

        #blueavg = np.mean(data[blueband])
        #redavg = np.mean(data[redband])

        blueend = np.where(wlarr <= skystart[i])[0]
        redend = np.where(wlarr >= skyend[i])[0]
        
        if (len(blueend) == 0) or (len(redend) == 0):
            continue
        #print len(blueend), len(redend), skystart[i], skyend[i]
        #continue
        blueavg = np.mean(data[blueend[-3:]])
        redavg = np.mean(data[redend[:3]])

        mainpass = np.where((wlarr >= skystart[i]) & (wlarr <= skyend[i]))[0]

        #mpl.plot(wlarr[blueend[-1] - 30:redend[0]+30],data[blueend[-1] - 30:redend[0] + 30])
        #mpl.show()

        #Do the linear fit
        #pf = np.polyfit([skystart[i]-2, skyend[i]+2],[blueavg, redavg], 1)
        pf = np.polyfit([skystart[i], skyend[i]],[blueavg, redavg], 1)
        polyfit = np.poly1d(pf)
        cont = polyfit(wlarr[mainpass])

        #print "Plotting skymask"
        #mpl.plot(wlarr[mainpass], cont,'g--')
        print("Correcting skyfeature: %s, %s, %s" % (str(skystart[i]), galaxy, band))
        data[mainpass] = cont
        
    return wlarr, data

