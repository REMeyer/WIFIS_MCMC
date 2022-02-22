################################################################################
# Compilation of spectral handling functions for the MCMC module
################################################################################

from __future__ import print_function

from astropy.io import fits
import numpy as np

def preparespecwifis(fl, z, baseforce = False):
    '''
    Reads input data file for WIFIS spectra. The format of the fits file
    should be:
        extension 0: spectrum
        extension 1: full wavelength array
        extension 2: uncertainty/error array
        extension 3: OPTIONAL mask array
    If there is a masking array the spectrum will be masked with np.nan 
    objects. The non-masked spectrum is also returned.
    '''

    if baseforce:
        base = baseforce
    else:
        pass

    ff = fits.open(fl)
    data = ff[0].data
    wl = ff[1].data 
    errors = ff[2].data
    try:
        mask = np.array(ff[3].data, dtype = bool)
    except:
        print("...No mask included in fits")
        mask = np.ones(data.shape, dtype = bool)

    data_nomask = np.array(data)
    data[mask == False] = np.nan

    wl = wl / (1. + z)

    return wl, data, errors, data_nomask, mask

def splitspec(wl, data, linedefsfull, err=False, scale = False, usecont=True, returncont=False):
    '''
    Splits the spectrum into the requested features and returns the segmented spectra.
    The splitting is also done on the wavelength and error arrays.

    Inputs:
        wl: Wavelength array
        data: Spectrum
        linedefsfull: Full spectral feature definitions
        err: Uncertainty/error array
        scale: Depreciated parameter to scale the spectra by a set value
        usecont: Flag to normalize the spectra by the continuum as done when
                  measuring spectral indices.
        returncont: Flag to return the continuum array. Used for debugging.
    '''

    linedefs = linedefsfull[0]
    line_names = linedefsfull[1]
    index_names = linedefsfull[2]

    databands = []
    wlbands = []
    errorbands = []
    contarray = []

    for i in range(len(linedefs[6])):
        if type(linedefs) == list:
            wh = np.where(np.logical_and(wl >= linedefs[6][i],wl <= linedefs[7][i]))[0]
        else:
            wh = np.where(np.logical_and(wl >= linedefs[6,i],wl <= linedefs[7,i]))[0]

        dataslice = data[wh]
        wlslice = wl[wh]
        wlbands.append(wlslice)

        if usecont:
            if linedefs[8,i] == 1:
                #Define the bandpasses for each line 
                k = np.where(index_names == line_names[i])[0]

                bluepass = np.where((wl >= linedefs[0,k]) & (wl <= linedefs[1,k]))[0]
                redpass = np.where((wl >= linedefs[4,k]) & (wl <= linedefs[5,k]))[0]
                fullpass = np.where((wl >= linedefs[0,k]) & (wl <= linedefs[5,k]))[0]

                #Cacluating center value of the blue and red bandpasses
                blueavg = np.mean([linedefs[0,k], linedefs[1,k]])
                redavg = np.mean([linedefs[4,k], linedefs[5,k]])

                blueval = np.nanmean(data[bluepass])
                redval = np.nanmean(data[redpass])

                pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)
                contarray.append(polyfit)

                if scale:
                    newdata = np.array(data)
                    newdata[fullpass] -= scale*polyfit(wl[fullpass])

                    blueval = np.nanmean(newdata[bluepass])
                    redval = np.nanmean(newdata[redpass])

                    pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                    polyfit = np.poly1d(pf)
                    cont = polyfit(wlslice)

            else:
                pf = np.polyfit(wlslice, dataslice, linedefs[8,i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)

            #databands.append(1. - (dataslice / cont))
            databands.append(dataslice / cont)
            if type(err) != bool:
                errslice = err[wh]
                #errorbands.append(1 - (errslice / cont))
                errorbands.append(errslice / cont)
        else:
            databands.append(dataslice)
            if type(err) != bool:
                errslice = err[wh]
                errorbands.append(errslice)
    
    if returncont:
        return wlbands, databands, errorbands, contarray
    else:
        return wlbands, databands, errorbands

def convolvemodels(wlfull, datafull, veldisp, reglims = False):
    '''
    Performs gaussian broadening of the input spectrum to match it to
    a particular velocity dispersion.

    Inputs:
        wlfull: Input wavegrid
        datafull: Input spectrum
        veldisp: Target velocity dispersion
        reglims: Bandpass limits

    Returns:
        wlfull: Output wavegrid
        out: Broadened spectrum
    '''

    if reglims:
        reg = (wlfull >= reglims[0]) & (wlfull <= reglims[1])
        m_center = reglims[0] + (reglims[1] - reglims[0])/2.
        #print("Reglims")
    else:
        reg = (wlfull >= 9500) & (wlfull <= 13500)
        m_center = 11500
        #print("Not Reglims")
    
    wl = wlfull[reg]
    dw = wl[1]-wl[0]
    data = datafull[reg]

    c = 299792.458

    #Sigma from description of models
    m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    #f = m_center + m_sigma
    #v = c * ((f/m_center) - 1)
    
    sigma_gal = np.abs((m_center / (veldisp/c + 1.)) - m_center)
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    #convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    #gaussplot = gauss_nat(convolvex, [sigma_conv,0.])

    #out = np.convolve(datafull, gaussplot, mode='same')
    out = scipy.ndimage.gaussian_filter(datafull, sigma_conv / dw)

    return wlfull, out

def removeLineSlope(wlc, mconv, linedefs, i):
    '''
    Function that performs a spectral index style normalization
    by normalalizing the feature by a linear fit between the 
    two continuum regions

    Inputs:
        wlc: input wavegrid
        mconv: input spectrum
        linedefs: spectral feature bandpass definitions

    Returns:
        polyfit: numpy linear polyfit object
    '''

    bluelow,bluehigh,redlow,redhigh = linedefs

    #Define the bandpasses for each line 
    bluepass = np.where((wlc >= bluelow[i]) & (wlc <= bluehigh[i]))[0]
    redpass = np.where((wlc >= redlow[i]) & (wlc <= redhigh[i]))[0]

    #Cacluating center value of the blue and red bandpasses
    blueavg = np.mean([bluelow[i],bluehigh[i]])
    redavg = np.mean([redlow[i],redhigh[i]])

    blueval = np.mean(mconv[bluepass])
    redval = np.mean(mconv[redpass])

    pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
    polyfit = np.poly1d(pf)

    return polyfit

###############
# DEPRECIATED #
###############

def preparespec(galaxy, baseforce = False):
    '''
    Legacy function to input gemini data. Depreciated
    '''

    if baseforce:
        base = baseforce
    else:
        pass
        #global base

    if galaxy == 'M87':
        z = 0.004283
        ejf = base+'data/M87J_errors.fits'
        ezf = base+'data/M87Z_errors.fits'
        scale = 1.0
        contcorr = False
        flz = base+'data/20150602_obs60_merged_reduced.fits'
        flj = base+'data/20150605_obs52_merged_reduced.fits'

    if galaxy == 'M85':
        z = 0.002432
        ejf = base+'data/M85J_errors.fits'
        ezf = base+'data/M85Z0527_errors.fits'
        scale = 1.0
        contcorr = False
        flj = base+'data/20150508_obs36_merged_reduced.fits'
        flz = base+'data/20150527_obs44_merged_reduced.fits'
        flNa = base+'data/20150527_obs44_merged_reduced_NAFIX.fits'

    fz = fits.open(flz)
    dataz = fz[0].data
    errz = fits.open(ezf)
    errorsz = errz[0].data
    wlz = genwlarr(fz[0])

    if galaxy == 'M87':
        wlznew, dataz = reduce_resolution(wlz, dataz)
        wlzerrors, errorsz = reduce_resolution(wlz, errorsz)
        wlz = wlznew
        
    wlz = wlz / (1 + z)
    dwz = wlz[50] - wlz[49]
    wlz, dataz = nm.skymask(wlz, dataz, galaxy, 'Z')

    if galaxy == 'M85':
        print("Replacing NA spectrum")
        whna = np.where((wlz >= bluelow[2]) & (wlz <= redhigh[2]))[0]
        #mpl.plot(wlz[whna], dataz[whna])
        fna = fits.open(flNa)
        datana = fna[0].data
        dataz[whna] = datana 
        #mpl.plot(wlz[whna], dataz[whna])
        #mpl.show()

    #Opening and de-redshifting the J-band spectra
    fj = fits.open(flj)
    dataj = fj[0].data
    errj = fits.open(ejf)
    errorsj = errj[0].data    
    wlj = genwlarr(fj[0])

    if galaxy == 'M87':
        wljnew, dataj = reduce_resolution(wlj, dataj)
        wljerrors, errorsj = reduce_resolution(wlj, errorsj)
        wlj = wljnew

    wlj = wlj / (1 + z)
    dwj = wlj[50] - wlj[49]
    wlj, dataj = nm.skymask(wlj, dataj, galaxy, 'J')

    #Cropping the j-band spectrum so no bandpasses overlap between the two
    zendval = 11500
    zend = np.where(wlz < zendval)[0][-1]
    wlz = wlz[:zend]
    dataz = dataz[:zend]
    errorsz = errorsz[:zend]

    jstartval = 11500
    jstart = np.where(wlj > jstartval)[0][0]
    wlj = wlj[jstart:]
    dataj = dataj[jstart:]
    errorsj = errorsj[jstart:]

    finalwl = np.concatenate((wlz,wlj))
    finaldata = np.concatenate((dataz,dataj))
    finalerr = np.concatenate((errorsz,errorsj))

    return finalwl, finaldata, finalerr

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
