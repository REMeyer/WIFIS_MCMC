from __future__ import print_function

from astropy.io import fits
import numpy as np
import nifsmethods as nm

def preparespec(galaxy, baseforce = False):

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
    wlz = nm.genwlarr(fz[0])

    if galaxy == 'M87':
        wlznew, dataz = nm.reduce_resolution(wlz, dataz)
        wlzerrors, errorsz = nm.reduce_resolution(wlz, errorsz)
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
    wlj = nm.genwlarr(fj[0])

    if galaxy == 'M87':
        wljnew, dataj = nm.reduce_resolution(wlj, dataj)
        wljerrors, errorsj = nm.reduce_resolution(wlj, errorsj)
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

def preparespecwifis(fl, z, baseforce = False):

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

    #bluelow, bluehigh, linelow, linehigh, redlow, redhigh, \
    #      mlow, mhigh, morder, line_name, index_name = linedefs

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




