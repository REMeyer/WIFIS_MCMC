}###############################################################################
#   WIFIS MCMC spectral fitting - 'full-index' version
#   Algorithm based on the emcee affine-invariant ensemble sampler mcmc 
#       implementation by Foreman-Mackey et al. (2013)
#   Author: Elliot Meyer, Dept Astronomy & Astrophysics University of Toronto
################################################################################
#
# This program is run on the command line by invoking python mcmc_fullindex.py
# 
# Please refer to the __main__ component for the beginning of the program.
# An input file placed in the inputs directory is used to provide the 
#   required input parameters for the mcmc simulation
# A logfile keeps track of the mcmc runs in the logs directory.
#
# Please refer to the functions in the mcmc_spectra.py module for the format of
#   the input spectra.
# 
# This suite is programmed to work with the Conroy et al. (2018) SPS models.
#
################################################################################

from __future__ import print_function

import numpy as np
import pandas as pd
import emcee
import time
import matplotlib.pyplot as mpl
import scipy.interpolate as spi
import warnings
import sys, os
import mcmc_support as mcsupp
import mcmc_spectra as mcspec
import plot_corner as plcr
import logging
from random import uniform
from multiprocessing import Pool
from astropy.io import fits
from sys import exit
from glob import glob

warnings.simplefilter('ignore', np.RankWarning)

# DEFINES THE BASE PATH -- NEEDS UPDATING FOR ALL SYSTEMS
base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

# MCMC Parameters
# Metallicity: -0.2 < [Z/H] < 0.2 (Continuous)
# Age: 1.0--13.5 (Continuous)
# IMF - x1 & x2: both from 0.5-3.5 in steps of 0.2
# Various chemical abundances: -0.5 -- [X/H] -- 0.5

#Global imf slope array
x1_m = 0.5 + np.arange(16)/5.0
x2_m = 0.5 + np.arange(16)/5.0

#Dictionary to help easily access the IMF index
imfsdict = {}
for i in range(16):
    for j in range(16):
        imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

#Init dict for global access to interpolated models
#vcj = {}

def preload_vcj(overwrite_base = False, sauron=False, saurononly=False, 
        MLR=False):
    '''Loads the SSP models, performs multidimensional interpolation,
    then creates a dictionary to speed up model creation in the mcmc.

    Inputs: 
        overwrite_base: forcing a base path for running the code
        sauron: including sauron spectra, will extend the output model bandpass
        saurononly: only for sauron spectra, will restrict the model bandpass
        MLR: for calculating K-band based MLR estimates
    
    Return:
        vcj: a dictionary of both the imf and chemical abundance models
                keyed by Age_Z
        imf_interp: An array of interpolated models for each x1 & x2 pair
        ele_interp: An array of interpolated models for each chemical 
                        abundance
    '''

    #global vcjfull
    #global base

    vcj = {}

    if overwrite_base:
        base = overwrite_base
    else:
        base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

    chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', \
            'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', \
            'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+','V+', 'Cu+', 'Na+0.6',\
            'Na+0.9']

    # Loading the IMF models for each Age/Z pair 
    print("PRELOADING SSP MODELS INTO MEMORY")
    fls = glob(base+'models/vcj_ssp/*')    
    for fl in fls:
        flspl = fl.split('/')[-1]
        mnamespl = flspl.split('_')

        age = float(mnamespl[3][1:])
        Zsign = mnamespl[4][1]
        Zval = float(mnamespl[4][2:5])

        if Zsign == "m":
            Zval = -1.0 * Zval

        x = pd.read_csv(fl, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj["%.1f_%.1f" % (age, Zval)] = [x[:,1:]]

    # Loading the chemical abundance models for each Age/Z pair
    print("PRELOADING ABUNDANCE MODELS INTO MEMORY")
    fls = glob(base+'models/atlas/*')    
    for fl in fls:
        flspl = fl.split('/')[-1]
        mnamespl = flspl.split('_')

        age = float(mnamespl[2][1:])
        Zsign = mnamespl[3][1]
        Zval = float(mnamespl[3][2:5])

        if Zsign == "m":
            Zval = -1.0 * Zval
        if age == 13.0: # Small correction based on the model age range differences
            age = 13.5

        x = pd.read_csv(fl, skiprows=2, names = chem_names, 
                delim_whitespace = True, header=None)
        x = np.array(x)
        vcj["%.1f_%.1f" % (age, Zval)].append(x[:,1:])

    # Saving the model wavelength array
    vcj["WL"] = x[:,0]
    print("FINISHED LOADING MODELS")

    # Determine the bandpass for the model output
    wlfull = vcj["WL"]
    if sauron:
        bp = np.where((wlfull > 4000) & (wlfull < 14000))[0]
    elif saurononly:
        bp = np.where((wlfull > 4000) & (wlfull < 6000))[0]
    elif MLR:
        bp = np.where((wlfull > 20000) & (wlfull < 24000))[0]
    else:
        bp = np.where((wlfull > 8500) & (wlfull < 14000))[0]
    wl = wlfull[bp]

    # Grids for model age and metallicity for multi-variate interpolation
    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])

    # Create the imf interpolation array
    print("Calculating IMF interpolators")
    imf_interp = []
    for k in range(vcj['3.0_0.0'][0].shape[1]):
        out = np.meshgrid(fullage,fullZ,wl)
        grid = (out[0],out[1],out[2])
        newgrid = np.zeros(out[0].shape)
        for i,age in enumerate(fullage):
            for j,z in enumerate(fullZ):
                #print(vcj["%.1f_%.1f" % (age, z)][0][bp,73].shape)
                newgrid[j,i,:] = vcj["%.1f_%.1f" % (age, z)][0][bp,k]
        fulli = spi.RegularGridInterpolator((fullZ,fullage,wl), newgrid)
        imf_interp.append(fulli)
    
    # Create the chemical abundance interpolation array
    print("Calculating elemental interpolators")
    ele_interp = []
    for k in range(vcj['3.0_0.0'][1].shape[1]):
        out = np.meshgrid(fullage,fullZ,wl)
        grid = (out[0],out[1],out[2])
        newgrid = np.zeros(out[0].shape)
        for i,age in enumerate(fullage):
            for j,z in enumerate(fullZ):
                newgrid[j,i,:] = vcj["%.1f_%.1f" % (age, z)][1][bp,k]
        fulli = spi.RegularGridInterpolator((fullZ,fullage,wl), newgrid)
        ele_interp.append(fulli)
    
    return vcj, imf_interp, ele_interp

def model_spec(inputs, paramnames, paramdict, saurononly = False, 
        vcjset = False, timing = False, full = False, MLR=False, 
        fixZ = False):
    '''
    Core function which takes input model parameters, acquires the appropriate
    base model (age, Z, imf), then adjusts the model for the chemical
    abundances. Returns an un-broadened model spectrum.

    Inputs:
        inputs: The input parameter values
        paramnames: The model fit parameters (e.g. age, z, x1, x2, etc)
        paramdict: Model fit parameter dictionary, used to set constant values
        saurononly: Flag to only fit sauron spectra
        vcjset: Parameter to pass the model variables so this function can be
                    used outside the main mcmc program
        timing (debug): Flag to output timing information
        full: Flag for sauron + WIFIS spectra
        MLR: Flag for calculating K-band MLR estimates
        fixZ: Flag to fix all chemical abundances to the general metallicity
    
    Returns:
        wl: The model wavelength array
        newm: The chemically adjusted model based on the input parameters
        basemodel: The base model (see above) with a Kroupa IMF
    '''

    global vcj

    if vcjset:
        vcj = vcjset

    if timing:
        print("Starting Model Spec Time")
        t1 = time.time()

    # Separating input parameters into variables
    for j in range(len(paramnames)):
        if paramnames[j] == 'Age':
            Age = inputs[j]
        elif paramnames[j] == 'Z':
            Z = inputs[j]
        elif paramnames[j] == 'x1':
            x1 = inputs[j]
        elif paramnames[j] == 'x2':
            x2 = inputs[j]
        elif paramnames[j] == 'Na':
            Na = inputs[j]
        elif paramnames[j] == 'K':
            K = inputs[j]
        elif paramnames[j] == 'Fe':
            Fe = inputs[j]
        elif paramnames[j] == 'Ca':
            Ca = inputs[j]
        elif paramnames[j] == 'Alpha':
            alpha = inputs[j]

    # Action in case x1 or x2 not in the list of parameters
    # Generally used to set x1 = x2 for special case IMF
    if 'x1' not in paramnames:
        x1 = 1.3
    if 'x2' not in paramnames:
        #x2 = 2.3
        x2 = x1

    # If any of the items in the paramdict are not None, 
    #   set the associated variable to that value
    for key in paramdict.keys():
        if paramdict[key] == None:
            continue

        if key == 'Age':
            Age = paramdict[key]
        elif key == 'Z':
            Z = paramdict[key]
        elif key == 'x1':
            x1 = paramdict[key]
        elif key == 'x2':
            x2 = paramdict[key]
        elif key == 'Na':
            Na = paramdict[key]
        elif key == 'K':
            K = paramdict[key]
        elif key == 'Fe':
            Fe = paramdict[key]
        elif key == 'Ca':
            Ca = paramdict[key]
        elif key == 'Alpha':
            alpha = paramdict[key]

    # Finding closest IMF values in the IMF grid
    if x1 not in x1_m:
        x1min = np.argmin(np.abs(x1_m - x1))
        x1 = x1_m[x1min]
    if x2 not in x2_m:
        x2min = np.argmin(np.abs(x2_m - x2))
        x2 = x2_m[x2min]

    if timing:
        t2 = time.time()
        print("MSPEC T1: ",t2 - t1)
    
    # Finding the relevant bandpass to adjust the 
    # computational complexity
    wlfull = vcj[0]["WL"]
    if full:
        rel = np.where((wlfull > 4000) & (wlfull < 14000))[0]
    elif saurononly:
        rel = np.where((wlfull > 4000) & (wlfull < 6000))[0]
    elif MLR:
        rel = np.where((wlfull > 20000) & (wlfull < 24000))[0]
    else:
        rel = np.where((wlfull > 8500) & (wlfull < 14000))[0]
    wl = vcj[0]["WL"][rel]
        
    # Some indexing for the chemical abundances
    abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3,8,7,18,17,14,13,21]
    # Getting the appropriate base model including one with a Kroupa IMF
    # Input parameter base model
    imfinterp = vcj[1][imfsdict[(x1,x2)]]
    mimf = imfinterp((Z,Age,wl))
    # Kroupa IMF base model
    baseinterp = vcj[1][imfsdict[(1.3,2.3)]]
    basemodel = baseinterp((Z,Age,wl))

    # Create empty array to hold interpolated chemical abdundance response
    #   functions
    chems = np.zeros((len(wl), vcj[0]['3.0_0.0'][1].shape[1]))
    # Load in interpolated response functions
    for k in abundi:
        chems[:,k] = vcj[2][k]((Z,Age,wl))

    if timing:
        t3 = time.time()
        print("MSPEC T2: ", t3 - t2)
        
    # Chemical abundances just for sauron spectra
    ### Old Code -- Do not follow ###
    if saurononly:
        if 'Alpha' in paramdict.keys():
            alpha_contribution = 0.0

            #Ca
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,3])),\
                    kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #C
            Cextend = (c[:,7]-c[:,0])*(0.3/0.15) + c[:,0]
            interp = spi.interp2d(wl, [0.0,0.15,0.3], 
                    np.stack((c[:,0],c[:,7],Cextend)), kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #Mg
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,15])), \
                    kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #Si
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,17])), \
                    kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            basemodel = basemodel*(1 + alpha_contribution)
        return wl, mimf, basemodel
    ###############################

    # Reminder of the abundance model columns
    # ['Solar', 'Na+', 'Na-',   'Ca+',  'Ca-', 'Fe+', 'Fe-', 'C+',  'C-',  'a/Fe+', 
    #  'N+',    'N-',  'as/Fe+','Ti+',  'Ti-', 'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 
    #  'T-',    'Cr+', 'Mn+',   'Ba+',  'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #  'V+',    'Cu+', 'Na+0.6','Na+0.9']

    # DETERMINING THE ABUNDANCE RATIO EFFECTS
    # We assume that the abundance effects scale linearly, except for sodium
    # as it covers a wider range of abundance values.
    # The models are interpolated between the given abundances to allow for 
    # continuous variables.
    # The interpolated function is then normalized by the base metallicity model
    # 
    # The chemical contribution is calculated by multiplying the individual
    # chemical contributions together. 

    #Na adjustment
    ab_contribution = np.ones(chems[:,0].shape)

    if 'Na' in paramdict.keys():
        Naextend = (chems[:,2]-chems[:,0])*(-0.5/-0.3) + chems[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.6,0.9], 
                np.stack((Naextend,chems[:,2],chems[:,0],chems[:,1],
                            chems[:,-2],chems[:,-1])), 
                kind = 'cubic')
        if fixZ:
            ab_contribution *= interp(wl,Z) / chems[:,0]
        else:
            ab_contribution *= interp(wl,Na) / chems[:,0]

    #K adjustment (assume symmetrical K adjustment)
    if 'K' in paramdict.keys():
        Kminus = (2. - (chems[:,29] / chems[:,0]))*chems[:,0]
        Kmextend = (Kminus - chems[:,0])*(-0.5/-0.3) + chems[:,0]
        Kpextend = (chems[:,29] - chems[:,0]) * (0.5/0.3) + chems[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], 
                np.stack((Kmextend,Kminus,chems[:,0],chems[:,29],Kpextend)), 
                kind = 'linear')
        if fixZ:
            ab_contribution *= interp(wl,Z) / chems[:,0]
        else:
            ab_contribution *= interp(wl,K) / chems[:,0] 

    #Fe Adjustment
    if 'Fe' in paramdict.keys():
        Femextend = (chems[:,6] - chems[:,0])*(-0.5/-0.3) + chems[:,0]
        Fepextend = (chems[:,5] - chems[:,0])*(0.5/0.3) + chems[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], 
                np.stack((Femextend,chems[:,6], chems[:,0],chems[:,5],Fepextend)), 
                kind = 'linear')
        #FeP = interp(wl,Fe) / c[:,0] - 1.
        #ab_contribution += FeP
        if fixZ:
            ab_contribution *= interp(wl,Z) / chems[:,0]
        else:
            ab_contribution *= interp(wl,Fe) / chems[:,0]

    #Ca Adjustment
    if 'Ca' in paramdict.keys():
        Cmextend = (chems[:,4] - chems[:,0])*(-0.5/-0.3) + chems[:,0]
        Cpextend = (chems[:,3] - chems[:,0])*(0.5/0.3) + chems[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], 
                np.stack((Cmextend,chems[:,4], chems[:,0],chems[:,3],Cpextend)), 
                kind = 'linear')
        if fixZ:
            ab_contribution *= interp(wl,Z) / chems[:,0]
        else:
            ab_contribution *= interp(wl,Ca) / chems[:,0]

    # Special alpha parameter that combines positive abundances in several
    # included alpha elements
    if 'Alpha' in paramdict.keys():
        alpha_contribution = np.ones(chems[:,0].shape)

        #Ca
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((chems[:,0],chems[:,3])), 
                kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / chems[:,0] 

        #C
        Cextend = (chems[:,7]-chems[:,0])*(0.3/0.15) + chems[:,0]
        interp = spi.interp2d(wl, [0.0,0.15,0.3], 
                np.stack((chems[:,0],chems[:,7],Cextend)), kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / chems[:,0]

        #Mg
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((chems[:,0],chems[:,15])), 
                kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / chems[:,0]

        #Si
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((chems[:,0],chems[:,17])), 
                kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / chems[:,0]

        ab_contribution *= alpha_contribution

    ### Depreciated ###
    #model_ratio = mimf / basemodel
    #model_ratio = basemodel / mimf
    #model_ratio = 1.0
    ####
     
    # The model formula is as follows....
    # The model is = the base IMF model * abundance effects. 
    # The abundance effect percentages are scaled by the ratio of the selected IMF 
    #   model to the Kroupa IMF model
    # The formula ensures that if the abundances are the 0 (i.e. base)
    # then the base IMF  model is recovered. 

    #newm = mimf*(1. + model_ratio*ab_contribution) ## Depreciated ##
    #newm = mimf*(1. + ab_contribution) ## Depreciated ##
    newm = mimf*ab_contribution

    if timing:
        print("MSPEC T3: ", time.time() - t3)

    return wl, newm, basemodel

def calc_chisq(params, wl, data, err, veldisp, paramnames, paramdict, 
        lineinclude, linedefsfull, sauron, saurononly, plot=False, 
        timing = False):
    '''
    Main function that produces the chisquared value that drives the mcmc
    simulation, i.e. the likelihood of the mcmc algorithm.
    Takes the input parameter values, creates a model, broadens the model,
    then compares the data and model spectrum. Returns the chisq value.

    Inputs:
        params: Input parameter values
        wl: wavelength array for the observed spectrum
        data: observed spectrum
        err: uncertainty array for the observed spectrum
        veldisp: velocity dispersion of the observed spectrum (if not a
                    fitting parameter)
        paramnames: The fitting parameters
        paramdict: Dictionary of the parameters including fixed inputs
        lineinclude: spectral features used in the fitting
        linedefsfull: line bandpass definitions
        sauron: Array of sauron information (if used) including wavelength
                    array, spectrum, uncertainty, and included lines
        saurononly: Flag for only sauron fitting
        plot (debug): Plot spectra for debugging
        timing (debug): Timing for debugging and optimizing
    '''

    # Unpack line defintiions
    linedefs = linedefsfull[0]
    line_names = linedefsfull[1]
    index_names = linedefsfull[2]

    if timing:
        t1 = time.time()

    # Creating model spectrum based on the input parameters
    if sauron:
        if saurononly:
            wlm, newm, base = model_spec(params, paramnames, paramdict, 
                    saurononly = True, timing=timing)
        else:
            wlm, newm, base = model_spec(params, paramnames, paramdict, 
                    full = True, timing=timing)
    else:
        wlm, newm, base = model_spec(params, paramnames, paramdict, 
                timing=timing)

    # Convolve model to observed velocity dispersion
    # Depends on whether sauron spectra is included and whether
    # velocity dispersion is a parameter
    if not saurononly:
        if 'VelDisp' in paramdict.keys():
            if 'VelDisp' in paramnames:
                whsig = np.where(np.array(paramnames) == 'VelDisp')[0]
                wlc, mconv = mcspec.convolvemodels(wlm, newm, params[whsig])
            else:
                wlc, mconv = mcspec.convolvemodels(wlm, newm, 
                        paramdict['VelDisp'])
        else:
            wlc, mconv = mcspec.convolvemodels(wlm, newm, veldisp)

    # Do the same for the sauron spectra (differing velocity dispersions)
    if sauron:
        if saurononly:
            if 'VelDisp' in paramdict.keys():
                if 'VelDisp' in paramnames:
                    whsig = np.where(np.array(paramnames) == 'VelDisp')[0]
                    wlc_s, mconv_s = mcspec.convolvemodels(wlm, base, 
                            params[whsig], reglims=[4000,6000])
                else:
                    wlc_s, mconv_s = mcspec.convolvemodels(wlm, base, 
                            paramdict['VelDisp'], reglims=[4000,6000])
            else:
                wlc_s, mconv_s = mcspec.convolvemodels(wlm, base, sauron[4], 
                        reglims=[4000,6000])
        else:
            wlc_s, mconv_s = mcspec.convolvemodels(wlm, base, sauron[4],
                    reglims=[4000,6000])

    # Handling the 'f' error adjustment parameter
    if 'f' in paramdict.keys():
        if 'f' in paramnames:
            whf = np.where(np.array(paramnames) == 'f')[0][0]
            f = params[whf]
        else:
            f = paramdict['f']
    
    if timing:
        t2 = time.time()
        print("CHISQ T1: ", t2 - t1)

    # Interpolating the broadened model spectrum and placing it on the
    # observed (and sauron) wavegrid.
    if not saurononly:
        mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', 
                bounds_error=False)
    if sauron:
        mconvinterp_s = spi.interp1d(wlc_s, mconv_s, kind='cubic', 
                bounds_error=False)

    if timing:
        t3 = time.time()
        print("CHISQ T2: ", t3 - t2)
    
    # Measuring the chisq by comparing the model and observed spectra
    chisq = 0

    if not saurononly:
        for i in range(len(linedefs[0,:])):
            if line_names[i] not in lineinclude:
                #print line_name[i]
                continue

            #Getting a slice of the model
            wli = wl[i]

            modelslice = mconvinterp(wli)

            #Removing a high-order polynomial from the slice
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= linedefs[0,i]) & \
                    (wlc <= linedefs[1,i]))[0]
            redpass = np.where((wlc >= linedefs[4,i]) & \
                    (wlc <= linedefs[5,i]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([linedefs[0,i], linedefs[1,i]])
            redavg = np.mean([linedefs[4,i], linedefs[5,i]])

            blueval = np.mean(mconv[bluepass])
            redval = np.mean(mconv[redpass])

            pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
            polyfit = np.poly1d(pf) 
            cont = polyfit(wli)

            #Normalizing the model
            modelslice = modelslice / cont

            #if 'f' in paramdict.keys():
            #    errterm = (err[i] ** 2.0) + (data[i]**2.0 * np.exp(2*f))
            #    addterm = np.log(2.0 * np.pi * errterm)
            #else:
            #    errterm = err[i] ** 2.0
            #    addterm = err[i] * 0.0
            errterm = err[i] ** 2.0
            addterm = err[i] * 0.0

            #Performing the chisq calculation
            chisq += np.nansum(((data[i] - modelslice)**2.0 / errterm) +\
                    addterm)

            if plot:
                mpl.plot(wl[i], modelslice, 'r')
                mpl.plot(wl[i], data[i], 'b')

    if sauron:
        for i in range(len(sauron[0])):

            #Getting a slice of the model
            wli = sauron[0][i]

            modelslice = mconvinterp_s(wli)

            linedefs_s = [sauron[3][0][0,:], sauron[3][0][1,:],\
                    sauron[3][0][4,:], sauron[3][0][5,:]]

            polyfit_model = mcspec.removeLineSlope(wlc_s, mconv_s, linedefs_s, i)
            cont = polyfit_model(wli)

            #Normalizing the model
            modelslice = modelslice / cont

            if 'f' in paramdict.keys():
                errterm = (sauron[2][i] ** 2.0) + modelslice**2.0 * \
                        np.exp(2*np.log(f))
                addterm = np.log(2.0 * np.pi * errterm)
            else:
                errterm = sauron[2][i] ** 2.0
                addterm = sauron[2][i] * 0.0

            #Performing the chisq calculation
            chisq += np.nansum(((sauron[1][i] - modelslice)**2.0 / errterm) +\
                    addterm)

    if plot:
        mpl.show()

    if timing:
        t4 = time.time()
        print("CHISQ T3: ", t4 - t3)

    # Returns chisq
    return -0.5*chisq

def lnprior(theta, paramnames):
    '''
    Ensuring the inputs fit the priors for the mcmc. 
    Returns 0.0 if the inputs are clean and -inf if otherwise.
    
    Inputs:
        theta: parameter values
        paramnames: names of the input parameters
    '''

    goodpriors = True
    for j in range(len(paramnames)):
        if paramnames[j] == 'Age':
            if not (1.0 <= theta[j] <= 13.5):
                goodpriors = False
        elif paramnames[j] == 'Z':
            #if not (-1.5 <= theta[j] <= 0.2):
            #    goodpriors = False
            if not (-0.25 <= theta[j] <= 0.2):
                goodpriors = False
        elif paramnames[j] == 'Alpha':
            if not (0.0 <= theta[j] <= 0.3):
                goodpriors = False
        elif paramnames[j] in ['x1', 'x2']:
            if not (0.5 <= theta[j] <= 3.5):
                goodpriors = False
        elif paramnames[j] == 'Na':
            if not (-0.5 <= theta[j] <= 0.9):
                goodpriors = False
        elif paramnames[j] in ['K','Ca','Fe','Mg']:
            if not (-0.5 <= theta[j] <= 0.5):
                goodpriors = False
        elif paramnames[j] == 'VelDisp':
            if not (120 <= theta[j] <= 390):
                goodpriors = False
        elif paramnames[j] == 'Vel':
            if not (0.0001 <= theta[j] <= 0.03):
                goodpriors = False
        elif paramnames[j] == 'f':
            if not (-10. <= np.log(theta[j]) <= 1.):
                goodpriors = False

    if goodpriors == True:
        return 0.0
    else:
        return -np.inf

def lnprob(theta, wl, data, err, paramnames, paramdict, lineinclude, linedefs,
        veldisp, sauron, saurononly):
    '''
    Primary function of the mcmc. 
    Checks priors and returns the likelihood. Essentially a wrapper for 
    calc_chisq above.
    
    Inputs:
        theta: Input parameter values
        wl: wavelength array for the observed spectrum
        data: observed spectrum
        err: uncertainty array for the observed spectrum
        veldisp: velocity dispersion of the observed spectrum (if not a
                    fitting parameter)
        paramnames: The fitting parameters
        paramdict: Dictionary of the parameters including fixed inputs
        lineinclude: spectral features used in the fitting
        linedefs: line bandpass definitions
        sauron: Array of sauron information (if used) including wavelength
                    array, spectrum, uncertainty, and included lines
        saurononly: Flag for only sauron fitting
    '''
    
    # Check priors
    lp = lnprior(theta, paramnames)
    if not np.isfinite(lp):
        return -np.inf

    # Calculate chisq
    chisqv = calc_chisq(theta, wl, data, err, veldisp, 
            paramnames, paramdict, lineinclude, linedefs, 
            sauron, saurononly, timing=False)

    # Return likelihood
    return lp + chisqv

def do_mcmc(gal, nwalkers, n_iter, z, veldisp, paramdict, lineinclude,
        threads = 6, restart=False, scale=False, fl=None, sauron=None, 
        sauron_z=None, sauron_veldisp=None, saurononly=False,
        comments='No Comment', logger=None):
    '''Main program. Extracts the observed spectral data and features and
    sets up the mcmc call. Handles logging and other outputs.
    
    Inputs:
        gal: Name of target galaxy
        nwalkers: Number of mcmc walkers
        n_iter: Number of iterations of the mcmc
        z: redshift of galaxy (currently not a fitting parameter
        veldisp: velocity dispersion of the galaxy (can be a fitting 
                    parameter)
        paramdict: dictionary of input parameters that can include
                    fixed values for particular parameters
        lineinclude: spectral features included in this analysis
        threads: number of cpu threads to employ
        restart: clean output file that did not finish (POTENTIALLY BROKEN)
        scale: scale the spectrum (basic adjustment DEPRECIATED)
        fl: observed spectrum filepath
        sauron: sauron spectrum filepath (if included)
        sauron_z: sauron spectrum redshift
        sauron_veldisp: sauron spectrum velocity dispersion
        saurononly: only running on sauron spectrum
        comments: Comment to include in output file
        logger: python logger instance
    '''

    # handing inputs
    if fl == None:
        print('Please input filename for WIFIS data')
        return

    if sauron.lower() == 'none':
        sauron = None

    #Handle parameters and walker initialization
    paramnames = []
    for key in paramdict.keys():
        if paramdict[key] == None:
            paramnames.append(key)
    if saurononly:
        for param in paramnames:
            if param in ['Ca','Mg','Fe','x1','x2','K']:
                print("Please remove elemental abundances")
                return

    #Line definitions & other definitions
    #WIFIS Defs
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, \
                    12240, 11905]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, \
                    12260, 11935]
    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, \
                    12309, 11935]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, \
                    12333, 11965]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, \
                    12360, 12005]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, \
                    12390, 12025]

    #mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    #mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]
    mlow, mhigh = [],[]
    for i in zip(bluelow, redhigh):
        mlow.append(i[0])
        mhigh.append(i[1])
    morder = [1,1,1,1,1,1,1,1,1,1]

    # Line name definitions
    line_name = np.array(['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB',\
            'NaI127', 'NaI123','CaII119'])

    # Line bandpass summary array
    linedefs = [np.array([bluelow, bluehigh, linelow, linehigh, redlow,\
            redhigh, mlow, mhigh, morder]), line_name, line_name]

    # Load wifis data
    wl, data, err, datanomask, mask = preps.preparespecwifis(fl, z)

    # Split the spectrum into individual features including wavelength
    #   and uncertainty arrays
    if scale:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, 
                scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err)

    # Handling sauron data (Currently only HBeta, but can be extended to 
    #   other features. 
    if sauron != None:
        print("Using SAURON data")
        #SAURON Lines
        bluelow_s =  [4827.875]#, 4946.500, 5142.625]
        bluehigh_s = [4847.875]#, 4977.750, 5161.375]
        linelow_s =  [4847.875]#, 4977.750, 5160.125]
        linehigh_s = [4876.625]#, 5054.000, 5192.625]
        redlow_s =   [4876.625]#, 5054.000, 5191.375]
        redhigh_s =  [4891.625]#, 5065.250, 5206.375]

        mlow_s, mhigh_s = [],[]
        for i in zip(bluelow_s, redhigh_s):
            mlow_s.append(i[0])
            mhigh_s.append(i[1])
        morder_s = [1]#,1,1]
        line_names_s = np.array(['HBeta'])#,'Fe5015','MgB'])

        sauronlines = [np.array([bluelow_s,bluehigh_s,linelow_s,linehigh_s,\
                redlow_s,redhigh_s,mlow_s,mhigh_s,morder_s]),line_names_s,\
                line_names_s]

        ff = fits.open(sauron)
        spec_s = ff[0].data
        wl_s = ff[1].data
        if 'Vel' not in paramdict.keys():
            wl_s = wl_s / (1. + sauron_z)
        noise_s = ff[2].data

        wl_s, data_s, err_s = preps.splitspec(wl_s, spec_s, sauronlines, \
                err = noise_s)
    else:
        print("No SAURON")

    ndim = len(paramnames)

    print(paramnames)
    print(lineinclude)

    # Initializing the mcmc walker starting positions
    # Random valid values of every parameter for every walker
    pos = []
    if not restart:
        for i in range(nwalkers):
            newinit = []
            for j in range(len(paramnames)):
                if paramnames[j] == 'Age':
                    newinit.append(np.random.random()*12.5 + 1.0)
                elif paramnames[j] == 'Z':
                    newinit.append(np.random.random()*0.45 - 0.25)
                    #newinit.append(np.random.random()*0.1 + 0.1)
                elif paramnames[j] == 'Alpha':
                    newinit.append(np.random.random()*0.3)
                elif paramnames[j] in ['x1', 'x2']:
                    newinit.append(np.random.choice(x1_m))
                elif paramnames[j] == 'Na':
                    newinit.append(np.random.random()*1.3 - 0.3)
                elif paramnames[j] in ['K','Ca','Fe','Mg']:
                    newinit.append(np.random.random()*0.6 - 0.3)
                elif paramnames[j] == 'VelDisp':
                    newinit.append(np.random.random()*240 + 120)
                elif paramnames[j] == 'Vel':
                    newinit.append(np.random.random()*0.015 + 0.002)
                elif paramnames[j] == 'f':
                    #newinit.append(np.random.random()*11 - 10.)
                    newinit.append(np.random.random())
            pos.append(np.array(newinit))
    else:
       realdata, postprob, infol, lastdata = mcsupp.load_mcmc_file(restart)
       pos = lastdata

    # Handle output file and print important header information
    savefl_end = time.strftime("%Y%m%dT%H%M%S")+\
            "_%s_fullindex.dat" % (gal)
    savefl = base + "mcmcresults/"+ savefl_end
    f = open(savefl, "w")
    strparams = '\t'.join(paramnames)
    strparamdict = '    '.join(['%s: %s' % (key, paramdict[key]) \
            for key in paramdict.keys()])
    if sauron:
        strlines = '\t'.join(lineinclude+list(sauronlines[1]))
    else:
        strlines = '\t'.join(lineinclude)
    f.write("#NWalk\tNStep\tGal\tFit\n")
    f.write("#%d\t%d\t%s\tFullIndex\n" % (nwalkers, n_iter,gal))
    f.write("#%s\n" % (strparams))
    f.write("#%s\n" % (strlines))
    f.write("#%s\n" % (strparamdict))
    f.write('#'+comments+'\n')
    f.close()

    # Handle logging 
    if "VelDisp" in paramdict.keys():
        veldisp_str = 'Fit Parameter'
    else:
        veldisp_str = str(veldisp)

    logger.info(savefl_end)
    logger.info("NWalk: %i, NStep: %i, Gal: %s, Fit: FullIndex" % (nwalkers,
                    n_iter, gal))
    logger.info("Galaxy File: " + fl)
    logger.info("Galaxy z: " + str(z))
    logger.info("Galaxy sigma: "+veldisp_str)
    logger.info("Sauron File: " + str(sauron))
    logger.info("Sauron z: " + str(sauron_z))
    logger.info("Sauron sigma: " + str(sauron_veldisp))
    logger.info("Lines: " + strlines)
    logger.info("Params: " + strparams)
    logger.info("ParamDict: " + strparamdict)
    logger.info("Sauron Only: " + str(saurononly))
    logger.info("Comments: " + comments)

    # Initialize MCMC sampler
    pool = Pool(processes=16)

    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
    #        (wl, data, err, gal, paramnames, lineinclude, linedefs), \
    #        threads=threads)
    if not sauron:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
                (wl, data, err, paramnames, paramdict, lineinclude, linedefs, \
                veldisp, False, saurononly), pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
                (wl, data, err, paramnames, paramdict, lineinclude, linedefs, 
                veldisp, [wl_s, data_s, err_s, sauronlines, \
                sauron_veldisp], saurononly), pool=pool)
    print("Starting MCMC...")

    # Run the MCMC and output the step data
    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        #position = result[0]
        position = result.coords
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            #f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result[1][k]))
            f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), 
                result.log_prob[k]))
        f.close()

        # Print status every 100 steps
        if (i+1) % 100 == 0:
            ct = time.time() - t1
            pfinished = (i+1.)*100. / float(n_iter)
            print(ct / 60., " Minutes")
            print(pfinished, "% Finished")
            print(((ct / (pfinished/100.)) - ct) / 60., "Minutes left")
            print(((ct / (pfinished/100.)) - ct) / 3600., "Hours left")
            print()

    # Finish logging to confirm mcmc completion
    logger.info("Runtime: " + str(ct / 60.) + " Minutes")
    logger.info("")

    return sampler

if __name__ == '__main__':
    # Preload the model files
    vcj = preload_vcj(sauron=True, saurononly=False) 

    # Load the inputs for each MCMC run
    #inputfl = 'inputs/20210326_PaperPaBTest.txt'
    #inputfl = 'inputs/20210324_Paper.txt'
    #inputfl = 'inputs/20210613_Paper.txt'
    #inputfl = 'inputs/20210614_OtherIMFPaper.txt'
    inputfl = 'inputs/20220210_revisedpaper_alpha.txt'
    mcmcinputs = mcsupp.load_mcmc_inputs(inputfl)

    # Set up logging
    logging.basicConfig(filename="logs/mcmc_runinfo.log",format='%(message)s',
            filemode='a')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    #Run the MCMC for each set of inputs
    for i in range(len(mcmcinputs)):
        try:
            if mcmcinputs[i]['skip'] == 1:
                print("Skipping: ",i)
                continue
        except:
            print("No skip parameter, not skipping")
        print(mcmcinputs[i])
        sampler = do_mcmc(mcmcinputs[i]['target'], mcmcinputs[i]['workers'],
            mcmcinputs[i]['steps'],mcmcinputs[i]['targetz'],
            mcmcinputs[i]['targetsigma'],mcmcinputs[i]['paramdict'],
            mcmcinputs[i]['lineinclude'],threads = 16, 
            fl = mcmcinputs[i]['fl'],sauron=mcmcinputs[i]['sfl'],
            sauron_z = mcmcinputs[i]['sz'],
            sauron_veldisp=mcmcinputs[i]['ssigma'],
            saurononly=mcmcinputs[i]['saurononly'],
            comments = mcmcinputs[i]['comments'], logger=logger) 
