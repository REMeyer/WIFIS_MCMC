####################
#   MCMC program using Prof Charlie Conroy's SSP Models
#   Based off of emcee 
#   Author: Elliot Meyer, Dept Astronomy & Astrophysics University of Toronto
###################

from __future__ import print_function

import numpy as np
from astropy.io import fits
from sys import exit
from glob import glob
import pandas as pd
import emcee
import time
import matplotlib.pyplot as mpl
import nifsmethods as nm
import scipy.interpolate as spi
import warnings
import sys, os
import mcmc_support as mcsp
import prepare_spectra as preps
import plot_corner as plcr
from random import uniform
from multiprocessing import Pool


warnings.simplefilter('ignore', np.RankWarning)

# DEFINES THE BASE PATH -- NEEDS UPDATING FOR ALL SYSTEMS
base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

# MCMC Parameters
# Metallicity: -1.5 < [Z/H] < 0.2 steps of 0.1?
# Age: depends on galaxy, steps of 1 Gyr?
# IMF: x1 and x2 full range
# [Na/H]: -0.4 <-> +1.3

lineexclude = False
ratiofit = False

#Setting some of the mcmc priors
Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
x1_m = 0.5 + np.arange(16)/5.0
x2_m = 0.5 + np.arange(16)/5.0

Z_pm = np.array(['m','m','m','m','p','p','p'])
ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

#Dictionary to help easily access the IMF index
imfsdict = {}
for i in range(16):
    for j in range(16):
        imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

vcj = {}

def preload_vcj(overwrite_base = False, sauron=False, saurononly=False, MLR=False):
    '''Loads the SSP models into memory so the mcmc model creation takes a
    shorter time. Returns a dict with the filenames as the keys'''
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
            'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+','V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    print("PRELOADING SSP MODELS INTO MEMORY")
    fls = glob(base+'spec/vcj_ssp/*')    

    for fl in fls:
        #print fl
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

    print("PRELOADING ABUNDANCE MODELS INTO MEMORY")
    fls = glob(base+'spec/atlas/*')    
    for fl in fls:
        #print fl
        flspl = fl.split('/')[-1]

        mnamespl = flspl.split('_')
        age = float(mnamespl[2][1:])
        Zsign = mnamespl[3][1]
        Zval = float(mnamespl[3][2:5])
        if Zsign == "m":
            Zval = -1.0 * Zval
        if age == 13.0:
            age = 13.5

        x = pd.read_csv(fl, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj["%.1f_%.1f" % (age, Zval)].append(x[:,1:])

    vcj["WL"] = x[:,0]
    print("FINISHED LOADING MODELS")

    print("Calculating IMF interpolators")
    wlfull = vcj["WL"]
    if sauron:
        rel = np.where((wlfull > 4000) & (wlfull < 14000))[0]
    elif saurononly:
        rel = np.where((wlfull > 4000) & (wlfull < 6000))[0]
    elif MLR:
        rel = np.where((wlfull > 20000) & (wlfull < 24000))[0]
    else:
        rel = np.where((wlfull > 8500) & (wlfull < 14000))[0]
    wl = wlfull[rel]

    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])

    imf_interp = []
    for k in range(vcj['3.0_0.0'][0].shape[1]):
        out = np.meshgrid(fullage,fullZ,wl)
        grid = (out[0],out[1],out[2])
        newgrid = np.zeros(out[0].shape)
        for i,age in enumerate(fullage):
            for j,z in enumerate(fullZ):
                #print(vcj["%.1f_%.1f" % (age, z)][0][rel,73].shape)
                newgrid[j,i,:] = vcj["%.1f_%.1f" % (age, z)][0][rel,k]
        fulli = spi.RegularGridInterpolator((fullZ,fullage,wl), newgrid)
        imf_interp.append(fulli)
    
    print("Calculating elemental interpolators")
    ele_interp = []
    for k in range(vcj['3.0_0.0'][1].shape[1]):
        out = np.meshgrid(fullage,fullZ,wl)
        grid = (out[0],out[1],out[2])
        newgrid = np.zeros(out[0].shape)
        for i,age in enumerate(fullage):
            for j,z in enumerate(fullZ):
                newgrid[j,i,:] = vcj["%.1f_%.1f" % (age, z)][1][rel,k]
        fulli = spi.RegularGridInterpolator((fullZ,fullage,wl), newgrid)
        ele_interp.append(fulli)
    
    return vcj, imf_interp, ele_interp

def select_model_file(Z, Age):
    '''Selects the model file for a given Age and [Z/H]. If the requested values
    are between two models it returns two filenames for each model set.'''

    #Acceptable parameters...also global variables but restated here...
    Z_m = np.array([-1.5,-1.25, -1.0, -0.75, -0.5, -0.25, 0.0, 0.1, 0.2])
    Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
    x1_m = 0.5 + np.arange(16)/5.0
    x2_m = 0.5 + np.arange(16)/5.0
    Z_pm = np.array(['m','m','m','m','m','m','p','p','p'])
    ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])
    
    #Matching parameters to the nearest acceptable value (for Age, Z, x1, and x2)
    #if Z not in Z_m:
    #    Zmin = np.argmin(np.abs(Z_m - Z))
    #    Z = Z_m[Zmin]
    #if Age not in Age_m:
    #    Agemin = np.argmin(np.abs(Age_m - Age))
    #    Age = Age_m[Agemin]

    mixage = False
    mixZ = False
    agep = 0
    agem = 0
    zp = 0
    zm = 0
    if Age not in fullage:
        mixage = True
        ageminsort = np.argsort(np.abs(fullage - Age))
        if ageminsort[0] > ageminsort[1]:
            agep = ageminsort[0]
            agem = ageminsort[1]
        else:
            agep = ageminsort[1]
            agem = ageminsort[0]
    if Z not in fullZ:
        mixZ = True
        zminsort = np.argsort(np.abs(fullZ - Z))
        if zminsort[0] > zminsort[1]:
            zp = zminsort[0]
            zm = zminsort[1]
        else:
            zp = zminsort[1]
            zm = zminsort[0]

    #whAge = np.where(Age_m == Age)[0][0]
    #whZ = np.where(Z_m == Z)[0][0]        

    if mixage and mixZ:
        fl1 = "%.1f_%.1f" % (fullage[agem],fullZ[zm])
        fl2 = "%.1f_%.1f" % (fullage[agep],fullZ[zm])
        fl3 = "%.1f_%.1f" % (fullage[agem],fullZ[zp])
        fl4 = "%.1f_%.1f" % (fullage[agep],fullZ[zp])
    elif mixage:
        fl1 = "%.1f_%.1f" % (fullage[agem],Z)
        fl2 = "%.1f_%.1f" % (fullage[agep],Z)
        fl3 = ''
        fl4 = ''
    elif mixZ:
        fl1 = "%.1f_%.1f" % (Age,fullZ[zm])
        fl2 = "%.1f_%.1f" % (Age,fullZ[zp])
        fl3 = ''
        fl4 = ''
    else:
        fl1 = "%.1f_%.1f" % (Age,Z)
        fl2 = ''
        fl3 = ''
        fl4 = ''

    return fl1, fl2, fl3, fl4, agem, agep, zm, zp, mixage, mixZ

def model_spec(inputs, paramnames, paramdict, saurononly = False, vcjset = False, timing = False, \
        full = False, MLR=False, fixZ = False):
    '''Core function which takes the input model parameters, finds the appropriate models,
    and adjusts them for the input abundance ratios. Returns a broadened model spectrum 
    to be matched with a data spectrum.'''

    global vcj

    #fixZ = True

    if vcjset:
        vcj = vcjset
    #else:
    #    global vcj

    if timing:
        print("Starting Model Spec Time")
        t1 = time.time()

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

    #if 'Age' not in paramnames:
    #    Age = 3.82
    #    if gal == 'M85':
    #        Age = 5.0
    #    elif gal == 'M87':
    #        Age = 13.5
    #if 'Z' not in paramnames:
    #    Z = 0.0
    if 'x1' not in paramnames:
        x1 = 1.3
    if 'x2' not in paramnames:
        #x2 = 2.3
        x2 = x1

    if x1 not in x1_m:
        x1min = np.argmin(np.abs(x1_m - x1))
        x1 = x1_m[x1min]
    if x2 not in x2_m:
        x2min = np.argmin(np.abs(x2_m - x2))
        x2 = x2_m[x2min]

    #Finding the appropriate base model files.
    #fl1, fl2, fl3, fl4, agem, agep, zm, zp, mixage, mixZ = select_model_file(Z, Age)

    if timing:
        t2 = time.time()
        print("MSPEC T1: ",t2 - t1)
    
    wlfull = vcj[0]["WL"]
    #Finding the relevant section of the models to reduce the computational complexity
    if full:
        rel = np.where((wlfull > 4000) & (wlfull < 14000))[0]
    elif saurononly:
        rel = np.where((wlfull > 4000) & (wlfull < 6000))[0]
    elif MLR:
        rel = np.where((wlfull > 20000) & (wlfull < 24000))[0]
    else:
        rel = np.where((wlfull > 8500) & (wlfull < 14000))[0]
    wl = vcj[0]["WL"][rel]
        
    # If the Age of Z is inbetween models then this will average the respective models to produce 
    # one that is closer to what is expected.
    # NOTE THAT THIS IS AN ASSUMPTION AND THE CHANGE IN THE MODELS IS NOT NECESSARILY LINEAR
    #fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    #fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])
    #abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3]

    abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3,8,7,18,17,14,13,21]
    imfinterp = vcj[1][imfsdict[(x1,x2)]]
    mimf = imfinterp((Z,Age,wl))
    baseinterp = vcj[1][imfsdict[(1.3,2.3)]]
    basemodel = baseinterp((Z,Age,wl))

    c = np.zeros((len(wl), vcj[0]['3.0_0.0'][1].shape[1]))
    for k in abundi:
        c[:,k] = vcj[2][k]((Z,Age,wl))

    if timing:
        t3 = time.time()
        print("MSPEC T2: ", t3 - t2)

        
    # If the Age of Z is inbetween models then this will average the respective models to produce 
    # one that is closer to what is expected.
    # NOTE THAT THIS IS AN ASSUMPTION AND THE CHANGE IN THE MODELS IS NOT NECESSARILY LINEAR

    if saurononly:
        if 'Alpha' in paramdict.keys():
            alpha_contribution = 0.0

            #Ca
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,3])), kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #C
            Cextend = (c[:,7]-c[:,0])*(0.3/0.15) + c[:,0]
            interp = spi.interp2d(wl, [0.0,0.15,0.3], np.stack((c[:,0],c[:,7],Cextend)), kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #Mg
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,15])), kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            #Si
            interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,17])), kind = 'linear')
            alpha_contribution += interp(wl, alpha) / c[:,0] - 1.

            basemodel = basemodel*(1 + alpha_contribution)
        return wl, mimf, basemodel

    # Reminder of the abundance model columns
    # ['Solar', 'Na+', 'Na-',   'Ca+',  'Ca-', 'Fe+', 'Fe-', 'C+',  'C-',  'a/Fe+', 
    #  'N+',    'N-',  'as/Fe+','Ti+',  'Ti-', 'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 
    #  'T-',    'Cr+', 'Mn+',   'Ba+',  'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #  'V+',    'Cu+', 'Na+0.6','Na+0.9']

    # DETERMINING THE ABUNDANCE RATIO EFFECTS
    # The assumption here is that the abundance effects scale linearly...for all except sodium which covers a much wider
    # range of acceptable values.
    # The models are interpolated at the various given abundances to allow for abundance values to be continuous.
    # The interpolated value is then normalized by the solar metallicity model and then subtracted by 1 to 
    # retain the percentage
    #if fitmode in [True, False, 'NoAge', 'NoAgeVelDisp']:

        #Na adjustment
    ab_contribution = np.ones(c[:,0].shape)
    if 'Na' in paramdict.keys():
        Naextend = (c[:,2]-c[:,0])*(-0.5/-0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.6,0.9], np.stack((Naextend,c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        #NaP = interp(wl,Na) / c[:,0] - 1.
        #ab_contribution += NaP
        if fixZ:
            ab_contribution *= interp(wl,Z) / c[:,0]
        else:
            ab_contribution *= interp(wl,Na) / c[:,0]

        #K adjustment (assume symmetrical K adjustment)
    if 'K' in paramdict.keys():
        Kminus = (2. - (c[:,29] / c[:,0]))*c[:,0]
        Kmextend = (Kminus - c[:,0])*(-0.5/-0.3) + c[:,0]
        Kpextend = (c[:,29] - c[:,0]) * (0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Kmextend,Kminus,c[:,0],c[:,29],Kpextend)), kind = 'linear')
        #KP = interp(wl,K) / c[:,0] - 1.
        #ab_contribution += KP
        if fixZ:
            ab_contribution *= interp(wl,Z) / c[:,0]
        else:
            ab_contribution *= interp(wl,K) / c[:,0] 

        #Fe Adjustment
    if 'Fe' in paramdict.keys():
        Femextend = (c[:,6] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Fepextend = (c[:,5] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Femextend,c[:,6], c[:,0],c[:,5],Fepextend)), kind = 'linear')
        #FeP = interp(wl,Fe) / c[:,0] - 1.
        #ab_contribution += FeP
        if fixZ:
            ab_contribution *= interp(wl,Z) / c[:,0]
        else:
            ab_contribution *= interp(wl,Fe) / c[:,0]

        #Ca Adjustment
    if 'Ca' in paramdict.keys():
        Cmextend = (c[:,4] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Cpextend = (c[:,3] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Cmextend,c[:,4], c[:,0],c[:,3],Cpextend)), kind = 'linear')
        #CaP = interp(wl,Ca) / c[:,0] - 1.
        #ab_contribution += CaP
        if fixZ:
            ab_contribution *= interp(wl,Z) / c[:,0]
        else:
            ab_contribution *= interp(wl,Ca) / c[:,0]
        
    if 'Alpha' in paramdict.keys():
        alpha_contribution = np.ones(c[:,0].shape)

        #Ca
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,3])), kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / c[:,0] 

        #C
        Cextend = (c[:,7]-c[:,0])*(0.3/0.15) + c[:,0]
        interp = spi.interp2d(wl, [0.0,0.15,0.3], np.stack((c[:,0],c[:,7],Cextend)), kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / c[:,0]

        #Mg
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,15])), kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / c[:,0]

        #Si
        interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,17])), kind = 'linear')
        alpha_contribution *= interp(wl, alpha) / c[:,0]

        ab_contribution *= alpha_contribution


    #model_ratio = mimf / basemodel
    #model_ratio = basemodel / mimf
    #model_ratio = 1.0

    # The model formula is as follows....
    # The new model is = the base IMF model * abundance effects. 
    # The abundance effect %ages are scaled by the ratio of the selected IMF model to the Kroupa IMF model
    # The formula ensures that if the abundances are solar then the base IMF model is recovered. 
    #newm = mimf*(1. + model_ratio*ab_contribution)
    #newm = mimf*(1. + ab_contribution)
    newm = mimf*ab_contribution

    if timing:
        print("MSPEC T3: ", time.time() - t3)

    return wl, newm, basemodel

def calc_chisq(params, wl, data, err, veldisp, paramnames, paramdict, lineinclude, \
        linedefsfull, sauron, saurononly, plot=False, timing = False):
    ''' Important function that produces the value that essentially
    represents the likelihood of the mcmc equation. Produces the model
    spectrum then returns a normal chisq value.'''

    linedefs = linedefsfull[0]
    line_names = linedefsfull[1]
    index_names = linedefsfull[2]

    if timing:
        t1 = time.time()

    #Creating model spectrum then interpolating it so that it can be easily matched with the data.
    if sauron:
        if saurononly:
            wlm, newm, base = model_spec(params, paramnames, paramdict, saurononly = True, timing=timing)
        else:
            wlm, newm, base = model_spec(params, paramnames, paramdict, full = True, timing=timing)
    else:
        wlm, newm, base = model_spec(params, paramnames, paramdict, timing=timing)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    if not saurononly:
        if 'VelDisp' in paramdict.keys():
            if 'VelDisp' in paramnames:
                whsig = np.where(np.array(paramnames) == 'VelDisp')[0]
                wlc, mconv = mcsp.convolvemodels(wlm, newm, params[whsig])
            else:
                wlc, mconv = mcsp.convolvemodels(wlm, newm, paramdict['VelDisp'])
        else:
            wlc, mconv = mcsp.convolvemodels(wlm, newm, veldisp)

    if sauron:
        if saurononly:
            if 'VelDisp' in paramdict.keys():
                if 'VelDisp' in paramnames:
                    whsig = np.where(np.array(paramnames) == 'VelDisp')[0]
                    wlc_s, mconv_s = mcsp.convolvemodels(wlm, base, params[whsig], reglims=[4000,6000])
                else:
                    wlc_s, mconv_s = mcsp.convolvemodels(wlm, base, paramdict['VelDisp'], reglims=[4000,6000])
            else:
                wlc_s, mconv_s = mcsp.convolvemodels(wlm, base, sauron[4], reglims=[4000,6000])
        else:
            wlc_s, mconv_s = mcsp.convolvemodels(wlm, base, sauron[4], reglims=[4000,6000])

    if 'f' in paramdict.keys():
        if 'f' in paramnames:
            whf = np.where(np.array(paramnames) == 'f')[0][0]
            f = params[whf]
        else:
            f = paramdict['f']
    
    if timing:
        t2 = time.time()
        print("CHISQ T1: ", t2 - t1)

    if not saurononly:
        mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    if sauron:
        mconvinterp_s = spi.interp1d(wlc_s, mconv_s, kind='cubic', bounds_error=False)

    if timing:
        t3 = time.time()
        print("CHISQ T2: ", t3 - t2)
    
    #bluelow, bluehigh, linelow, linehigh, redlow, redhigh, \
            #mlow, mhigh, morder, line_name, index_name

    #Measuring the chisq
    chisq = 0

    if not saurononly:
        for i in range(len(linedefs[0,:])):
            if line_names[i] not in lineinclude:
                #print line_name[i]
                continue

            #line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
            if lineexclude:
                if 'Na' not in paramnames:
                    if line_names[i] == 'NaI':
                        continue
                if 'Ca' not in paramnames:
                    if line_names[i] == 'CaI':
                        continue
                if 'Fe' not in paramnames:
                    if line_names[i] == 'FeH':
                        continue
                if 'K' not in paramnames:
                    if line_names[i] in ['KI_a','KI_b','KI_1.25']:
                        continue

            #Getting a slice of the model
            wli = wl[i]

            modelslice = mconvinterp(wli)

            #Removing a high-order polynomial from the slice
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= linedefs[0,i]) & (wlc <= linedefs[1,i]))[0]
            redpass = np.where((wlc >= linedefs[4,i]) & (wlc <= linedefs[5,i]))[0]

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
            chisq += np.nansum(((data[i] - modelslice)**2.0 / errterm) + addterm)

            if plot:
                mpl.plot(wl[i], modelslice, 'r')
                mpl.plot(wl[i], data[i], 'b')

    if sauron:
        for i in range(len(sauron[0])):

            #Getting a slice of the model
            wli = sauron[0][i]

            modelslice = mconvinterp_s(wli)

            linedefs_s = [sauron[3][0][0,:], sauron[3][0][1,:], sauron[3][0][4,:], sauron[3][0][5,:]]

            polyfit_model = mcsp.removeLineSlope(wlc_s, mconv_s, linedefs_s, i)
            cont = polyfit_model(wli)

            #Normalizing the model
            modelslice = modelslice / cont

            if 'f' in paramdict.keys():
                errterm = (sauron[2][i] ** 2.0) + modelslice**2.0 * np.exp(2*np.log(f))
                addterm = np.log(2.0 * np.pi * errterm)
            else:
                errterm = sauron[2][i] ** 2.0
                addterm = sauron[2][i] * 0.0

            #Performing the chisq calculation
            chisq += np.nansum(((sauron[1][i] - modelslice)**2.0 / errterm) + addterm)

    if plot:
        mpl.show()

    if timing:
        t4 = time.time()
        print("CHISQ T3: ", t4 - t3)

    return -0.5*chisq

def lnprior(theta, paramnames):
    '''Setting the priors for the mcmc. Returns 0.0 if fine and -inf if otherwise.'''

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

def lnprob(theta, wl, data, err, paramnames, paramdict, lineinclude, linedefs, veldisp, \
        sauron,saurononly):
    '''Primary function of the mcmc. Checks priors and returns the likelihood'''

    lp = lnprior(theta, paramnames)
    if not np.isfinite(lp):
        return -np.inf

    chisqv = calc_chisq(theta, wl, data, err, veldisp, \
            paramnames, paramdict, lineinclude, linedefs, \
            sauron, saurononly, timing=False)
    return lp + chisqv

def do_mcmc(gal, nwalkers, n_iter, z, veldisp, paramdict, lineinclude,\
        threads = 6, restart=False, scale=False, fl=None, sauron=None, sauron_z=None, \
        sauron_veldisp=None, saurononly=False,comments='No Comment'):
    '''Main program. Runs the mcmc'''

    if fl == None:
        print('Please input filename for WIFIS data')
        return

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
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240, 11905]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260, 11935]
    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309, 11935]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333, 11965]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360, 12005]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390, 12025]

    #mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    #mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]
    mlow, mhigh = [],[]
    for i in zip(bluelow, redhigh):
        mlow.append(i[0])
        mhigh.append(i[1])
    morder = [1,1,1,1,1,1,1,1,1,1]

    line_name = np.array(['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123','CaII119'])

    linedefs = [np.array([bluelow, bluehigh, linelow, linehigh, redlow,\
            redhigh, mlow, mhigh, morder]), line_name, line_name]

    wl, data, err = preps.preparespecwifis(fl, z)

    if scale:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err)

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
                redlow_s,redhigh_s,mlow_s,mhigh_s,morder_s]),line_names_s,line_names_s]

        ff = fits.open(sauron)
        spec_s = ff[0].data
        wl_s = ff[1].data
        if 'Vel' not in paramdict.keys():
            wl_s = wl_s / (1. + sauron_z)
        noise_s = ff[2].data

        wl_s, data_s, err_s = preps.splitspec(wl_s, spec_s, sauronlines, err = noise_s)
    else:
        print("No SAURON")

    #    if 'Alpha' in paramdict.keys():
    #        paramnames = ['Age','Z','Alpha']
    #        paramdict = {'Age':None, 'Z':None, 'Alpha':None}
    #    else:
    #        paramnames = ['Age','Z']
    #        paramdict = {'Age':None, 'Z':None}
    ndim = len(paramnames)

    print(paramnames)
    print(lineinclude)

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
                    if paramdict['Vel'] != None:
                        newinit.append(paramdict['Vel'] + np.random.random()*0.004 - 0.002)
                    else:
                        newinit.append(np.random.random()*0.008 + 0.002)
                elif paramnames[j] == 'f':
                    #newinit.append(np.random.random()*11 - 10.)
                    newinit.append(np.random.random())
            pos.append(np.array(newinit))
    else:
       realdata, postprob, infol, lastdata = mcsp.load_mcmc_file(restart)
       pos = lastdata

    savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_fullindex.dat" % (gal)
    f = open(savefl, "w")
    strparams = '\t'.join(paramnames)
    strparamdict = '    '.join(['%s: %s' % (key, paramdict[key]) for key in paramdict.keys()])
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

    pool = Pool(processes=16)

    #sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
    #        (wl, data, err, gal, paramnames, lineinclude, linedefs), \
    #        threads=threads)
    if not sauron:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
                (wl, data, err, paramnames, paramdict, lineinclude, linedefs, veldisp, False, saurononly), \
                pool=pool)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
                (wl, data, err, paramnames, paramdict, lineinclude, linedefs, veldisp, \
                [wl_s, data_s, err_s, sauronlines, sauron_veldisp], saurononly), pool=pool)
    print("Starting MCMC...")

    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        #position = result[0]
        position = result.coords
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            #f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result[1][k]))
            f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result.log_prob[k]))
        f.close()

        if (i+1) % 100 == 0:
            ct = time.time() - t1
            pfinished = (i+1.)*100. / float(n_iter)
            print(ct / 60., " Minutes")
            print(pfinished, "% Finished")
            print(((ct / (pfinished/100.)) - ct) / 60., "Minutes left")
            print(((ct / (pfinished/100.)) - ct) / 3600., "Hours left")
            print()

    return sampler

if __name__ == '__main__':
    vcj = preload_vcj(sauron=True, saurononly=False) #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    #fl_R1 = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20171229_R1.fits'
    #fl_R1 = '/data2/wifis_reduction/elliot/NGC5845/20180531/processed/NGC5845_combined_cube_1_telluricreduced_20200604_R1.fits'

    #lineinclude =   ['FeH', 'CaI','NaI','KI_a','KI_b','KI_1.25','NaI123']
    #lineinclude =   ['FeH', 'NaI','KI_a','KI_b','KI_1.25', 'NaI123']
    #lineinclude =   ['FeH', 'CaI','NaI','KI_a','KI_b','KI_1.25','NaI123','PaB']
    #sampler = do_mcmc('M85', 512, 4000, 0.00235, 178, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/M85_ATLAS3D_R1_2.fits', sauron_z = 0.0022, sauron_veldisp=155) 
    #params =        {'Age':None, 'Z':None, 'Alpha':None}
    #params =        {'Age':None, 'Z':None,'x1':None,'x2':None,'Na':None,'Ca':None,'Fe':None,'K':None,'f':None}

    #lineinclude =   ['FeH', 'CaI', 'NaI', 'KI_a', 'KI_b', 'KI_1.25', 'NaI127']
    #params =        {'Age':None, 'Z':None,'x1':None,'Na':None,'Ca':None,'Fe':None,'K':None}

    #fl_R1 = '/data2/wifis_reduction/elliot/NGC4262/20180402/processed/NGC4262_combined_cube_1_telluricreduced_20200716_R1.fits'

    #sampler = do_mcmc('NGC4262', 512, 4000, 0.004612, 189, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC4262_ATLAS3D_R1.fits',\
    #        sauron_z = 0.004262, sauron_veldisp=187, saurononly=False, \
    #        comments= 'R1 5 spaxels, Hbeta only, 20200716 extraction, equivalent regions, new broadening, multabundance, x1 only') 

    fl_R1 = '/data2/wifis_reduction/elliot/NGC5557/20180302/processed/NGC5557_combined_cube_1_telluricreduced_20200709_R1.fits'
    fl_R2 = '/data2/wifis_reduction/elliot/NGC5557/20180302/processed/NGC5557_combined_cube_1_telluricreduced_20200709_R2.fits'
    lineinclude =   ['FeH', 'NaI', 'KI_a', 'KI_b', 'KI_1.25', 'NaI127']
    #lineinclude =   ['HBeta', 'Fe5015','MgB']
    #params =        {'Age':None, 'Z':None, 'Alpha':None}
    params =        {'Age':None, 'Z':None,'x1':None,'x2':2.3,'Na':None,'K':None,'Fe':None}

    sfl1 = '/home/elliot/sauronspec/NGC5557_ATLAS3D_R1.fits'
    sampler = do_mcmc('NGC5557', 512, 4000, 0.01076, 224, params, lineinclude, threads = 16, \
            fl = fl_R1, sauron=sfl1,\
            sauron_z = 0.01048, sauron_veldisp=269, saurononly=False, \
            comments= 'R1 5 spaxels, 20200709 extraction, equivalent regions, new broadening, multabundance, FixedConvolve, Sauron, x1 only') 

    sfl2 = '/home/elliot/sauronspec/NGC5557_ATLAS3D_R2.fits'
    #params =        {'Age':9.5, 'Z':-0.03,'x1':None,'x2':None,'Na':None,'K':None,'Fe':None}
    sampler = do_mcmc('NGC5557', 512, 4000, 0.01076, 226, params, lineinclude, threads = 16,\
            fl = fl_R2, sauron=sfl2,\
            sauron_z = 0.010513, sauron_veldisp=201, saurononly=False,\
            comments= 'R2 20-5 spaxels, 20200709 extraction, equivalent regions, new broadening, multabundance, fixedconvolve, Sauron, x1 only') 
    


    fl_R1 = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20200528_R1.fits'
    fl_R2 = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20200528_R2.fits'

    lineinclude =   ['FeH', 'CaI', 'NaI', 'KI_a', 'KI_b', 'KI_1.25', 'NaI123']
    params =        {'Age':None, 'Z':None,'x1':None,'x2':2.3,'Na':None,'Ca':None,'K':None,'Fe':None}
    #params =        {'Age':None, 'Z':None, 'Alpha':None}
    #params =        {'Age':None, 'Z':None,'x1':None,'x2':None}#'Na':None,'Fe':None,'K':None}

    sfl1 = '/home/elliot/sauronspec/M85_ATLAS3D_R1.fits'
    sampler = do_mcmc('M85', 512, 4000, 0.00235, 178, params, lineinclude, threads = 16,\
            fl = fl_R1, sauron=sfl1, sauron_z = 0.002169, sauron_veldisp=200, saurononly=False, \
            comments = 'R1 5 spaxels, 20200528 extraction, equivalent regions, new broadening, multabundance, Sauron, x1 only') 

    sfl2 = '/home/elliot/sauronspec/M85_ATLAS3D_R2.fits'
    sampler = do_mcmc('M85', 512, 4000, 0.00228, 163, params, lineinclude, threads = 16,\
            fl = fl_R2, sauron=sfl2, sauron_z = 0.002177, sauron_veldisp=200, saurononly=False, \
            comments = 'R2 20-5 spaxels, 20200528 extraction, equivalent regions, new broadening, multabundance, Sauron, x1 only')
    



    fl_R1 = '/data2/wifis_reduction/elliot/NGC5557/20180302/processed/NGC5557_combined_cube_1_telluricreduced_20200709_R1.fits'
    fl_R2 = '/data2/wifis_reduction/elliot/NGC5557/20180302/processed/NGC5557_combined_cube_1_telluricreduced_20200709_R2.fits'
    lineinclude =   ['FeH', 'NaI', 'KI_a', 'KI_b', 'KI_1.25', 'NaI127', 'PaB']
    #lineinclude =   ['HBeta', 'Fe5015','MgB']
    #params =        {'Age':None, 'Z':None, 'Alpha':None}
    params =        {'Age':None, 'Z':None,'x1':None,'Na':None,'K':None,'Fe':None}

    sfl1 = '/home/elliot/sauronspec/NGC5557_ATLAS3D_R1.fits'
    sampler = do_mcmc('NGC5557', 512, 4000, 0.01076, 224, params, lineinclude, threads = 16, \
            fl = fl_R1, sauron=sfl1,\
            sauron_z = 0.01048, sauron_veldisp=269, saurononly=False, \
            comments= 'R1 5 spaxels, 20200709 extraction, equivalent regions, new broadening, multabundance, FixedConvolve, Sauron, Single Slope') 

    sfl2 = '/home/elliot/sauronspec/NGC5557_ATLAS3D_R2.fits'
    #params =        {'Age':9.5, 'Z':-0.03,'x1':None,'x2':None,'Na':None,'K':None,'Fe':None}
    sampler = do_mcmc('NGC5557', 512, 4000, 0.01076, 226, params, lineinclude, threads = 16,\
            fl = fl_R2, sauron=sfl2,\
            sauron_z = 0.010513, sauron_veldisp=201, saurononly=False,\
            comments= 'R2 20-5 spaxels, 20200709 extraction, equivalent regions, new broadening, multabundance, fixedconvolve, Sauron, Single Slope') 

    fl_R1 = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20200528_R1.fits'
    fl_R2 = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20200528_R2.fits'

    lineinclude =   ['FeH', 'CaI', 'NaI', 'KI_a', 'KI_b', 'KI_1.25', 'NaI123', 'PaB']
    params =        {'Age':None, 'Z':None,'x1':None,'Na':None,'Ca':None,'K':None,'Fe':None}
    #params =        {'Age':None, 'Z':None, 'Alpha':None}
    #params =        {'Age':None, 'Z':None,'x1':None,'x2':None}#'Na':None,'Fe':None,'K':None}

    sfl1 = '/home/elliot/sauronspec/M85_ATLAS3D_R1.fits'
    sampler = do_mcmc('M85', 512, 4000, 0.00235, 178, params, lineinclude, threads = 16,\
            fl = fl_R1, sauron=sfl1, sauron_z = 0.002169, sauron_veldisp=200, saurononly=False, \
            comments = 'R1 5 spaxels, 20200528 extraction, equivalent regions, new broadening, multabundance, Sauron, Single Slope') 

    sfl2 = '/home/elliot/sauronspec/M85_ATLAS3D_R2.fits'
    sampler = do_mcmc('M85', 512, 4000, 0.00228, 163, params, lineinclude, threads = 16,\
            fl = fl_R2, sauron=sfl2, sauron_z = 0.002177, sauron_veldisp=200, saurononly=False, \
            comments = 'R2 20-5 spaxels, 20200528 extraction, equivalent regions, new broadening, multabundance, Sauron, Single Slope')

    #######################################
    #sampler = do_mcmc('NGC5845', 512, 4000, 0.004967, 283, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5845_ATLAS3D_R1.fits',sauron_z = 0.0045, sauron_veldisp=225) 
    #sampler = do_mcmc('NGC5557', 512, 4000, 0.0108, 229, params, 'wifis', lineinclude, threads = 16, fl = fl_R1) 

    #lineinclude =   ['FeH', 'CaI','NaI','KI_a','KI_b','KI_1.25','NaI123']
    #params =        {'Age':4.0, 'Z':None,'x1':None,'x2':None,'Na':None,'Ca':None,'Fe':None,'K':None}
    #sampler = do_mcmc('M85', 512, 4000, 0.00235, 178, params, 'wifis', lineinclude, threads = 16, fl = fl_R2) 
    
    #lineinclude =   ['FeH','NaI','CaI','KI_a','KI_b','KI_1.25','NaI127']
    #params =        {'Age': 11.0, 'Z':None,'x1':None,'x2':None,'Na':None,'Ca':None,'Fe':None,'K':None}
    
    #sampler = do_mcmc('NGC5845', 512, 4000, 0.004967, 283, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5845_ATLAS3D_Re8.fits',\
    #        sauron_z = 0.00452, sauron_veldisp=267., saurononly=True) 
    #sampler = do_mcmc('NGC5845', 512, 4000, 0.004967, 283, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5845_ATLAS3D_Re2.fits',\
    #        sauron_z = 0.0045168, sauron_veldisp=267., saurononly=True) 
    #sampler = do_mcmc('NGC5845', 512, 4000, 0.004967, 283, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5845_ATLAS3D_Re.fits',\
    #        sauron_z = 0.00453, sauron_veldisp=267., saurononly=True) 
    #params =        {'Age':None, 'Z':None,'Alpha':None, 'x1':None,'x2':None,'Na':None,'Ca':None,'Fe':None,'K':None}

    #sampler = do_mcmc('NGC5557', 512, 4000, 0.0108, 229, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5557_ATLAS3D_Re8_test.fits',\
    #        sauron_z = 0.0105, sauron_veldisp=295, saurononly=True) 
    #sampler = do_mcmc('NGC5557', 512, 4000, 0.0108, 229, params, lineinclude, threads = 16, fl = fl_R1,\
            #sauron='/home/elliot/sauronspec/NGC5557_ATLAS3D_Re2.fits',\
            #sauron_z = 0.0105, sauron_veldisp=295, saurononly=True) 
    #sampler = do_mcmc('NGC5557', 512, 4000, 0.0108, 229, params, lineinclude, threads = 16, fl = fl_R1,\
    #        sauron='/home/elliot/sauronspec/NGC5557_ATLAS3D_Re.fits',\
    #        sauron_z = 0.0105, sauron_veldisp=295, saurononly=True) 
