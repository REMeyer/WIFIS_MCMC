####################
#   MCMC program using Prof Charlie Conroy's SSP Models
#   Based off of emcee 
#   Author: Elliot Meyer, Dept Astronomy & Astrophysics University of Toronto
###################

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
import plot_corner as plcr
from random import uniform

warnings.simplefilter('ignore', np.RankWarning)

# DEFINES THE BASE PATH -- NEEDS UPDATING FOR ALL SYSTEMS
#base = '/home/elliot/mcmcgemini/'
#base = '/Users/relliotmeyer/mcmcgemini/'
base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

# MCMC Parameters
# Metallicity: -1.5 < [Z/H] < 0.2 steps of 0.1?
# Age: depends on galaxy, steps of 1 Gyr?
# IMF: x1 and x2 full range
# [Na/H]: -0.4 <-> +1.3

linefit = True
lineexclude = False
ratiofit = False
#instrument = 'nifs'

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

#elif not linefit:
#    linelow = [9905,10337,11372,11680,11765,12505,13115]
#    linehigh = [9935,10360,11415,11705,11793,12545,13165]
#
#    bluelow = [9855,10300,11340,11667,11710,12460,13090]
#    bluehigh = [9880,10320,11370,11680,11750,12495,13113]

#    redlow = [9940,10365,11417,11710,11793,12555,13165]
#    redhigh = [9970,10390,11447,11750,11810,12590,13175]

#    line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']

#    mlow = [9700,10550,11340,11550,12350,12665]
#    mhigh = [10450,10965,11447,12200,12590,13180]
#    morder = [8,4,1,7,2,5]

#Dictionary to help easily access the IMF index
imfsdict = {}
for i in range(16):
    for j in range(16):
        imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

vcj = {}

def preload_vcj(overwrite_base = False):
    '''Loads the SSP models into memory so the mcmc model creation takes a
    shorter time. Returns a dict with the filenames as the keys'''
    global vcj
    #global base

    if overwrite_base:
        base = overwrite_base
    else:
        base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

    chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-',\
            'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    print "PRELOADING SSP MODELS INTO MEMORY"
    fls = glob(base+'spec/vcj_ssp/*')    

    vcj = {}
    for fl in fls:
        #print fl
        flspl = fl.split('/')[-1]
        mnamespl = flspl.split('_')
        age = float(mnamespl[3][1:])
        Zsign = mnamespl[4][1]
        Zval = float(mnamespl[4][2:5])
        if Zsign == "m":
            Zval = -1.0 * Zval
        x = pd.read_table(fl, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj["%.1f_%.1f" % (age, Zval)] = [x[:,1:]]

    print "PRELOADING ABUNDANCE MODELS INTO MEMORY"
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

        x = pd.read_table(fl, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj["%.1f_%.1f" % (age, Zval)].append(x[:,1:])

    vcj["WL"] = x[:,0]

    print "FINISHED LOADING MODELS"

    return vcj

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

def model_spec(inputs, gal, paramnames, vcjset = False, timing = False, full = False):
    '''Core function which takes the input model parameters, finds the appropriate models,
    and adjusts them for the input abundance ratios. Returns a broadened model spectrum 
    to be matched with a data spectrum.'''

    if vcjset:
        vcj = vcjset
    else:
        global vcj

    if timing:
        print "Starting Model Spec Time"
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
        elif paramnames[j] == 'Mg':
            Mg = inputs[j]
        elif paramnames[j] == 'VelDisp':
            veldisp = inputs[j]

    if 'Age' not in paramnames:
        if gal == 'M85':
            Age = 5.0
        elif gal == 'M87':
            Age = 13.5
    if 'Z' not in paramnames:
        Z = 0.0

    if 'x1' not in paramnames:
        x1 = 3.0
    if 'x2' not in paramnames:
        x2 = 2.3

    if x1 not in x1_m:
        x1min = np.argmin(np.abs(x1_m - x1))
        x1 = x1_m[x1min]
    if x2 not in x2_m:
        x2min = np.argmin(np.abs(x2_m - x2))
        x2 = x2_m[x2min]


    #Finding the appropriate base model files.
    fl1, fl2, fl3, fl4, agem, agep, zm, zp, mixage, mixZ = select_model_file(Z, Age)

    if timing:
        t2 = time.time()
        print "MSPEC T1: ",t2 - t1
        
    # If the Age of Z is inbetween models then this will average the respective models to produce 
    # one that is closer to what is expected.
    # NOTE THAT THIS IS AN ASSUMPTION AND THE CHANGE IN THE MODELS IS NOT NECESSARILY LINEAR
    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])
    abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3]
    if mixage and mixZ:
        # Reading models. This step was vastly improved by pre-loading the models prior to running the mcmc
        fm1 = vcj[fl1][0]
        fm2 = vcj[fl2][0]
        fm3 = vcj[fl3][0]
        fm4 = vcj[fl4][0]

        fc1 = vcj[fl1][1]
        fc2 = vcj[fl2][1]
        fc3 = vcj[fl3][1]
        fc4 = vcj[fl4][1]

        wlc1 = vcj["WL"]

        #Finding the relevant section of the models to reduce the computational complexity
        if not full:
            rel = np.where((wlc1 > 8500) & (wlc1 < 14500))[0]
        else:
            rel = np.where((wlc1 > 3000) & (wlc1 < 24000))[0]

        fc1 = fc1[rel,:]
        fc2 = fc2[rel,:]
        fc3 = fc3[rel,:]
        fc4 = fc4[rel,:]

        fm1 = fm1[rel, :]
        fm2 = fm2[rel, :]
        fm3 = fm3[rel, :]
        fm4 = fm4[rel, :]

        wl = wlc1[rel]

        #m = np.zeros(fm1.shape)
        c = np.zeros(fc1.shape)
        imf_i = imfsdict[(x1,x2)]

        #For the x1x2 model
        interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm1[:,imf_i],fm2[:,imf_i])), kind = 'linear')
        interp2 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm3[:,imf_i],fm4[:,imf_i])), kind = 'linear')
        age1 = interp1(wl,Age)
        age2 = interp2(wl,Age)

        interp3 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], np.stack((age1,age2)), kind = 'linear')
        mimf = interp3(wl,Z)

        #For the basemodel
        interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm1[:,73],fm2[:,73])), kind = 'linear')
        interp2 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm3[:,73],fm4[:,73])), kind = 'linear')
        age1 = interp1(wl,Age)
        age2 = interp2(wl,Age)

        interp3 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], np.stack((age1,age2)), kind = 'linear')
        basemodel = interp3(wl,Z)

        #Taking the average of the models (could be improved?)
        #mimf = (fm1[rel,imfsdict[(x1,x2)]] + fm2[rel,imfsdict[(x1,x2)]])/2.0
        #basemodel = (fm1[rel,73] + fm2[rel,73])/2.0
        for i in abundi:
            interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                    np.stack((fc1[:,i],fc2[:,i])), kind = 'linear')
            interp2 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                    np.stack((fc3[:,i],fc4[:,i])), kind = 'linear')
            age1 = interp1(wl,Age)
            age2 = interp2(wl,Age)

            interp3 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], np.stack((age1,age2)), kind = 'linear')
            c[:,i] = interp3(wl,Z)

        #c = (fc1 + fc2)/2.0

        # Setting the models to the proper length
        #c = c[rel,:]
        #mimf = m[rel,imfsdict[(x1,x2)]]
    elif mixage:
        fm1 = vcj[fl1][0]
        fm2 = vcj[fl2][0]

        fc1 = vcj[fl1][1]
        fc2 = vcj[fl2][1]

        wlc1 = vcj["WL"]

        #Finding the relevant section of the models to reduce the computational complexity
        if not full:
            rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]
        else:
            rel = np.where((wlc1 > 6500) & (wlc1 < 24000))[0]

        fc1 = fc1[rel,:]
        fc2 = fc2[rel,:]

        fm1 = fm1[rel, :]
        fm2 = fm2[rel, :]

        wl = wlc1[rel]

        #m = np.zeros(fm1.shape)
        c = np.zeros(fc1.shape)

        #For the x1x2 model
        interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm1[:,imfsdict[(x1,x2)]],fm2[:,imfsdict[(x1,x2)]])), kind = 'linear')
        mimf = interp1(wl,Age)

        #For the basemodel
        interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                np.stack((fm1[:,73],fm2[:,73])), kind = 'linear')
        basemodel = interp1(wl,Age)

        #Taking the average of the models (could be improved?)
        #mimf = (fm1[rel,imfsdict[(x1,x2)]] + fm2[rel,imfsdict[(x1,x2)]])/2.0
        #basemodel = (fm1[rel,73] + fm2[rel,73])/2.0

        for i in abundi:
            interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                    np.stack((fc1[:,i],fc2[:,i])), kind = 'linear')
            c[:,i] = interp1(wl,Age)
    elif mixZ:
        fm1 = vcj[fl1][0]
        fm2 = vcj[fl2][0]

        fc1 = vcj[fl1][1]
        fc2 = vcj[fl2][1]

        wlc1 = vcj["WL"]

        #Finding the relevant section of the models to reduce the computational complexity
        if not full:
            rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]
        else:
            rel = np.where((wlc1 > 6500) & (wlc1 < 24000))[0]

        fc1 = fc1[rel,:]
        fc2 = fc2[rel,:]

        fm1 = fm1[rel, :]
        fm2 = fm2[rel, :]

        wl = wlc1[rel]

        #m = np.zeros(fm1.shape)
        c = np.zeros(fc1.shape)

        #For the x1x2 model
        interp1 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], \
                np.stack((fm1[:,imfsdict[(x1,x2)]],fm2[:,imfsdict[(x1,x2)]])), kind = 'linear')
        mimf = interp1(wl,Z)

        #For the basemodel
        interp1 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], \
                np.stack((fm1[:,73],fm2[:,73])), kind = 'linear')
        basemodel = interp1(wl,Z)

        #Taking the average of the models (could be improved?)
        #mimf = (fm1[rel,imfsdict[(x1,x2)]] + fm2[rel,imfsdict[(x1,x2)]])/2.0
        #basemodel = (fm1[rel,73] + fm2[rel,73])/2.0

        for i in abundi:
            interp1 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], \
                    np.stack((fc1[:,i],fc2[:,i])), kind = 'linear')
            c[:,i] = interp1(wl,Z)
    else:
        #If theres no need to mix models then just read them in and set the length
        m = vcj[fl1][0]
        wlc1 = vcj["WL"]
        c = vcj[fl1][1]

        if not full:
            rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]
        else:
            rel = np.where((wlc1 > 6500) & (wlc1 < 24000))[0]

        mimf = m[rel,imfsdict[(x1,x2)]]
        c = c[rel,:]
        wl = wlc1[rel]
        basemodel = m[rel,73]

    if timing:
        t3 = time.time()
        print "MSPEC T2: ", t3 - t2

    # Reminder of the abundance model columns
    # ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
    #                'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #                'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    # DETERMINING THE ABUNDANCE RATIO EFFECTS
    # The assumption here is that the abundance effects scale linearly...for all except sodium which covers a much wider
    # range of acceptable values.
    # The models are interpolated at the various given abundances to allow for abundance values to be continuous.
    # The interpolated value is then normalized by the solar metallicity model and then subtracted by 1 to 
    # retain the percentage
    #if fitmode in [True, False, 'NoAge', 'NoAgeVelDisp']:

        #Na adjustment
    ab_contribution = 0.0
    if 'Na' in paramnames:
        Naextend = (c[:,2]-c[:,0])*(-0.5/-0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.6,0.9], np.stack((Naextend,c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        NaP = interp(wl,Na) / c[:,0] - 1.
        ab_contribution += NaP

        #K adjustment (assume symmetrical K adjustment)
    if 'K' in paramnames:
        Kminus = (2. - (c[:,29] / c[:,0]))*c[:,0]
        Kmextend = (Kminus - c[:,0])*(-0.5/-0.3) + c[:,0]
        Kpextend = (c[:,29] - c[:,0]) * (0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Kmextend,Kminus,c[:,0],c[:,29],Kpextend)), kind = 'linear')
        KP = interp(wl,K) / c[:,0] - 1.
        ab_contribution += KP

    if 'Mg' in paramnames:
        #Mg adjustment (only for full fitting)
        #if fitmode == False:
        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,16], c[:,0],c[:,15])), kind = 'linear')
        MgP = interp(wl,Mg) / c[:,0] - 1.
        ab_contribution += MgP

    if 'Fe' in paramnames:
        #Fe Adjustment
        Femextend = (c[:,6] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Fepextend = (c[:,5] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Femextend,c[:,6], c[:,0],c[:,5],Fepextend)), kind = 'linear')
        FeP = interp(wl,Fe) / c[:,0] - 1.
        ab_contribution += FeP

    if 'Ca' in paramnames:
        #Ca Adjustment
        Cmextend = (c[:,4] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Cpextend = (c[:,3] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], np.stack((Cmextend,c[:,4], c[:,0],c[:,3],Cpextend)), kind = 'linear')
        CaP = interp(wl,Ca) / c[:,0] - 1.
        ab_contribution += CaP

    model_ratio = mimf / basemodel

    # The model formula is as follows....
    # The new model is = the base IMF model * abundance effects. 
    # The abundance effect %ages are scaled by the ratio of the selected IMF model to the Kroupa IMF model
    # The formula ensures that if the abundances are solar then the base IMF model is recovered. 
    newm = mimf*(1. + model_ratio*ab_contribution)
    #newm = mimf*(1. + ab_contribution)

    #if fitmode in [True, 'NoAge', 'NoAgeVelDisp']:
    #    newm = mimf*(1. + model_ratio*(NaP + KP + CaP + FeP))
    #elif fitmode in ['limited', 'LimitedVelDisp']:
    #    newm = mimf
    #else:
    #    newm = mimf*(1. + model_ratio(NaP + KP + MgP + CaP + FeP))
        
    if timing:
        print "MSPEC T3: ", time.time() - t3

    return wl, newm

def calc_chisq(params, wl, data, err, gal, paramnames, lineinclude, \
        linedefs, plot=False, timing = False):
    ''' Important function that produces the value that essentially
    represents the likelihood of the mcmc equation. Produces the model
    spectrum then returns a normal chisq value.'''

    linelow, linehigh, bluelow, bluehigh, redlow, redhigh, line_name, mlow, mhigh, morder = linedefs
    
    if timing:
        t1 = time.time()

    #Creating model spectrum then interpolating it so that it can be easily matched with the data.
    wlm, newm = model_spec(params, gal, paramnames, timing=timing)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    if 'VelDisp' in paramnames:
        whsig = np.where(paramnames == 'VelDisp')[0]
        wlc, mconv = convolvemodels(wlm, newm, params[whsig])
    else:
        if gal == 'M85':
            #wlc, mconv = convolvemodels(wl, newm, 176.)
            #wlc, mconv = convolvemodels(wlm, newm, 170.)
            wlc, mconv = convolvemodels(wlm, newm, 140.)
        elif gal == 'M87':
            #wlc, mconv = convolvemodels(wlm, newm, 308.)
            wlc, mconv = convolvemodels(wlm, newm, 370.)

    if 'f' in paramnames:
        whf = np.where(np.array(paramnames) == 'f')[0][0]
        f = params[whf]
    
    if timing:
        t2 = time.time()
        print "CHISQ T1: ", t2 - t1

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    if timing:
        t3 = time.time()
        print "CHISQ T2: ", t3 - t2
    
    #Measuring the chisq
    chisq = 0
    for i in range(len(mlow)):
        if (gal == 'M87') and linefit:
            if line_name[i] == 'KI_1.25':
                continue

        if line_name[i] not in lineinclude:
            #print line_name[i]
            continue

        #line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
        if lineexclude:
            if (gal == 'M85') and linefit and ('Na' not in paramnames):
                if line_name[i] == 'NaI':
                    continue
            if (gal == 'M85') and linefit and ('Ca' not in paramnames):
                if line_name[i] == 'CaI':
                    continue
            if (gal == 'M85') and linefit and ('Fe' not in paramnames):
                if line_name[i] == 'FeH':
                    continue
            if (gal == 'M85') and linefit and ('K' not in paramnames):
                if line_name[i] in ['KI_a','KI_b','KI_1.25']:
                    continue

        #Getting a slice of the model
        wli = wl[i]
        #if (gal == 'M85') and linefit:
        #    if i in [3,4]:
        #        wli = np.array(wli)
        #        wli -= 2.0
            #elif i == 5:
            #    wli = np.array(wli)
            #    wli += 1.0
            #elif i == 1:
            #    wli = np.array(wli)
            #    wli += 1.0

        modelslice = mconvinterp(wli)

        #Removing a high-order polynomial from the slice
        if linefit:
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
            cont = polyfit(wli)

        else:
            if i == 2:
                if (gal == 'M85') and ('Na' not in paramnames):
                    continue
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
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wli, modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wli)

        #Normalizing the model
        if not linefit:
            mdratio = data[i] / modelslice
            pf = np.polyfit(wli, mdratio, morder[i])
            polyfit = np.poly1d(pf)
            cont = polyfit(wli)
        elif not linefit and (i == 2):
            modelslice = modelslice / cont
        else:
            modelslice = modelslice / cont

        if 'f' in paramnames:
            errterm = (err[i] ** 2.0) + (data[i]**2.0 * np.exp(2*f))
            addterm = np.log(2.0 * np.pi * errterm)
        else:
            errterm = err[i] ** 2.0
            addterm = err[i] * 0.0

        #Performing the chisq calculation
        if linefit and (line_name[i] == 'FeH'):
            whbad = (wl[i] > 9894) & (wl[i] < 9908)
            wg = np.logical_not(whbad)
            chisq += np.sum(((modelslice[wg] - data[i][wg])**2.0 / errterm[wg]) + addterm[wg])
        elif not linefit and (i==2):
            chisq += np.sum((modelslice - data[i])**2.0 / (err[i]**2.0))
        elif not linefit:
            chisq += np.sum(((modelslice/cont) - (data[i]/cont))**2.0 / (err[i]**2.0))
        else:
            chisq += np.sum(((modelslice - data[i])**2.0 / errterm) + addterm)

        if plot:
            mpl.plot(wl[i], modelslice, 'r')
            mpl.plot(wl[i], data[i], 'b')

    if plot:
        mpl.show()

    if timing:
        t4 = time.time()
        print "CHISQ T3: ", t4 - t3

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
        elif paramnames[j] == 'f':
            if not (-10. <= np.log(theta[j]) <= 1.):
                goodpriors = False

    if goodpriors == True:
        return 0.0
    else:
        return -np.inf

def lnprob(theta, wl, data, err, gal, paramnames, lineinclude, linedefs):
    '''Primary function of the mcmc. Checks priors and returns the likelihood'''

    lp = lnprior(theta, paramnames)
    if not np.isfinite(lp):
        return -np.inf

    chisqv = calc_chisq(theta, wl, data, err, gal, \
            paramnames, lineinclude, linedefs, timing=False)
    return lp + chisqv

def do_mcmc(gal, nwalkers, n_iter, paramnames, instrument, lineinclude,\
        threads = 6, restart=False, scale=False,fl=None):
    '''Main program. Runs the mcmc'''

    #Line definitions & other definitions
    if instrument == 'nifs' and linefit:
        linelow = [9905,10337,11372,11680,11765,12505,13115, 12810, 12670]
        linehigh = [9935,10360,11415,11705,11793,12545,13165, 12840, 12690]

        bluelow = [9855,10300,11340,11667,11710,12460,13090, 12780, 12648]
        bluehigh = [9880,10320,11370,11680,11750,12495,13113, 12800, 12660]

        redlow = [9940,10365,11417,11710,11793,12555,13165, 12855, 12700]
        redhigh = [9970,10390,11447,11750,11810,12590,13175, 12880, 12720]

        line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI', 'PaB', 'NaI127']

        mlow = [9855,10300,11340,11667,11710,12460,13090,12780, 12648]
        mhigh = [9970,10390,11447,11750,11810,12590,13175, 12880, 12720]
        morder = [1,1,1,1,1,1,1,1]
        linedefs = [linelow, linehigh, bluelow, bluehigh, redlow, redhigh, line_name, mlow, mhigh, morder]

    elif instrument == 'wifis' and linefit:
        linelow = [9905,10337,11372,11680,11765,12505,13115, 12810, 12670, 12309]
        linehigh = [9935,10360,11415,11705,11793,12545,13165, 12840, 12690, 12333]

        bluelow = [9855,10300,11340,11667,11710,12460,13090, 12780, 12648, 12240]
        bluehigh = [9880,10320,11370,11680,11750,12495,13113, 12800, 12660, 12260]

        redlow = [9940,10365,11417,11710,11793,12555,13165, 12860, 12700, 12360]
        redhigh = [9970,10390,11447,11750,11810,12590,13175, 12870, 12720, 12390]

        line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI', 'PaB', 'NaI127', 'NaI123']

        mlow = [9855,10300,11340,11667,11710,12460,13090,12780, 12648, 12240]
        mhigh = [9970,10390,11447,11750,11810,12590,13175, 12880, 12720, 12390]
        morder = [1,1,1,1,1,1,1,1,1]
        linedefs = [linelow, linehigh, bluelow, bluehigh, redlow, redhigh, line_name, mlow, mhigh, morder]

    if instrument == 'nifs':
        wl, data, err = preparespec(gal)
    elif instrument == 'wifis':
        if fl == None:
            print('Please input filename for WIFIS data')
            return
        wl, data, err = preparespecwifis(gal, fl)

    if scale:
        wl, data, err = splitspec(wl, data, linedefs, err = err, scale = scale)
    else:
        wl, data, err = splitspec(wl, data, linedefs, err = err)

    ndim = len(paramnames)

    print paramnames
    print lineinclude

    pos = []
    if not restart:
        for i in range(nwalkers):
            newinit = []
            for j in range(len(paramnames)):
                if paramnames[j] == 'Age':
                    newinit.append(np.random.random()*12.5 + 1.0)
                elif paramnames[j] == 'Z':
                    newinit.append(np.random.random()*0.45 - 0.25)
                elif paramnames[j] in ['x1', 'x2']:
                    newinit.append(np.random.choice(x1_m))
                elif paramnames[j] == 'Na':
                    newinit.append(np.random.random()*1.3 - 0.3)
                elif paramnames[j] in ['K','Ca','Fe','Mg']:
                    newinit.append(np.random.random()*0.6 - 0.3)
                elif paramnames[j] == 'VelDisp':
                    newinit.append(np.random.random()*240 + 120)
                elif paramnames[j] == 'f':
                    newinit.append(np.random.random())
            pos.append(np.array(newinit))
    else:
       realdata, postprob, infol, lastdata = mcsp.load_mcmc_file(restart)
       pos = lastdata

    savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_fullfit.dat" % (gal)
    f = open(savefl, "w")
    strparams = '\t'.join(paramnames)
    strlines = '\t'.join(lineinclude)
    f.write("#NWalk\tNStep\tGal\tFit\n")
    f.write("#%d\t%d\t%s\n" % (nwalkers, n_iter,gal))
    f.write("#%s\n" % (strparams))
    f.write("#%s\n" % (strlines))
    f.close()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
            (wl, data, err, gal, paramnames, lineinclude, linedefs), \
            threads=threads)
    print "Starting MCMC..."

    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        position = result[0]
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result[1][k]))
        f.close()

        if (i+1) % 100 == 0:
            ct = time.time() - t1
            pfinished = (i+1.)*100. / float(n_iter)
            print ct / 60., " Minutes"
            print pfinished, "% Finished"
            print ((ct / (pfinished/100.)) - ct) / 60., "Minutes left"
            print ((ct / (pfinished/100.)) - ct) / 3600., "Hours left"
            print 

    return sampler

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
        print "Replacing NA spectrum"
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

def preparespecwifis(galaxy, fl, baseforce = False):

    if baseforce:
        base = baseforce
    else:
        pass
        #global base

    if galaxy == 'M85':
        z = 0.002432
        datafl = fl
        scale = 1.0
        contcorr = False

    if galaxy == 'M87':
        z = 0.005#0.004283
        datafl = fl
        scale = 1.0
        contcorr = False

    ff = fits.open(datafl)
    data = ff[0].data
    wl = ff[1].data 
    errors = ff[2].data

    wl = wl / (1 + z)
    #wlz, dataz = nm.skymask(wlz, dataz, galaxy, 'Z')

    gd = ~np.isnan(data)
    data = data[gd]
    wl = wl[gd]
    errors = errors[gd]

    gd2 = ~np.isnan(errors)
    data = data[gd2]
    wl = wl[gd2]
    errors = errors[gd2]

    return wl, data, errors

def splitspec(wl, data, linedefs, err=False, scale = False):

    linelow, linehigh, bluelow, bluehigh, redlow, redhigh, line_name, mlow, mhigh, morder = linedefs
    lines = linefit

    databands = []
    wlbands = []
    errorbands = []

    for i in range(len(mlow)):
        wh = np.where( (wl >= mlow[i]) & (wl <= mhigh[i]))[0]
        dataslice = data[wh]
        wlslice = wl[wh]
        wlbands.append(wlslice)

        if lines:
            #Define the bandpasses for each line 
            bluepass = np.where((wl >= bluelow[i]) & (wl <= bluehigh[i]))[0]
            redpass = np.where((wl >= redlow[i]) & (wl <= redhigh[i]))[0]
            fullpass = np.where((wl >= bluelow[i]) & (wl <= redhigh[i]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([bluelow[i],bluehigh[i]])
            redavg = np.mean([redlow[i],redhigh[i]])

            blueval = np.mean(data[bluepass])
            redval = np.mean(data[redpass])

            pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
            polyfit = np.poly1d(pf)
            cont = polyfit(wlslice)

            
            #if i in [3,4]:
            #    mpl.plot(wlslice, data[fullpass])
            #    mpl.plot(wlslice, cont)
            #    mpl.show()

            if scale:
                newdata = np.array(data)
                newdata[fullpass] -= scale*polyfit(wl[fullpass])

                blueval = np.mean(newdata[bluepass])
                redval = np.mean(newdata[redpass])

                pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)

                #if i in [3,4]:
                #    mpl.plot(wlslice, newdata[fullpass])
                #    mpl.plot(wlslice, cont)
                #    mpl.show()

        else:
            if i == 2:
                #Define the bandpasses for each line 
                bluepass = np.where((wl >= bluelow[2]) & (wl <= bluehigh[2]))[0]
                redpass = np.where((wl >= redlow[2]) & (wl <= redhigh[2]))[0]

                #Cacluating center value of the blue and red bandpasses
                blueavg = np.mean([bluelow[2],bluehigh[2]])
                redavg = np.mean([redlow[2],redhigh[2]])

                blueval = np.mean(data[bluepass])
                redval = np.mean(data[redpass])

                pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)

                if scale:
                    data[fullpass] -= scale*polyfit(wl[fullpass])

                    blueval = np.mean(data[bluepass])
                    redval = np.mean(data[redpass])
                    pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                    polyfit = np.poly1d(pf)
                    cont = polyfit(wlslice)
            else:
                pf = np.polyfit(wlslice, dataslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)

        databands.append(data[wh] / cont)

        if type(err) != bool:
            errslice = err[wh]
            errorbands.append(errslice / cont)
    
    return wlbands, databands, errorbands

def convolvemodels(wlfull, datafull, veldisp, reglims = False):

    if reglims:
        reg = (wlfull >= reglims[0]) & (wlfull <= reglims[1])
        #print("Reglims")
    else:
        reg = (wlfull >= 9500) & (wlfull <= 13500)
        #print("Not Reglims")
    
    wl = wlfull[reg]
    data = datafull[reg]

    c = 299792.458

    #Sigma from description of models
    m_center = 11500
    m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    
    sigma_gal = np.abs((m_center / (veldisp/c + 1.)) - m_center)
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    gaussplot = mcsp.gauss_nat(convolvex, [sigma_conv,0.])

    out = np.convolve(datafull, gaussplot, mode='same')

    return wlfull, out

def test_chisq(paramnames, lineinclude, vcj, trials = 5, timing = True):
    d = preparespec('M85')
    wl, data, err = splitspec(d[0], d[1], err = d[2], scale = False, lines = linefit)

    for i in range(trials):
        newinit = []
        for j in range(len(paramnames)):
            if paramnames[j] == 'Age':
                newinit.append(np.random.random()*12.5 + 1.0)
            elif paramnames[j] == 'Z':
                newinit.append(np.random.random()*0.45 - 0.25)
            elif paramnames[j] in ['x1', 'x2']:
                newinit.append(np.random.choice(x1_m))
            elif paramnames[j] == 'Na':
                newinit.append(np.random.random()*1.3 - 0.3)
            elif paramnames[j] in ['K','Ca','Fe','Mg']:
                newinit.append(np.random.random()*0.6 - 0.3)
            elif paramnames[j] == 'VelDisp':
                newinit.append(np.random.random()*240 + 120)
            elif paramnames[j] == 'f':
                newinit.append(np.random.random()*11 - 10)

        chisquared = chisq(newinit, wl, data, err, 'M85', paramnames, lineinclude)
        print(chisquared)

    return

def removeLineSlope(wlc, mconv,i):
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

if __name__ == '__main__':
    vcj = preload_vcj() #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    
    lineinclude =   ['FeH', 'NaI', 'KI_a','KI_b','KI_1.25','PaB', 'NaI123']
    params =        ['Age','Z','x1','x2','Na','Fe','K']
    sampler = do_mcmc('M85', 512, 4000, params, 'wifis', lineinclude,\
            threads = 16, \
            fl = '/home/elliot/M85_combined_cube_1_telluricreduced_r1.fits')

    lineinclude =   ['FeH', 'NaI', 'KI_a','KI_b','KI_1.25','PaB', 'NaI123']
    params =        ['Age','Z','x1','x2','Na','Fe','K']
    sampler = do_mcmc('M85', 512, 4000, params, 'wifis', lineinclude, \
            threads = 16, \
            fl = '/home/elliot/M85_combined_cube_1_telluricreduced_r2.fits')

    lineinclude =   ['FeH', 'NaI', 'KI_a','KI_b','KI_1.25','PaB', 'NaI123']
    params =        ['Age','Z','x1','x2','Na','Fe','K']
    sampler = do_mcmc('M85', 512, 4000, params, 'wifis', lineinclude,\
            threads = 16, \
            fl = '/home/elliot/M85_combined_cube_1_telluricreduced_r3.fits')

    #lineinclude =   ['FeH','NaI', 'KI_a','KI_b','KI_1.25','PaB', 'NaI123']
    #params =        ['Age','Z','x1','x2','Na','Fe','K']
    #sampler = do_mcmc('M85', 512, 4000, params, 'wifis', lineinclude, \
    #        threads = 16, fl = '/home/elliot/M85_combined_cube_1_extracted_r1.fits')

    #lineinclude =   ['FeH','NaI', 'KI_a','KI_b','KI_1.25','PaB', 'NaI123']
    #params =        ['Age','Z','x1','x2','Na','Fe','K']
    #sampler = do_mcmc('M85', 512, 4000, params, 'wifis', lineinclude, \
    #        threads = 16, fl = '/home/elliot/M85_combined_cube_1_extracted_r2.fits')

    #lineinclude =   ['FeH','CaI', 'NaI', 'KI_a','KI_b','PaB', 'NaI127']
    #params =        ['Age','Z','x1','x2','Na','Fe','Ca','K']
    #sampler = do_mcmc('M87', 512, 4000, params, 'wifis', lineinclude, \
    #        threads = 16, fl = '/home/elliot/M87_combined_cube_1_extracted_r1.fits')

    #lineinclude =   ['FeH','CaI', 'NaI', 'KI_a','KI_b','PaB', 'NaI127']
    #params =        ['Age','Z','x1','x2','Na','Fe','Ca','K']
    #sampler = do_mcmc('M87', 512, 4000, params, 'wifis', lineinclude, \
    #        threads = 16, fl = '/home/elliot/M87_combined_cube_1_extracted_r2.fits')

