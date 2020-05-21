####################
#   MCMC program using Prof Charlie Conroy's SSP Models
#   Based off of emcee 
#   Author: Elliot Meyer, Dept Astronomy & Astrophysics University of Toronto
###################
from __future__ import print_function

from astropy.io import fits
from sys import exit
from glob import glob
from random import uniform
from multiprocessing import Pool

import numpy as np
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
import prepare_spectra as preps

warnings.simplefilter('ignore', np.RankWarning)

# DEFINES THE BASE PATH -- NEEDS UPDATING FOR ALL SYSTEMS
base = os.path.dirname(os.path.realpath(sys.argv[0])) + '/'

# MCMC Parameters
# Metallicity: -1.5 < [Z/H] < 0.2 steps of 0.1?
# Age: depends on galaxy, steps of 1 Gyr?
# IMF: x1 and x2 full range
# [Na/H]: -0.4 <-> +1.3

linefit = False
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

chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', \
        'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-','Mg+', 'Mg-', 'Si+', 'Si-', \
        'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
        'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

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

def model_spec(inputs, paramnames, vcjset = False, timing = False, full = False):
    '''Core function which takes the input model parameters, finds the appropriate models,
    and adjusts them for the input abundance ratios. Returns a broadened model spectrum 
    to be matched with a data spectrum.'''

    if vcjset:
        vcj = vcjset
    else:
        global vcj

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
        elif paramnames[j] == 'Mg':
            Mg = inputs[j]
        elif paramnames[j] == 'C':
            Carbon = inputs[j]
        elif paramnames[j] == 'Si':
            Si = inputs[j]
        elif paramnames[j] == 'Ti':
            Ti = inputs[j]
        elif paramnames[j] == 'Cr':
            Cr = inputs[j]
        elif paramnames[j] == 'VelDisp':
            veldisp = inputs[j]

    if 'Z' not in paramnames:
        Z = 0.0
    if 'x1' not in paramnames:
        x1 = 1.3
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
        print("MSPEC T1: ",t2 - t1)
        
    # If the Age of Z is inbetween models then this will average the respective models to produce 
    # one that is closer to what is expected.
    # NOTE THAT THIS IS AN ASSUMPTION AND THE CHANGE IN THE MODELS IS NOT NECESSARILY LINEAR
    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -1.0, -0.5, 0.0, 0.2])
    #abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3]
    abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3,8,7,18,17,14,13,21]
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

        if timing:
            t2_1 = time.time()
            print("MSPEC T1_1: ", t2_1 - t2)

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

        if timing:
            t2_2 = time.time()
            print("MSPEC T1_2: ", t2_2 - t2_1)

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

        if timing:
            t2_1 = time.time()
            print("MSPEC T1_1: ", t2_1 - t2)

        for i in abundi:
            interp1 = spi.interp2d(wl, [fullage[agem],fullage[agep]], \
                    np.stack((fc1[:,i],fc2[:,i])), kind = 'linear')
            c[:,i] = interp1(wl,Age)

        if timing:
            t2_2 = time.time()
            print("MSPEC T1_2: ", t2_2 - t2_1)
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

        if timing:
            t2_1 = time.time()
            print("MSPEC T1_1: ", t2_1 - t2)

        for i in abundi:
            interp1 = spi.interp2d(wl, [fullZ[zm],fullZ[zp]], \
                    np.stack((fc1[:,i],fc2[:,i])), kind = 'linear')
            c[:,i] = interp1(wl,Z)

        if timing:
            t2_2 = time.time()
            print("MSPEC T1_2: ", t2_2 - t2_1)
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
        print("MSPEC T2: ", t3 - t2)

    # Reminder of the abundance model columns
    # ['Solar', 'Na+', 'Na-',   'Ca+',  'Ca-', 'Fe+', 'Fe-', 'C+',  'C-',  'a/Fe+', 
    #  'N+',    'N-',  'as/Fe+','Ti+',  'Ti-', 'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 
    #  'T-',    'Cr+', 'Mn+',   'Ba+',  'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #  'V+',    'Cu+', 'Na+0.6','Na+0.9']

    # DETERMINING THE ABUNDANCE RATIO EFFECTS
    # The assumption here is that the abundance effects scale linearly
    # for all elements except sodium which covers a much wider range of abundance values.
    # The models are interpolated at the various given abundances to allow for abundance values to be continuous.
    # The interpolated value is then normalized by the solar metallicity model and then subtracted by 1 to 
    # retain the percentage
    #if fitmode in [True, False, 'NoAge', 'NoAgeVelDisp']:

    #Na adjustment
    ab_contribution = 0.0
    if 'Na' in paramnames:
        Naextend = (c[:,2]-c[:,0])*(-0.5/-0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.6,0.9], \
                np.stack((Naextend,c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        NaP = interp(wl,Na) / c[:,0] - 1.
        ab_contribution += NaP

    #K adjustment (assume symmetrical K adjustment)
    if 'K' in paramnames:
        Kminus = (2. - (c[:,29] / c[:,0]))*c[:,0]
        Kmextend = (Kminus - c[:,0])*(-0.5/-0.3) + c[:,0]
        Kpextend = (c[:,29] - c[:,0]) * (0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((Kmextend,Kminus,c[:,0],c[:,29],Kpextend)), kind = 'linear')
        KP = interp(wl,K) / c[:,0] - 1.
        ab_contribution += KP

    #Mg adjustment
    if 'Mg' in paramnames:
        mextend = (c[:,16] - c[:,0])*(-0.5/-0.3) + c[:,0]
        pextend = (c[:,15] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((mextend,c[:,16], c[:,0],c[:,15],pextend)), kind = 'linear')
        MgP = interp(wl,Mg) / c[:,0] - 1.
        ab_contribution += MgP

    #Fe Adjustment
    if 'Fe' in paramnames:
        Femextend = (c[:,6] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Fepextend = (c[:,5] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((Femextend,c[:,6], c[:,0],c[:,5],Fepextend)), kind = 'linear')
        FeP = interp(wl,Fe) / c[:,0] - 1.
        ab_contribution += FeP

    #Ca Adjustment
    if 'Ca' in paramnames:
        Cmextend = (c[:,4] - c[:,0])*(-0.5/-0.3) + c[:,0]
        Cpextend = (c[:,3] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((Cmextend,c[:,4], c[:,0],c[:,3],Cpextend)), kind = 'linear')
        CaP = interp(wl,Ca) / c[:,0] - 1.
        ab_contribution += CaP

    #C Adjustment
    if 'C' in paramnames:
        mextend = (c[:,8] - c[:,0])*(-0.3/-0.15) + c[:,0]
        pextend = (c[:,7] - c[:,0])*(0.3/0.15) + c[:,0]
        interp = spi.interp2d(wl, [-0.3,-0.15,0.0,0.15,0.3], \
                np.stack((mextend,c[:,8], c[:,0],c[:,7],pextend)), kind = 'linear')
        Abval = interp(wl,Carbon) / c[:,0] - 1.
        ab_contribution += Abval
    abundi = [0,1,2,-2,-1,29,16,15,6,5,4,3,8,7]

    #Si Adjustment
    if 'Si' in paramnames:
        mextend = (c[:,18] - c[:,0])*(-0.5/-0.3) + c[:,0]
        pextend = (c[:,17] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((mextend,c[:,18], c[:,0],c[:,17],pextend)), kind = 'linear')
        Abval = interp(wl,Si) / c[:,0] - 1.
        ab_contribution += Abval

    #Ti Adjustment
    if 'Ti' in paramnames:
        mextend = (c[:,14] - c[:,0])*(-0.5/-0.3) + c[:,0]
        pextend = (c[:,13] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((mextend,c[:,14], c[:,0],c[:,13],pextend)), kind = 'linear')
        Abval = interp(wl,Ti) / c[:,0] - 1.
        ab_contribution += Abval

    #Cr Adjustment
    if 'Cr' in paramnames:
        Crminus = (2. - (c[:,21] / c[:,0]))*c[:,0]
        mextend = (Crminus - c[:,0])*(-0.5/-0.3) + c[:,0]
        pextend = (c[:,21] - c[:,0])*(0.5/0.3) + c[:,0]
        interp = spi.interp2d(wl, [-0.5,-0.3,0.0,0.3,0.5], \
                np.stack((mextend,Crminus, c[:,0],c[:,21],pextend)), kind = 'linear')
        Abval = interp(wl,Cr) / c[:,0] - 1.
        ab_contribution += Abval

    #model_ratio = mimf / basemodel
    model_ratio = basemodel / mimf
    #model_ratio = 1.0

    # The model formula is as follows....
    # The new model is = the base IMF model * abundance effects. 
    # The abundance effect %ages are scaled by the ratio of the selected IMF model to the Kroupa IMF model
    # The formula ensures that if the abundances are solar then the base IMF model is recovered. 
    newm = mimf*(1. + (model_ratio*ab_contribution))

    if timing:
        print("MSPEC T3: ", time.time() - t3)

    return wl, newm

def calc_chisq(params, wl, data, err, paramnames,\
        linedefs, veldisp, plot=False, timing = False):
    ''' Important function that produces the value that essentially
    represents the likelihood of the mcmc equation. Produces the model
    spectrum then returns a normal chisq value.'''
    timing=True

    linelow, linehigh, bluelow, bluehigh, redlow, redhigh, line_name, \
            index_name, mlow, mhigh, morder = linedefs
    
    if timing:
        t1 = time.time()

    #Creating model spectrum then interpolating it so that it can be easily matched with the data.
    wlm, newm = model_spec(params, paramnames, timing=timing)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    if 'VelDisp' in paramnames:
        whsig = np.where(np.array(paramnames) == 'VelDisp')[0]
        wlc, mconv = mcsp.convolvemodels(wlm, newm, params[whsig])
    else:
        wlc, mconv = mcsp.convolvemodels(wlm, newm, veldisp)

    if 'f' in paramnames:
        whf = np.where(np.array(paramnames) == 'f')[0][0]
        f = params[whf]
    
    if timing:
        t2 = time.time()
        print("CHISQ T1: ", t2 - t1)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    if timing:
        t3 = time.time()
        print("CHISQ T2: ", t3 - t2)
    
    #Measuring the chisq
    chisq = 0
    for i in range(len(mlow)):
        #Getting a slice of the model
        wli = wl[i]
        dataslice = data[i]
        errslice = err[i]

        modelslice = mconvinterp(wli)

        if morder[i] == 1:
            k = np.where(np.array(index_name) == line_name[i])[0][0]
            linedefs = [bluelow,bluehigh,redlow,redhigh]
            polyfit_model = mcsp.removeLineSlope(wlc, mconv, linedefs, k)
            cont = polyfit_model(wli)

            #bluepass = np.where((wlc >= bluelow[k]) & (wlc <= bluehigh[k]))[0]
            #redpass = np.where((wlc >= redlow[k]) & (wlc <= redhigh[k]))[0]

            #Cacluating center value of the blue and red bandpasses
            #blueavg = np.mean([bluelow[k],bluehigh[k]])
            #redavg = np.mean([redlow[k],redhigh[k]])

            #blueval = np.mean(mconv[bluepass])
            #redval = np.mean(mconv[redpass])

            #pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
            #polyfit = np.poly1d(pf)

            polyfit_data = mcsp.removeLineSlope(wli,dataslice,linedefs,k)
            dataslice = dataslice / polyfit_data(wli)
        else:
            mdratio = modelslice / dataslice
            pf = np.polyfit(wli, mdratio, morder[i])
            polyfit = np.poly1d(pf)
            cont = polyfit(wli)

        modelslice = modelslice / cont

        if 'f' in paramnames:
            #errterm = (errslice ** 2.0) + (dataslice**2.0 * np.exp(2*f))
            #errterm = (errslice ** 2.0)*(1. + np.exp(2*f))
            errterm = (errslice ** 2.0) + (modelslice**2.0)*np.exp(2*f)
            addterm = np.log(2.0 * np.pi * errterm)
        else:
            errterm = errslice ** 2.0
            addterm = errslice * 0.0

        #Performing the chisq calculation
        chisq += np.sum(((dataslice - modelslice)**2.0 / errterm) + addterm)

        if plot:
            mpl.plot(wl[i], modelslice, 'r')
            mpl.plot(wl[i], dataslice, 'b')

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
        elif paramnames[j] in ['x1', 'x2']:
            if not (0.5 <= theta[j] <= 3.5):
                goodpriors = False
        elif paramnames[j] == 'Na':
            if not (-0.5 <= theta[j] <= 0.9):
                goodpriors = False
        elif paramnames[j] in ['K','Ca','Fe','Mg','Si','Ti','Cr']:
            if not (-0.5 <= theta[j] <= 0.5):
                goodpriors = False
        elif paramnames[j] == 'C':
            if not (-0.3 <= theta[j] <= 0.3):
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

def lnprob(theta, wl, data, err, paramnames, linedefs, veldisp):
    '''Primary function of the mcmc. Checks priors and returns the likelihood'''

    lp = lnprior(theta, paramnames)
    if not np.isfinite(lp):
        return -np.inf

    chisqv = calc_chisq(theta, wl, data, err, \
            paramnames, linedefs, veldisp, timing=False)
    return lp + chisqv

def do_mcmc(gal, nwalkers, n_iter, z, veldisp, paramnames, threads = 6, fl = None,\
        restart=False, scale=False):
    '''Main program. Runs the mcmc'''

    #Line definitions & other definitions
    #mlow = [9700,10550,11340,11550,12350,12665]
    #mhigh = [10450,10965,11447,12200,12590,13180]
    #morder = [8,4,1,7,2,5]

    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333]
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]
    index_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']

    #mlow = [9700,10550,11550,12350,12665]
    #mhigh = [10450,10965,12200,12590,13050]
    #morder = [8,4,7,2,5]
    mlow =   [9700,  10120, 11340, 11550, 12240, 12665]
    mhigh =  [10000, 10450, 11447, 12070, 12590, 13050]
    #morder = [3,3,1,5,4,5]
    morder = [5,5,1,7,6,7]
    #mlow =   [9855, 10300, 11340, 11550, 12350, 12665]
    #mhigh =  [9970, 10390, 11447, 12070, 12590, 13050]
    #morder = [1,1,1,5,2,4]
    line_name = ['Band1','Band2','NaI','Band3','Band4','Band5']

    linedefs = [linelow, linehigh, bluelow, bluehigh, redlow, redhigh,\
            line_name, index_name, mlow, mhigh, morder]

    ### INSERT YOUR SPECTRA LOADING SCRIPT HERE ######
    #wl, data, err = preps.preparespec(gal) ### NIFS
    if fl == None:
        print('Please input filename for the spectrum.')
        return
    wl, data, err = preps.preparespecwifis(fl, z)
    ##################################################

    if scale:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, \
                usecont = False, scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, \
                usecont = False)

    ndim = len(paramnames)
    print("Input fit parameters: ", paramnames)

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
                    #newinit.append(np.random.random()*1.3 - 0.3)
                    newinit.append(np.random.random()*0.9 - 0.3)
                elif paramnames[j] == 'C':
                    newinit.append(np.random.random()*0.3 - 0.15)
                elif paramnames[j] in ['K','Ca','Fe','Mg','Si','Ti','Cr']:
                    newinit.append(np.random.random()*0.6 - 0.3)
                elif paramnames[j] == 'VelDisp':
                    vdisp = np.random.random()*240 + 130
                    newinit.append(vdisp)
                elif paramnames[j] == 'f':
                    newinit.append(np.random.random())
            pos.append(np.array(newinit))
    else:
       realdata, postprob, infol, lastdata = mcsp.load_mcmc_file(restart)
       pos = lastdata

    savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_fullfit.dat" % (gal)
    f = open(savefl, "w")
    strparams = '\t'.join(paramnames)
    #strlines = '\t'.join(lineinclude)
    f.write("#NWalk\tNStep\tGal\tFit\n")
    f.write("#%d\t%d\t%s\n" % (nwalkers, n_iter,gal))
    f.write("#%s\n" % (strparams))
    #f.write("#%s\n" % (strlines))
    f.close()

    print("Starting MCMC...")
    pool = Pool(processes=threads)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = \
            (wl, data, err, paramnames, linedefs, veldisp), pool=pool)

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
    vcj = preload_vcj() #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    
    #params = ['Age','Z','x1','x2','Ca','Na','Fe','K','Mg','C','Ti','Cr','Si']
    params = ['Age','Z','x1','x2','Ca','Na','Fe','K','Mg','C','Ti','Si']
    sampler = do_mcmc('M85', 512, 10000, 0.00230, 157, params, threads = 16, \
        fl = '/data2/wifis_reduction/elliot/M85/20171229/science/processed/M85_combined_cube_1_telluricreduced_20171229_R1.fits')
