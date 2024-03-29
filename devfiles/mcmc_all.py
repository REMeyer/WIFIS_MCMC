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

#Setting some of the mcmc priors
Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
x1_m = 0.5 + np.arange(16)/5.0
x2_m = 0.5 + np.arange(16)/5.0

Z_pm = np.array(['m','m','m','m','p','p','p'])
ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

#Line definitions & other definitions
linelow = [9905,10337,11372,11680,11765,12505,13115]
linehigh = [9935,10360,11415,11705,11793,12545,13165]

bluelow = [9855,10300,11340,11667,11710,12460,13090]
bluehigh = [9880,10320,11370,11680,11750,12495,13113]

redlow = [9940,10365,11417,11710,11793,12555,13165]
redhigh = [9970,10390,11447,11750,11810,12590,13175]

line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

#Definitions for the fitting bandpasses
linefit = True
if linefit:
    mlow = [9855,10300,11340,11667,11710,12460,13090]
    mhigh = [9970,10390,11447,11750,11810,12590,13175]
    #mlow = [9905,10337,11372,11680,11765,12505,13115]
    #mhigh = [9935,10360,11415,11705,11793,12545,13165]
    morder = [1,1,1,1,1,1,1]
else:
    mlow = [9700,10550,11340,11550,12350]
    mhigh = [10450,10993,11447,12200,13180]
    morder = [8,9,1,7,8]

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
    global base

    if overwrite_base:
        base = overwrite_base

    chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-',\
            'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    print "PRELOADING SSP MODELS INTO MEMORY"
    fls = glob(base+'spec/vcj_ssp/*')    
    #vcj = {}
    for fl in fls:
        print fl
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
        print fl
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

def select_model_file(Z, Age, fitmode):
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
    if Z not in Z_m:
        Zmin = np.argmin(np.abs(Z_m - Z))
        Z = Z_m[Zmin]
    if Age not in Age_m:
        Agemin = np.argmin(np.abs(Age_m - Age))
        Age = Age_m[Agemin]

    mixage = False
    mixZ = False
    if Age not in fullage:
        mixage = True
    if Z not in fullZ:
        mixZ = True

    if fitmode in [False, 'limited', 'LimitedVelDisp', 'NoAgeLimited']:
        noZ = False
    else:
        noZ = True

    #Long set of if-elif to determine the proper model file...unoptimized.
    if noZ:
        if mixage:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        
            #fl1 = "%.1f_%.1f" % (Age_m[whAge-1],0.0)
            #fl2 = "%.1f_%.1f" % (Age_m[whAge+1],0.0)

            if Age < 9.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge+1])       
                cfl1 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge+1])            
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge+1])
                cfl1 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge+1])
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (Age_m[whAge+1])
                cfl1 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.s100' % (ChemAge_m[whAge+1]) 
        else:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''        
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Zp0.0.ssp.imf_varydoublex.s100' % (agemod, Age,)
            cfl1 = 'atlas_ssp_t%s_Zp0.0.abund.krpa.s100' % (agemod + str(ChemAge_m[whAge]))
            fl2 = ''
            cfl2 = ''
    else:
        if mixage and mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]
            
            if Age < 9.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]), abs(Z_m[whZ+1])) 
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))          
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]),  abs(Z_m[whZ+1]))
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))          
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))
                cfl1 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))           
        elif mixage and not mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        

            if Age < 9.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))       
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z))            
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z))
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))
                cfl1 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.s100' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z)) 

        elif not mixage and mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (agemod,Age, str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
            fl2 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (agemod,Age, str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))      
            cfl1 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.s100' % (agemod+str(ChemAge_m[whAge]), str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
            cfl2 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.s100' % (agemod+str(ChemAge_m[whAge]), str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))
        else:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''        
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.s100' % (agemod, Age, str(Z_pm[whZ]), abs(Z))
            cfl1 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.s100' % (agemod + str(ChemAge_m[whAge]), str(Z_pm[whZ]), abs(Z))
            fl2 = ''
            cfl2 = ''

    return fl1, cfl1, fl2, cfl2, mixage, mixZ

def select_model_file_new(Z, Age):
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
    if Z not in Z_m:
        Zmin = np.argmin(np.abs(Z_m - Z))
        Z = Z_m[Zmin]
    if Age not in Age_m:
        Agemin = np.argmin(np.abs(Age_m - Age))
        Age = Age_m[Agemin]

    mixage = False
    mixZ = False
    if Age not in fullage:
        mixage = True
    if Z not in fullZ:
        mixZ = True

    #if fitmode in [False, 'limited', 'LimitedVelDisp', 'NoAgeLimited']:
    #    noZ = False
    #else:
    #    noZ = True
    #    Z = 0.0

    whAge = np.where(Age_m == Age)[0][0]
    whZ = np.where(Z_m == Z)[0][0]        

    if mixage and mixZ:
        fl1 = "%.1f_%.1f" % (Age_m[whAge-1],Z_m[whZ-1])
        fl2 = "%.1f_%.1f" % (Age_m[whAge+1],Z_m[whZ-1])
        fl3 = "%.1f_%.1f" % (Age_m[whAge-1],Z_m[whZ+1])
        fl4 = "%.1f_%.1f" % (Age_m[whAge+1],Z_m[whZ+1])
    elif mixage:
        fl1 = "%.1f_%.1f" % (Age_m[whAge-1],Z)
        fl2 = "%.1f_%.1f" % (Age_m[whAge+1],Z)
        fl3 = ''
        fl4 = ''
    elif mixZ:
        fl1 = "%.1f_%.1f" % (Age,Z_m[whZ-1])
        fl2 = "%.1f_%.1f" % (Age,Z_m[whZ+1])
        fl3 = ''
        fl4 = ''
    else:
        fl1 = "%.1f_%.1f" % (Age,Z)
        fl2 = ''
        fl3 = ''
        fl4 = ''

    return fl1, fl2, fl3, fl4, whAge, whZ, mixage, mixZ

def select_model_file_new_2(Z, Age):
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

def model_spec(inputs, gal, paramnames, vcjset = False, timing = False):
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

    #Determining the fitting parameters from the fitmode variable
    #if fitmode == True:
    #    Age, x1, x2, Na, K, Ca, Fe = inputs
    #    Z = 0.0
    #elif fitmode == 'limited':
    #    Z, Age, x1, x2 = inputs
    #elif fitmode == 'LimitedVelDisp':
    #    Z, Age, x1, x2,veldisp = inputs
    #elif fitmode == 'NoAge':
    #    x1, x2, Na, K, Ca, Fe = inputs
    #    Z = 0.0
    #    if gal == 'M85':
    #        Age = 5.0
    #    elif gal == 'M87':
    #        Age = 13.5
    #elif fitmode == 'NoAgeVelDisp':
    #    x1, x2, Na, K, Ca, Fe, veldisp = inputs
    #    Z = 0.0
    #    if gal == 'M85':
    #        Age = 5.0
    #    elif gal == 'M87':
    #        Age = 13.5
    #else:
    #    Z, Age, x1, x2, Na, K, Mg, Fe, Ca = inputs

    if x1 not in x1_m:
        x1min = np.argmin(np.abs(x1_m - x1))
        x1 = x1_m[x1min]
    if x2 not in x2_m:
        x2min = np.argmin(np.abs(x2_m - x2))
        x2 = x2_m[x2min]

    #Finding the appropriate base model files.
    #fl1, cfl1, fl2, cfl2, mixage, mixZ = select_model_file(Z, Age, fitmode)
    #fl1, fl2, fl3, fl4, whAge, whZ, mixage, mixZ = select_model_file_new_2(Z, Age)
    fl1, fl2, fl3, fl4, agem, agep, zm, zp, mixage, mixZ = select_model_file_new_2(Z, Age)

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
        rel = np.where((wlc1 > 8500) & (wlc1 < 14500))[0]

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
        rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]

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
        rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]

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

        rel = np.where((wlc1 > 6500) & (wlc1 < 16000))[0]

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
        interp = spi.interp2d(wl, [-0.3,0.0,0.3,0.6,0.9], np.stack((c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        NaP = interp(wl,Na) / c[:,0] - 1.
        ab_contribution += NaP

        #K adjustment (assume symmetrical K adjustment)
    if 'K' in paramnames:
        Kminus = (2. - (c[:,29] / c[:,0]))*c[:,0]
        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((Kminus,c[:,0],c[:,29])), kind = 'linear')
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
        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,6], c[:,0],c[:,5])), kind = 'linear')
        FeP = interp(wl,Fe) / c[:,0] - 1.
        ab_contribution += FeP

    if 'Ca' in paramnames:
        #Ca Adjustment
        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,4], c[:,0],c[:,3])), kind = 'linear')
        CaP = interp(wl,Ca) / c[:,0] - 1.
        ab_contribution += CaP

    model_ratio = mimf / basemodel

    # The model formula is as follows....
    # The new model is = the base IMF model * abundance effects. 
    # The abundance effect %ages are scaled by the ratio of the selected IMF model to the Kroupa IMF model
    # The formula ensures that if the abundances are solar then the base IMF model is recovered. 
    newm = mimf*(1. + model_ratio*ab_contribution)

    #if fitmode in [True, 'NoAge', 'NoAgeVelDisp']:
    #    newm = mimf*(1. + model_ratio*(NaP + KP + CaP + FeP))
    #elif fitmode in ['limited', 'LimitedVelDisp']:
    #    newm = mimf
    #else:
    #    newm = mimf*(1. + model_ratio(NaP + KP + MgP + CaP + FeP))
        
    if timing:
        print "MSPEC T3: ", time.time() - t3

    return wl, newm

def chisq(params, wl, data, err, gal, paramnames, plot=False, timing = False, widths = False):
    ''' Important function that produces the value that essentially
    represents the likelihood of the mcmc equation. Produces the model
    spectrum then returns a normal chisq value.'''

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
            wlc, mconv = convolvemodels(wlm, newm, 170.)
        elif gal == 'M87':
            #wlc, mconv = convolvemodels(wlm, newm, 308.)
            wlc, mconv = convolvemodels(wlm, newm, 370.)
    
    if timing:
        t2 = time.time()
        print "CHISQ T1: ", t2 - t1

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    if timing:
        t3 = time.time()
        print "CHISQ T2: ", t3 - t2
    
    #Measuring the chisq
    chisq = 0

    if not widths:
        for i in range(len(mlow)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = wl[i]
            if (gal == 'M85') and linefit:
                if i in [3,4]:
                    wli = np.array(wli)
                    wli -= 2.0
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
                pf = np.polyfit(wli, modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wli)

            #Normalizing the model
            modelslice = modelslice / cont

            #Performing the chisq calculation
            if linefit and (line_name[i] == 'FeH'):
                whbad = (wl[i] > 9894) & (wl[i] < 9908)
                wg = np.logical_not(whbad)
                chisq += np.sum((modelslice[wg] - data[i][wg])**2.0 / (err[i][wg]**2.0))
            else:
                chisq += np.sum((modelslice - data[i])**2.0 / (err[i]**2.0))

            if plot:
                mpl.plot(wl[i], modelslice, 'r')
                mpl.plot(wl[i], data[i], 'b')
    else:
		for i in range(len(bluelow)):
			eqw_model = measure_line(wlc, mconv, i, errors = False)       #wlc, mconv
        	chisq += (eqw_model - data[i])**2.0 / err[i]**2.0
        
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
            if not (-0.3 <= theta[j] <= 0.9):
                goodpriors = False
        elif paramnames[j] in ['K','Ca','Fe','Mg']:
            if not (-0.3 <= theta[j] <= 0.3):
                goodpriors = False
        elif paramnames[j] == 'VelDisp':
            if not (120 <= theta[j] <= 390):
                goodpriors = False

    if goodpriors == True:
        return 0.0
    else:
        return -np.inf

    #if fitmode == True:
    #    Age, x1, x2, Na, K, Ca, Fe = theta 
    #    if (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
    #            (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3) and (-0.3 <= Ca <= 0.3) and\
    #            (-0.3 <= Fe <= 0.3):
    #        return 0.0

    #elif fitmode == 'limited':
    #    Z, Age, x1, x2 = theta 
    #    if (-1.5 <= Z <= 0.2) and (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
    #            (0.5 <= x2 <= 3.5):
    #        return 0.0

    #elif fitmode == 'LimitedVelDisp':
    #    Z, Age, x1, x2, veldisp = theta 
    #    if (-1.5 <= Z <= 0.2) and (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
    #            (0.5 <= x2 <= 3.5) and (120 <= veldisp <= 390):
    #        return 0.0

    #elif fitmode == 'NoAge':
    #    x1, x2, Na, K, Ca, Fe = theta 
    #    if (0.5 <= x1 <= 3.5) and (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3)\
    #            and (-0.3 <= Ca <= 0.3) and (-0.3 <= Fe <= 0.3):
    #        return 0.0

    #elif fitmode == 'NoAgeVelDisp':
    #    x1, x2, Na, K, Ca, Fe, veldisp = theta 
    #    if (0.5 <= x1 <= 3.5) and (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3)\
    #            and (-0.3 <= Ca <= 0.3) and (-0.3 <= Fe <= 0.3) and (120 <= veldisp <= 390):
    #        return 0.0

    #else:
    #    Z, Age, x1, x2, Na, K, Mg, Fe, Ca = theta 
    #    if (-1.5 <= Z <= 0.2) and (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
    #            (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3) and (-0.3 <= Mg <= 0.3) and\
    #            (-0.3 <= Fe <= 0.3) and (-0.3 <= Ca <= 0.3):
    #        return 0.0
    #return -np.inf

def lnprob(theta, wl, data, err, gal, paramnames):
    '''Primary function of the mcmc. Checks priors and returns the likelihood'''

    lp = lnprior(theta, paramnames)
    if not np.isfinite(lp):
        return -np.inf

    chisqv = chisq(theta, wl, data, err, gal, paramnames)
    return lp + chisqv

def lnprobwidth(theta, y, yerr, gal, paramnames):
    lp = lnprior(theta, paramnames)

    if not np.isfinite(lp):
        return -np.inf

    return lp + chisq(theta, [], y, yerr, gal, paramnames, widths = True)

def do_mcmc(gal, nwalkers, n_iter, paramnames, threads = 6, restart=False, scale=False, widths = False):
    '''Main program. Runs the mcmc'''

    if widths:
        fl = base + 'widths/eqwidths.txt'
        yfull = load_eqw(fl)
        if gal == 'M85':
            y = np.array(yfull.M85[yfull.Model == 'M85'])
            yerr = np.array(yfull.Error[yfull.Model == 'M85'])
        elif gal =='M87':
            y = np.array(yfull.M87[yfull.Model == 'M87'])
            yerr = np.array(yfull.Error[yfull.Model == 'M87'])
    else:
        wl, data, err = preparespec(gal)
        if scale:
            wl, data, err = splitspec(wl, data, err = err, lines=linefit, scale = scale)
            #wl, data, err = splitspec(wl, data, err = err, lines=linefit)
        else:
            wl, data, err = splitspec(wl, data, err = err, lines=linefit)

    ndim = len(paramnames)

    pos = []
    if not restart:
        for i in range(nwalkers):
            newinit = []
            for j in range(len(paramnames)):
                if paramnames[j] == 'Age':
                    #newinit.append(np.random.choice(Age_m))
                    newinit.append(np.random.random()*12.5 + 1.0)
                elif paramnames[j] == 'Z':
                    #newinit.append(np.random.choice(Z_m))
                    newinit.append(np.random.random()*0.45 - 0.25)
                elif paramnames[j] in ['x1', 'x2']:
                    newinit.append(np.random.choice(x1_m))
                elif paramnames[j] == 'Na':
                    newinit.append(np.random.random()*1.3 - 0.3)
                elif paramnames[j] in ['K','Ca','Fe','Mg']:
                    newinit.append(np.random.random()*0.6 - 0.3)
                elif paramnames[j] == 'VelDisp':
                    newinit.append(np.random.random()*240 + 120)
            pos.append(np.array(newinit))
    else:
       realdata, postprob, infol, lastdata = mcsp.load_mcmc_file(restart)
       pos = lastdata

    if widths:
        savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_widthfit.dat" % (gal)
    else:
        savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_fullfit.dat" % (gal)
    f = open(savefl, "w")
    strparams = '\t'.join(paramnames)
    f.write("#NWalk\tNStep\tGal\tFit\n")
    f.write("#%d\t%d\t%s\n" % (nwalkers, n_iter,gal))
    f.write("#%s\n" % (strparams))
    f.close()

    if widths:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprobwidth, args = (y, yerr, gal, paramnames), threads=threads)
    else:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (wl, data, err, gal, paramnames), threads=threads)
    print "Starting MCMC..."

    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        position = result[0]
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result[1][k]))
        f.close()

        if (i+1) % 10 == 0:
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
        global base

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
    wlj, dataj = nm.skymask(wlj, dataj, galaxy, 'J')

    finalwl = np.concatenate((wlz,wlj))
    finaldata = np.concatenate((dataz,dataj))
    finalerr = np.concatenate((errorsz,errorsj))

    return finalwl, finaldata, finalerr

def splitspec(wl, data, err=False, lines = False, scale = False):

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

            if scale:
                data[fullpass] -= scale*polyfit(wl[fullpass])

                blueval = np.mean(data[bluepass])
                redval = np.mean(data[redpass])
                pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                polyfit = np.poly1d(pf)
                cont = polyfit(wlslice)

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

        #if scale:
        #    dataslice -= scale*cont

        databands.append(data[wh] / cont)

        if type(err) != bool:
            errslice = err[wh]
            errorbands.append(errslice / cont)
    
    return wlbands, databands, errorbands

def read_ssp(fl):

    chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-',\
            'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    spl = fl.split('/')[-1]
    spll = spl.split('_')[0]
    if spll == 'VCJ':
        #x = np.loadtxt(fl)
        x = pd.read_table(fl, delim_whitespace = True, header=None)
        x = np.array(x)
        return x[:,1:]
    elif spll == 'atlas':
        #x = np.loadtxt(fl, skiprows = 2)
        x = pd.read_table(fl, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
        x = np.array(x)
        return np.array(x[:,0]), x[:,1:]
    else:
        print "MODEL READ FAIL"
        return

def convolvemodels(wlfull, datafull, veldisp):

    reg = (wlfull >= 9500) & (wlfull <= 13500)
    
    wl = wlfull[reg]
    data = datafull[reg]

    c = 299792.458

    #Measuring on KIa line
    #mainpass = np.where((wl >= bluelow[3]) & (wl <= redhigh[3]))[0]

    #linedata = data[mainpass]
    #linewl = wl[mainpass]
    #mpl.plot(linewl, linedata)
    #mpl.show()

    #try:
    #    popt, pcov = mcsp.gaussian_fit_os(linewl, linedata, [-0.005, 5.0, 11693, 0.0695])
        #popt, pcov = gf.gaussian_fit_os(linewl, linedata, [-0.005, 5.0 ,12525, 0.064])
        #Sigma measured from fit
    #    m_sigma = popt[1]
    #    m_center = popt[2]
    #except:
    #    print "LINEFIT DIDNT WORK -- GAUSSIAN"
    #    return wl, data

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
    #mpl.plot(out)
    #mpl.plot(datafull)
    #mpl.show()
    
    #print len(wl), len(out), len(reg), len(out[reg])
    #return wl, out[reg]
    return wlfull, out

def test_chisq(paramnames, vcj, trials = 5, timing = True):
    d = preparespec('M85')
    wl, data, err = splitspec(d[0], d[1], err = d[2], scale = False, lines = linefit)

    for i in range(trials):
        newinit = []
        for j in range(len(paramnames)):
            if paramnames[j] == 'Age':
                #newinit.append(np.random.choice(Age_m))
                newinit.append(np.random.random()*12.5 + 1.0)
            elif paramnames[j] == 'Z':
                #newinit.append(np.random.choice(Z_m))
                newinit.append(np.random.random()*0.45 - 0.25)
            elif paramnames[j] in ['x1', 'x2']:
                newinit.append(np.random.choice(x1_m))
            elif paramnames[j] == 'Na':
                newinit.append(np.random.random()*1.3 - 0.3)
            elif paramnames[j] in ['K','Ca','Fe','Mg']:
                newinit.append(np.random.random()*0.6 - 0.3)
            elif paramnames[j] == 'VelDisp':
                newinit.append(np.random.random()*240 + 120)

        inputs = newinit

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

        if x1 not in x1_m:
            x1min = np.argmin(np.abs(x1_m - x1))
            x1 = x1_m[x1min]
        if x2 not in x2_m:
            x2min = np.argmin(np.abs(x2_m - x2))
            x2 = x2_m[x2min]

        #Finding the appropriate base model files.
        #fl1, cfl1, fl2, cfl2, mixage, mixZ = select_model_file(Z, Age, fitmode)
        #fl1, fl2, fl3, fl4, whAge, whZ, mixage, mixZ = select_model_file_new_2(Z, Age)
        fl1, fl2, fl3, fl4, agem, agep, zm, zp, mixage, mixZ = select_model_file_new_2(Z, Age)
        return fl1,fl2,fl3,fl4,agem,agep,zm,zp,mixage,mixZ, Age, Z
            #chisq(mockdata, wl, data, err, 'M85', fitmode, plot=False, timing = True)

    return mockdata, wl, data, err, vcj

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

def measure_line(wl, spec, i, errors = False, contcorrect=False):

    #Define the bandpasses for each line 
    bluepass = np.where((wl >= bluelow[i]) & (wl <= bluehigh[i]))[0]
    redpass = np.where((wl >= redlow[i]) & (wl <= redhigh[i]))[0]
    mainpass = np.where((wl >= linelow[i]) & (wl <= linehigh[i]))[0]
    fullpass = np.where((wl >= bluelow[i]) & (wl <= redhigh[i]))[0]

    #Cacluating center value of the blue and red bandpasses
    blueavg = np.mean([bluelow[i],bluehigh[i]])
    redavg = np.mean([redlow[i],redhigh[i]])

    #Calculate the mean of each continuum bandpass for fitting
    blueval = np.mean(spec[bluepass])
    redval = np.mean(spec[redpass])

    if errors:
        blueerrorarr = errors[bluepass]
        rederrorarr = errors[redpass]
        errorarr = errors[mainpass]
        eqwerrprop = 0
        meanSN = np.median(spec[mainpass] / errorarr)
        #conterror = np.mean([bluestd, redstd]) / np.sqrt(len(data[bluepass]))

    if contcorrect:
        bluevalM87 = 0.15*blueval
        redvalM87 = 0.15*redval
        #blueval -= bluevalM87
        #redval -= redvalM87

        #Do the linear fit
        pf = np.polyfit([blueavg, redavg], [bluevalM87, redvalM87], 1)
        polyfit = np.poly1d(pf)
        cont = polyfit(wl[fullpass])

        #linedata -= cont
        spec[fullpass] -= cont

    #The main line bandpass
    linewl = wl[mainpass]
    linedata = spec[mainpass]
    
    #Do the linear fit
    pf = np.polyfit([blueavg, redavg], [blueval, redval], 1)
    polyfit = np.poly1d(pf)
    cont = polyfit(linewl)
    
    if errors:
        bluecont = polyfit(wl[bluepass])
        redcont = polyfit(wl[redpass])
        bluestd = rms(bluecont - spec[bluepass])
        redstd = rms(redcont - spec[redpass])
        conterr = np.max([bluestd,redstd])

    #Calculate the equivalent width
    eqw = 0
    for j,k in enumerate(mainpass):
        eqw += (1 - (linedata[j]/cont[j])) * (wl[k+1] - wl[k])
        if errors:
            eqwerrprop += ((errorarr[j]**2 / cont[j]**2) + ((conterr*linedata[j])/cont[j]**2)**2) * (wl[k+1] - wl[k])**2

    if errors:
		eqwerrprop = np.sqrt(eqwerrprop)
		return eqw, eqwerrprop
    else:
		return eqw

def rms(a):

    lena = len(a)
    rms = 0
    for val in a:
        rms += val**2.0

    return np.sqrt(rms / lena)

def load_eqw(fl):
 
    data = pd.read_table(fl, names=['Model','Line','Raw','Error','M85','M87']) 
    return data

if __name__ == '__main__':
    vcj = preload_vcj() #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    #ret = test_chisq(['Age','Z','x1','x2'],vcj, trials = 1)
    #test_chisq('NoAge',trials = 2)

    sampler = do_mcmc('M85', 512, 25000, ['Age','x1','x2','Na','Fe','Ca','K'], threads = 18, widths = True)
    sampler = do_mcmc('M87', 512, 25000, ['Z', 'Age','x1','x2'], threads = 18, widths = True)
    #sampler = do_mcmc('M87', 512, 25000, ['Age','Z','x1','x2','Na'], threads = 18, scale = 0.15)
    #sampler = do_mcmc('M87', 512, 25000, ['Age','x1','x2','Fe','Na'], threads = 18)
    #sampler = do_mcmc('M87', 512, 25000, ['Age','Z','x1','x2'], threads = 18)
    #sampler = do_mcmc('M87', 512, 20000, fitmode = '', threads = 18)
    #compare_bestfit('20180807T031027_M85_fullfit.dat')
    #compare_bestfit('20180727T104340_M87_fullfit.dat')
