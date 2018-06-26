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

warnings.simplefilter('ignore', np.RankWarning)

# MCMC Parameters
# Metallicity: -1.5 < [Z/H] < 0.2 steps of 0.1?
# Age: depends on galaxy, steps of 1 Gyr?
# IMF: x1 and x2 full range
# [Na/H]: -0.4 <-> +1.3
Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
x1_m = 0.5 + np.arange(16)/5.0
x2_m = 0.5 + np.arange(16)/5.0
Na_m = np.arange(-0.4,1.0,0.1)
K_m = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
Mg_m = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
Fe_m = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])
Ca_m = np.array([-0.4,-0.3,-0.2,-0.1,0.0,0.1,0.2,0.3,0.4])

Z_pm = np.array(['m','m','m','m','p','p','p'])
ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

#base = '/Users/relliotmeyer/Thesis_Work/ssp_models/'
#base = '/Users/relliotmeyer/gemini2015a/mcmcgemini/'
#base = '/home/elliot/mcmcgemini/'
base = '/Users/relliotmeyer/mcmcgemini/'

imffiles = glob(base + 'widths/VCJ_v8_mcut0.08_t*')
chemfiles = glob(base + 'widths/atlas*')

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

imfsdict = {}

mlow = [9700,10550,11550,12350]
mhigh = [10450,11450,12200,13180]
morder = [8,9,7,8]


for i in range(16):
    for j in range(16):
        imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

def preload_vcj():

    chem_names = ['WL', 'Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-',\
            'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
            'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
            'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

    print "PRELOADING SSP MODELS INTO MEMORY"
    fls = glob(base+'spec/vcj_ssp/*')    
    vcj = {}
    for fl in fls:
        flspl = fl.split('/')[-1]
        x = pd.read_table(fl, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj[flspl] = x[:,1:]

    print "PRELOADING ABUNDANCE MODELS INTO MEMORY"
    fls = glob(base+'spec/atlas/*')    
    for fl in fls:
        flspl = fl.split('/')[-1]
        x = pd.read_table(fl, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
        x = np.array(x)
        vcj[flspl] = x[:,1:]
    vcj["WL"] = x[:,0]

    print "FINISHED LOADING MODELS"

    return vcj

vcj = preload_vcj()

def select_model_file(Z, Age, mixZ, mixage, noZ = False):

    Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
    Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
    x1_m = 0.5 + np.arange(16)/5.0
    x2_m = 0.5 + np.arange(16)/5.0
    Z_pm = np.array(['m','m','m','m','p','p','p'])
    ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

    if noZ:
        if mixage:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        

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

    return fl1, cfl1, fl2, cfl2

def model_spec(inputs, gal, masklines = False, smallfit = False):

    #t1 = time.time()
    if smallfit == True:
        Age, x1, x2, Na, K, Ca, Fe = inputs
        Z = 0.0
    elif smallfit == 'limited':
        Z, Age, x1, x2 = inputs
    else:
        Z, Age, x1, x2, Na, K, Mg, Fe, Ca = inputs

    Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
    Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])
    x1_m = 0.5 + np.arange(16)/5.0
    x2_m = 0.5 + np.arange(16)/5.0
    Z_pm = np.array(['m','m','m','m','p','p','p'])
    ChemAge_m = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13])

    fullage = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])
    fullZ = np.array([-1.5, -0.5, 0.0, 0.2])
    
    if Z not in Z_m:
        Zmin = np.argmin(np.abs(Z_m - Z))
        Z = Z_m[Zmin]
    if Age not in Age_m:
        Agemin = np.argmin(np.abs(Age_m - Age))
        Age = Age_m[Agemin]
    if x1 not in x1_m:
        x1min = np.argmin(np.abs(x1_m - x1))
        x1 = x1_m[x1min]
    if x2 not in x2_m:
        x2min = np.argmin(np.abs(x2_m - x2))
        x2 = x2_m[x2min]

    mixage = False
    mixZ = False
    if Age not in fullage:
        mixage = True
    if Z not in fullZ:
        mixZ = True

    if smallfit in [False, 'limited']:
        noZ = False
    else:
        noZ = True
    fl1, cfl1, fl2, cfl2 = select_model_file(Z, Age, mixZ, mixage, noZ = noZ)
    #t2 = time.time()
    #print t2 - t1
        
    if mixage or mixZ:
        #wlm1, fm1 = read_ssp(base+'spec/vcj_ssp/'+fl1, 'vcj', imfsdict[(x1,x2)])
        fm1 = vcj[fl1]
        wlc1 = vcj["WL"]
        fc1 = vcj[cfl1]
        #wlc1, fc1 = read_ssp(base+'spec/atlas/'+cfl1, 'atlas')
        fm2 = vcj[fl2]
        #wlm2, fm2 = read_ssp(base+'spec/vcj_ssp/'+fl2, )
        fc2 = vcj[cfl2]
        #wlc2, fc2 = read_ssp(base+'spec/atlas/'+cfl2)

        rel = np.where((wlc1 > 8500) & (wlc1 < 14000))[0]

        m = np.zeros(fm1.shape)
        c = np.zeros(fc1.shape)

        for i in range(fm1.shape[1]):
            m[:,i] = (fm1[:,i] + fm2[:,i]) / 2.0
        #mimf = (fm1[:,imfsdict[(x1,x2)]] + fm2[:, imfsdict[(x1,x2)]]) / 2.0
        for i in range(fc1.shape[1]):
            c[:,i] = (fc1[:,i] + fc2[:,i]) / 2.0
        c = c[rel,:]
        mimf = m[rel,imfsdict[(x1,x2)]]
        wl = wlc1

    else:
        #wlm1, m = read_ssp(base+'spec/vcj_ssp/'+fl1)
        m = vcj[fl1]
        wlc1 = vcj["WL"]
        c = vcj[cfl1]

        wl = wlc1
        rel = np.where((wlc1 > 8500) & (wlc1 < 14000))[0]

        #wlc1, c = read_ssp(base+'spec/atlas/'+cfl1)
        mimf = m[rel,imfsdict[(x1,x2)]]
        c = c[rel,:]

    #t3 = time.time()
    #print t3 - t2

    # ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
    #                'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #                'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
    # Z, Age, x1, x2, Na, K, Mg, Fe, Ca = theta 

    wl = wl[rel]
    if type(smallfit) == bool:
        interp = spi.interp2d(wl, [-0.3,0.0,0.3,0.6,0.9], np.stack((c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        NaP = interp(wl,Na) / c[:,0] - 1.

        interp = spi.interp2d(wl, [0.0,0.3], np.stack((c[:,0],c[:,29])), kind = 'linear')
        KP = interp(wl,K) / c[:,0] - 1.

        if not smallfit:
            interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,16], c[:,0],c[:,15])), kind = 'linear')
            MgP = interp(wl,Mg) / c[:,0] - 1.

        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,6], c[:,0],c[:,5])), kind = 'linear')
        FeP = interp(wl,Fe) / c[:,0] - 1.

        #if not smallfit:
        interp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,4], c[:,0],c[:,3])), kind = 'linear')
        CaP = interp(wl,Ca) / c[:,0] - 1.

    basemodel = m[rel,73]
    #mimf = mimf[rel]

    model_ratio = mimf / basemodel

    if smallfit == True:
        newm = mimf*(1. + model_ratio*(NaP + KP + CaP + FeP))
    elif smallfit == 'limited':
        newm = newm
    else:
        newm = mimf*model_ratio*(1. + NaP + KP + MgP + CaP + FeP)
        
    if gal == 'M85':
        wlc, mconv = convolvemodels(wl, newm, 176.)
    elif gal == 'M87':
        wlc, mconv = convolvemodels(wl, newm, 360.)
    
    #print time.time() - t3
    return wlc, mconv

def chisq(params, wl, data, err, gal, smallfit):

    if gal == 'M87':
        masklines = ['KI_1.25']
    else:
        masklines = False

    wlc, mconv = model_spec(params, gal, masklines=masklines, smallfit = smallfit)
    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    #wlnew, mnew, errnew = splitspec(wlc, mconv, err=False)
    
    chisq = 0
    for i in range(len(mlow)):
        modelslice = mconvinterp(wl[i])
        pf = np.polyfit(wl[i], modelslice, morder[i])
        polyfit = np.poly1d(pf)
        cont = polyfit(wl[i])
        modelslice = modelslice / cont

        chisq += np.sum((modelslice - data[i])**2.0 / err[i]**2.0)
        #mpl.plot(wl[i], modelslice, 'r')
        #mpl.plot(wl[i], data[i], 'b')
    #mpl.show()

    return -0.5*chisq

def lnprior(theta, smallfit):

    if smallfit == True:
        Age, x1, x2, Na, K, Ca, Fe = theta 
        if (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
                (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3) and (-0.3 <= Ca <= 0.3) and\
                (-0.3 <= Fe <= 0.3):
            return 0.0

    elif smallfit == 'limited':
        Z, Age, x1, x2 = theta 
        if (-1.5 <= Z <= 0.2) and (7.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
                (0.5 <= x2 <= 3.5):
            return 0.0

    else:
        Z, Age, x1, x2, Na, K, Mg, Fe, Ca = theta 
        if (-1.5 <= Z <= 0.2) and (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
                (0.5 <= x2 <= 3.5) and (-0.3 <= Na <= 0.9) and (-0.3 <= K <= 0.3) and (-0.3 <= Mg <= 0.3) and\
                (-0.3 <= Fe <= 0.3) and (-0.3 <= Ca <= 0.3):
            return 0.0
    return -np.inf

def lnprob(theta, wl, data, err, gal, smallfit):
    lp = lnprior(theta, smallfit)
    if not np.isfinite(lp):
        return -np.inf
    t2 = time.time()
    chisqv = chisq(theta, wl, data, err, gal, smallfit)
    return lp + chisqv

def do_mcmc(gal, nwalkers, n_iter, smallfit = False, threads = 6):

    wl, data, err = preparespec(gal)
    wl, data, err = splitspec(wl, data, err = err)

    if smallfit == True:
        ndim = 7
    elif smallfit == 'limited':
        ndim = 4
    else:
        ndim = 9

    pos = []
    for i in range(nwalkers):
        newinit = []
        if smallfit != True:
            newinit.append(np.random.choice(Z_m))
        newinit.append(np.random.choice(Age_m))
        newinit.append(np.random.choice(x1_m))
        newinit.append(np.random.choice(x2_m))
        if smallfit != 'limited':
            newinit.append(np.random.random()*1.3 - 0.3)
            newinit.append(np.random.random()*0.6 - 0.3)
            newinit.append(np.random.random()*0.6 - 0.3)
            newinit.append(np.random.random()*0.6 - 0.3)
        if not smallfit:
            newinit.append(np.random.random()*0.6 - 0.3)
        pos.append(np.array(newinit))


    savefl = base + "mcmcresults/"+time.strftime("%Y%m%dT%H%M%S")+"_%s_fullfit.dat" % (gal)
    f = open(savefl, "w")
    f.write("#NWalk\tNStep\tGal\tFit\n")
    f.write("#%d\t%d\t%s\t%s\n" % (nwalkers, n_iter,gal,str(smallfit)))
    f.close()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (wl, data, err, gal, smallfit), threads=threads)
    print "Starting MCMC..."

    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        position = result[0]
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            f.write("%d\t%s\t%s\n" % (k, " ".join(map(str, position[k])), result[1][k]))
        f.close()

        if (i+1) % 1 == 0:
            print (time.time() - t1) / 60., " Minutes"
            print (i+1.)*100. / float(n_iter), "% Finished\n"
    
    return sampler

def load_mcmc_file(fl):
    '''Loads the chain file (fl) as output by the do_mcmc program. Returns
    the walker data, the probability data, and the run info.'''

    f = open(fl,'r')
    f.readline()
    info = f.readline()
    f.close()
    info = info[1:]
    values = info.split()
    nworkers = int(values[0])
    niter = int(values[1])
    gal = values[2]
    fit = values[3]
    infol = [nworkers, niter, gal, fit]

    data = np.loadtxt(fl, skiprows = 2)
    folddata = data.reshape((niter, nworkers,data.shape[1]))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]

    return [realdata, postprob, infol]

def preparespec(galaxy):

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

def splitspec(wl, data, err=False):

    mlow = [9700,10550,11550,12350]
    mhigh = [10450,11450,12200,13180]
    morder = [8,9,7,8]

    databands = []
    wlbands = []
    errorbands = []
    for i in range(len(mlow)):
        wh = np.where( (wl >= mlow[i]) & (wl <= mhigh[i]))[0]
        dataslice = data[wh]
        wlbands.append(wl[wh])

        pf = np.polyfit(wl[wh], dataslice, morder[i])
        polyfit = np.poly1d(pf)
        cont = polyfit(wl[wh])
        databands.append(dataslice / cont)
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

    reg = (wlfull > 9500) & (wlfull < 13500)
    
    wl = wlfull[reg]
    data = datafull[reg]

    c = 299792.458

    #Measuring on KIa line
    mainpass = np.where((wl >= bluelow[3]) & (wl <= redhigh[3]))[0]

    linedata = data[mainpass]
    linewl = wl[mainpass]

    try:
        popt, pcov = gf.gaussian_fit_os(linewl, linedata, [-0.005, 5.0,11693, 0.0695])
        #popt, pcov = gf.gaussian_fit_os(linewl, linedata, [-0.005, 5.0 ,12525, 0.064])
        #Sigma measured from fit
        m_sigma = popt[1]
        m_center = popt[2]
    except:
        return wl, data

    #Sigma from description of models
    m_sigma = np.abs(m_center / (1 + 100./c) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    
    sigma_gal = (m_center * (veldisp/c + 1.)) - m_center
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    convolvex = np.arange(-4*sigma_conv,4*sigma_conv, 2)
    gaussplot = gf.gauss_nat(convolvex, [sigma_conv,0.])

    out = np.convolve(datafull, gaussplot, mode='same')

    return wl, out[reg]

if __name__ == '__main__':
    #sampler = do_mcmc('M85', 512, 30000, smallfit = True, threads = 6)
    sampler = do_mcmc('M85', 100, 3000, smallfit = True, threads = 6)


