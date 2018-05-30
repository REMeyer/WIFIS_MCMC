import numpy as np
from astropy.io import fits
from sys import exit
from glob import glob
import pandas as pd
import emcee
import time
import matplotlib.pyplot as mpl

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

modelloc = '/home/elliot/mcmcgemini/widths/'

imffiles = glob(modelloc + 'VCJ_v8_mcut0.08_t*')
chemfiles = glob(modelloc + 'atlas*')

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

#base = '/Users/relliotmeyer/Thesis_Work/ssp_models/widths/'
base = modelloc

imfsdict = {}
for i in range(16):
    for j in range(16):
        imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

def load_eqw(fl):
 
    data = pd.read_table(fl, names=['Model','Line','Raw','Error','M85','M87']) 
    return data

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
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge+1])       
                cfl1 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge+1])            
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge+1])
                cfl1 = 'atlas_ssp_t0%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge+1])
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge-1])
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (Age_m[whAge+1])
                cfl1 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge-1])
                cfl2 = 'atlas_ssp_t%1i_Zp0.0.abund.krpa.txt' % (ChemAge_m[whAge+1]) 
        else:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''        
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Zp0.0.ssp.imf_varydoublex.txt' % (agemod, Age,)
            cfl1 = 'atlas_ssp_t%s_Zp0.0.abund.krpa.txt' % (agemod + str(ChemAge_m[whAge]))
            fl2 = ''
            cfl2 = ''
    else:
        if mixage and mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]
            
            if Age < 9.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]), abs(Z_m[whZ+1])) 
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))          
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]),  abs(Z_m[whZ+1]))
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))          
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], \
                        str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], \
                        str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))
                cfl1 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))           
        elif mixage and not mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        

            if Age < 9.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))       
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z))            
            elif Age == 10.0:
                fl1 = 'VCJ_v8_mcut0.08_t0%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))
                cfl1 = 'atlas_ssp_t0%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z))
            else:
                fl1 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                fl2 = 'VCJ_v8_mcut0.08_t%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (Age_m[whAge+1], str(Z_pm[whZ]), abs(Z))
                cfl1 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge-1], str(Z_pm[whZ]), abs(Z))
                cfl2 = 'atlas_ssp_t%1i_Z%s%.1f.abund.krpa.txt' % (ChemAge_m[whAge+1], str(Z_pm[whZ]), abs(Z)) 
        elif not mixage and mixZ:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (agemod,Age, str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
            fl2 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (agemod,Age, str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))      
            cfl1 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.txt' % (agemod+str(ChemAge_m[whAge]), str(Z_pm[whZ-1]), abs(Z_m[whZ-1]))
            cfl2 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.txt' % (agemod+str(ChemAge_m[whAge]), str(Z_pm[whZ+1]), abs(Z_m[whZ+1]))
        else:
            whAge = np.where(Age_m == Age)[0][0]
            whZ = np.where(Z_m == Z)[0][0]        
            if Age < 10:
                agemod = '0'
            else:
                agemod = ''        
            fl1 = 'VCJ_v8_mcut0.08_t%s%.1f_Z%s%.1f.ssp.imf_varydoublex.txt' % (agemod, Age, str(Z_pm[whZ]), abs(Z))
            cfl1 = 'atlas_ssp_t%s_Z%s%.1f.abund.krpa.txt' % (agemod + str(ChemAge_m[whAge]), str(Z_pm[whZ]), abs(Z))
            fl2 = ''
            cfl2 = ''

    return fl1, cfl1, fl2, cfl2

def model_width(inputs, gal, masklines = False, smallfit = False):

    if smallfit:
        Age, x1, x2, Na, K, Mg, Fe = inputs
        Z = 0.0
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

    fl1, cfl1, fl2, cfl2 = select_model_file(Z, Age, mixZ, mixage, noZ = smallfit)
        
    if mixage or mixZ:
        w1 = load_eqw(base+fl1)
        w2 = load_eqw(base+fl2)
        c1 = load_eqw(base+cfl1)
        c2 = load_eqw(base+cfl2)

        w = pd.DataFrame(data = w1)
        c = pd.DataFrame(data = c1)

        if gal == 'M85':
            w.M85 = (w1.M85 + w2.M85) / 2.0
            c.M85 = (c1.M85 + c2.M85) / 2.0
        else:
            w.M87 = (w1.M87 + w2.M87) / 2.0
            c.M87 = (c1.M87 + c2.M87) / 2.0
    else:
        w = load_eqw(base+fl1)
        c = load_eqw(base+cfl1)

    # ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
    #                'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
    #                'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
    # Z, Age, x1, x2, Na, K, Mg, Fe, Ca = theta 
     
    modellines = []    
    for line in line_name:
        if masklines:
            if line in masklines:
                continue
        linearr = np.array(c[gal][c.Line == line])
        #NaP.append((linearr[1]-linearr[0])/linearr[0]/3.)
        #NaM.append((linearr[2]-linearr[0])/linearr[0]/3.)
        #KP.append((linearr[29]-linearr[0])/linearr[0]/3.)
        #MgP.append((linearr[15]-linearr[0])/linearr[0]/3.)
        #MgM.append((linearr[16]-linearr[0])/linearr[0]/3.)
        #FeP.append((linearr[5]-linearr[0])/linearr[0]/3.)
        #FeM.append((linearr[6]-linearr[0])/linearr[0]/3.)
        pf = np.polyfit([-0.3,0.0,0.3,0.6,0.9], [linearr[2], linearr[1], linearr[0], linearr[-2], linearr[-1]], 3)
        polyfit = np.poly1d(pf)
        NaP = (polyfit(Na) - linearr[0]) / linearr[0]
        #if Na >= 0:
        #    NaP = (linearr[1]-linearr[0])/linearr[0]/0.3
        #else:
        #    NaP = (linearr[2]-linearr[0])/linearr[0]/0.3

        pf = np.polyfit([0.0,0.3], [linearr[0], linearr[29]], 1)
        polyfit = np.poly1d(pf)
        KP = (polyfit(K) - linearr[0]) / linearr[0]
        #KP = (linearr[29]-linearr[0])/linearr[0]/0.3

        pf = np.polyfit([-0.3,0.0,0.3], [linearr[16], linearr[0], linearr[15]], 2)
        polyfit = np.poly1d(pf)
        MgP = (polyfit(Mg) - linearr[0]) / linearr[0]
        #if Mg >= 0:
        #    MgP = (linearr[15]-linearr[0])/linearr[0]/0.3
        #else:
        #    MgP = (linearr[16]-linearr[0])/linearr[0]/0.3

        pf = np.polyfit([-0.3,0.0,0.3], [linearr[6], linearr[0], linearr[5]], 2)
        polyfit = np.poly1d(pf)
        FeP = (polyfit(Fe) - linearr[0]) / linearr[0]
        #if Fe >= 0:
        #    FeP = (linearr[5]-linearr[0])/linearr[0]/0.3
        #else:
        #    FeP = (linearr[6]-linearr[0])/linearr[0]/0.3

        if not smallfit:
            pf = np.polyfit([-0.3,0.0,0.3], [linearr[4], linearr[0], linearr[3]], 2)
            polyfit = np.poly1d(pf)
            CaP = (polyfit(Ca) - linearr[0]) / linearr[0]
        #if Ca >= 0:
        #    CaP = (linearr[3]-linearr[0])/linearr[0]/0.3
        #else:
        #    CaP = (linearr[4]-linearr[0])/linearr[0]/0.3

        lwidths = np.array(w[gal][w.Line == line])
        basewidth = lwidths[73]
        imfwidth = lwidths[imfsdict[(x1,x2)]]
        #chemratio = linearr[0] / basewidth
        imfratio = basewidth / imfwidth

        #newwidth = imfwidth*(1 + imfratio*(NaP*Na + K*KP + Mg*MgP + Fe*FeP))
        if smallfit:
            newwidth = imfwidth*(1 + imfratio*(NaP + KP + MgP + FeP))
        else:
            newwidth = imfwidth*(1 + imfratio*(NaP + KP + MgP + FeP + CaP))

        modellines.append(newwidth)

    return modellines

def chisq(params, width, err, gal, smallfit):

    if gal == 'M87':
        masklines = ['KI_1.25']
    else:
        masklines = False

    widthmodels = model_width(params, gal, masklines=masklines, smallfit = smallfit)
    
    chisq = 0
    for i,w in enumerate(widthmodels):
        chisq += (w - width[i])**2.0 / err[i]**2.0

    return -1*chisq

def lnprior(theta, smallfit):

    Z_m = np.array([-1.5,-1.0, -0.5, -0.25, 0.0, 0.1, 0.2])
    Age_m = np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,11.0,12.25,13.5])

    if smallfit:
        Age, x1, x2, Na, K, Mg, Fe = theta 
        if (9.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
                (0.5 <= x2 <= 3.5) and (-0.4 <= Na <= 1.0) and (-0.4 <= K <= 0.4) and (-0.4 <= Mg <= 0.4) and\
                (-0.4 <= Fe <= 0.4):
            return 0.0
    else:
        Z, Age, x1, x2, Na, K, Mg, Fe, Ca = theta 
        if (-1.5 <= Z <= 0.2) and (1.0 <= Age <= 13.5) and (0.5 <= x1 <= 3.5) and\
                (0.5 <= x2 <= 3.5) and (-0.4 <= Na <= 1.0) and (-0.4 <= K <= 0.4) and (-0.4 <= Mg <= 0.4) and\
                (-0.4 <= Fe <= 0.4) and (-0.4 <= Ca <= 0.4):
            return 0.0
    return -np.inf

def lnprob(theta, y, yerr, gal, smallfit):
    lp = lnprior(theta, smallfit)
    if not np.isfinite(lp):
        return -np.inf
    return lp + chisq(theta, y, yerr, gal, smallfit)

def do_mcmc(gal):

    fl = '/home/elliot/mcmcgemini/eqwidths.txt'
    yfull = load_eqw(fl)
    if gal == 'M85':
        y = np.array(yfull.M85[yfull.Model == 'M85'])
        yerr = np.array(yfull.Error[yfull.Model == 'M85'])
    elif gal =='M87':
        y = np.array(yfull.M87[yfull.Model == 'M87'])
        yerr = np.array(yfull.Error[yfull.Model == 'M87'])

    nwalkers = 500
    n_iter = 5000
    smallfit = True
    if smallfit:
        ndim = 7
    else:
        ndim = 9

    pos = []
    for i in range(nwalkers):
        newinit = []
        if not smallfit:
            newinit.append(np.random.choice(Z_m))
        newinit.append(np.random.choice(Age_m))
        #newinit.append(np.random.choice(np.array([-0.25, 0.0,0.1, 0.1])))
        #newinit.append(np.random.choice(np.array([1.0,2.0,3.0,4.0,5.0,6.0,7.0])))
        newinit.append(np.random.choice(x1_m))
        newinit.append(np.random.choice(x2_m))
        newinit.append(np.random.random()*1.4 - 0.4)
        newinit.append(np.random.random()*0.8 - 0.4)
        newinit.append(np.random.random()*0.8 - 0.4)
        newinit.append(np.random.random()*0.8 - 0.4)
        if not smallfit:
            newinit.append(np.random.random()*0.8 - 0.4)
        pos.append(np.array(newinit))

    #savefl = "/Users/relliotmeyer/Desktop/chainM87.dat"
    savefl = "/home/elliot/mcmcresults/result.dat"
    f = open(savefl, "w")
    f.close()

    threads = 6
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args = (y, yerr, gal, smallfit), threads=threads)
    print "Starting MCMC..."

    t1 = time.time() 
    for i, result in enumerate(sampler.sample(pos, iterations=n_iter)):
        position = result[0]
        f = open(savefl, "a")
        for k in range(position.shape[0]):    
            f.write("%d\t%s\n" % (k, " ".join(map(str, position[k]))))
        print zip(range(len(result)), result)
        f.close()
        
        #if (i - 4) == 0:
        #    print "Finished first three iterations...running"
        #    print

        if (i+1) % 100 == 0:
            #position = result[0]
            #for k in range(position.shape[0]):
            #    print "%d\t%s" % (k, " ".join(map(str, position[k])))
            print (time.time() - t1) / 60., " Minutes"
            print (i+1.)*100. / float(n_iter), "% Finished\n"
    
    #sampler.run_mcmc(pos, 5000)

    return sampler

def newchemfile():
    np.set_printoptions(precision=4)

    Z = ['m1.5','m1.0','m0.5','p0.0','p0.2']
    for met in Z:
        fl1 = '/Users/relliotmeyer/Thesis_Work/ssp_models/atlas/atlas_ssp_t05_Z'+met+'.abund.krpa.s100'
        fl2 = '/Users/relliotmeyer/Thesis_Work/ssp_models/atlas/atlas_ssp_t09_Z'+met+'.abund.krpa.s100'

        fulldata1 = np.loadtxt(fl1, skiprows=2)
        fulldata2 = np.loadtxt(fl2, skiprows=2)

        newdata = np.zeros(fulldata1.shape)
        newdata[:,0] = fulldata1[:,0]
        for i in range(1,len(newdata[0,:])):
            meanspec = (fulldata1[:,i] + fulldata2[:,i]) / 2.0
            newdata[:,i] = meanspec
        np.savetxt('/Users/relliotmeyer/Thesis_Work/ssp_models/atlas/atlas_ssp_t07_Z'+met+'.abund.krpa.s100', newdata\
                , header= '# lam, Solar, Na+, Na-, Ca+, Ca-, Fe+, Fe-, C+, C-, a/Fe+, N+, N-, as/Fe+, Ti+, Ti-, Mg+, Mg-, Si+, Si-, T+, T-, Cr+, Mn+, Ba+, Ba-, Ni+, Co+, Eu+, Sr+, K+, V+, Cu+, Na+0.6, Na+0.9\n# Kroupa IMF, 13.5 Gyr; +/- is 0.3 dex except for C which is 0.15 dex; T+/- is 50K', fmt = ['%.3f','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E','%.4E'])

if __name__ == '__main__':
    sampler = do_mcmc('M87')
    #newchemfile()
    #for i in Z_m:
    #    for j in Age_m:
    #        print model_width(i, j, 1.3, 1.3, 0.7, 0.2, 0.2, 0.2, 0.2, 'M85')
    #print model_width(-1.5, 2.0, 1.3, 1.3, 0.7, 0.2, 0.2, 0.2, 0.2, 'M85')







        


