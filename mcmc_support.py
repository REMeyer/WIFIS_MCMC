import numpy as np
import scipy.optimize as spo
import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd

linefit = True
#Line definitions & other definitions
linelow = [9905,10337,11372,11680,11765,12505,13115]
linehigh = [9935,10360,11415,11705,11793,12545,13165]

bluelow = [9855,10300,11340,11667,11710,12460,13090]
bluehigh = [9880,10320,11370,11680,11750,12495,13113]

redlow = [9940,10365,11417,11710,11793,12555,13165]
redhigh = [9970,10390,11447,11750,11810,12590,13175]

chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']

def gaussian(xs, a, sigma, x0):
    second = ((xs - x0)/sigma)**2.0
    full = a*np.exp((-1.0/2.0) * second)
    
    return full

def gaussian_os(xs, a, sigma, x0, b):
    second = ((xs - x0)/sigma)**2.0
    full = b + a*np.exp((-1.0/2.0) * second)
        
    return full

def gaussian_fit_os(xdata,ydata,p0, gaussian=gaussian_os):
    '''Performs a very simple gaussian fit using scipy.optimize.curvefit. Returns 
        the fit parameters and the covariance matrix'''

    #first = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    #second = ((x - x0)/sigma)**2.0
        
    #gaussian = lambda x,a,sigma,x0: a * np.exp((-1.0/2.0) * ((x - x0)/sigma)**2.0)
    popt, pcov = spo.curve_fit(gaussian, xdata, ydata, p0=p0)
    return [popt, pcov]

def gauss_nat(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    return  (1 / (2.*np.pi*p0[0]**2.))* np.exp((-1.0/2.0) * ((xs - p0[1])/p0[0])**2.0)

def load_mcmc_file(fl):
    '''Loads the chain file (fl) as output by the do_mcmc program. Returns
    the walker data, the probability data, and the run info.'''

    f = open(fl,'r')
    f.readline()
    info = f.readline()
    firstline = f.readline()
    n_values = len(firstline.split())

    #Get line count to diagnose
    lc = 1
    for line in f:
        lc += 1
    f.close()
    info = info[1:]
    values = info.split()
    nworkers = int(values[0])
    niter = int(values[1])
    gal = values[2]
    fitmode = values[3]

    if fitmode == 'True':
        names = ['Age','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]']
    elif fitmode == 'False':
        names = ['[Z/H]','Age','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Mg/H]','[Fe/H]']
    elif fitmode == 'NoAge':
        names = ['x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]']
    else:
        names = ['[Z/H]','Age','x1','x2']

    names.insert(0,"Worker")
    names.insert(len(names), "ChiSq")
    print "MODE: ", fitmode

    #N lines should be nworkers*niter
    n_lines = nworkers*niter
    if lc < nworkers:
        print "FILE DOES NOT HAVE ONE STEP...RETURNING"
        return
    elif lc % nworkers != 0:
        print "FILE HAS INCOMPLETE STEP...REMOVING"
        n_steps = int(lc / nworkers)
        initdata = pd.read_table(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        #initdata = np.loadtxt(fl)
        data = np.array(initdata)
        data = data[:n_steps*nworkers,:]
    elif lc != n_lines:
        print "FILE NOT COMPLETE"
        initdata = pd.read_table(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        data = np.array(initdata)
        #data = np.loadtxt(fl)
        n_steps = int(data.shape[0]/nworkers)
    else:
        initdata = pd.read_table(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        data = np.array(initdata)
        #data = np.loadtxt(fl)
        n_steps = niter

    names = names[1:-1]
    infol = [nworkers, niter, gal, fitmode, names]

    folddata = data.reshape((n_steps, nworkers,len(names)+2))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]
    lastdata = realdata[-1,:,:]
    print "DATASHAPE: ", realdata.shape

    return [realdata, postprob, infol, lastdata]

def bestfitPrepare(fl, burnin):

    dataall = load_mcmc_file(fl)
    data = dataall[0]
    fitmode = dataall[2][3]
    names = dataall[2][4]
    gal = dataall[2][2]

    if gal == 'M85':
        galveldisp = 145.
    elif gal == 'M87':
        galveldisp = 370.

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"
    #infol = [nworkers, niter, gal, fitmode, names]

    if fitmode == 'limited':
        mcmctype = 'Base Params'
    elif fitmode == 'True':
        mcmctype = 'Abundance Fit'
        fitmode = True
    elif fitmode == 'False':
        mcmctype = 'Full Fit'
        fitmode = False
    elif fitmode == 'NoAge':
        mcmctype = 'NoAge'

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    fitvalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    fitvalues = np.array(fitvalues)
    truevalues = fitvalues[:,0]

    return [data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp]

def compare_bestfit(fl, burnin=-1000, onesigma = False, addshift = False, vcjset = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = mcsp.bestfitPrepare(fl, burnin)

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)

    convr = np.arange(120.,395.,25.)
    plot = True
    if plot: 
        fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
        axes = axes.flatten()
        fig.delaxes(axes[-1])

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.05)
    #shifts = [1.75,-1.35,-1.3,-0.9,0.45,0.35,-1.35]

    for i in range(len(mlow)):
        if (gal == 'M87') and linefit:
            if line_name[i] == 'KI_1.25':
                continue

        #Getting a slice of the model
        wli = wl[i]

        if addshift:
            #wli += shifts[i]
            if i in [3,4]:
                wli = np.array(wli)
                wli -= 2.0
            elif i == 5:
                wli = np.array(wli)
                wli += 1.0
            elif i == 1:
                wli = np.array(wli)
                wli += 1.0

        modelslice = mconvinterp(wli)
        
        #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
        #Removing a high-order polynomial from the slice
        if onesigma:
            plusvalues = np.array(truevalues)
            negvalues = np.array(truevalues)

            #if i == 0:
            #    plusvalues[5] += fitvalues[5,1]
            #    negvalues[5] -= fitvalues[5,2]
            #elif i == 1:
            #    plusvalues[4] += fitvalues[4,1]
            #    negvalues[4] -= fitvalues[4,2]
            #elif i == 2:
            #    plusvalues[2] += fitvalues[2,1]
            #    negvalues[2] -= fitvalues[2,2]
            #elif i in [3,4,5]:
            #    plusvalues[3] += fitvalues[3,1]
            #    negvalues[3] -= fitvalues[3,2]
            #plusvalues[0] += fitvalues[0,1]
            #negvalues[0] -= fitvalues[0,2]
            plusvalues[1] += fitvalues[1,1]
            negvalues[1] -= fitvalues[1,2]

            wlgneg, newmneg = model_spec(negvalues, gal, fitmode = fitmode, vcjset = vcjset)
            wlgplus, newmplus = model_spec(plusvalues, gal, fitmode = fitmode, vcjset = vcjset)

        if linefit:
            if onesigma:
                wlcneg, mconvneg = convolvemodels(wlgneg, newmneg, galveldisp)
                mconvinterpneg = spi.interp1d(wlcneg, mconvneg, kind='cubic', bounds_error=False)

                wlcplus, mconvplus = convolvemodels(wlgplus, newmplus, galveldisp)
                mconvinterpplus = spi.interp1d(wlcplus, mconvplus, kind='cubic', bounds_error=False)

                polyfitneg = removeLineSlope(wlcneg, mconvneg,i)
                contneg = polyfitneg(wli)

                polyfitplus = removeLineSlope(wlcplus, mconvplus,i)
                contplus = polyfitplus(wli)

            polyfit = removeLineSlope(wlc, mconv,i)
            cont = polyfit(wli)

        else:
            pf = np.polyfit(wl[i], modelslice, morder[i])
            polyfit = np.poly1d(pf)
            cont = polyfit(wl[i])

        modelslice = modelslice / cont
        
        if plot:
            axes[i].plot(wl[i], modelslice, 'r')
            if onesigma:
                modelsliceneg = mconvinterpneg(wli) / contneg
                modelsliceplus = mconvinterpplus(wli) / contplus
                axes[i].plot(wl[i], modelsliceneg, 'g')
                axes[i].plot(wl[i], modelsliceplus, 'm')
            axes[i].plot(wl[i], data[i],'b')
            axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
            axes[i].set_title(line_name[i])
            if linefit:
                axes[i].set_xlim((bluelow[i],redhigh[i]))
    if plot:
        mpl.show()

def deriveShifts(fl, burnin = -1000, vcjset = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = bestfitPrepare(fl, burnin)

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.1)
    #shifts = [1.75,-1.35,-1.3,-0.9,0.45,0.35,-1.35]

    for sh in shifts: 
        fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
        axes = axes.flatten()
        fig.delaxes(axes[-1])
        for i in range(len(wl)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = wl[i]
            wli += sh

            modelslice = mconvinterp(wli)
            
            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            if linefit:
                polyfit = mcfs.removeLineSlope(wlc, mconv,i)
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wl[i], modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wl[i])

            modelslice = modelslice / cont
            
            axes[i].plot(wl[i], modelslice, 'r')
            axes[i].plot(wl[i], data[i],'b')
            axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
            axes[i].set_title(line_name[i] + ' '+ str(np.std(data[i] / modelslice)))
            if linefit:
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        mpl.show()

if __name__ == '__main__':
    vcj = mcfs.preload_vcj() #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    deriveShifts('20180807T031027_M85_fullfit.dat', vcjset = vcj)
