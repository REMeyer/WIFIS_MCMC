import numpy as np
import scipy.optimize as spo
import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd
import plot_corner as pc
import imf_mass as imf

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

#Line definitions & other definitions
#linelow = [9905,10337,11372,11680,11765,12505,13115]
#linehigh = [9935,10360,11415,11705,11793,12545,13165]

#bluelow = [9855,10300,11340,11667,11710,12460,13090]
#bluehigh = [9880,10320,11370,11680,11750,12495,13113]

#redlow = [9940,10365,11417,11710,11793,12555,13165]
#redhigh = [9970,10390,11447,11750,11810,12590,13175]

linelow = [9905,10337,11372,11680,11765,12505,13115, 12810, 12670]
linehigh = [9935,10360,11415,11705,11793,12545,13165, 12840, 12690]

bluelow = [9855,10300,11340,11667,11710,12460,13090, 12780, 12648]
bluehigh = [9880,10320,11370,11680,11750,12495,13113, 12800, 12660]

redlow = [9940,10365,11417,11710,11793,12555,13165, 12855, 12700]
redhigh = [9970,10390,11447,11750,11810,12590,13175, 12880, 12720]

line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI', 'PaB', 'NaI127']
nicenames = [r'FeH',r'CaI',r'NaI',r'KI a',r'KI b', r'KI 1.25', r'AlI', r'PaB', r'NaI127']

mlow = [9855,10300,11340,11667,11710,12460,13090,12780, 12648]
mhigh = [9970,10390,11447,11750,11810,12590,13175, 12880, 12720]
morder = [1,1,1,1,1,1,1,1]

chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
#line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']

#Definitions for the fitting bandpasses
linefit = True
#if linefit:
#    mlow = [9855,10300,11340,11667,11710,12460,13090]
#    mhigh = [9970,10390,11447,11750,11810,12590,13175]
    #mlow = [9905,10337,11372,11680,11765,12505,13115]
    #mhigh = [9935,10360,11415,11705,11793,12545,13165]
#    morder = [1,1,1,1,1,1,1]
#else:
#    mlow = [9700,10550,11340,11550,12350,12665]
#    mhigh = [10450,10965,11447,12200,12590,13180]
#    morder = [8,4,1,7,2,5]

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
    secondline = f.readline()

    if secondline[0] == '#':
        legacy = False
        paramstr = firstline[1:]
        linestr = secondline[1:]
        nextline = f.readline()
        n_values = len(nextline.split())
    else:
        legacy = True
        paramstr = firstline[1:]
        linestr = []
        n_values = len(secondline.split())

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

    if not legacy:
        paramnames = paramstr.split()
        linenames = linestr.split()
    else:
        paramnames = paramstr.split()
        linenames = []

    names = []
    high = []
    low = []
    for j in range(len(paramnames)):
        if paramnames[j] == 'Age':
            names.append('Age')
            high.append(13.5)
            low.append(1.0)
        elif paramnames[j] == 'Z':
            names.append('[Z/H]')
            high.append(0.2)
            #low.append(-1.5)
            low.append(-0.25)
        elif paramnames[j] in ['x1', 'x2']:
            if paramnames[j] == 'x1':
                names.append('x1')
            else:
                names.append('x2')
            high.append(3.5)
            low.append(0.5)
        elif paramnames[j] == 'Na':
            names.append('[%s/H]' % (paramnames[j]))
            high.append(0.9)
            low.append(-0.5)
        elif paramnames[j] in ['K','Ca','Fe','Mg']:
            names.append('[%s/H]' % (paramnames[j]))
            high.append(0.5)
            low.append(-0.5)
        elif paramnames[j] == 'VelDisp':
            names.append('$\sigma$')
            high.append(390)
            low.append(120)
        elif paramnames[j] == 'f':
            names.append('f')
            high.append(1.0)
            low.append(-1.0)

    names.insert(0,"Worker")
    names.insert(len(names), "ChiSq")
    print "Params: ", paramnames
    if not legacy:
        print "Lines: ", linenames

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
    for k,name in enumerate(names):
        if name == 'x1':
            names[k] = r'\textbf{$x_{1}$}'
        if name == 'x2':
            names[k] = r'\textbf{$x_{2}$}'

    infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, legacy]

    folddata = data.reshape((n_steps, nworkers,len(names)+2))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]
    lastdata = realdata[-1,:,:]
    print "DATASHAPE: ", realdata.shape

    return [realdata, postprob, infol, lastdata]

def bestfitPrepare(fl, burnin):

    #infol = [nworkers, niter, gal, names, high, low, paramnames,linenames,legacy]
    #return [realdata, postprob, infol, lastdata]
    dataall = load_mcmc_file(fl)
    data = dataall[0]
    names = dataall[2][3]
    high = dataall[2][4]
    low = dataall[2][5]
    legacy = dataall[2][8]
    gal = dataall[2][2]
    paramnames = dataall[2][6]
    linenames = dataall[2][7]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"

    print names
    print data.shape

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    fitvalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    fitvalues = np.array(fitvalues)
    truevalues = fitvalues[:,0]

    return [data, names, gal, datatype, fitvalues, truevalues, paramnames, linenames]

def compare_bestfit(fl, instrument = 'nifs', burnin=-1000, onesigma = False, addshift = False, vcjset = False):

    #Load the necessary information
    data, names, gal, datatype, fitvalues, truevalues,paramnames, linenames = bestfitPrepare(fl, burnin)

    if gal == 'M85':
        #galveldisp = 145.
        #galveldisp = 161.
        galveldisp = 170
    elif gal == 'M87':
        galveldisp = 370.
        #galveldisp = 307.

    if instrument == 'nifs':
        wl, data, err = mcfs.preparespec(gal)
    else:
        wl, data, err = mcfs.preparespecwifis(gal)

    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, paramnames, vcjset = vcjset)

    #convr = np.arange(120.,395.,25.)
    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()

    fig2, axes2 = mpl.subplots(2,5,figsize = (16,6.5))
    axes2 = axes2.flatten()
    #fig.delaxes(axes[-1])

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    for i in range(len(mlow)):
        if (gal == 'M87') and linefit:
            if line_name[i] == 'KI_1.25':
                continue

        print line_name[i]

        if line_name[i] not in linenames:
            print "Skipping "+line_name[i]
            continue

        #Getting a slice of the model
        wli = wl[i]

        if addshift and (instrument == 'nifs'):
            if i in [3,4]:
                wli = np.array(wli)
                wli -= 2.0

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

                polyfitneg = mcfs.removeLineSlope(wlcneg, mconvneg,i)
                contneg = polyfitneg(wli)

                polyfitplus = mcfs.removeLineSlope(wlcplus, mconvplus,i)
                contplus = polyfitplus(wli)

            polyfit = mcfs.removeLineSlope(wlc, mconv,i)
            cont = polyfit(wli)

        else:
            if i == 2:
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

        modelslice = modelslice / cont
        
        #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
        if onesigma:
            modelsliceneg = mconvinterpneg(wli) / contneg
            modelsliceplus = mconvinterpplus(wli) / contplus
            axes[i].plot(wl[i], modelsliceneg, 'g')
            axes[i].plot(wl[i], modelsliceplus, 'm')

        axes[i].plot(wl[i], data[i],'b')
        axes[i].plot(wl[i], modelslice, 'r')
        axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
        axes[i].set_title(nicenames[i])
        axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.25)

        axes2[i].plot(wl[i], (data[i]/modelslice) - 1,'b')
        axes2[i].fill_between(wl[i],(data[i] + err[i])/modelslice - 1 ,(data[i]-err[i])/modelslice - 1, facecolor = 'gray', alpha=0.5)
        axes2[i].set_title(nicenames[i])
        axes2[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.25)
        axes2[i].axhspan(-0.01,0.01, facecolor='red', alpha = 0.5)

        if linefit:
            axes[i].set_xlim((bluelow[i],redhigh[i]))
            axes2[i].set_xlim((bluelow[i],redhigh[i]))

    fig.savefig(fl[:-4] + '_bestfitlines.pdf')
    fig2.savefig(fl[:-4] + '_bestfitresiduals.pdf')

def deriveShifts(fl, burnin = -1000, vcjset = False, plot = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = bestfitPrepare(fl, burnin)

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.1)
    shifts = [0.9,-0.8,-1.5,-2.8,-1.1,1.0,-0.6]

    shfit = np.zeros((len(shifts),7))
    for j,sh in enumerate(shifts): 
        if plot:
            fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
            #fig.suptitle(str(sh))
            axes = axes.flatten()
            fig.delaxes(axes[-1])
        for i in range(len(wl)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = np.array(wl[i])
            wli += sh
            #wli += shifts[i]

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
            shfit[j,i] = np.std(data[i] / modelslice)
            
            if plot:
                axes[i].plot(wl[i], modelslice, 'r')
                axes[i].plot(wl[i], data[i],'b')
                axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
                axes[i].set_title(line_name[i] + ' %.2f' % (np.std(data[i] / modelslice)))
                if linefit:
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

        if plot:
            mpl.show()
    for i in range(len(wl)):
        print shifts[np.argmin(shfit[:,i])]

    return shfit

def deriveVelDisp(gal):

    wl, data, err = mcfs.preparespec(gal)
    mpl.plot(wl,data)
    mpl.show()
    sys.exit()

    mlow = [9700,10550,11550,12350]
    mhigh = [10450,11450,12200,13180]
    morder = [8,9,7,8]

    databands = []
    wlbands = []
    errorbands = []

    for i in range(len(mlow)):
        wh = np.where( (wl >= mlow[i]) & (wl <= mhigh[i]))[0]
        dataslice = data[wh]
        wlslice = wl[wh]
        wlbands.append(wlslice)

        pf = np.polyfit(wlslice, dataslice, morder[i])
        polyfit = np.poly1d(pf)
        cont = polyfit(wlslice)

        databands.append(dataslice / cont)

        if type(err) != bool:
            errslice = err[wh]
            errorbands.append(errslice / cont)

    data = databands[0]
    wl = wlbands[0]
    
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=False)

    #wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #veldists = np.arange(120,390,10)
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    #mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    #9820, 9855
    #10850,10910
    #11800,11870
    #11870,11920
    #12780,12860
    #13010,13070
    #reg = (wl >= 9820) & (wl <= 9855)
    
    #wl = wl[reg]
    #data = data[reg]
    #popt, pcov = gaussian_fit_os(wl, data, [-0.005, 5.0, 9840, 1.0])
    #gaus = gaussian_os(wl, popt[0], popt[1], popt[2], popt[3])
    #gaus2 = gaussian_os(wl, -0.035, 8.25, popt[2], popt[3])

    #print popt

    #mpl.plot(wl, data)
    #mpl.plot(wl, gaus)
    #mpl.plot(wl,gaus2)
    #mpl.show()

    c = 299792.458

    m_center = 11834
    m_sigma = 6.8
    #m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    print v
    
    #sigma_gal = np.abs((m_center * (veldisp/c + 1.)) - m_center)
    #sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    #convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    #gaussplot = mcsp.gauss_nat(convolvex, [sigma_conv,0.])

    #out = np.convolve(datafull, gaussplot, mode='same')

def testVelDisp(fl, burnin = -1000, vcjset = False, plot = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = bestfitPrepare(fl, burnin)

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.1)
    shifts = [0.9,-0.8,-1.5,-2.8,-1.1,1.0,-0.6]

    shfit = np.zeros((len(shifts),7))
    for j,sh in enumerate(shifts): 
        if plot:
            fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
            #fig.suptitle(str(sh))
            axes = axes.flatten()
            fig.delaxes(axes[-1])
        for i in range(len(wl)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = np.array(wl[i])
            wli += sh
            #wli += shifts[i]

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
            shfit[j,i] = np.std(data[i] / modelslice)
            
            if plot:
                axes[i].plot(wl[i], modelslice, 'r')
                axes[i].plot(wl[i], data[i],'b')
                axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
                axes[i].set_title(line_name[i] + ' %.2f' % (np.std(data[i] / modelslice)))
                if linefit:
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

        if plot:
            mpl.show()
    for i in range(len(wl)):
        print shifts[np.argmin(shfit[:,i])]

    return shfit

def calculate_MLR_test():

    oldm = np.loadtxt('t13.5_solar.ssp')
    wlm = oldm[:,0]
    bh = oldm[:,2]
    salp = oldm[:,3]
    chab = oldm[:,4]

    #Get interpolation function
    interps = imf.mass_ratio_prepare_isochrone()
    #Calculate Kroupa IMF integrals
    MWremaining, to_mass, normconstant = imf.determine_mass_ratio_isochrone(1.3,2.3, 13.5, interps)
    #Calculate Remnant mass at turnoff mass
    massremnant = imf.massremnant(interps[-1], to_mass)
    #Caluclate final remaining mass
    MWremaining += massremnant*normconstant
    print "MW Mass: ", MWremaining
    
    whK = np.where((wlm >= 20300) & (wlm <= 23700))[0] 

    MLR_MW = np.sum(chab[whK])
    mlrarr = []

    vals = [(1.3,1.3),(1.3,2.3),(2.3,2.3),(2.9,2.9),(3.1,3.1),(3.5,3.5)]
    MLRDict = {}

    MLR_BH = np.sum(bh[whK])
    MLR_SALP = np.sum(salp[whK])

            #if (x1,x2) in vals:
            #    MLRDict[(x1,x2)] = MLR_IMF

    IMFBH, to_massBH, normconstantBH = imf.determine_mass_ratio_isochrone(3.0,3.0, 13.5, interps)
    IMFSALP, to_massSALP, normconstantSALP = imf.determine_mass_ratio_isochrone(2.3,2.3, 13.5, interps)
    IMFBH += massremnant*normconstantBH
    IMFSALP += massremnant*normconstantSALP

    alphaBH = MLR_MW / MLR_BH
    alphaSALP = MLR_MW / MLR_SALP

    alphaAdjBH = (IMFBH * MLR_MW) / (MWremaining * MLR_BH)
    alphaAdjSALP =(IMFSALP * MLR_MW) / (MWremaining * MLR_SALP)

    print alphaBH, alphaAdjBH
    print alphaSALP, alphaAdjSALP

def calculate_MLR(fl, instrument = 'nifs', burnin = -1000, vcjset = None):

    #Load the best-fit parameters
    data, names, gal, datatype, fitvalues, truevalues,paramnames, linenames= bestfitPrepare(fl, burnin)

    #Generate the IMF exponent arrays and get indices for various parameters
    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)
    ix1 = np.where(paramnames == 'x1')[0][0]
    ix2 = np.where(paramnames == 'x2')[0][0]
    iage = np.where(paramnames == 'Age')[0][0]
    
    MLR = np.zeros((len(x_m),len(x_m))) #Array to hold the M/L values
    MWvalues = np.array(truevalues)
    MWvalues[ix1] = 1.3
    MWvalues[ix2] = 2.3
    wlgMW, newmMW = mcfs.model_spec(MWvalues, gal, paramnames, vcjset = vcjset, full = True) #Generate the MW spectrum
    
    #Get the best fit age
    bestage = MWvalues[iage]
    print "Age: ", bestage
    
    #Get interpolation function
    interps = imf.mass_ratio_prepare_isochrone()
    #Calculate Kroupa IMF integrals
    MWremaining, to_mass, normconstant = imf.determine_mass_ratio_isochrone(1.3,2.3, bestage, interps)
    #Calculate Remnant mass at turnoff mass
    massremnant = imf.massremnant(interps[-1], to_mass)
    #Caluclate final remaining mass
    MWremaining += massremnant*normconstant
    print "MW Mass: ", MWremaining
    
    whK = np.where((wlgMW >= 20300) & (wlgMW <= 23700))[0] 

    MLR_MW = np.sum(newmMW[whK])
    mlrarr = []

    vals = [(0.5,0.5),(1.3,1.3),(1.3,2.3),(2.3,2.3),(2.9,2.9),(3.1,3.1),(3.5,3.5)]
    MLRDict = {}

    for i in range(len(x_m)):
        for j in range(len(x_m)):
            x1 = x_m[i]
            x2 = x_m[j]
            tempvalues = np.array(truevalues)
            tempvalues[ix1] = x1
            tempvalues[ix2] = x2
            wlg, newm = mcfs.model_spec(tempvalues, gal, paramnames, vcjset = vcjset, full = True)
            MLR_IMF = np.sum(newm[whK])

            #if (x1 >= 3.0) and (x2 >= 3.0):
            #    mpl.plot(wlg[whK], newm[whK],linestyle='dashed')
            #else:
            #    mpl.plot(wlg[whK], newm[whK])

            #if (x1,x2) in vals:
            #    MLRDict[(x1,x2)] = MLR_IMF

            IMFremaining, to_massimf, normconstantimf = imf.determine_mass_ratio_isochrone(x1,x2, bestage, interps)
            IMFremaining += massremnant*normconstantimf
            print x1, x2, IMFremaining, MWremaining, MLR_IMF, MLR_MW

            mlrarr.append(IMFremaining)
            if (x1,x2) in vals:
                MLRDict[(x1,x2)] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = MLR_MW/ MLR_IMF

            MLR[i,j] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
            #MLR[i,j] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
            #MLR[i,j] = MLR_MW / MLR_IMF
    #mpl.show()
    #sys.exit()

    mpl.close('all')
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    print samples.shape
    
    x1 = samples[:,ix1]
    x2 = samples[:,ix2]

    x_mbins = 0.4 + np.arange(17)/5.0
    histprint = mpl.hist2d(x1,x2, bins = x_mbins)

    fullMLR = []
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            n_val = int(histprint[0][i,j])
            addlist = [float(MLR[i,j])] * n_val
            fullMLR.extend(addlist)

    mpl.close('all')
    percentiles = np.percentile(fullMLR, [16,50,84], axis = 0)
    print fl, np.percentile(fullMLR, [16, 50, 84], axis=0)

    return MLR, MLRDict, paramnames, truevalues, histprint, mlrarr, percentiles, fullMLR

def plotMLRhist(M87, M85):

    fig, ax = mpl.subplots(figsize = (7,6))

    h1 = ax.hist(M87[-1], bins = 15, alpha = 0.7, label = 'M87')
    h2 = ax.hist(M85[-1], bins = 20, alpha = 0.7, label = 'M85')
    print np.median(M87[-1])
    print np.median(M85[-1])
    print np.percentile(M85[-1], [16, 50, 84], axis=0)
    print np.percentile(M87[-1], [16, 50, 84], axis=0)

    ax.axvline(np.median(M87[-1]), linestyle = '--', color = 'b', linewidth=2)#, label='M87 Mean')
    ax.axvline(np.median(M85[-1]), linestyle = '--', color = 'g', linewidth=2)#, label = 'M85 Mean')
    ax.axvline(1.0, color = 'k')
    ax.text(0.7, ax.get_ylim()[1]/1.75,'Kroupa', rotation = 'vertical', fontsize=13)
    #ax.axvline(1.581, color = 'k')
    ax.axvline(1.8, color = 'k')
    #ax.text(1.6, ax.get_ylim()[1]/1.75,'Salpeter', rotation = 'vertical', fontsize = 13)
    ax.text(1.85, ax.get_ylim()[1]/1.75,'Salpeter', rotation = 'vertical', fontsize = 13)

    #ax.set_xlabel('$(M/L)_{K}/(M/L)_{K,MW}$')
    ax.set_xlabel(r'$\alpha_{K}$')
    ax.set_yticklabels([])
    ax.set_xlim((np.min(M85[-1]),np.max(M87[-1])))

    mpl.rc('axes', labelsize=15)
    mpl.rc('axes', titlesize=17)
    mpl.rc('xtick',labelsize=13)
    mpl.rc('ytick',labelsize=13)

    mpl.legend()
    #mpl.show()
    mpl.savefig('/home/elliot/MLR.pdf', dpi = 600)

def plotModels(modelfile, chemfile, gal):

    mpl.close('all')

    #fig.delaxes(axes[-1])

    wl, data, err = mcfs.preparespec(gal)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    x = pd.read_table(modelfile, delim_whitespace = True, header=None)
    x = np.array(x)
    mwl = x[:,0]
    mdata = x[:,74]

    chem_names = ['WL','Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a_o_Fe+', 'N+', 'N-', 'as_o_Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
    #chem_names = ['WL','Solar', 'Nap', 'Nam', 'Cap', 'Cam', 'Fep', 'Fem', 'Cp', 'Cm', 'a/Fep', 'Np', 'Nm', 'as/Fep', 'Tip', 'Tim',\
    #                'Mgp', 'Mgm', 'Sip', 'Sim', 'Tp', 'Tm', 'Crp', 'Mnm', 'Bap', 'Bam', 'Nip', 'Cop', 'Eup', 'Srp', 'Kp',\
    #                'Vp', 'Cup', 'Nap0.6', 'Nap0.9']

    x = pd.read_table(chemfile, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
    chems = np.array(x)
    chemwl = chems[:,0]
    chemsolar = chems[:,1]

    for j in range(2, len(chem_names)):
        print chem_names[j]
        fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
        axes = axes.flatten()

        Feplus = chems[:,j]
        Feeffect = Feplus/chemsolar

        mconvinterp_chem = spi.interp1d(chemwl,Feeffect , kind='cubic', bounds_error=False)
        adjust = mconvinterp_chem(wl)

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)

        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)
        wls2, datas2, err2 = mcfs.splitspec(wl, newmdata * adjust, err = False, lines=linefit)

        for i in range(len(mlow)):
            #print line_name[i]

            modelslice = datas[i]
            
            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            polyfit = mcfs.removeLineSlope(wl, newmdata,i)
            cont = polyfit(wls[i])
            polyfit = mcfs.removeLineSlope(wl, newmdata * adjust,i)
            cont2 = polyfit(wls[i])

            modelslice = modelslice / cont
            otherslice = datas2[i] / cont2 
            
            #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
            axes[i].plot(wls[i], datas[i],'b')
            axes[i].plot(wls[i], datas2[i],'g')
            #axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
            axes[i].set_title(line_name[i])
            axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
            axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
            axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
            
            if linefit:
                axes[i].set_xlim((bluelow[i],redhigh[i]))
        
        mpl.title(chem_names[j])
        mpl.savefig('/home/elliot/chemplots/'+ chem_names[j]+'.png',dpi=500)
        mpl.close('all')

def parsemodelname(modelname):

        end = modelname.split('/')[-1][:-5]
        namespl = end.split('_')
        age = float(namespl[3][1:])
        zfull = namespl[4]

        if zfull[1] == 'p':
            pm = 1.0
        else:
            pm = -1.0
        Z = pm * float(zfull[2:5])

        return age, Z

def plotZlines(Age, gal):

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    wl, data, err = mcfs.preparespec(gal)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)


    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    plotstyle = False
    for j in range(len(modelfiles)):
        mage, mz = parsemodelname(modelfiles[j])
        if mage != Age:
            continue
        if mz < -0.6:
            continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdata = x[:,74]

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)
        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)

        for i in range(len(mlow)):

            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            polyfit = mcfs.removeLineSlope(wl, newmdata, i)
            cont = polyfit(wls[i])

            modelslice = datas[i] / cont
            
            if i == 0:
                if mz == 0.2:
                    axes[i].plot(wls[i], datas[i],'b', label = str(mz))
                elif mz == 0.0:
                    axes[i].plot(wls[i], datas[i],'g', label = str(mz))
                elif mz == -0.5:
                    axes[i].plot(wls[i], datas[i],'r', label = str(mz))
            else:
                if mz == 0.2:
                    axes[i].plot(wls[i], datas[i],'b')
                elif mz == 0.0:
                    axes[i].plot(wls[i], datas[i],'g')
                elif mz == -0.5:
                    axes[i].plot(wls[i], datas[i],'r')

            if not plotstyle:
                axes[i].set_title(line_name[i])
                axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
                axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
                axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        plotstyle = True
        
    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    mpl.savefig('/home/elliot/chemplots/Age_'+ str(Age)+'.png',dpi=500)
    mpl.close('all')

def plotAgeZcontours():

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    wl, data, err = mcfs.preparespec('M85')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).


    plotstyle = False
    widths = {}
    for j in range(len(modelfiles)):
        print modelfiles[j]
        mage, mz = parsemodelname(modelfiles[j])
        widths[(mage,mz)] = []

        #if mage != Age:
        #    continue
        #if mz < -0.6:
        #    continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdata = x[:,221]

        #mwl, mdata = mcfs.convolvemodels(mwl, mdata, 370.)
        #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)
        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)

        for i in range(len(mlow)):
            eqw = measure_line(wl, newmdata,i)
            widths[(mage,mz)].append(eqw)

    #Z_m = np.array([-1.5,-1.0, -0.5, 0.0, 0.2])
    Z_m = np.array([-0.5, 0.0, 0.2])
    Age_m = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])

    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    for i in range(len(mlow)):
        print line_name[i]
        linewidths = []
        for z in Z_m:
            lw_z = []
            for age in Age_m:
                lw_z.append(widths[(age,z)][i])
            linewidths.append(lw_z)
        finalwidths = np.array(linewidths)
        CS = axes[i].contour(np.array(Age_m), np.array(Z_m), finalwidths)
        axes[i].clabel(CS, inline=1, fontsize=10, colors='k')
        axes[i].set_title(line_name[i])

    #fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    mpl.savefig('/home/elliot/chemplots/ContoursBottomHeavy.png',dpi=500)
    mpl.close('all')

    return widths
    
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

def compareModels(paramslist,pdescrip, gal, vcjset, runvalues = False):

    mpl.close('all')
    if runvalues:
        data, names, gal, datatype, fitvalues, truevalues,paramnames, linenames =\
                bestfitPrepare(runvalues, burnin)
        paramslist.append(truevalues)
        pdescript.append("Best-Fit")

    paramnames = ['Age','Z','x1','x2','Na','Fe','Ca','K']

    wl, data, err = mcfs.preparespec(gal)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    mdataarr = []
    for params in paramslist:
        wl1, newm1 = mcfs.model_spec(params1, gal, paramnames, vcjset = vcjset)
        wlg1, data1, err1 = mcfs.splitspec(wl1, newm1, err = False, lines=linefit)
        mdataarr.append(data1)

    #wl2, newm2 = mcfs.model_spec(params2, gal, paramnames, vcjset = vcjset)
    #wlg2, data2, err2 = mcfs.splitspec(wl2, newm2, err = False, lines=linefit)

    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    styling = False
    for modeldata in mdataarr:
        for i in range(len(mlow)):
            #print line_name[i]

            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            #polyfit = mcfs.removeLineSlope(wl1, data1, i)
            #cont1 = polyfit(wlg1[i])

            #polyfit = mcfs.removeLineSlope(wl1, data2, i)
            #cont2 = polyfit(wls[i])

            #modelslice = modelslice / cont
            #otherslice = datas2[i] / cont2 
            
            #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
            axes[i].plot(wlg1[i], modeldata[i], label = pdescrip[i])
            #axes[i].plot(wlg1[i], data2[i],'g', label = '2')
            #axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)

            if not styling:
                axes[i].set_title(line_name[i])
                axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.2)
                axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.2)
                axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.2)
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        styling = True
        
        #mpl.title(chem_names[j])
        #mpl.savefig('/home/elliot/chemplots/'+ chem_names[j]+'.png',dpi=500)
    mpl.show()
    mpl.close('all')

if __name__ == '__main__':

    vcj = mcfs.preload_vcj(overwrite_base='/home/elliot/mcmcgemini/') #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    #compare_bestfit('20190103T035753_M85_fullfit.dat', vcjset = vcj, addshift = True)
    #compare_bestfit('20190103T052837_M85_fullfit.dat', vcjset = vcj, addshift = True)
    #compare_bestfit('20181204T211732_M85_fullfit.dat', vcjset = vcj, addshift = True)
    #compare_bestfit('20181204T195631_M87_fullfit.dat', vcjset = vcj, addshift = True)
    
    #calculate_MLR_test()
    #sys.exit()
    #widths = plotAgeZcontours()
    #sys.exit()

    #plotZlines(3.0, 'M85')
    #plotZlines(13.5, 'M85')
    #sys.exit()

    #modelfile = '/home/elliot/mcmcgemini/spec/vcj_ssp/VCJ_v8_mcut0.08_t03.0_Zp0.2.ssp.imf_varydoublex.s100'
    #modelfile = '/home/elliot/mcmcgemini/spec/vcj_ssp/VCJ_v8_mcut0.08_t13.5_Zp0.2.ssp.imf_varydoublex.s100'
    #chemfile = '/home/elliot/mcmcgemini/spec/atlas/atlas_ssp_t03_Zp0.2.abund.krpa.s100'
    #plotModels(modelfile, chemfile, 'M85')
    #sys.exit()

    #compare_bestfit('mcmcresults/20181129T014719_M85_fullfit.dat', instrument = 'nifs',burnin=-1000, onesigma = False, addshift = True, vcjset = vcj)
    #compare_bestfit('mcmcresults/20181201T142457_M87_fullfit.dat', instrument = 'nifs',burnin=-1000, onesigma = False, addshift = True, vcjset = vcj)

##
    #M85base = calculate_MLR('mcmcresults/20181130T161910_M85_fullfit.dat', vcjset = vcj)
    #M87base = calculate_MLR('mcmcresults/20181201T003101_M87_fullfit.dat', vcjset = vcj)
    #M85base = calculate_MLR('mcmcresults/20181129T093431_M85_fullfit.dat', vcjset = vcj)
    #M87base = calculate_MLR('mcmcresults/20181129T032004_M87_fullfit.dat', vcjset = vcj)
    #M85extended = calculate_MLR('mcmcresults/20181204T170523_M85_fullfit.dat', vcjset = vcj)
    #M87extended = calculate_MLR('mcmcresults/20181204T183647_M87_fullfit.dat', vcjset = vcj)
    #M85 = calculate_MLR('mcmcresults/20181204T211732_M85_fullfit.dat', vcjset = vcj)
    #M87 = calculate_MLR('mcmcresults/20181204T195631_M87_fullfit.dat', vcjset = vcj)
##
    WIFISM851 = calculate_MLR('mcmcresults/20190319T233200_M85_fullfit.dat',vcjset = vcj)
    WIFISM852 = calculate_MLR('mcmcresults/20190320T014215_M85_fullfit.dat',vcjset = vcj)
    WIFISM871 = calculate_MLR('mcmcresults/20190321T025651_M87_fullfit.dat',vcjset = vcj)
    WIFISM872 = calculate_MLR('mcmcresults/20190321T040458_M87_fullfit.dat',vcjset = vcj)


    #plotMLRhist(M87, M85)

    #shfit = deriveShifts('20180807T090719_M85_fullfit.dat', vcjset = vcj, plot = False)
    #deriveVelDisp('M85')
