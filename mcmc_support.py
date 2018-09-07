import numpy as np
import scipy.optimize as spo
import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd

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

#Definitions for the fitting bandpasses
linefit = False
if linefit:
    mlow = [9855,10300,11340,11667,11710,12460,13090]
    mhigh = [9970,10390,11447,11750,11810,12590,13175]
    #mlow = [9905,10337,11372,11680,11765,12505,13115]
    #mhigh = [9935,10360,11415,11705,11793,12545,13165]
    morder = [1,1,1,1,1,1,1]
else:
    mlow = [9700,10550,11340,11550,12350,12665]
    mhigh = [10450,10965,11447,12200,12590,13180]
    morder = [8,4,1,7,2,5]

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
    if firstline[0] == '#':
        legacy = False
        paramstr = firstline[1:]
        nextline = f.readline()
        n_values = len(nextline.split())
    else:
        legacy = True
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
    if not legacy:
        fitmode = ''
        paramnames = paramstr.split()
    else:
        fitmode = values[3]
        paramnames = []

    if legacy:
        if fitmode == 'True':
            names = ['Age','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]']
            high = [13.5,3.5,3.5,0.9,0.3,0.3,0.3]
            low = [1.0,0.5,0.5,-0.3,-0.3,-0.3,-0.3]
        elif fitmode == 'False':
            names = ['[Z/H]','Age','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Mg/H]','[Fe/H]']
            high = [0.2,13.5,3.5,3.5,0.9,0.3,0.3,0.3,0.3]
            low = [-1.5,1.0,0.5,0.5,-0.3,-0.3,-0.3,-0.3,-0.3]
        elif fitmode == 'NoAge':
            names = ['x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]']
            high = [3.5,3.5,0.9,0.3,0.3,0.3]
            low = [0.5,0.5,-0.3,-0.3,-0.3,-0.3]
        elif fitmode == 'NoAgeVelDisp':
            names = ['x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]', 'VelDisp']
            high = [3.5,3.5,0.9,0.3,0.3,0.3, 390]
            low = [0.5,0.5,-0.3,-0.3,-0.3,-0.3,120]
        elif fitmode == 'LimitedVelDisp':
            names = ['[Z/H]','Age','x1','x2','VelDisp']
            high = [0.2,13.5,3.5,3.5,390]
            low = [-1.5,1.0,0.5,0.5,120]
        elif fitmode == 'NoAgeLimited':
            names = ['[Z/H]','x1','x2','[Na/H]']
            high = [0.2,3.5,3.5,0.9]
            low = [-1.5,0.5,0.5,-0.3]
        elif fitmode == 'VeryBase':
            names = ['[Z/H]','x1','x2']
            high = [0.2,3.5,3.5]
            low = [-1.5,0.5,0.5]
        else:
            names = ['[Z/H]','Age','x1','x2']
            high = [0.2,13.5,3.5,3.5]
            low = [-1.5,1.0,0.5,0.5]

    else:
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

    names.insert(0,"Worker")
    names.insert(len(names), "ChiSq")
    if legacy:
        print "MODE: ", fitmode
    else:
        print "Params: ", paramnames

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
    infol = [nworkers, niter, gal, fitmode, names, high, low, paramnames,legacy]

    folddata = data.reshape((n_steps, nworkers,len(names)+2))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]
    lastdata = realdata[-1,:,:]
    print "DATASHAPE: ", realdata.shape

    return [realdata, postprob, infol, lastdata]

def bestfitPrepare(fl, burnin):

    dataall = load_mcmc_file(fl)
    data = dataall[0]
    names = dataall[2][4]
    high = dataall[2][5]
    low = dataall[2][6]
    legacy = dataall[2][8]
    gal = dataall[2][2]
    paramnames = dataall[2][7]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"
    #infol = [nworkers, niter, gal, fitmode, names]

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    fitvalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    fitvalues = np.array(fitvalues)
    truevalues = fitvalues[:,0]

    return [data, names, gal, datatype, fitvalues, truevalues, paramnames]

def compare_bestfit(fl, burnin=-1000, onesigma = False, addshift = False, vcjset = False):

    #Load the necessary information
    data, names, gal, datatype, fitvalues, truevalues,paramnames\
            = bestfitPrepare(fl, burnin)

    if gal == 'M85':
        #galveldisp = 145.
        #galveldisp = 161.
        galveldisp = 170
    elif gal == 'M87':
        galveldisp = 370.
        #galveldisp = 307.

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, paramnames, vcjset = vcjset)

    #convr = np.arange(120.,395.,25.)
    plot = True
    if plot: 
        fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
        axes = axes.flatten()
        fig.delaxes(axes[-1])

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    for i in range(len(mlow)):
        if (gal == 'M87') and linefit:
            if line_name[i] == 'KI_1.25':
                continue

        #Getting a slice of the model
        wli = wl[i]

        if addshift:
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

if __name__ == '__main__':
    vcj = mcfs.preload_vcj() #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    compare_bestfit('20180907T001647_M85_fullfit.png', burnin=-300, onesigma = False, addshift = True, vcjset = vcj)

    #shfit = deriveShifts('20180807T090719_M85_fullfit.dat', vcjset = vcj, plot = False)
    #deriveVelDisp('M85')
