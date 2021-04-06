from __future__ import print_function

import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as mpl
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd
import plot_corner as pc
import imf_mass as imf
import scipy.ndimage

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

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
    #return  (1 / (2.*np.pi*p0[0]**2.))* np.exp((-1.0/2.0) * ((xs - p0[1])/p0[0])**2.0)
    return  (1 / (np.sqrt(2.*np.pi)*p0[0])) * np.exp( (-1.0/2.0) * ((xs - p0[1])/p0[0])**2.0 )

def load_mcmc_file(fl):
    '''Loads the mcmc chain ouput. Returns
    the full walker data, the calculated statistcal data, 
    and the run info.
    
    Inputs:
        fl -- The mcmc chain data file.'''

    fittype = fl.split('_')[-1][:-4]

    f = open(fl,'r')
    f.readline()
    info = f.readline()

    extralines = 0
    if fittype == 'fullindex':
        lines = True

        paramline = f.readline()
        paramnames = paramline[1:].split()

        linesline = f.readline()
        linenames = linesline[1:].split()

        dictline = f.readline()
        if dictline[0] != '#':
            extralines += 1

        commentline = f.readline()
        if commentline[0] != '#':
            extralines += 1
        
    else:
        lines = False 

        paramline = f.readline()
        paramnames = paramline[1:].split()

        linenames = []

        dictline = f.readline()
        if dictline[0] != '#':
            extralines += 1

        commentline = f.readline()
        if commentline[0] != '#':
            extralines += 1

    n_values = len(paramnames)

    #Get line count to diagnose
    lc = extralines
    for line in f:
        lc += 1
    f.close()

    #Get MCMC run info
    info = info[1:]
    values = info.split()
    nworkers = int(values[0])
    niter = int(values[1])
    gal = values[2]

    #Parse paramnames and assign ploting text and limits
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
        elif paramnames[j] in ['K','Ca','Fe','Mg','Si','Ti','Cr']:
            names.append('[%s/H]' % (paramnames[j]))
            high.append(0.5)
            low.append(-0.5)
        elif paramnames[j] == 'C':
            names.append('[%s/H]' % (paramnames[j]))
            high.append(0.3)
            low.append(-0.3)
        elif paramnames[j] == 'Vel':
            names.append('z')
            high.append(0)
            low.append(0.015)
        elif paramnames[j] == 'VelDisp':
            names.append('Veldisp')
            high.append(390)
            low.append(120)
        elif paramnames[j] == 'f':
            names.append('log-f')
            high.append(1.0)
            low.append(-10.0)
        elif paramnames[j] == 'Alpha':
            names.append('[Alpha/H]')
            high.append(0.3)
            low.append(0.0)

    names.insert(0,"Worker")
    names.insert(len(names), "ChiSq")
    print("Params: ", paramnames)
    if lines:
        print("Lines: ", linenames)

    #N lines should be nworkers*niter
    n_lines = nworkers*niter
    if lc < nworkers:
        print("FILE DOES NOT HAVE ONE STEP...RETURNING")
        return
    elif lc % nworkers != 0:
        print("FILE HAS INCOMPLETE STEP...REMOVING")
        n_steps = int(lc / nworkers)
        initdata = pd.read_csv(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        #initdata = np.loadtxt(fl)
        data = np.array(initdata)
        data = data[:n_steps*nworkers,:]
    elif lc != n_lines:
        print("FILE NOT COMPLETE")
        initdata = pd.read_csv(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        data = np.array(initdata)
        #data = np.loadtxt(fl)
        n_steps = int(data.shape[0]/nworkers)
    else:
        initdata = pd.read_csv(fl, comment='#', header = None, \
                names=names, delim_whitespace=True)
        data = np.array(initdata)
        #data = np.loadtxt(fl)
        n_steps = niter

    paramorder = np.array(['Age','Z','Alpha','x1','x2','Na','K','Ca','Fe','Mg','Si','C','Ti','Cr','Vel','VelDisp','f'])
    rearrange = []
    #for param in paramnames:
    #   o = np.where(paramorder == param)[0][0]
    #   rearrange.append(o)
    paramnames = np.array(paramnames)
    for param in paramorder:
        o = np.where(paramnames == param)[0]
        if len(o) > 0:
            rearrange.append(o[0])

    rearrange = np.array(rearrange)
    high = np.array(high)[rearrange]
    low = np.array(low)[rearrange]

    names = names[1:-1]
    for k,name in enumerate(names):
        if name == 'x1':
            names[k] = r'\textbf{$x_{1}$}'
        if name == 'x2':
            names[k] = r'\textbf{$x_{2}$}'
    names = np.array(names)[rearrange]
    paramnames = np.array(paramnames)[rearrange]

    infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, lines]

    folddata = data.reshape((n_steps, nworkers,len(names)+2))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]
    realdata = realdata[:,:,rearrange]
    lastdata = realdata[-1,:,:]
    print("DATASHAPE: ", realdata.shape)

    return [realdata, postprob, infol, lastdata]

def convolvemodels(wlfull, datafull, veldisp, reglims = False):

    if reglims:
        reg = (wlfull >= reglims[0]) & (wlfull <= reglims[1])
        m_center = reglims[0] + (reglims[1] - reglims[0])/2.
        #print("Reglims")
    else:
        reg = (wlfull >= 9500) & (wlfull <= 13500)
        m_center = 11500
        #print("Not Reglims")
    
    wl = wlfull[reg]
    dw = wl[1]-wl[0]
    data = datafull[reg]

    c = 299792.458

    #Sigma from description of models
    m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    #f = m_center + m_sigma
    #v = c * ((f/m_center) - 1)
    
    sigma_gal = np.abs((m_center / (veldisp/c + 1.)) - m_center)
    sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    #convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    #gaussplot = gauss_nat(convolvex, [sigma_conv,0.])

    #out = np.convolve(datafull, gaussplot, mode='same')
    out = scipy.ndimage.gaussian_filter(datafull, sigma_conv / dw)

    return wlfull, out

def removeLineSlope(wlc, mconv, linedefs, i):
    bluelow,bluehigh,redlow,redhigh = linedefs

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
    print("MW Mass: ", MWremaining)
    
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

    print(alphaBH, alphaAdjBH)
    print(alphaSALP, alphaAdjSALP)

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

def preload_vcj(overwrite_base = False):
    '''Loads the SSP models into memory so the mcmc model creation takes a
    shorter time. Returns a dict with the filenames as the keys'''

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

def load_mcmc_inputs(fl):

    f = open(fl,'r')
    lines = f.readlines()
    f.close()

    inputs = []
    inputset = {}
    for i, line in enumerate(lines):

        if line == '\n' and inputset != {}:
            inputs.append(inputset)
            inputset = {}
            continue

        line_split = line.split()
        key = line_split[0]

        if key == 'fl':
            inputset['fl'] = line_split[1]
        elif key == 'lineinclude':
            inputset['lineinclude'] = line_split[1:]
        elif key == 'params':
            paramdict = {}
            for s in line_split[1:]:
                k_v = s.split(':')
                if k_v[1] == 'None':
                    paramdict[k_v[0]] = None
                else:
                    paramdict[k_v[0]] = float(k_v[1])
            inputset['paramdict'] = paramdict
        elif key == 'target':
            inputset['target'] = line_split[1]
        elif key == 'workers':
            inputset['workers'] = int(line_split[1])
        elif key == 'steps':
            inputset['steps'] = int(line_split[1])
        elif key == 'targetz':
            inputset['targetz'] = float(line_split[1])
        elif key == 'targetsigma':
            inputset['targetsigma'] = float(line_split[1])
        elif key == 'sfl':
            inputset['sfl'] = line_split[1]
        elif key == 'sz':
            inputset['sz'] = float(line_split[1])
        elif key == 'ssigma':
            inputset['ssigma'] = float(line_split[1])
        elif key == 'saurononly':
            if line_split[1] == 'False':
                inputset['saurononly'] = False
            else:
                inputset['saurononly'] = True
        elif key == 'comments':
            inputset['comments'] = ' '.join(line_split[1:])
        elif key == 'skip':
            inputset['skip'] = bool(line_split[1])
        
    inputs.append(inputset)

    return inputs

if __name__ == '__main__':

    vcj = mcfs.preload_vcj(overwrite_base='/home/elliot/mcmcgemini/') #Preload the model files so the mcmc runs rapidly (<0.03s per iteration)
    wl, gal = compare_bestfit('mcmcresults/20181204T211732_M85_fullfit.dat', vcjset = vcj, addshift = True)
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
    #WIFISM851 = calculate_MLR('mcmcresults/20190319T233200_M85_fullfit.dat',vcjset = vcj)
    #WIFISM852 = calculate_MLR('mcmcresults/20190320T014215_M85_fullfit.dat',vcjset = vcj)
    #WIFISM871 = calculate_MLR('mcmcresults/20190321T025651_M87_fullfit.dat',vcjset = vcj)
    #WIFISM872 = calculate_MLR('mcmcresults/20190321T040458_M87_fullfit.dat',vcjset = vcj)


    #plotMLRhist(M87, M85)

    #shfit = deriveShifts('20180807T090719_M85_fullfit.dat', vcjset = vcj, plot = False)
    #deriveVelDisp('M85')
