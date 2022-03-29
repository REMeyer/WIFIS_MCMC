################################################################################
# Compilation of support functions for the WIFIS mcmc module
################################################################################

from __future__ import print_function

import corner
import sys
import scipy.ndimage
import numpy as np
import scipy.optimize as spo
import matplotlib.pyplot as mpl
import scipy.interpolate as spi
import pandas as pd
import plot_corner as pc
from glob import glob

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

# Gaussian fitting functions
def gaussian(xs, a, sigma, x0):
    second = ((xs - x0)/sigma)**2.0
    full = a*np.exp((-1.0/2.0) * second)
    
    return full

def gaussian_os(xs, a, sigma, x0, b):
    second = ((xs - x0)/sigma)**2.0
    full = b + a*np.exp((-1.0/2.0) * second)
        
    return full

def gaussian_fit_os(xdata,ydata,p0, gaussian=gaussian_os):
    '''
    Performs a very simple gaussian fit using scipy.optimize.curvefit. 
    Returns the fit parameters and the covariance matrix'''

    popt, pcov = spo.curve_fit(gaussian, xdata, ydata, p0=p0)
    return [popt, pcov]

# Natural Gaussian fitting functions
def gauss_nat(xs, p0):
    '''
    Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]
    '''
    return  (1 / (np.sqrt(2.*np.pi)*p0[0])) * np.exp( (-1.0/2.0) * \
                ((xs - p0[1])/p0[0])**2.0 )

def load_mcmc_file(fl, linesoverride=False):
    '''Loads the mcmc chain ouput. Returns
    the full walker data, the calculated statistcal data, 
    and the run info.
   
    Note: this is somewhat hamfisted as it was adjusted over several years
          to handle an evolving mcmc output file. It contains legacy code
          to read those old formats.

    Inputs:
        fl: The mcmc chain data file.
        linesoverride: Flag for old implementations
    '''

    fittype = fl.split('_')[-1][:-4]

    f = open(fl,'r')
    f.readline()
    info = f.readline()

    extralines = 0
    if (fittype == 'fullindex') or linesoverride:
        lines = True

        paramline = f.readline()
        paramnames = paramline[1:].split()

        linesline = f.readline()
        linenames = linesline[1:].split()

        dictline = f.readline()
        paramdict = {}
        if dictline[0] != '#':
            extralines += 1
            for param in paramnames:
                paramdict[param] = None
        else:
            dictline = dictline[1:].split()
            for k in range(len(dictline)-1):
                if k % 2 == 0:
                    if dictline[k+1] == 'None':
                        paramdict[dictline[k][:-1]] = None
                    else:
                        paramdict[dictline[k][:-1]] = float(dictline[k+1])
        print(paramdict)

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
            names.append(r'$\sigma_{v}$')
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

    paramorder = np.array(['Age','Z','Alpha','x1','x2','Na','K','Ca',\
            'Fe','Mg','Si','C','Ti','Cr','Vel','VelDisp','f'])
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

    infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, lines, paramdict]

    folddata = data.reshape((n_steps, nworkers,len(names)+2))
    postprob = folddata[:,:,-1]
    realdata = folddata[:,:,1:-1]
    realdata = realdata[:,:,rearrange]
    lastdata = realdata[-1,:,:]
    print("DATASHAPE: ", realdata.shape)

    return [realdata, postprob, infol, lastdata]

def save_sliced_mcmc(mcmcfile, save, start, end=None):
    '''
    Loads an MCMC results file, truncates the file accordingly, 
    then saves two new files with the trucated data and information.
    '''

    data, postprob, infolist, lastdata = load_mcmc_file(mcmcfile)

    names = infolist[3]
    paramnames = infolist[6]
    linenames = infolist[7]
    lines = infolist[8]

    if start < 0:
        samples = data[start:,:,:].reshape((-1,len(names)))
    else:
        samples = data[start:end+1,:,:].reshape((-1,len(names)))
        
    savefl = open(save+'.mcmc', 'w')
    savefl.write('\t'.join(paramnames) + '\n')
    for s in range(samples.shape[0]):
        savefl.write('\t'.join([str(float(i)) for i in samples[s,:]]) + '\n')

    savefl.close()
        
    return

def parsemodelname(modelname):
    '''
    Helper function to parse the Conroy et al. SED model filenames.
    '''

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

def load_mcmc_inputs(fl):
    '''
    Reads and parses mcmc input files. MCMC input files simplify the process
    of running multiple mcmc simulations. Each input block is parsed into a
    dict then added to a list.

    Inputs:
        fl: filepath of input file

    Returns:
        inputs: list of input dicts
    '''

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
            inputset['skip'] = int(line_split[1])
        elif key == 'twossp':
            if line_split[1] == 'False':
                inputset['twossp'] = False
            else:
                inputset['twossp'] = True

    inputs.append(inputset)

    return inputs

if __name__ == '__main__':

    pass
