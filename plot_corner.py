from __future__ import print_function

import matplotlib.pyplot as mpl
import numpy as np
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd
import mcmc_support as mcsp

from matplotlib import rc

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

def plot_corner(fl, burnin, variables=[], nolimits=False, latexout=False, \
        alpha=False, annotate=True, save=True):

    rc('text', usetex=True)

    mpl.close('all')
    #if fl.lower() == 'last':
        #fls = np.sort(glob(
    print(fl.split('/')[-1])
    dataall = mcsp.load_mcmc_file(fl)

    data = dataall[0]
    names = dataall[2][3]
    high = dataall[2][4]
    low = dataall[2][5]
    paramnames = dataall[2][6]
    linenames = dataall[2][7]
    lines = dataall[2][8]

    #print(paramnames)
    #print(linenames)
    #print(lines)

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]

    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"

    if type(alpha) != bool:
        samples = data[burnin:,:,:].reshape((-1,len(names)))
        names = np.append(names, r'$\alpha_{K}$')
        paramnames = np.append(paramnames,'alpha')
        samples = np.c_[samples, alpha]
    else:
        samples = data[burnin:,:,:].reshape((-1,len(names)))

    if 'f' in paramnames:
        whf = np.where(paramnames == 'f')[0][0]
        samples[:,whf] = np.log(samples[:,whf])

    #sys.exit()

    if len(variables) > 0:
        print("Using only variables: ", variables)
        var_i = []
        var_names = []
        for j, var in enumerate(paramnames):
            if var in variables:
                var_i.append(j)
                var_names.append(names[j])
        var_i = np.array(var_i)

        names = var_names
        samples = samples[:,var_i]
        print(samples.shape)
        #if 'f' in paramnames:
            #samples[
        low = np.array(low)[var_i]
        high = np.array(high)[var_i]

    truevalues = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0))))

    figure = corner.corner(samples, labels = names, label_kwargs=\
            {'fontsize': 23})
    #figure.suptitle(dataall[2][2] + ' ' + flspl + ' ' + mcmctype + ' ' + \
    # str(datatype))

    # Extract the axes
    axes = np.array(figure.axes).reshape((len(names), len(names)))

    #mpl.rc('axes', labelsize=17)
    #mpl.rc('axes', titlesize=17)
    mpl.rc('xtick',labelsize=17)
    mpl.rc('ytick',labelsize=17)
    #text = r''

    # Loop over the diagonal
    for i in range(len(names)):
        print(paramnames[i]+':\t', np.round(truevalues[i][0] - \
                truevalues[i][2],2), np.round(truevalues[i][0],2),\
                np.round(truevalues[i][0] + truevalues[i][1],2))

        midval = str(np.round(truevalues[i][0],2))
        lowval = str(np.round(truevalues[i][2],2))
        highval = str(np.round(truevalues[i][1],2))

        if annotate:
            textstr = r'%s = $%s^{%s}_{%s}$' % (names[i],midval, highval, \
                    lowval)
            mpl.text(len(names)-1.25, 0.75-i*0.25, textstr, fontsize = 25,\
                    transform=axes[0,0].transAxes)

        ax = axes[i, i]
        ax.axvline(truevalues[i][0], color="r")
        ax.axvline(truevalues[i][0] + truevalues[i][1], color="g")
        ax.axvline(truevalues[i][0] - truevalues[i][2], color="g")

        if paramnames[i] in ['x1','x2','VelDisp']:
            ax.set_title(names[i], fontsize = 30)
        else:
            ax.set_title(names[i], fontsize = 23)
        if not nolimits:
            ax.set_xlim((low[i],high[i]))
        elif paramnames[i] == 'alpha':
            ax.set_xlim((0,4))

        ax.tick_params(axis='both', which='major', labelsize=20)

    # Loop over the histograms
    for yi in range(len(names)):
        for xi in range(yi):
            ax = axes[yi, xi]
            ax.axvline(truevalues[xi][0], color="r")
            ax.axhline(truevalues[yi][0], color="r")
            ax.plot(truevalues[xi][0], truevalues[yi][0], "sr")
            ax.axvline(truevalues[xi][0] + truevalues[xi][1], color="g")
            ax.axvline(truevalues[xi][0] - truevalues[xi][2], color="g")
            ax.axhline(truevalues[yi][0] + truevalues[yi][1], color="g")
            ax.axhline(truevalues[yi][0] - truevalues[yi][2], color="g")
            if not nolimits:
                ax.set_ylim((low[yi], high[yi]))
                ax.set_xlim((low[xi], high[xi]))
            elif paramnames[yi] == 'alpha':
                ax.set_ylim((0,4))
            elif paramnames[xi] == 'alpha':
                ax.set_xlim((0,4))

            ax.tick_params(axis='both', which='major', labelsize=20)

            if np.logical_and(paramnames[yi] == 'x2', paramnames[xi] == 'x1'):
                ax.plot(1.3,2.3,marker='o',color='b', markersize=10)
                ax.plot(2.3,2.3,marker='o',color='g', markersize=10)
                ax.plot(3.0,3.0,marker='o',color='r', markersize=10)
            #xlab = ax.get_xticklabels()
            #ylab = ax.get_yticklabels()
            #xlab.set_fontsize(20)
            #ylab.set_fontsize(20
            if (xi == 0) and (yi != 0):
                if names[yi] in ['x1','x2','VelDisp']:
                    ax.set_ylabel(names[yi], fontsize = 30)
                else:
                    ax.set_ylabel(names[yi], fontsize = 23)
            if (yi == 0) :
                if names[yi] in ['x1','x2','VelDisp']:
                    ax.set_xlabel(names[yi], fontsize = 30)
                else:
                    ax.set_xlabel(names[yi], fontsize = 23)
    if latexout:
        print()
        for i in range(len(names)):
            print(paramnames[i]+':\t', "$"+str(np.round(truevalues[i][0],2))+
                    "_{-"+str(np.round(truevalues[i][2],2))+"}^{+"+\
                            str(np.round(truevalues[i][1],2))+"}$")

    if save:
        figure.savefig(fl[:-4]+'.pdf')

def burnin_test(fl, burnin, step=1000, variables=[]):

    mpl.close('all')
    dataall = mcsp.load_mcmc_file(fl)

    #infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, legacy]
    data = dataall[0]
    #smallfit = dataall[2][3]
    names = dataall[2][3]
    high = dataall[2][4]
    low = dataall[2][5]
    paramnames = dataall[2][6]
    linenames = dataall[2][7]
    lines = dataall[2][8]

    print(paramnames)
    print(linenames)
    print(lines)

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]

    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"

    for j in range(int(dataall[2][1])/step - 1):
        print(j*step, (j+1)*step)
        samples = data[j*step:(j+1)*step,:,:].reshape((-1,len(names)))
        truevalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
                zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        figure = corner.corner(samples, labels = names)
        #figure.suptitle(dataall[2][2] + ' ' + flspl + ' ' + mcmctype + ' ' + str(datatype))

        # Extract the axes
        axes = np.array(figure.axes).reshape((len(names), len(names)))

        # Loop over the diagonal
        for i in range(len(names)):
            ax = axes[i, i]
            ax.axvline(truevalues[i][0], color="r")
            ax.axvline(truevalues[i][0] + truevalues[i][1], color="g")
            ax.axvline(truevalues[i][0] - truevalues[i][2], color="g")
            #ax.set_title(names[i]+"=$%s_{-%s}^{+%s}$" % (np.round(truevalues[i][0],3), \
            #        np.round(truevalues[i][2],3), np.round(truevalues[i][1],3)))
            #ax.set_xlim((low[i],high[i]))

        # Loop over the histograms
        for yi in range(len(names)):
            for xi in range(yi):
                ax = axes[yi, xi]
                ax.axvline(truevalues[xi][0], color="r")
                ax.axhline(truevalues[yi][0], color="r")
                ax.plot(truevalues[xi][0], truevalues[yi][0], "sr")
                ax.axvline(truevalues[xi][0] + truevalues[xi][1], color="g")
                ax.axvline(truevalues[xi][0] - truevalues[xi][2], color="g")
                ax.axhline(truevalues[yi][0] + truevalues[yi][1], color="g")
                ax.axhline(truevalues[yi][0] - truevalues[yi][2], color="g")
                #ax.set_ylim((low[yi], high[yi]))
                #ax.set_xlim((low[xi], high[xi]))

        figure.savefig(fl[:-4]+'_%s.png' % (str(j)))
        mpl.close('all')

def print_bestfit(fl, burnin=-1000):
    dataall = mcsp.load_mcmc_file(fl)

    data = dataall[0]
    names = dataall[2][3]

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    truevalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))

    for val in truevalues:
        print(val)

def get_hist(fl, burnin=-300):
    mpl.close('all')
    dataall = mcsp.load_mcmc_file(fl)
    data = dataall[0]
    smallfit = dataall[2][3]
    names = dataall[2][4]
    high = dataall[2][5]
    low = dataall[2][6]
    legacy = dataall[2][8]
    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    
    x1 = samples[:,2]
    x2 = samples[:,3]
    x_m = 0.4 + np.arange(17)/5.0
    histprint = mpl.hist2d(x1,x2, bins = x_m)
    print(histprint)
    mpl.show()

    return histprint
