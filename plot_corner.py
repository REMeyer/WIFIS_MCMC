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

def plot_corner(fl, burnin, burnintest = False, variables=[]):

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

    if burnintest:
        for j in range(int(dataall[2][1])/1000 - 1):
            print j*1000, (j+1)*1000
            samples = data[j*1000:(j+1)*1000,:,:].reshape((-1,len(names)))
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

    else:
        samples = data[burnin:,:,:].reshape((-1,len(names)))

        if len(variables) > 0:
            var_i = []
            var_names = []
            for j, var in enumerate(paramnames):
                if var in variables:
                    var_i.append(j)
                    var_names.append(names[j])
            var_i = np.array(var_i)
            print names
            print variables
            print var_i

            names = var_names
            samples = samples[:,var_i]
            low = np.array(low)[var_i]
            high = np.array(high)[var_i]

        truevalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
                zip(*np.percentile(samples, [16, 50, 84], axis=0)))

        figure = corner.corner(samples, labels = names, label_kwargs={'fontsize': 20})
        #figure.suptitle(dataall[2][2] + ' ' + flspl + ' ' + mcmctype + ' ' + str(datatype))

        # Extract the axes
        axes = np.array(figure.axes).reshape((len(names), len(names)))

        #mpl.rc('axes', labelsize=17)
        #mpl.rc('axes', titlesize=17)
        mpl.rc('xtick',labelsize=17)
        mpl.rc('ytick',labelsize=17)

        # Loop over the diagonal
        for i in range(len(names)):
            ax = axes[i, i]
            ax.axvline(truevalues[i][0], color="r")
            ax.axvline(truevalues[i][0] + truevalues[i][1], color="g")
            ax.axvline(truevalues[i][0] - truevalues[i][2], color="g")
            #ax.set_title(names[i]+"=$%s_{-%s}^{+%s}$" % (np.round(truevalues[i][0],3), \
            #        np.round(truevalues[i][2],3), np.round(truevalues[i][1],3)))
            ax.set_title(names[i], fontsize = 20)
            ax.set_xlim((low[i],high[i]))

            #xlab = ax.get_xticklabels()
            #ylab = ax.get_yticklabels()
            #xlab.set_fontsize(20)
            #ylab.set_fontsize(20
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
                ax.set_ylim((low[yi], high[yi]))
                ax.set_xlim((low[xi], high[xi]))
                ax.tick_params(axis='both', which='major', labelsize=20)
                #xlab = ax.get_xticklabels()
                #ylab = ax.get_yticklabels()
                #xlab.set_fontsize(20)
                #ylab.set_fontsize(20

        figure.savefig(fl[:-4]+'.pdf')

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
    print histprint
    mpl.show()

    return histprint
