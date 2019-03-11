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

def plot_corner(fl, burnin, burnintest = False):

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
    legacy = dataall[2][8]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]

    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"

    #if legacy:
    #    if smallfit == 'limited':
    #        mcmctype = 'Base Params'
    #    if smallfit == 'LimitedVelDisp':
    #        mcmctype = 'Base + VelDisp'
    #    elif smallfit == 'True':
    #        mcmctype = 'Abundance Fit'
    #    elif smallfit == 'False':
    #        mcmctype = 'Full Fit'
    #    elif smallfit == 'NoAge':
    #        mcmctype = 'Abundance Fit (no Age)'
    #    elif smallfit == 'NoAgeVelDisp':
    #        mcmctype = 'No Age / VelDisp'
    #    elif smallfit == 'NoAgeLimited':
    #        mcmctype = 'Base - Age + [Na/H]'
    #    elif smallfit == 'VeryBase':
    #        mcmctype = 'Very Base Params'
    #else:
    #    mcmctype = ''
                                         
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
                #xlab = ax.get_xticklabels()
                #ylab = ax.get_yticklabels()
                #xlab.set_fontsize(20)
                #ylab.set_fontsize(20

        figure.savefig(fl[:-4]+'.pdf')

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
