import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import numpy as np
import corner
from glob import glob

line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
mlow = [9855,10300,11340,11667,11710,12460,13090]
mhigh = [9970,10390,11447,11750,11810,12590,13175]
morder = [1,1,1,1,1,1,1]
linefit = True

def plot_corner(fl, burnin):

    mpl.close('all')
    dataall = mcfs.load_mcmc_file(fl)
    data = dataall[0]
    smallfit = dataall[2][3]
    names = dataall[2][4]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"
    #infol = [nworkers, niter, gal, smallfit, names]

    if smallfit == 'limited':
        mcmctype = 'Base Params'
    elif smallfit == 'True':
        mcmctype = 'Abundance Fit'
    elif smallfit == 'False':
        mcmctype = 'Full Fit'

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    truevalues = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))
                                         
    figure = corner.corner(samples, labels = names)
    figure.suptitle(dataall[2][2] + ' ' + flspl + ' ' + mcmctype + ' ' + str(datatype))

    # Extract the axes
    axes = np.array(figure.axes).reshape((len(names), len(names)))

    # Loop over the diagonal
    for i in range(len(names)):
        ax = axes[i, i]
        ax.axvline(truevalues[i][0], color="r")
        ax.axvline(truevalues[i][0] + truevalues[i][1], color="g")
        ax.axvline(truevalues[i][0] - truevalues[i][2], color="g")

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

    figure.savefig(fl[:-4]+'.png')



def multiplot(fls, burnin = -1000):
    for fl in fls:
        plot_corner(fl, burnin)

if __name__=='__main__':
    multiplot(glob('/home/elliot/mcmcgemini/mcmcresults/201807*.dat'))

