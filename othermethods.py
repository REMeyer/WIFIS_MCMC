import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import numpy as np
import corner

def plot_corner(fl, burnin):

    dataall = mcfs.load_mcmc_file(fl)
    data = dataall[0]
    smallfit = dataall[2][3]
    names = dataall[2][4]

    flspl = fl.split('/')[-1]
    flspl = flspl.split('_')[0]
    #infol = [nworkers, niter, gal, smallfit, names]

    samples = data[burnin:,:,:].reshape((-1,len(names)))

    fig = corner.corner(samples, labels = names)
    fig.suptitle(dataall[2][2] + ' ' + flspl + ' ' + str(smallfit))

    fig.savefig(fl[:-4]+'.png')

