import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import numpy as np
import corner

def plot_corner(fl, burnin, smallfit = True):

    dataall = mcfs.load_mcmc_file(fl)
    data = dataall[0]

    if smallfit = True:
        names = ['Age','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Fe/H]']
    else:
        names = ['Age','[Z/H]','x1','x2','[Na/H]','[K/H]','[Ca/H]','[Mg/H]','[Fe/H]']

    samples = data[burnin:,:,:].reshape((-1,len(names)))

    fig = corner.corner(samples, labels = names)

    fig.savefig(fl[:-4]+'.png')

