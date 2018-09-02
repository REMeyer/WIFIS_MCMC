import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import numpy as np
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import mcmc_support as mcsp
from mcmc_support import load_mcmc_file

line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
mlow = [9855,10300,11340,11667,11710,12460,13090]
mhigh = [9970,10390,11447,11750,11810,12590,13175]
morder = [1,1,1,1,1,1,1]
linefit = True

def plot_corner(fl, burnin):

    mpl.close('all')
    dataall = load_mcmc_file(fl)
    data = dataall[0]
    smallfit = dataall[2][3]
    names = dataall[2][4]
    high = dataall[2][5]
    low = dataall[2][6]
    legacy = dataall[2][8]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"
    #infol = [nworkers, niter, gal, smallfit, names]

    if legacy:
        if smallfit == 'limited':
            mcmctype = 'Base Params'
        if smallfit == 'LimitedVelDisp':
            mcmctype = 'Base + VelDisp'
        elif smallfit == 'True':
            mcmctype = 'Abundance Fit'
        elif smallfit == 'False':
            mcmctype = 'Full Fit'
        elif smallfit == 'NoAge':
            mcmctype = 'Abundance Fit (no Age)'
        elif smallfit == 'NoAgeVelDisp':
            mcmctype = 'No Age / VelDisp'
        elif smallfit == 'NoAgeLimited':
            mcmctype = 'Base - Age + [Na/H]'
        elif smallfit == 'VeryBase':
            mcmctype = 'Very Base Params'
    else:
        mcmctype = ''

    samples = data[burnin:,:,:].reshape((-1,len(names)))
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
        ax.set_title(names[i]+"=$%s_{-%s}^{+%s}$" % (np.round(truevalues[i][0],3), \
                np.round(truevalues[i][2],3), np.round(truevalues[i][1],3)))
        ax.set_xlim((low[i],high[i]))

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

    figure.savefig(fl[:-4]+'.png')

def multiplot(fls, burnin = -1000):
    for fl in fls:
        plot_corner(fl, burnin)

def testChemVariance(Z, Age, fitmode):

    vcj = mcfs.preload_vcj(overwrite_base='/home/elliot/mcmcgemini/')
    fl1, cfl1, fl2, cfl2, mixage, mixZ = mcfs.select_model_file(Z, Age, fitmode)

    x1_m = 0.5 + np.arange(16)/5.0
    x2_m = 0.5 + np.arange(16)/5.0
    imfsdict = {}
    for i in range(16):
        for j in range(16):
            imfsdict[(x1_m[i],x1_m[j])] = i*16 + j

    if mixage or mixZ:
        # Reading models. This step was vastly improved by pre-loading the models prior to running the mcmc
        fm1 = vcj[fl1]
        wlc1 = vcj["WL"]
        fc1 = vcj[cfl1]
        fm2 = vcj[fl2]
        fc2 = vcj[cfl2]

        #Finding the relevant section of the models to reduce the computational complexity
        rel = np.where((wlc1 > 8500) & (wlc1 < 14000))[0]

        m = np.zeros(fm1.shape)
        c = np.zeros(fc1.shape)

        #Taking the average of the models (could be improved?)
        for i in range(fm1.shape[1]):
            m[:,i] = (fm1[:,i] + fm2[:,i]) / 2.0
        for i in range(fc1.shape[1]):
            c[:,i] = (fc1[:,i] + fc2[:,i]) / 2.0

        # Setting the models to the proper length
        c = c[rel,:]
        wl = wlc1[rel]
    else:
        #If theres no need to mix models then just read them in and set the length
        m = vcj[fl1]
        wlc1 = vcj["WL"]
        c = vcj[cfl1]

        rel = np.where((wlc1 > 8500) & (wlc1 < 14000))[0]

        c = c[rel,:]
        wl = wlc1[rel]

    if fitmode in [True, False, 'NoAge']:
        #Na adjustment
        Narange = [-0.3,0.0,0.3,0.6,0.9]
        Nainterp = spi.interp2d(wl, [-0.3,0.0,0.3,0.6,0.9], np.stack((c[:,2],c[:,0],c[:,1],c[:,-2],c[:,-1])), kind = 'cubic')
        #NaP = Nainterp(wl,Na) / c[:,0] - 1.

        #K adjustment
        Krange = [-0.3,0.0,0.3]
        Kminus = (2. - (c[:,29] / c[:,0]))*c[:,0]
        Kinterp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((Kminus,c[:,0],c[:,29])), kind = 'linear')
        #KP = interp(wl,K) / c[:,0] - 1.

        #Mg adjustment (only for full fitting)
        if fitmode == False:
            Mginterp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,16], c[:,0],c[:,15])), kind = 'linear')
            #MgP = interp(wl,Mg) / c[:,0] - 1.

        #Fe Adjustment
        Ferange = [-0.3,0.0,0.3]
        Feinterp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,6], c[:,0],c[:,5])), kind = 'linear')
        #FeP = interp(wl,Fe) / c[:,0] - 1.

        #Ca Adjustment
        Carange = [-0.3,0.0,0.3]
        Cainterp = spi.interp2d(wl, [-0.3,0.0,0.3], np.stack((c[:,4], c[:,0],c[:,3])), kind = 'linear')
        #CaP = interp(wl,Ca) / c[:,0] - 1.

    ranges = [Narange, Krange, Ferange, Carange]
    interps = [Nainterp, Kinterp, Feinterp, Cainterp]

    #mpl.plot(wl, Kminus)
    #mpl.show()
    #sys.exit()
    for k in Krange:
        mpl.plot(wl, (Kinterp(wl, k) / c[:,0]), label=str(k))
    mpl.legend()
    mpl.show()

    sys.exit()

    for i in range(len(ranges)):
        for k in ranges[i]:
            mpl.plot(wl, (interps[i](wl, k) / c[:,0]), label=str(k))
        mpl.legend()
        mpl.show()

def measureveldisp(gal):
    c = 299792.458

    wl, data, err = mcfs.preparespec(gal, baseforce = '/home/elliot/mcmcgemini/')
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=True)

    if gal == 'M85':
        kwl = wl[4]
        kdata = data[4]
        gausreturn = mcsp.gaussian_fit_os(kwl, kdata, [-0.04,10.0,11780,1.0])
        print gausreturn[0]
    if gal == 'M87':
        kwl = wl[4]
        kdata = data[4]
        gausreturn = mcsp.gaussian_fit_os(kwl, kdata, [-0.04,10.0,11780,1.0])
        print gausreturn[0]

    popt = list(gausreturn[0])
    popt.insert(0, kwl)
    mpl.plot(kwl, kdata, 'b')
    mpl.plot(kwl, mcsp.gaussian_os(*popt))
    mpl.show()

    m_sigma = popt[2]
    m_center = popt[3]

    #Sigma from description of models
    #m_sigma = np.abs(m_center / (1 + 100./c) - m_center)
    #f = m_center + m_sigma
    #v = c * ((f/m_center) - 1)
    
    #sigma_gal = (m_center * (veldisp/c + 1.)) - m_center
    #sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)
    print c * (m_sigma/m_center)

if __name__=='__main__':
    #multiplot(glob('/home/elliot/mcmcgemini/mcmcresults/201807*.dat'))
    measureveldisp('M87')

