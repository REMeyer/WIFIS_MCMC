import mcmc_fullspec as mcfs
import matplotlib.pyplot as mpl
import numpy as np
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import mcmc_support as mcsp

line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'AlI']
mlow = [9855,10300,11340,11667,11710,12460,13090]
mhigh = [9970,10390,11447,11750,11810,12590,13175]
morder = [1,1,1,1,1,1,1]
linefit = True

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
    #get_hist('20180902T235307_M85_fullfit.dat')
    h = get_hist('20180821T100923_M87_fullfit.dat')

    #multiplot(glob('/home/elliot/mcmcgemini/mcmcresults/201807*.dat'))
    #measureveldisp('M87')

