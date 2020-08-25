import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
#from matplotlib.colors import Listedcolormap, BoundaryNorm
#from mcmc_fullspec import preload_vcj

def compare_vcj(vcj):

    plt.close('all')
    wl = vcj['WL']
    wh = np.where(np.logical_and(wl >= 9000, wl <= 13100))[0]
    base = vcj['3.0_0.0'][0][:,74]
    ratio = vcj['13.5_0.0'][0][:,74] / base

    pf = np.polyfit(wl[wh], ratio[wh], 9)
    polyfit = np.poly1d(pf)
    cont = polyfit(wl[wh])
    rat = ratio[wh] / cont
    
    #norm = plt.Normalize(rat.min(), rat.max())
    #lc = LineCollection(

    fig, ax = plt.subplots(figsize = (12,8))
    ax.plot(wl[wh], ratio[wh]/cont)
    ax.set_title('13.5 / 3.0 Gyr, Z/H = 0.0')
    fig.savefig('/home/elliot/vcj_sensitivity/age.pdf')
    #plt.plot(wl[wh], cont)
    plt.show()


