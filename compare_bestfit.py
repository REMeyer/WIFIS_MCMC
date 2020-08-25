from __future__ import print_function
import numpy as np
import scipy.optimize as spo

import mcmc_fullindex as mcfi
import mcmc_support as mcsp
import prepare_spectra as preps

import matplotlib.pyplot as mpl
import corner
from glob import glob
import scipy.interpolate as spi
import sys
import pandas as pd
import plot_corner as pc
import imf_mass as imf

from astropy.io import fits

from matplotlib import rc
from matplotlib import animation

rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)


chem_names = ['Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', \
        'C+', 'C-', 'a/Fe+', 'N+', 'N-', 'as/Fe+', 'Ti+', 'Ti-',\
        'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', \
        'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
        'V+', 'Cu+', 'Na+0.6', 'Na+0.9']

def bestfitPrepare(fl, burnin):
    '''Loads the mcmc results file, returns the data, statistics, and other
    header information'''

    #infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, lines]
    #return [realdata, postprob, infol, lastdata]
    data, postprob, info, lastdata = mcsp.load_mcmc_file(fl)
    gal = info[2]
    names = info[3]
    high = info[4]
    low = info[5]
    paramnames = info[6]
    linenames = info[7]
    lines = info[8]

    flsplname = fl.split('/')[-1]
    flspl = flsplname.split('_')[0]
    datatype = flsplname.split('_')[-1][:-4]
    if datatype == "widthfit":
        datatype = "Widths"
    elif datatype == "fullfit":
        datatype = "Spectra"

    print(names)
    print(data.shape)

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    fitvalues = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    fitvalues = np.array(fitvalues)
    midvalues = fitvalues[:,0]

    return [data, names, gal, datatype, fitvalues, midvalues, paramnames, linenames, lines]

def compare_bestfit(datafl, mcmcfl, z, veldisp, burnin=-1000, \
        onesigma = False, scale = False, addshift = False, vcjset = False):

    #Load the necessary information
    data,names,gal,datatype,fitvalues,midvalues,paramnames, linenames = \
            bestfitPrepare(mcmcfl, burnin)

    if fl == None:
        print('Please input filename for WIFIS data')
        return
    wl, data, err = preps.preparespecwifis(gal, fl)

    if scale:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err)

    if (gal == 'M85') and ('VelDisp' not in paramnames):
        #galveldisp = 145.
        #galveldisp = 161.
        galveldisp = 170
    elif (gal == 'M87') and ('VelDisp' not in paramnames):
        galveldisp = 370.
        #galveldisp = 307.
    else:
        w = np.where(np.array(paramnames) == 'VelDisp')
        galveldisp = midvalues[w]

    if lines:
        wlg, newm = mcfi.model_spec(midvalues, gal, paramnames, vcjset = vcjset, full=True)
    else:
        wlg, newm = mcfi.model_spec(midvalues, gal, paramnames, vcjset = vcjset, full=True)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfi.convolvemodels(wlg, newm, galveldisp, reglims=[4000,13500])

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    for i in range(len(mlow)):
        #Getting a slice of the model
        wli = wlc[i]
        if gal == 'M87':
            if line_name[i] == 'KI_1.25':
                continue

        if line_name[i] not in linenames:
            print("Skipping "+line_name[i])
            continue


        if addshift and (instrument == 'nifs'):
            if i in [3,4]:
                wli = np.array(wli)
                wli -= 2.0

        modelslice = mconvinterp(wli)
        
        #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
        #Removing a high-order polynomial from the slice
        if onesigma:
            plusvalues = np.array(truevalues)
            negvalues = np.array(truevalues)

            plusvalues[1] += fitvalues[1,1]
            negvalues[1] -= fitvalues[1,2]

            wlgneg, newmneg = model_spec(negvalues, gal, fitmode = fitmode, vcjset = vcjset)
            wlgplus, newmplus = model_spec(plusvalues, gal, fitmode = fitmode, vcjset = vcjset)

        if linefit:
            if onesigma:
                wlcneg, mconvneg = convolvemodels(wlgneg, newmneg, galveldisp)
                mconvinterpneg = spi.interp1d(wlcneg, mconvneg, kind='cubic', bounds_error=False)

                wlcplus, mconvplus = convolvemodels(wlgplus, newmplus, galveldisp)
                mconvinterpplus = spi.interp1d(wlcplus, mconvplus, kind='cubic', bounds_error=False)

                polyfitneg = mcfi.removeLineSlope(wlcneg, mconvneg,i)
                contneg = polyfitneg(wli)

                polyfitplus = mcfi.removeLineSlope(wlcplus, mconvplus,i)
                contplus = polyfitplus(wli)

            polyfit = mcfi.removeLineSlope(wlc, mconv,i)
            cont = polyfit(wli)

        else:
            if i == 1:
                #Define the bandpasses for each line 
                bluepass = np.where((wlc >= bluelow[i]) & (wlc <= bluehigh[i]))[0]
                redpass = np.where((wlc >= redlow[i]) & (wlc <= redhigh[i]))[0]

                #Cacluating center value of the blue and red bandpasses
                blueavg = np.mean([bluelow[i],bluehigh[i]])
                redavg = np.mean([redlow[i],redhigh[i]])

                blueval = np.mean(mconv[bluepass])
                redval = np.mean(mconv[redpass])

                pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
                polyfit = np.poly1d(pf)
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wli, modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wli)

        modelslice = modelslice / cont
        
        #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
        if onesigma:
            modelsliceneg = mconvinterpneg(wli) / contneg
            modelsliceplus = mconvinterpplus(wli) / contplus
            axes[i].plot(wl[i], modelsliceneg, 'g')
            axes[i].plot(wl[i], modelsliceplus, 'm')

        axes[i].plot(wl[i], data[i],'b')
        axes[i].plot(wl[i], modelslice, 'r')
        axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
        axes[i].set_title(nicenames[i])
        axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.25)

        axes2[i].plot(wl[i], (data[i]/modelslice) - 1,'b')
        axes2[i].fill_between(wl[i],(data[i] + err[i])/modelslice - 1 ,(data[i]-err[i])/modelslice - 1, facecolor = 'gray', alpha=0.5)
        axes2[i].set_title(nicenames[i])
        axes2[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.25)
        axes2[i].axhspan(-0.01,0.01, facecolor='red', alpha = 0.5)

        if linefit:
            axes[i].set_xlim((bluelow[i],redhigh[i]))
            axes2[i].set_xlim((bluelow[i],redhigh[i]))

    fig.savefig(fl[:-4] + '_bestfitlines.pdf')
    fig2.savefig(fl[:-4] + '_bestfitresiduals.pdf')

def compare_bestfit2(datafl, mcmcfl, z, veldisp, vcj, burnin=-1000, \
        scale = False, linefit = False):

    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333]
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

    nicenames = [r'FeH',r'CaI',r'NaI',r'KI a',r'KI b', r'KI 1.25', \
             r'Pa$\beta$', r'NaI127',r'NaI123']
    index_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']

    if linefit:
        line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']
        index_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']
        mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
        mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12880, 12720, 12390]
        morder =   [1,1,1,1,1,1,1,1]
    else:
        line_name = ['Band1','Band2','Band3','Band4','Band5']
        #mlow = [9700,10550,11550,12350,12665]
        #mhigh = [10450,10965,12200,12590,13050]
        #morder = [8,4,7,2,5]
        mlow = [9700,10120,11550,12350,12665]
        mhigh = [10000,10450,12070,12590,13050]
        morder = [3,3,5,2,5]

        line_name = ['Band1','Band2','Band3','Band4','Band5']

    #Load the necessary information
    data,names,gal,datatype,fitvalues,midvalues,paramnames, linenames, lines = \
            bestfitPrepare(mcmcfl, burnin)

    if datafl == None:
        print('Please input filename for data')
        return
    wl, data, err = preps.preparespecwifis(datafl, z)

    linedefs = [linelow, linehigh, bluelow, bluehigh, redlow, redhigh,\
            line_name, index_name, mlow, mhigh, morder]

    if linefit:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, \
               usecont = False, scale = scale)

    if 'VelDisp' in paramnames:
        w = np.where(np.array(paramnames) == 'VelDisp')
        veldisp = midvalues[w]

    if lines:
        wlg, newm = mcfi.model_spec(midvalues, paramnames, vcjset = vcj, full=True)
    else:
        wlg, newm = mcfi.model_spec(midvalues, paramnames, vcjset = vcj, full=True)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcsp.convolvemodels(wlg, newm, veldisp, reglims=[4000,15000])

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

    modelproc = []

    mpl.close('all')
    fig,ax = mpl.subplots(figsize= (16,8))
    for i in range(len(mlow)):
        #Getting a slice of the model
        wli = wl[i]

        if linefit and (line_name[i] not in linenames):
            print("Skipping "+line_name[i])
            continue

        modelslice = mconvinterp(wli)
        
        #Removing a high-order polynomial from the slice
        if linefit:
            linedefs = [bluelow,bluehigh,redlow,redhigh]
            polyfit = mcsp.removeLineSlope(wlc, mconv,linedefs,i)
            cont = polyfit(wli)
        else:
            if morder[i] == 1:
                linedefs = [bluelow,bluehigh,redlow,redhigh]
                polyfit = mcsp.removeLineSlope(wlc, mconv,linedefs,i)
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wli, modelslice/data[i], morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wli)

        modelslice = modelslice / cont
        modelproc.append(modelslice)
        
        ax.plot(wl[i], data[i],'b')
        ax.plot(wl[i], modelslice, 'r')
        ax.fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
        #ax.set_title(nicenames[i])

        #ax.plot(wl[i], (data[i]/modelslice) - 1,'b')
        #ax.fill_between(wl[i],(data[i] + err[i])/modelslice - 1 ,(data[i]-err[i])/modelslice - 1, facecolor = 'gray', alpha=0.5)
        #ax.set_title(nicenames[i])

    mpl.savefig('/home/elliot/M85_model_comparison.pdf')
    mpl.show()

    return wl, data, modelproc, [wlc,mconv]

def compare_bestfit_animation(datafl, mcmcfl, z, veldisp, vcj, burnin=-1000, \
        scale = False, linefit = False):

    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333]
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

    nicenames = [r'FeH',r'CaI',r'NaI',r'KI a',r'KI b', r'KI 1.25', \
             r'Pa$\beta$', r'NaI127',r'NaI123']
    index_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']

    if linefit:
        line_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']
        index_name = ['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123']
        mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
        mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12880, 12720, 12390]
        morder =   [1,1,1,1,1,1,1,1]
    else:
        line_name = ['Band1','Band2','Band3','Band4','Band5']
        #mlow = [9700,10550,11550,12350,12665]
        #mhigh = [10450,10965,12200,12590,13050]
        #morder = [8,4,7,2,5]
        mlow = [9700,10120,11550,12350,12665]
        mhigh = [10000,10450,12070,12590,13050]
        morder = [3,3,5,2,5]

        line_name = ['Band1','Band2','Band3','Band4','Band5']

    #Load the necessary information
    mcmcdata,names,gal,datatype,fitvalues,midvalues,paramnames, linenames, lines = \
            bestfitPrepare(mcmcfl, burnin)

    if datafl == None:
        print('Please input filename for data')
        return
    wl, data, err = preps.preparespecwifis(datafl, z)

    linedefs = [linelow, linehigh, bluelow, bluehigh, redlow, redhigh,\
            line_name, index_name, mlow, mhigh, morder]

    if linefit:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, scale = scale)
    else:
        wl, data, err = preps.splitspec(wl, data, linedefs, err = err, \
               usecont = False, scale = scale)

    if 'VelDisp' in paramnames:
        w = np.where(np.array(paramnames) == 'VelDisp')
        veldisp = midvalues[w]

    worker = mcmcdata[:,0,:]

    mpl.close('all')
    fig,axes = mpl.subplots(2,3, figsize= (20,12))
    axes = axes.flatten()
    line1, = axes[0].plot([],[],'r', linewidth=3)
    line2, = axes[1].plot([],[],'r', linewidth=3)
    line3, = axes[2].plot([],[],'r', linewidth=3)
    line4, = axes[3].plot([],[],'r', linewidth=3)
    line5, = axes[4].plot([],[],'r', linewidth=3)
    label = axes[5].text(0.2, 0.5, 'INIT', fontsize=12)

    linearr = [line1,line2,line3,line4,line5, label]
    strs = []
    for i in range(worker.shape[0]):
        inputdata = worker[i,:]
        anitext = [str(i)]
        for a,b in zip(inputdata, paramnames):
            anitext.append(str(b)+': '+str(a))
        strs.append('\n'.join(anitext))

    # initialization function: plot the background of each frame
    def init():
        for line in linearr[:-1]:
            line.set_data([], [])
        linearr[-1].set_text('')
        return linearr

    # animation function.  This is called sequentially
    def animate(j):
        inputdata = worker[j,:]
        
        #anitext = ''
        #for a,b in zip(inputdata, paramnames):
        #    anitext += str(b)+': '+str(a)+'\n'
        #linearr[-1].set_text(str(anitext))
        linearr[-1].set_text(strs[j])

        if lines:
            wlg, newm = mcfi.model_spec(inputdata, paramnames, vcjset = vcj, full=True)
        else:
            wlg, newm = mcfi.model_spec(inputdata, paramnames, vcjset = vcj, full=True)

        # Convolve model to previously determined velocity dispersion
        wlc, mconv = mcsp.convolvemodels(wlg, newm, veldisp, reglims=[4000,15000])

        mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)

        modelproc = []

        wls = np.array([])
        models = np.array([])
        for i in range(len(mlow)):
            #Getting a slice of the model
            wli = np.array(wl[i])

            if linefit and (line_name[i] not in linenames):
                print("Skipping "+line_name[i])
                continue

            modelslice = mconvinterp(wli)
            
            #Removing a high-order polynomial from the slice
            if linefit:
                linedefs = [bluelow,bluehigh,redlow,redhigh]
                polyfit = mcsp.removeLineSlope(wlc, mconv,linedefs,i)
                cont = polyfit(wli)
            else:
                if morder[i] == 1:
                    linedefs = [bluelow,bluehigh,redlow,redhigh]
                    polyfit = mcsp.removeLineSlope(wlc, mconv,linedefs,i)
                    cont = polyfit(wli)
                else:
                    pf = np.polyfit(wli, modelslice/data[i], morder[i])
                    polyfit = np.poly1d(pf)
                    cont = polyfit(wli)

            modelslice = modelslice / cont
            #wls = np.append(wls, wli)
            #models = np.append(models, modelslice)
            
            linearr[i].set_data(wli, modelslice)
        return linearr

    for k in range(len(mlow)):
        #Getting a slice of the model
        wli = wl[k]

        if linefit and (line_name[i] not in linenames):
            print("Skipping "+line_name[i])
            continue

        axes[k].plot(wl[k], data[k],'k')
        axes[k].fill_between(wli,data[k] + err[k],data[k]-err[k], facecolor = 'gray', alpha=0.5)

    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                       frames=worker.shape[0], interval=100, blit=True)
    #mpl.savefig('/home/elliot/M85_model_comparison.pdf')
    anim.save('/home/elliot/basic_animation.mp4', writer='ffmpeg')#extra_args=['--verbose-debug','libx264'])
    #return wl, data, modelproc, [wlc,mconv]

def compare_bestfit3(datafl, mcmcfl, z, veldisp, vcj, burnin=-1000, \
        sauron = False, sauron_z=False, sauron_veldisp=False, save = False):

    mpl.close('all')
    #if fl.lower() == 'last':
        #fls = np.sort(glob(
    print(mcmcfl.split('/')[-1])
    dataall = mcsp.load_mcmc_file(mcmcfl)

    #infol = [nworkers, niter, gal, names, high, low, paramnames, linenames, legacy]
    data = dataall[0]
    names = dataall[2][3]
    high = dataall[2][4]
    low = dataall[2][5]
    paramnames = dataall[2][6]
    linenames = dataall[2][7]
    lines = dataall[2][8]

    samples = data[burnin:,:,:].reshape((-1,len(names)))
    truevalues = np.array(list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0)))))
    params_median = truevalues[:,0]
    params_low = params_median - truevalues[:,2]
    params_high = params_median + truevalues[:,1]
    print(params_low)
    print(params_median)
    print(params_high)

    #Line definitions & other definitions
    #WIFIS Defs
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240, 11905]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260, 11935]
    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309, 11935]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333, 11965]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360, 12005]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390, 12025]

    #mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    #mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]
    mlow, mhigh = [],[]
    for i in zip(bluelow, redhigh):
        mlow.append(i[0])
        mhigh.append(i[1])
    morder = [1,1,1,1,1,1,1,1,1,1]

    line_name = np.array(['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123','CaII119'])
    nice_name = np.array(['FeH',r'Ca$\,$I',r'Na$\,$I',r'K$\,$I$\,$a',r'K$\,$I$\,$b',\
                          r'K$\,$I$\,$1.25', r'Pa$\,\beta$',\
                          r'Na$\,$I$\,$1.27', r'Na$\,$I$\,$1.23',r'Ca$\,$II$\,$1.19'])

    linedefs = [np.array([bluelow, bluehigh, linelow, linehigh, redlow,\
            redhigh, mlow, mhigh, morder]), line_name, line_name]

    wl, data, err = preps.preparespecwifis(datafl, z)
    wl, data, err = preps.splitspec(wl, data, linedefs, err = err)

    if sauron != None:
        #SAURON Lines
        bluelow_s =  [4827.875]#, 4946.500, 5142.625]
        bluehigh_s = [4847.875]#, 4977.750, 5161.375]
        linelow_s =  [4847.875]#, 4977.750, 5160.125]
        linehigh_s = [4876.625]#, 5054.000, 5192.625]
        redlow_s =   [4876.625]#, 5054.000, 5191.375]
        redhigh_s =  [4891.625]#, 5065.250, 5206.375]

        mlow_s, mhigh_s = [],[]
        for i in zip(bluelow_s, redhigh_s):
            mlow_s.append(i[0])
            mhigh_s.append(i[1])
        morder_s = [1]#,1,1]
        line_names_s = np.array(['HBeta'])#,'Fe5015','MgB'])

        sauronlines = [np.array([bluelow_s,bluehigh_s,linelow_s,linehigh_s,\
                redlow_s,redhigh_s,mlow_s,mhigh_s,morder_s]),line_names_s,line_names_s]

        ff = fits.open(sauron)
        spec_s = ff[0].data
        wl_s = ff[1].data
        wl_s = wl_s / (1. + sauron_z)
        noise_s = ff[2].data

        wl_s, data_s, err_s = preps.splitspec(wl_s, spec_s, sauronlines, err = noise_s)

        
    paramdict = {}
    for p in paramnames:
        paramdict[p] = None
    
    if sauron:
        wlm, newm_median, base_median = mcfi.model_spec(params_median, paramnames, paramdict, full = True, vcjset=vcj)
        wlm, newm_low, base_low = mcfi.model_spec(params_low, paramnames, paramdict, full = True, vcjset=vcj)
        wlm, newm_high, base_high = mcfi.model_spec(params_high, paramnames, paramdict, full = True, vcjset=vcj)

        # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
        wlc, mconv_median = mcsp.convolvemodels(wlm, newm_median, veldisp)
        mconvinterp_median = spi.interp1d(wlc, mconv_median, kind='cubic', bounds_error=False)
        wlc, mconv_low = mcsp.convolvemodels(wlm, newm_low, veldisp)
        mconvinterp_low = spi.interp1d(wlc, mconv_low, kind='cubic', bounds_error=False)
        wlc, mconv_high = mcsp.convolvemodels(wlm, newm_high, veldisp)
        mconvinterp_high = spi.interp1d(wlc, mconv_high, kind='cubic', bounds_error=False)

        wlc_s, mconv_s_median = mcsp.convolvemodels(wlm, base_median, sauron_veldisp,\
                                                   reglims=(4000,6000))
        mconvinterp_s_median = spi.interp1d(wlc_s, mconv_s_median, kind='cubic', bounds_error=False)
        wlc_s, mconv_s_low = mcsp.convolvemodels(wlm, base_low, sauron_veldisp,\
                                                reglims=(4000,6000))
        mconvinterp_s_low = spi.interp1d(wlc_s, mconv_s_low, kind='cubic', bounds_error=False)
        wlc_s, mconv_s_high = mcsp.convolvemodels(wlm, base_high, sauron_veldisp,\
                                                 reglims=(4000,6000))
        mconvinterp_s_high = spi.interp1d(wlc_s, mconv_s_high, kind='cubic', bounds_error=False)


    else:
        wlm, newm_median, base_median = model_spec(params_median, paramnames, paramdict, vcjset=vcj)
        wlm, newm_low, base_low = model_spec(params_low, paramnames, paramdict, vcjset=vcj)
        wlm, newm_high, base_high = model_spec(params_high, paramnames, paramdict, vcjset=vcj)

        # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
        wlc, mconv_median = mcsp.convolvemodels(wlm, newm_median, veldisp)
        mconvinterp_median = spi.interp1d(wlc, mconv_median, kind='cubic', bounds_error=False)
        wlc, mconv_low = mcsp.convolvemodels(wlm, newm_low, veldisp)
        mconvinterp_low = spi.interp1d(wlc, mconv_low, kind='cubic', bounds_error=False)
        wlc, mconv_high = mcsp.convolvemodels(wlm, newm_high, veldisp)
        mconvinterp_high = spi.interp1d(wlc, mconv_high, kind='cubic', bounds_error=False)
        
    fig, axes = mpl.subplots(2,4,figsize = (16,8))
    axes = axes.flatten()

    verbose = False
    for k in range(len(linenames)):
        if verbose:
            print("Working on ", linenames[k])
        if linenames[k] != 'HBeta':
            i = np.where(line_name == linenames[k])[0][0]
            wh = np.where((wlc >= bluelow[i]) & (wlc <= redhigh[i]))[0]
        else:
            wh = np.where((wlc_s >= bluelow_s[0]) & (wlc_s <= redhigh_s[0]))[0]
        
        #if (self.line_name[i] == 'Pa Beta') and self.pa_raw:
        #    print('Non-telluric PaB enabled')
        #    dataslice = self.target.spectrum[wh]
        #else:
        #    dataslice = data[wh] #/ np.median(data[wh])
            
        #errslice = err[wh] / np.median(data[wh])

        if linenames[k] != 'HBeta':
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= bluelow[i]) & (wlc <= bluehigh[i]))[0]
            redpass = np.where((wlc >= redlow[i]) & (wlc <= redhigh[i]))[0]
            mainpass = np.where((wlc >= linelow[i]) & (wlc <= linehigh[i]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([bluelow[i], bluehigh[i]])
            redavg = np.mean([redlow[i], redhigh[i]])
            
            blueval_median = np.nanmean(mconv_median[bluepass])
            redval_median = np.nanmean(mconv_median[redpass])
            blueval_low = np.nanmean(mconv_low[bluepass])
            redval_low = np.nanmean(mconv_low[redpass])
            blueval_high = np.nanmean(mconv_high[bluepass])
            redval_high = np.nanmean(mconv_high[redpass])
            
        else:
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= bluelow_s[0]) & (wlc <= bluehigh_s[0]))[0]
            redpass = np.where((wlc >= redlow_s[0]) & (wlc <= redhigh_s[0]))[0]
            mainpass = np.where((wlc >= linelow_s[0]) & (wlc <= linehigh_s[0]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([bluelow_s[0], bluehigh_s[0]])
            redavg = np.mean([redlow_s[0], redhigh_s[0]])

            blueval_median = np.nanmean(mconv_s_median[bluepass])
            redval_median = np.nanmean(mconv_s_median[redpass])
            blueval_low = np.nanmean(mconv_s_low[bluepass])
            redval_low = np.nanmean(mconv_s_low[redpass])
            blueval_high = np.nanmean(mconv_s_high[bluepass])
            redval_high = np.nanmean(mconv_s_high[redpass])

        pf_med = np.polyfit([blueavg, redavg], [blueval_median,redval_median], 1)
        pf_low = np.polyfit([blueavg, redavg], [blueval_low,redval_low], 1)
        pf_high = np.polyfit([blueavg, redavg], [blueval_high,redval_high], 1)
        polyfit_med = np.poly1d(pf_med)
        polyfit_low = np.poly1d(pf_low)
        polyfit_high = np.poly1d(pf_high)

        if linenames[k] != 'HBeta':
            dataslice_med = mconvinterp_median(wl[i]) / polyfit_med(wl[i])
            dataslice_low = mconvinterp_low(wl[i]) / polyfit_low(wl[i])
            dataslice_high = mconvinterp_high(wl[i]) / polyfit_high(wl[i])
        else:
            dataslice_med = mconvinterp_s_median(wl_s[0]) / polyfit_med(wl_s[0])
            dataslice_low = mconvinterp_s_low(wl_s[0]) / polyfit_low(wl_s[0])
            dataslice_high = mconvinterp_s_high(wl_s[0]) / polyfit_high(wl_s[0])
        #errslice = self.reducederr[wh]/polyfit(wlslice)
        
        #whm = np.where((mwl >= self.bluelow[i]) & (mwl <= self.redhigh[i]))[0]
        #wlmslice = mwl[whm]
        
        # PLOT GALAXY AND MODELS
        if linenames[k] != 'HBeta':
            axes[k].plot(wl[i], data[i], linewidth = 3.5, color='k', label = 'Galaxy')
            axes[k].fill_between(wl[i],data[i] + err[i], data[i]-err[i],\
                                 facecolor = 'gray', alpha = 0.5)
            axes[k].set_title(nice_name[i], fontsize = 17)

            #Plot REGIONS
            axes[k].axvspan(bluelow[i], bluehigh[i], facecolor='b',\
                            alpha=0.2)
            axes[k].axvspan(redlow[i], redhigh[i],facecolor='b', alpha=0.2)
            axes[k].axvspan(linelow[i], linehigh[i],facecolor='r',\
                            alpha=0.2)
            #axes[i].fill_between(wl[i], dataslice_low, dataslice_high,\
            #                     facecolor = '', alpha = 0.5)
            axes[k].set_xlim((bluelow[i], redhigh[i]))
            
            axes[k].plot(wl[i], dataslice_med, color='r', linewidth = 2.5, label='Model')
            axes[k].plot(wl[i], dataslice_low, color='r', linewidth = 2.5, \
                    linestyle='--')
            axes[k].plot(wl[i], dataslice_high, color='r', linewidth = 2.5, \
                    linestyle='--')

        else:
            axes[k].plot(wl_s[0], data_s[0], linewidth = 3.5, color='k', label = 'Galaxy')
            axes[k].fill_between(wl_s[0],data_s[0] + err_s[0], data_s[0]-err_s[0],\
                                 facecolor = 'gray', alpha = 0.5)
            axes[k].set_title(r'H$\beta$', fontsize = 17)

            #Plot REGIONS
            axes[k].axvspan(bluelow_s[0], bluehigh_s[0], facecolor='b',\
                            alpha=0.2)
            axes[k].axvspan(redlow_s[0], redhigh_s[0],facecolor='b', alpha=0.2)
            axes[k].axvspan(linelow_s[0], linehigh_s[0],facecolor='r',\
                            alpha=0.2)
            axes[k].set_xlim((bluelow_s[0], redhigh_s[0]))

            axes[k].plot(wl_s[0], dataslice_med, color='r', linewidth = 2.5, label='Model')
            axes[k].plot(wl_s[0], dataslice_low, color='r', linewidth = 2.5, \
                    linestyle='--')
            axes[k].plot(wl_s[0], dataslice_high, color='r', linewidth = 2.5, \
                    linestyle='--')


        axes[k].tick_params(axis='both', which='major', labelsize=15)
    mpl.subplots_adjust(hspace=0.2)

    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles,labels, bbox_to_anchor=(1.09, 0.25), fontsize='large')
    fig.legend(handles,labels, fontsize='large')
    #mpl.tight_layout()
    #fig.legend(handles, labels, fontsize='large')

    if save:
        mpl.savefig(save, dpi = 300)
    else:
        mpl.show()        
        
def imf_plots_nice(datafl, z, veldisp, vcj, linenames,\
        sauron = False, sauron_z=False, sauron_veldisp=False, save = False):

    mpl.close('all')
    #if fl.lower() == 'last':
        #fls = np.sort(glob(
    #Line definitions & other definitions
    #WIFIS Defs
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240, 11905]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260, 11935]
    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309, 11935]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333, 11965]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360, 12005]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390, 12025]

    #mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    #mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]
    mlow, mhigh = [],[]
    for i in zip(bluelow, redhigh):
        mlow.append(i[0])
        mhigh.append(i[1])
    morder = [1,1,1,1,1,1,1,1,1,1]

    line_name = np.array(['FeH','CaI','NaI','KI_a','KI_b', 'KI_1.25', 'PaB', 'NaI127', 'NaI123','CaII119'])
    nice_name = np.array(['FeH',r'Ca$\,$I',r'Na$\,$I',r'K$\,$I$\,$a',r'K$\,$I$\,$b',\
                          r'K$\,$I$\,$1.25', r'Pa$\,\beta$',\
                          r'Na$\,$I$\,$1.27', r'Na$\,$I$\,$1.23',r'Ca$\,$II$\,$1.19'])

    linedefs = [np.array([bluelow, bluehigh, linelow, linehigh, redlow,\
            redhigh, mlow, mhigh, morder]), line_name, line_name]

    wl, data, err = preps.preparespecwifis(datafl, z)
    wl, data, err = preps.splitspec(wl, data, linedefs, err = err)

    if sauron != None:
        #SAURON Lines
        bluelow_s =  [4827.875]#, 4946.500, 5142.625]
        bluehigh_s = [4847.875]#, 4977.750, 5161.375]
        linelow_s =  [4847.875]#, 4977.750, 5160.125]
        linehigh_s = [4876.625]#, 5054.000, 5192.625]
        redlow_s =   [4876.625]#, 5054.000, 5191.375]
        redhigh_s =  [4891.625]#, 5065.250, 5206.375]

        mlow_s, mhigh_s = [],[]
        for i in zip(bluelow_s, redhigh_s):
            mlow_s.append(i[0])
            mhigh_s.append(i[1])
        morder_s = [1]#,1,1]
        line_names_s = np.array(['HBeta'])#,'Fe5015','MgB'])

        sauronlines = [np.array([bluelow_s,bluehigh_s,linelow_s,linehigh_s,\
                redlow_s,redhigh_s,mlow_s,mhigh_s,morder_s]),line_names_s,line_names_s]

        ff = fits.open(sauron)
        spec_s = ff[0].data
        wl_s = ff[1].data
        wl_s = wl_s / (1. + sauron_z)
        noise_s = ff[2].data

        wl_s, data_s, err_s = preps.splitspec(wl_s, spec_s, sauronlines, err = noise_s)

    paramnames = ['Z','Age','x1','x2']
    params_median = [0.0,13.5,1.3,2.3]
    params_low = [0.0,13.5,2.3,2.3]
    params_high = [0.0,13.5,3.0,3.0]
        
    paramdict = {}
    for p in paramnames:
        paramdict[p] = None
    
    if sauron:
        wlm, newm_median, base_median = mcfi.model_spec(params_median, paramnames, paramdict, full = True, vcjset=vcj)
        wlm, newm_low, base_low = mcfi.model_spec(params_low, paramnames, paramdict, full = True, vcjset=vcj)
        wlm, newm_high, base_high = mcfi.model_spec(params_high, paramnames, paramdict, full = True, vcjset=vcj)

        # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
        wlc, mconv_median = mcsp.convolvemodels(wlm, newm_median, veldisp)
        mconvinterp_median = spi.interp1d(wlc, mconv_median, kind='cubic', bounds_error=False)
        wlc, mconv_low = mcsp.convolvemodels(wlm, newm_low, veldisp)
        mconvinterp_low = spi.interp1d(wlc, mconv_low, kind='cubic', bounds_error=False)
        wlc, mconv_high = mcsp.convolvemodels(wlm, newm_high, veldisp)
        mconvinterp_high = spi.interp1d(wlc, mconv_high, kind='cubic', bounds_error=False)

        wlc_s, mconv_s_median = mcsp.convolvemodels(wlm, base_median, sauron_veldisp,\
                                                   reglims=(4000,6000))
        mconvinterp_s_median = spi.interp1d(wlc_s, mconv_s_median, kind='cubic', bounds_error=False)
        wlc_s, mconv_s_low = mcsp.convolvemodels(wlm, base_low, sauron_veldisp,\
                                                reglims=(4000,6000))
        mconvinterp_s_low = spi.interp1d(wlc_s, mconv_s_low, kind='cubic', bounds_error=False)
        wlc_s, mconv_s_high = mcsp.convolvemodels(wlm, base_high, sauron_veldisp,\
                                                 reglims=(4000,6000))
        mconvinterp_s_high = spi.interp1d(wlc_s, mconv_s_high, kind='cubic', bounds_error=False)


    else:
        wlm, newm_median, base_median = model_spec(params_median, paramnames, paramdict, vcjset=vcj)
        wlm, newm_low, base_low = model_spec(params_low, paramnames, paramdict, vcjset=vcj)
        wlm, newm_high, base_high = model_spec(params_high, paramnames, paramdict, vcjset=vcj)

        # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
        wlc, mconv_median = mcsp.convolvemodels(wlm, newm_median, veldisp)
        mconvinterp_median = spi.interp1d(wlc, mconv_median, kind='cubic', bounds_error=False)
        wlc, mconv_low = mcsp.convolvemodels(wlm, newm_low, veldisp)
        mconvinterp_low = spi.interp1d(wlc, mconv_low, kind='cubic', bounds_error=False)
        wlc, mconv_high = mcsp.convolvemodels(wlm, newm_high, veldisp)
        mconvinterp_high = spi.interp1d(wlc, mconv_high, kind='cubic', bounds_error=False)
        
    fig, axes = mpl.subplots(2,4,figsize = (16,8))
    axes = axes.flatten()
    diff = 0
    if len(linenames) < 8:
        diff = 8 - len(linenames)
        for i in range(diff):
            axes[-1].set_visible(False)

    verbose = False
    for k in range(len(linenames)):
        if verbose:
            print("Working on ", linenames[k])
        if linenames[k] != 'HBeta':
            i = np.where(line_name == linenames[k])[0][0]
            wh = np.where((wlc >= bluelow[i]) & (wlc <= redhigh[i]))[0]
        else:
            wh = np.where((wlc_s >= bluelow_s[0]) & (wlc_s <= redhigh_s[0]))[0]
        
        #if (self.line_name[i] == 'Pa Beta') and self.pa_raw:
        #    print('Non-telluric PaB enabled')
        #    dataslice = self.target.spectrum[wh]
        #else:
        #    dataslice = data[wh] #/ np.median(data[wh])
            
        #errslice = err[wh] / np.median(data[wh])

        if linenames[k] != 'HBeta':
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= bluelow[i]) & (wlc <= bluehigh[i]))[0]
            redpass = np.where((wlc >= redlow[i]) & (wlc <= redhigh[i]))[0]
            mainpass = np.where((wlc >= linelow[i]) & (wlc <= linehigh[i]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([bluelow[i], bluehigh[i]])
            redavg = np.mean([redlow[i], redhigh[i]])
            
            blueval_median = np.nanmean(mconv_median[bluepass])
            redval_median = np.nanmean(mconv_median[redpass])
            blueval_low = np.nanmean(mconv_low[bluepass])
            redval_low = np.nanmean(mconv_low[redpass])
            blueval_high = np.nanmean(mconv_high[bluepass])
            redval_high = np.nanmean(mconv_high[redpass])
            
        else:
            #Define the bandpasses for each line 
            bluepass = np.where((wlc >= bluelow_s[0]) & (wlc <= bluehigh_s[0]))[0]
            redpass = np.where((wlc >= redlow_s[0]) & (wlc <= redhigh_s[0]))[0]
            mainpass = np.where((wlc >= linelow_s[0]) & (wlc <= linehigh_s[0]))[0]

            #Cacluating center value of the blue and red bandpasses
            blueavg = np.mean([bluelow_s[0], bluehigh_s[0]])
            redavg = np.mean([redlow_s[0], redhigh_s[0]])

            blueval_median = np.nanmean(mconv_s_median[bluepass])
            redval_median = np.nanmean(mconv_s_median[redpass])
            blueval_low = np.nanmean(mconv_s_low[bluepass])
            redval_low = np.nanmean(mconv_s_low[redpass])
            blueval_high = np.nanmean(mconv_s_high[bluepass])
            redval_high = np.nanmean(mconv_s_high[redpass])

        pf_med = np.polyfit([blueavg, redavg], [blueval_median,redval_median], 1)
        pf_low = np.polyfit([blueavg, redavg], [blueval_low,redval_low], 1)
        pf_high = np.polyfit([blueavg, redavg], [blueval_high,redval_high], 1)
        polyfit_med = np.poly1d(pf_med)
        polyfit_low = np.poly1d(pf_low)
        polyfit_high = np.poly1d(pf_high)

        if linenames[k] != 'HBeta':
            dataslice_med = mconvinterp_median(wl[i]) / polyfit_med(wl[i])
            dataslice_low = mconvinterp_low(wl[i]) / polyfit_low(wl[i])
            dataslice_high = mconvinterp_high(wl[i]) / polyfit_high(wl[i])
        else:
            dataslice_med = mconvinterp_s_median(wl_s[0]) / polyfit_med(wl_s[0])
            dataslice_low = mconvinterp_s_low(wl_s[0]) / polyfit_low(wl_s[0])
            dataslice_high = mconvinterp_s_high(wl_s[0]) / polyfit_high(wl_s[0])
        #errslice = self.reducederr[wh]/polyfit(wlslice)
        
        #whm = np.where((mwl >= self.bluelow[i]) & (mwl <= self.redhigh[i]))[0]
        #wlmslice = mwl[whm]
        
        # PLOT GALAXY AND MODELS
        if linenames[k] != 'HBeta':
            axes[k].plot(wl[i], data[i], linewidth = 3.5, color='k', label = 'Observed')
            axes[k].fill_between(wl[i],data[i] + err[i], data[i]-err[i],\
                                 facecolor = 'gray', alpha = 0.5)
            axes[k].set_title(nice_name[i], fontsize = 20)

            #Plot REGIONS
            axes[k].axvspan(bluelow[i], bluehigh[i], facecolor='b',\
                            alpha=0.2)
            axes[k].axvspan(redlow[i], redhigh[i],facecolor='b', alpha=0.2)
            axes[k].axvspan(linelow[i], linehigh[i],facecolor='r',\
                            alpha=0.2)
            #axes[i].fill_between(wl[i], dataslice_low, dataslice_high,\
            #                     facecolor = '', alpha = 0.5)
            axes[k].set_xlim((bluelow[i], redhigh[i]))
            
            axes[k].plot(wl[i], dataslice_med, color='g', linewidth = 2.5, \
                         linestyle='--', label='Kroupa')
            #axes[k].plot(wl[i], dataslice_low, color='r', linewidth = 2.5, \
            #        linestyle='--')
            axes[k].plot(wl[i], dataslice_high, color='r', linewidth = 2.5, \
                    linestyle='--', label='Bottom-Heavy')

        else:
            axes[k].plot(wl_s[0], data_s[0], linewidth = 3.5, color='k', label = 'Observed')
            axes[k].fill_between(wl_s[0],data_s[0] + err_s[0], data_s[0]-err_s[0],\
                                 facecolor = 'gray', alpha = 0.5)
            axes[k].set_title(r'H$\beta$', fontsize = 20)

            #Plot REGIONS
            axes[k].axvspan(bluelow_s[0], bluehigh_s[0], facecolor='b',\
                            alpha=0.2)
            axes[k].axvspan(redlow_s[0], redhigh_s[0],facecolor='b', alpha=0.2)
            axes[k].axvspan(linelow_s[0], linehigh_s[0],facecolor='r',\
                            alpha=0.2)
            axes[k].set_xlim((bluelow_s[0], redhigh_s[0]))

            axes[k].plot(wl_s[0], dataslice_med, color='g', linewidth = 2.5, \
                         linestyle='--', label='Kroupa')
            #axes[k].plot(wl_s[0], dataslice_low, color='r', linewidth = 2.5, \
             #       linestyle='--')
            axes[k].plot(wl_s[0], dataslice_high, color='r', linewidth = 2.5, \
                    linestyle='--', label='Bottom-Heavy')


        axes[k].tick_params(axis='both', which='major', labelsize=20)
    mpl.subplots_adjust(hspace=0.3, wspace=0.25)

    #handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles,labels, bbox_to_anchor=(1.09, 0.25), fontsize='large')
    if diff == 0:
        pass
        #axes[-diff-1].legend(fontsize=20)
    else:
        axes[-diff-1].legend(fontsize=20, loc=(1.2,0.2))
    #mpl.tight_layout()
    #fig.legend(handles, labels, fontsize='large')

    if save:
        mpl.savefig(save, dpi = 300)
    else:
        mpl.show()        
        
def removeLineSlope(self, wlc, mconv, i):
    
    #Define the bandpasses for each line 
    bluepass = np.where((wlc >= self.bluelow[i]) & (wlc <= self.bluehigh[i]))[0]
    redpass = np.where((wlc >= self.redlow[i]) & (wlc <= self.redhigh[i]))[0]
    mainpass = np.where((wlc >= self.linelow[i]) & (wlc <= self.linehigh[i]))[0]

    #Cacluating center value of the blue and red bandpasses
    blueavg = np.mean([self.bluelow[i], self.bluehigh[i]])
    redavg = np.mean([self.redlow[i], self.redhigh[i]])

    blueval = np.nanmean(mconv[bluepass])
    redval = np.nanmean(mconv[redpass])

    pf = np.polyfit([blueavg, redavg], [blueval,redval], 1)
    polyfit = np.poly1d(pf)

    return polyfit, [bluepass, redpass, mainpass]


if __name__ == '__main__':
    pass
