import numpy as np

from astropy.io import fits

import scipy.optimize as spo
import mcmc_fullspec as mcfs
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

def testVelDisp(fl, burnin = -1000, vcjset = False, plot = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = bestfitPrepare(fl, burnin)

    wl, data, err = mcfs.preparespec(gal)
    wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.1)
    shifts = [0.9,-0.8,-1.5,-2.8,-1.1,1.0,-0.6]

    shfit = np.zeros((len(shifts),7))
    for j,sh in enumerate(shifts): 
        if plot:
            fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
            #fig.suptitle(str(sh))
            axes = axes.flatten()
            fig.delaxes(axes[-1])
        for i in range(len(wl)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = np.array(wl[i])
            wli += sh
            #wli += shifts[i]

            modelslice = mconvinterp(wli)
            
            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            if linefit:
                polyfit = mcfs.removeLineSlope(wlc, mconv,i)
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wl[i], modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wl[i])

            modelslice = modelslice / cont
            shfit[j,i] = np.std(data[i] / modelslice)
            
            if plot:
                axes[i].plot(wl[i], modelslice, 'r')
                axes[i].plot(wl[i], data[i],'b')
                axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
                axes[i].set_title(line_name[i] + ' %.2f' % (np.std(data[i] / modelslice)))
                if linefit:
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

        if plot:
            mpl.show()
    for i in range(len(wl)):
        print shifts[np.argmin(shfit[:,i])]

    return shfit

def deriveShifts(fl, burnin = -1000, vcjset = False, plot = False):

    data, fitmode, names, gal, datatype, mcmctype, fitvalues, truevalues, galveldisp\
            = bestfitPrepare(fl, burnin)

    wl, data, err = preps.preparespec(gal)
    wl, data, err = preps.splitspec(wl, data, err = err, lines=linefit)

    wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    shifts = np.arange(-3.0,3.0,0.1)
    shifts = [0.9,-0.8,-1.5,-2.8,-1.1,1.0,-0.6]

    shfit = np.zeros((len(shifts),7))
    for j,sh in enumerate(shifts): 
        if plot:
            fig, axes = mpl.subplots(2,4,figsize = (16,6.5))
            #fig.suptitle(str(sh))
            axes = axes.flatten()
            fig.delaxes(axes[-1])
        for i in range(len(wl)):
            if (gal == 'M87') and linefit:
                if line_name[i] == 'KI_1.25':
                    continue

            #Getting a slice of the model
            wli = np.array(wl[i])
            wli += sh
            #wli += shifts[i]

            modelslice = mconvinterp(wli)
            
            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            if linefit:
                polyfit = mcfs.removeLineSlope(wlc, mconv,i)
                cont = polyfit(wli)
            else:
                pf = np.polyfit(wl[i], modelslice, morder[i])
                polyfit = np.poly1d(pf)
                cont = polyfit(wl[i])

            modelslice = modelslice / cont
            shfit[j,i] = np.std(data[i] / modelslice)
            
            if plot:
                axes[i].plot(wl[i], modelslice, 'r')
                axes[i].plot(wl[i], data[i],'b')
                axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
                axes[i].set_title(line_name[i] + ' %.2f' % (np.std(data[i] / modelslice)))
                if linefit:
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

        if plot:
            mpl.show()
    for i in range(len(wl)):
        print shifts[np.argmin(shfit[:,i])]

    return shfit

def deriveVelDisp(gal):

    wl, data, err = mcfs.preparespec(gal)
    mpl.plot(wl,data)
    mpl.show()
    sys.exit()

    mlow = [9700,10550,11550,12350]
    mhigh = [10450,11450,12200,13180]
    morder = [8,9,7,8]

    databands = []
    wlbands = []
    errorbands = []

    for i in range(len(mlow)):
        wh = np.where( (wl >= mlow[i]) & (wl <= mhigh[i]))[0]
        dataslice = data[wh]
        wlslice = wl[wh]
        wlbands.append(wlslice)

        pf = np.polyfit(wlslice, dataslice, morder[i])
        polyfit = np.poly1d(pf)
        cont = polyfit(wlslice)

        databands.append(dataslice / cont)

        if type(err) != bool:
            errslice = err[wh]
            errorbands.append(errslice / cont)

    data = databands[0]
    wl = wlbands[0]
    
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=False)

    #wlg, newm = mcfs.model_spec(truevalues, gal, fitmode = fitmode, vcjset = vcjset)
    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #veldists = np.arange(120,390,10)
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)

    #mconvinterp = spi.interp1d(wlc, mconv, kind='cubic', bounds_error=False)
    #9820, 9855
    #10850,10910
    #11800,11870
    #11870,11920
    #12780,12860
    #13010,13070
    #reg = (wl >= 9820) & (wl <= 9855)
    
    #wl = wl[reg]
    #data = data[reg]
    #popt, pcov = gaussian_fit_os(wl, data, [-0.005, 5.0, 9840, 1.0])
    #gaus = gaussian_os(wl, popt[0], popt[1], popt[2], popt[3])
    #gaus2 = gaussian_os(wl, -0.035, 8.25, popt[2], popt[3])

    #print popt

    #mpl.plot(wl, data)
    #mpl.plot(wl, gaus)
    #mpl.plot(wl,gaus2)
    #mpl.show()

    c = 299792.458

    m_center = 11834
    m_sigma = 6.8
    #m_sigma = np.abs((m_center / (1 + 100./c)) - m_center)
    f = m_center + m_sigma
    v = c * ((f/m_center) - 1)
    print v
    
    #sigma_gal = np.abs((m_center * (veldisp/c + 1.)) - m_center)
    #sigma_conv = np.sqrt(sigma_gal**2. - m_sigma**2.)

    #convolvex = np.arange(-5*sigma_conv,5*sigma_conv, 2.0)
    #gaussplot = mcsp.gauss_nat(convolvex, [sigma_conv,0.])

    #out = np.convolve(datafull, gaussplot, mode='same')

def compareModels(paramslist,pdescrip, gal, vcjset, runvalues = False):

    mpl.close('all')
    if runvalues:
        data, names, gal, datatype, fitvalues, truevalues,paramnames, linenames =\
                bestfitPrepare(runvalues, burnin)
        paramslist.append(truevalues)
        pdescript.append("Best-Fit")

    paramnames = ['Age','Z','x1','x2','Na','Fe','Ca','K']

    wl, data, err = mcfs.preparespec(gal)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    mdataarr = []
    for params in paramslist:
        wl1, newm1 = mcfs.model_spec(params1, gal, paramnames, vcjset = vcjset)
        wlg1, data1, err1 = mcfs.splitspec(wl1, newm1, err = False, lines=linefit)
        mdataarr.append(data1)

    #wl2, newm2 = mcfs.model_spec(params2, gal, paramnames, vcjset = vcjset)
    #wlg2, data2, err2 = mcfs.splitspec(wl2, newm2, err = False, lines=linefit)

    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    styling = False
    for modeldata in mdataarr:
        for i in range(len(mlow)):
            #print line_name[i]

            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            #polyfit = mcfs.removeLineSlope(wl1, data1, i)
            #cont1 = polyfit(wlg1[i])

            #polyfit = mcfs.removeLineSlope(wl1, data2, i)
            #cont2 = polyfit(wls[i])

            #modelslice = modelslice / cont
            #otherslice = datas2[i] / cont2 
            
            #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
            axes[i].plot(wlg1[i], modeldata[i], label = pdescrip[i])
            #axes[i].plot(wlg1[i], data2[i],'g', label = '2')
            #axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)

            if not styling:
                axes[i].set_title(line_name[i])
                axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.2)
                axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.2)
                axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.2)
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        styling = True
        
        #mpl.title(chem_names[j])
        #mpl.savefig('/home/elliot/chemplots/'+ chem_names[j]+'.png',dpi=500)
    mpl.show()
    mpl.close('all')

def plotModels(modelfile, chemfile, gal):

    mpl.close('all')

    #fig.delaxes(axes[-1])

    wl, data, err = mcfs.preparespec(gal)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    x = pd.read_table(modelfile, delim_whitespace = True, header=None)
    x = np.array(x)
    mwl = x[:,0]
    mdata = x[:,74]

    chem_names = ['WL','Solar', 'Na+', 'Na-', 'Ca+', 'Ca-', 'Fe+', 'Fe-', 'C+', 'C-', 'a_o_Fe+', 'N+', 'N-', 'as_o_Fe+', 'Ti+', 'Ti-',\
                    'Mg+', 'Mg-', 'Si+', 'Si-', 'T+', 'T-', 'Cr+', 'Mn+', 'Ba+', 'Ba-', 'Ni+', 'Co+', 'Eu+', 'Sr+', 'K+',\
                    'V+', 'Cu+', 'Na+0.6', 'Na+0.9']
    #chem_names = ['WL','Solar', 'Nap', 'Nam', 'Cap', 'Cam', 'Fep', 'Fem', 'Cp', 'Cm', 'a/Fep', 'Np', 'Nm', 'as/Fep', 'Tip', 'Tim',\
    #                'Mgp', 'Mgm', 'Sip', 'Sim', 'Tp', 'Tm', 'Crp', 'Mnm', 'Bap', 'Bam', 'Nip', 'Cop', 'Eup', 'Srp', 'Kp',\
    #                'Vp', 'Cup', 'Nap0.6', 'Nap0.9']

    x = pd.read_table(chemfile, skiprows=2, names = chem_names, delim_whitespace = True, header=None)
    chems = np.array(x)
    chemwl = chems[:,0]
    chemsolar = chems[:,1]

    for j in range(2, len(chem_names)):
        print chem_names[j]
        fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
        axes = axes.flatten()

        Feplus = chems[:,j]
        Feeffect = Feplus/chemsolar

        mconvinterp_chem = spi.interp1d(chemwl,Feeffect , kind='cubic', bounds_error=False)
        adjust = mconvinterp_chem(wl)

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)

        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)
        wls2, datas2, err2 = mcfs.splitspec(wl, newmdata * adjust, err = False, lines=linefit)

        for i in range(len(mlow)):
            #print line_name[i]

            modelslice = datas[i]
            
            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            polyfit = mcfs.removeLineSlope(wl, newmdata,i)
            cont = polyfit(wls[i])
            polyfit = mcfs.removeLineSlope(wl, newmdata * adjust,i)
            cont2 = polyfit(wls[i])

            modelslice = modelslice / cont
            otherslice = datas2[i] / cont2 
            
            #axes[i].plot(wl[i], (data[i] - modelslice)/data[i] * 100.0, 'r')
            axes[i].plot(wls[i], datas[i],'b')
            axes[i].plot(wls[i], datas2[i],'g')
            #axes[i].fill_between(wl[i],data[i] + err[i],data[i]-err[i], facecolor = 'gray', alpha=0.5)
            axes[i].set_title(line_name[i])
            axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
            axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
            axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
            
            if linefit:
                axes[i].set_xlim((bluelow[i],redhigh[i]))
        
        mpl.title(chem_names[j])
        mpl.savefig('/home/elliot/chemplots/'+ chem_names[j]+'.png',dpi=500)
        mpl.close('all')

def plotZlines(Age, gal):

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    wl, data, err = mcfs.preparespec(gal)

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)


    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    plotstyle = False
    for j in range(len(modelfiles)):
        mage, mz = parsemodelname(modelfiles[j])
        if mage != Age:
            continue
        if mz < -0.6:
            continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdata = x[:,74]

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)
        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)

        for i in range(len(mlow)):

            #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
            #Removing a high-order polynomial from the slice

            polyfit = mcfs.removeLineSlope(wl, newmdata, i)
            cont = polyfit(wls[i])

            modelslice = datas[i] / cont
            
            if i == 0:
                if mz == 0.2:
                    axes[i].plot(wls[i], datas[i],'b', label = str(mz))
                elif mz == 0.0:
                    axes[i].plot(wls[i], datas[i],'g', label = str(mz))
                elif mz == -0.5:
                    axes[i].plot(wls[i], datas[i],'r', label = str(mz))
            else:
                if mz == 0.2:
                    axes[i].plot(wls[i], datas[i],'b')
                elif mz == 0.0:
                    axes[i].plot(wls[i], datas[i],'g')
                elif mz == -0.5:
                    axes[i].plot(wls[i], datas[i],'r')

            if not plotstyle:
                axes[i].set_title(line_name[i])
                axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
                axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
                axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        plotstyle = True
        
    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    mpl.savefig('/home/elliot/chemplots/Age_'+ str(Age)+'.png',dpi=500)
    mpl.close('all')

def plotAgeZcontours():

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    wl, data, err = mcfs.preparespec('M85')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).


    plotstyle = False
    widths = {}
    for j in range(len(modelfiles)):
        print modelfiles[j]
        mage, mz = parsemodelname(modelfiles[j])
        widths[(mage,mz)] = []

        #if mage != Age:
        #    continue
        #if mz < -0.6:
        #    continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdata = x[:,221]

        #mwl, mdata = mcfs.convolvemodels(mwl, mdata, 370.)
        #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

        mconvinterp = spi.interp1d(mwl, mdata, kind='cubic', bounds_error=False)
        newmdata = mconvinterp(wl)
        wls, datas, err = mcfs.splitspec(wl, newmdata, err = False, lines=linefit)

        for i in range(len(mlow)):
            eqw = measure_line(wl, newmdata,i)
            widths[(mage,mz)].append(eqw)

    #Z_m = np.array([-1.5,-1.0, -0.5, 0.0, 0.2])
    Z_m = np.array([-0.5, 0.0, 0.2])
    Age_m = np.array([1.0,3.0,5.0,7.0,9.0,11.0,13.5])

    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    for i in range(len(mlow)):
        print line_name[i]
        linewidths = []
        for z in Z_m:
            lw_z = []
            for age in Age_m:
                lw_z.append(widths[(age,z)][i])
            linewidths.append(lw_z)
        finalwidths = np.array(linewidths)
        CS = axes[i].contour(np.array(Age_m), np.array(Z_m), finalwidths)
        axes[i].clabel(CS, inline=1, fontsize=10, colors='k')
        axes[i].set_title(line_name[i])

    #fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    mpl.savefig('/home/elliot/chemplots/ContoursBottomHeavy.png',dpi=500)
    mpl.close('all')

    return widths

def plotAgeLines(Age):

    linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309]
    linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333]
    bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260]
    redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360]
    redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

    mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
    mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

    line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25', 'PaB', 'NaI127', 'NaI123']

    morder = [1,1,1,1,1,1,1,1,1]
    linedefs = [linelow, linehigh, bluelow, bluehigh, redlow,\
            redhigh, line_name, line_name, mlow, mhigh, morder]
    linedefs2 = [bluelow, bluehigh, redlow, redhigh]

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    fig, axes = mpl.subplots(2,5,figsize = (16,6.5))
    axes = axes.flatten()
    fig.delaxes(axes[-1])

    plotstyle = False
    for j in range(len(modelfiles)):
        mage, mz = mcsp.parsemodelname(modelfiles[j])
        if mage != Age:
            continue
        if mz != 0.0:
            continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdatas = [x[:,73], x[:,153], x[:,221]]

        for j, mdata in enumerate(mdatas):
            wls, datas, errs = preps.splitspec(mwl, mdata, linedefs, err = False)

            for i in range(len(mlow)):

                #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
                #Removing a high-order polynomial from the slice

                polyfit = mcsp.removeLineSlope(mwl, mdata, linedefs2, i)
                cont = polyfit(wls[i])

                modelslice = datas[i] / cont
                
                if i == 0:
                    if j == 0:
                        axes[i].plot(wls[i], datas[i],'b', label = 'Kroupa')
                    elif j == 1:
                        axes[i].plot(wls[i], datas[i],'g', label = 'Salpeter')
                    elif j == 2:
                        axes[i].plot(wls[i], datas[i],'r', label = 'Bottom-Heavy')
                else:
                    if j == 0:
                        axes[i].plot(wls[i], datas[i],'b')
                    elif j == 1:
                        axes[i].plot(wls[i], datas[i],'g')
                    elif j == 2:
                        axes[i].plot(wls[i], datas[i],'r')

                if not plotstyle:
                    axes[i].set_title(line_name[i])
                    axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
                    axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
                    axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

            plotstyle = True
        
    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    mpl.savefig('/home/elliot/chemplots/AgeIMF_'+ str(Age)+'.png',dpi=500)
    mpl.close('all')

def plotIMFLines(Age, Z, sauron = False):

    if sauron:
        bluelow =  [4827.875, 4946.500, 5142.625]
        bluehigh = [4847.875, 4977.750, 5161.375]
        linelow =  [4847.875, 4977.750, 5160.125]
        linehigh = [4876.625, 5054.000, 5192.625]
        redlow =   [4876.625, 5054.000, 5191.375]
        redhigh =  [4891.625, 5065.250, 5206.375]

        mlow, mhigh = [],[]
        for i in zip(bluelow, redhigh):
            mlow.append(i[0])
            mhigh.append(i[1])
        morder = [1,1,1]
        line_name = np.array(['HBeta','Fe5015','MgB'])

        linedefs = [np.array([bluelow,bluehigh,linelow,linehigh,\
                redlow,redhigh,mlow,mhigh,morder]),line_name,line_name]
        #linedefs = [linelow, linehigh, bluelow, bluehigh, redlow,\
        #        redhigh, line_name, line_name, mlow, mhigh, morder]
        linedefs2 = [bluelow, bluehigh, redlow, redhigh]
    else:
        linelow =  [9905, 10337, 11372, 11680, 11765, 12505, 12810, 12670, 12309]
        linehigh = [9935, 10360, 11415, 11705, 11793, 12545, 12840, 12690, 12333]
        bluelow =  [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
        bluehigh = [9880, 10320, 11370, 11680, 11750, 12495, 12800, 12660, 12260]
        redlow =   [9940, 10365, 11417, 11710, 11793, 12555, 12860, 12700, 12360]
        redhigh =  [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

        mlow =     [9855, 10300, 11340, 11667, 11710, 12460, 12780, 12648, 12240]
        mhigh =    [9970, 10390, 11447, 11750, 11810, 12590, 12870, 12720, 12390]

        line_name = ['FeH','CaI','NaI','KI a','KI b', 'KI 1.25', 'PaB', 'NaI127', 'NaI123']

        morder = [1,1,1,1,1,1,1,1,1]
        linedefs = [np.array([bluelow,bluehigh,linelow,linehigh,\
                redlow,redhigh,mlow,mhigh,morder]),line_names,line_names]
        #linedefs = [linelow, linehigh, bluelow, bluehigh, redlow,\
        #        redhigh, line_name, line_name, mlow, mhigh, morder]
        linedefs2 = [bluelow, bluehigh, redlow, redhigh]

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    if sauron:
        fig, axes = mpl.subplots(1,3,figsize = (16,6.5))
    else:
        fig, axes = mpl.subplots(2,5,figsize = (16,6.5))

    axes = axes.flatten()

    if not sauron:
        fig.delaxes(axes[-1])

    plotstyle = False
    for j in range(len(modelfiles)):
        mage, mz = mcsp.parsemodelname(modelfiles[j])
        if mage != Age:
            continue
        if mz != Z:
            continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        mdatas = [x[:,68], x[:,73], x[:,153], x[:,221]]

        for j, mdata in enumerate(mdatas):
            wls, datas, errs = preps.splitspec(mwl, mdata, linedefs, err = False)

            for i in range(len(mlow)):

                #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
                #Removing a high-order polynomial from the slice

                polyfit = mcsp.removeLineSlope(mwl, mdata, linedefs2, i)
                cont = polyfit(wls[i])

                modelslice = datas[i] / cont
                
                if i == 0:
                    if j == 0:
                        axes[i].plot(wls[i], datas[i],'m', label = 'Bottom-Light')
                    elif j == 1:
                        axes[i].plot(wls[i], datas[i],'b', label = 'Kroupa')
                    elif j == 2:
                        axes[i].plot(wls[i], datas[i],'g', label = 'Salpeter')
                    elif j == 3:
                        axes[i].plot(wls[i], datas[i],'r', label = 'Bottom-Heavy')
                else:
                    if j == 0:
                        axes[i].plot(wls[i], datas[i],'m')
                    if j == 1:
                        axes[i].plot(wls[i], datas[i],'b')
                    elif j == 2:
                        axes[i].plot(wls[i], datas[i],'g')
                    elif j == 3:
                        axes[i].plot(wls[i], datas[i],'r')

                if not plotstyle:
                    axes[i].set_title(line_name[i])
                    axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
                    axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
                    axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
                    axes[i].set_xlim((bluelow[i],redhigh[i]))

            plotstyle = True
        
    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    if sauron:
        mpl.savefig('/home/elliot/chemplots/AgeZIMF_'+ str(Age)+'_'+str(Z)+'_SAURON.png',dpi=500)
    else:
        mpl.savefig('/home/elliot/chemplots/AgeZIMF_'+ str(Age)+'_'+str(Z)+'.png',dpi=500)
    mpl.close('all')


def CompareModels(Age, Z, wl1 = 9000, wl2 = 13500): 
    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wlc, mconv = mcfs.convolvemodels(wlg, newm, galveldisp)
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)
    fig, ax = mpl.subplots(figsize = (16,6.5))

    for j in range(len(modelfiles)):
        mage, mz = mcsp.parsemodelname(modelfiles[j])
        if mage != Age:
            continue
        if mz != Z:
            continue

        x = pd.read_table(modelfiles[j], delim_whitespace = True, header=None)
        x = np.array(x)
        mwl = x[:,0]
        #mdatas = [x[:,68], x[:,73], x[:,153], x[:,221]]
        mdata = x[:,73]

        wls, datas, errs = preps.splitspec(mwl, mdata, linedefs, err = False)


        #-1.8/1.75, -1.4/1.35,-1.35/1.3, -0.95/0.9,-0.5/0.45,-0.4/0.35,-1.4,1.35
        #Removing a high-order polynomial from the slice

        polyfit = mcsp.removeLineSlope(mwl, mdata, linedefs2, i)
        cont = polyfit(wls[i])

        modelslice = datas[i] / cont
        
        if i == 0:
            if j == 0:
                axes[i].plot(wls[i], datas[i],'m', label = 'Bottom-Light')
            elif j == 1:
                axes[i].plot(wls[i], datas[i],'b', label = 'Kroupa')
            elif j == 2:
                axes[i].plot(wls[i], datas[i],'g', label = 'Salpeter')
            elif j == 3:
                axes[i].plot(wls[i], datas[i],'r', label = 'Bottom-Heavy')
        else:
            if j == 0:
                axes[i].plot(wls[i], datas[i],'m')
            if j == 1:
                axes[i].plot(wls[i], datas[i],'b')
            elif j == 2:
                axes[i].plot(wls[i], datas[i],'g')
            elif j == 3:
                axes[i].plot(wls[i], datas[i],'r')

        if not plotstyle:
            axes[i].set_title(line_name[i])
            axes[i].axvspan(linelow[i], linehigh[i], facecolor='m', alpha=0.3)
            axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='b', alpha=0.3)
            axes[i].axvspan(redlow[i], redhigh[i], facecolor='r', alpha=0.3)
            axes[i].set_xlim((bluelow[i],redhigh[i]))

    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    if sauron:
        mpl.savefig('/home/elliot/chemplots/AgeZIMF_'+ str(Age)+'_'+str(Z)+'_SAURON.png',dpi=500)
    else:
        mpl.savefig('/home/elliot/chemplots/AgeZIMF_'+ str(Age)+'_'+str(Z)+'.png',dpi=500)
    mpl.close('all')

def CompareFitSauron(fl, sauron_z, veldisp, ageZ, vcj, save = False): 
    mpl.close('all')
    
    ff = fits.open(fl)
    spec_s = ff[0].data
    wl_s = ff[1].data
    wl_s = wl_s / (1. + sauron_z)
    noise_s = ff[2].data

    bluelow =  [4827.875, 4946.500, 5142.625]
    bluehigh = [4847.875, 4977.750, 5161.375]
    linelow =  [4847.875, 4977.750, 5160.125]
    linehigh = [4876.625, 5054.000, 5192.625]
    redlow =   [4876.625, 5054.000, 5191.375]
    redhigh =  [4891.625, 5065.250, 5206.375]

    mlow, mhigh = [],[]
    for i in zip(bluelow, redhigh):
        mlow.append(i[0])
        mhigh.append(i[1])
    morder = [1,1,1]
    line_name = np.array(['HBeta','Fe5015','MgB'])

    linedefs = [np.array([bluelow,bluehigh,linelow,linehigh,\
            redlow,redhigh,mlow,mhigh,morder]),line_name,line_name]
    #linedefs = [linelow, linehigh, bluelow, bluehigh, redlow,\
    #        redhigh, line_name, line_name, mlow, mhigh, morder]
    linedefs2 = [bluelow, bluehigh, redlow, redhigh]

    wls_s, datas_s, errs_s = preps.splitspec(wl_s, spec_s, linedefs, usecont=False)

    mpl.close('all')

    #fig.delaxes(axes[-1])
    modelfiles = glob('/home/elliot/mcmcgemini/spec/vcj_ssp/*')

    # Convolve model to previously determined velocity dispersion (we don't fit dispersion in this code).
    #wl, data, err = mcfs.splitspec(wl, data, err = err, lines=linefit)

    fig, axes = mpl.subplots(1,3,figsize = (16,6.5))
    axes = axes.flatten()

    for i in range(len(mlow)):

        polyfit = mcsp.removeLineSlope(wl_s, spec_s, linedefs2, i)
        cont = polyfit(wls_s[i])

        modelslice = datas_s[i] / cont

        if i == 0:
            axes[i].plot(wls_s[i], modelslice, 'k', label = 'Data')
        else:
            axes[i].plot(wls_s[i], modelslice, 'k')

    plotstyle = False
    for age, z in ageZ:

        truevalues = [age,z]
        paramnames = ['Age','Z']
        paramdict = {'Age':None,'Z':None}

        wlg, newm, base = mcfi.model_spec(truevalues, paramnames, paramdict, saurononly=True, vcjset = vcj)

        wlc, mconv = mcsp.convolvemodels(wlg, newm, veldisp)
        wls, datas, errs = preps.splitspec(wlc, mconv, linedefs, err = False, usecont=False)
        
        for i in range(len(mlow)):

            polyfit = mcsp.removeLineSlope(wlc, mconv, linedefs2, i)
            cont = polyfit(wls[i])

            modelslice = datas[i] / cont
            
            if i == 0:
                axes[i].plot(wls[i], modelslice, label = str(age)+', '+str(z))
            else:
                axes[i].plot(wls[i], modelslice)

            if not plotstyle:
                axes[i].set_title(line_name[i])
                axes[i].axvspan(linelow[i], linehigh[i], facecolor='b', alpha=0.2)
                axes[i].axvspan(bluelow[i], bluehigh[i], facecolor='m', alpha=0.2)
                axes[i].axvspan(redlow[i], redhigh[i], facecolor='m', alpha=0.2)
                axes[i].set_xlim((bluelow[i],redhigh[i]))

        plotstyle = True
        
    fig.legend(bbox_to_anchor=(0.9, 0.4))
    #mpl.title(chem_names[j])
    if save:
        mpl.savefig('/home/elliot/chemplots/AgeZCompareSAURON_'+save+'.png',dpi=500)
    else:
        mpl.show()
    mpl.close('all')
