from __future__ import print_function
import imf_mass as imf
import compare_bestfit as cbf 
import numpy as np
import matplotlib.pyplot as plt
import mcmc_fullindex as mcfi

def calculate_MLR(fl, burnin = -1000, vcjset = None, verbose = False):

    #Load the best-fit parameters
    data, names, gal, datatype, fitvalues, truevalues, \
    paramnames, linenames, lines = cbf.bestfitPrepare(fl, burnin)

    #Generate the IMF exponent arrays and get indices for various parameters
    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)
    ix1 = np.where(paramnames == 'x1')[0][0]
    ix2 = np.where(paramnames == 'x2')[0][0]
    iage = np.where(paramnames == 'Age')[0][0]
    
    paramdict = {}
    for n in paramnames:
        paramdict[n] = None
    
    #Generate 2d array for every IMF to hold M/L values
    MLR = np.zeros((len(x_m),len(x_m))) #Array to hold the M/L values
    MWvalues = np.array(truevalues) #Generate array with same file params but w/ MW IMF
    MWvalues[ix1] = 1.3 # Change IMF x1
    MWvalues[ix2] = 2.3 # Change IMF x2

    # Create model spectrum for best-fit run params & MW-imf
    wlgMW, newmMW, basemMW = mcfi.model_spec(MWvalues, paramnames, paramdict, \
                        vcjset = vcjset, MLR = True) #Generate the MW spectrum
    
    #Get the best fit age
    bestage = MWvalues[iage]
    print("Age: ", bestage)
    
    #Get interpolation function
    interps = imf.mass_ratio_prepare_isochrone()
    #Calculate Kroupa IMF integrals
    MWremaining, to_mass, normconstant = imf.determine_mass_ratio_isochrone(1.3,2.3, bestage, interps)
    #Calculate Remnant mass at turnoff mass
    massremnant = imf.massremnant(interps[-1], to_mass)
    #Caluclate final remaining mass
    MWremaining += massremnant*normconstant
    print("MW Mass: ", MWremaining)
    
    whK = np.where((wlgMW >= 20300) & (wlgMW <= 23700))[0] 

    MLR_MW = np.sum(newmMW[whK])
    mlrarr = []

    vals = [(0.5,0.5),(1.3,1.3),(1.3,2.3),(2.3,2.3),(2.9,2.9),(3.1,3.1),(3.5,3.5)]
    MLRDict = {}

    for i in range(len(x_m)):
        for j in range(len(x_m)):
            x1 = x_m[i]
            x2 = x_m[j]
            tempvalues = np.array(truevalues)
            tempvalues[ix1] = x1
            tempvalues[ix2] = x2
            wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                            vcjset = vcjset, MLR = True)
            MLR_IMF = np.sum(newm[whK])

            #if (x1 >= 3.0) and (x2 >= 3.0):
            #    mpl.plot(wlg[whK], newm[whK],linestyle='dashed')
            #else:
            #    mpl.plot(wlg[whK], newm[whK])

            #if (x1,x2) in vals:
            #    MLRDict[(x1,x2)] = MLR_IMF

            IMFremaining, to_massimf, normconstantimf = \
                    imf.determine_mass_ratio_isochrone(x1,x2, bestage, interps)
            IMFremaining += massremnant*normconstantimf
            if verbose:
                print(x1, x2, IMFremaining, MWremaining, MLR_IMF, MLR_MW)

            mlrarr.append(IMFremaining)
            if (x1,x2) in vals:
                MLRDict[(x1,x2)] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = MLR_MW/ MLR_IMF

            MLR[i,j] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
            #MLR[i,j] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
            #MLR[i,j] = MLR_MW / MLR_IMF
    #mpl.show()
    #sys.exit()

    plt.close('all')
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    print(samples.shape)
    
    x1 = samples[:,ix1]
    x2 = samples[:,ix2]

    x_mbins = 0.4 + np.arange(17)/5.0
    histprint = plt.hist2d(x1,x2, bins = x_mbins)

    fullMLR = []
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            n_val = int(histprint[0][i,j])
            addlist = [float(MLR[i,j])] * n_val
            fullMLR.extend(addlist)

    plt.close('all')
    percentiles = np.percentile(fullMLR, [16,50,84], axis = 0)
    print(fl, np.percentile(fullMLR, [16, 50, 84], axis=0))

    return MLR, MLRDict, paramnames, truevalues, histprint, mlrarr, percentiles, fullMLR

def calculate_MLR_func(paramnames, truevalues, names, samples, vcjset = None, verbose = False):

    #Load the best-fit parameters
    #data, names, gal, datatype, fitvalues, truevalues, \
    #paramnames, linenames, lines = cbf.bestfitPrepare(fl, burnin)

    #Generate the IMF exponent arrays and get indices for various parameters
    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)
    ix1 = np.where(paramnames == 'x1')[0][0]
    ix2 = np.where(paramnames == 'x2')[0][0]
    iage = np.where(paramnames == 'Age')[0][0]
    
    paramdict = {}
    for n in paramnames:
        paramdict[n] = None
    
    #Generate 2d array for every IMF to hold M/L values
    MLR = np.zeros((len(x_m),len(x_m))) #Array to hold the M/L values
    MWvalues = np.array(truevalues) #Generate array with same file params but w/ MW IMF
    MWvalues[ix1] = 1.3 # Change IMF x1
    MWvalues[ix2] = 2.3 # Change IMF x2

    # Create model spectrum for best-fit run params & MW-imf
    wlgMW, newmMW, basemMW = mcfi.model_spec(MWvalues, paramnames, paramdict, \
                        vcjset = vcjset, MLR = True) #Generate the MW spectrum
    
    #Get the best fit age
    bestage = MWvalues[iage]
    if verbose:
        print("Age: ", bestage)
    
    #Get interpolation function
    interps = imf.mass_ratio_prepare_isochrone()
    #Calculate Kroupa IMF integrals
    MWremaining, to_mass, normconstant = imf.determine_mass_ratio_isochrone(1.3,2.3, bestage, interps)
    #Calculate Remnant mass at turnoff mass
    massremnant = imf.massremnant(interps[-1], to_mass)
    #Caluclate final remaining mass
    MWremaining += massremnant*normconstant
    if verbose:
        print("MW Mass: ", MWremaining)
    
    whK = np.where((wlgMW >= 20300) & (wlgMW <= 23700))[0] 

    MLR_MW = np.sum(newmMW[whK])
    mlrarr = []

    vals = [(0.5,0.5),(1.3,1.3),(1.3,2.3),(2.3,2.3),(2.9,2.9),(3.1,3.1),(3.5,3.5)]
    MLRDict = {}

    for i in range(len(x_m)):
        for j in range(len(x_m)):
            x1 = x_m[i]
            x2 = x_m[j]
            tempvalues = np.array(truevalues)
            tempvalues[ix1] = x1
            tempvalues[ix2] = x2
            wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                            vcjset = vcjset, MLR = True)
            MLR_IMF = np.sum(newm[whK])

            #if (x1 >= 3.0) and (x2 >= 3.0):
            #    mpl.plot(wlg[whK], newm[whK],linestyle='dashed')
            #else:
            #    mpl.plot(wlg[whK], newm[whK])

            #if (x1,x2) in vals:
            #    MLRDict[(x1,x2)] = MLR_IMF

            IMFremaining, to_massimf, normconstantimf = \
                    imf.determine_mass_ratio_isochrone(x1,x2, bestage, interps)
            IMFremaining += massremnant*normconstantimf
            if verbose:
                print(x1, x2, IMFremaining, MWremaining, MLR_IMF, MLR_MW)

            mlrarr.append(IMFremaining)
            if (x1,x2) in vals:
                MLRDict[(x1,x2)] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
                #MLRDict[(x1,x2)] = MLR_MW/ MLR_IMF

            MLR[i,j] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
            #MLR[i,j] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
            #MLR[i,j] = MLR_MW / MLR_IMF
    #mpl.show()
    #sys.exit()

    plt.close('all')
    #print(burnin)
    #samples = data[burnin:,:,:].reshape((-1,len(names)))
    if verbose:
        print(samples.shape)
    
    x1 = samples[:,ix1]
    x2 = samples[:,ix2]

    x_mbins = 0.4 + np.arange(17)/5.0
    histprint = plt.hist2d(x1,x2, bins = x_mbins)

    fullMLR = []
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            n_val = int(histprint[0][i,j])
            addlist = [float(MLR[i,j])] * n_val
            fullMLR.extend(addlist)

    plt.close('all')
    percentiles = np.percentile(fullMLR, [16,50,84], axis = 0)
    if verbose:
        print(fl, np.percentile(fullMLR, [16, 50, 84], axis=0))

    return MLR, MLRDict, paramnames, truevalues, histprint, mlrarr, percentiles, fullMLR


def plotMLRhist(M87, M85):

    fig, ax = mpl.subplots(figsize = (7,6))

    h1 = ax.hist(M87[-1], bins = 15, alpha = 0.7, label = 'M87')
    h2 = ax.hist(M85[-1], bins = 20, alpha = 0.7, label = 'M85')

    print(np.median(M87[-1]))
    print(np.median(M85[-1]))
    print(np.percentile(M85[-1], [16, 50, 84], axis=0))
    print(np.percentile(M87[-1], [16, 50, 84], axis=0))

    ax.axvline(np.median(M87[-1]), linestyle = '--', color = 'b', linewidth=2)#, label='M87 Mean')
    ax.axvline(np.median(M85[-1]), linestyle = '--', color = 'g', linewidth=2)#, label = 'M85 Mean')
    ax.axvline(1.0, color = 'k')
    ax.text(0.7, ax.get_ylim()[1]/1.75,'Kroupa', rotation = 'vertical', fontsize=13)
    #ax.axvline(1.581, color = 'k')
    ax.axvline(1.8, color = 'k')
    #ax.text(1.6, ax.get_ylim()[1]/1.75,'Salpeter', rotation = 'vertical', fontsize = 13)
    ax.text(1.85, ax.get_ylim()[1]/1.75,'Salpeter', rotation = 'vertical', fontsize = 13)

    #ax.set_xlabel('$(M/L)_{K}/(M/L)_{K,MW}$')
    ax.set_xlabel(r'$\alpha_{K}$')
    ax.set_yticklabels([])
    ax.set_xlim((np.min(M85[-1]),np.max(M87[-1])))

    mpl.rc('axes', labelsize=15)
    mpl.rc('axes', titlesize=17)
    mpl.rc('xtick',labelsize=13)
    mpl.rc('ytick',labelsize=13)

    mpl.legend()
    #mpl.show()
    mpl.savefig('/home/elliot/imfplots/MLR.pdf', dpi = 600)

