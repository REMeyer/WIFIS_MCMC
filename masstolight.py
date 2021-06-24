from __future__ import print_function
import imf_mass as imf
#import compare_bestfit as cbf 
import numpy as np
import matplotlib.pyplot as plt
import mcmc_fullindex as mcfi
import mcmc_support as mcsp
from time import time

def rtwo(val):
    '''Convenience wrapper function. Returns the input number val truncated
    to two decimal places'''

    return np.round(val, 2)

def calculate_alpha(fl, burnin = -1000, vcjset = None, verbose = False):

    #Load the best-fit parameters
    data, postprob, info, lastdata = mcsp.load_mcmc_file(fl)
    gal = info[2]
    names = info[3]
    high = info[4]
    low = info[5]
    paramnames = info[6]
    linenames = info[7]
    lines = info[8]
    paramdict = info[9]

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
    print(samples.shape)
    fitvalues = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    fitvalues = np.array(fitvalues)
    midvalues = fitvalues[:,0]

    #return [data, names, gal, datatype, fitvalues, midvalues, paramnames, linenames, lines]
    #data, names, gal, datatype, fitvalues, truevalues, \
    #paramnames, linenames, lines = cbf.bestfitPrepare(fl, burnin)

#    paramdict = {}
#    for n in paramnames:
#        paramdict[n] = None

    #Generate the IMF exponent arrays and get indices for various parameters
    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)

    #Generate 2d array for every IMF to hold M/L values
    MWvalues = np.array(midvalues) #Generate array with same file params but w/ MW IMF

    if ('x1' in paramnames) and ('x2' in paramnames):
        ix1 = np.where(paramnames == 'x1')[0][0]
        ix2 = np.where(paramnames == 'x2')[0][0]
        MWvalues[ix1] = 1.3 # Change IMF x1
        MWvalues[ix2] = 2.3 # Change IMF x2
        MLR = np.zeros((len(x_m),len(x_m))) #Array to hold the M/L values
    else:
        ix1 = np.where(paramnames == 'x1')[0][0]
        MWvalues[ix1] = 1.3 # Change IMF x1
        MLR = np.zeros(len(x_m)) #Array to hold the M/L values
    if 'Age' in paramnames:
        iage = np.where(paramnames == 'Age')[0][0]
    
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

    if ('x1' in paramnames) and ('x2' in paramnames):
        for i in range(len(x_m)):
            for j in range(len(x_m)):
                x1 = x_m[i]
                x2 = x_m[j]
                tempvalues = np.array(midvalues)
                if 'x1' in paramnames:
                    tempvalues[ix1] = x1
                if 'x2' in paramnames:
                    tempvalues[ix2] = x2
                wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                                vcjset = vcjset, MLR = True)
                MLR_IMF = np.sum(newm[whK])

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
    elif 'x2' not in paramnames:
        if 'x2' in paramdict.keys():
            for i in range(len(x_m)):
                x1 = x_m[i]
                x2 = paramdict['x2']
                tempvalues = np.array(midvalues)
                tempvalues[ix1] = x1

                wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                                vcjset = vcjset, MLR = True)
                MLR_IMF = np.sum(newm[whK])

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

                MLR[i] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
                #MLR[i,j] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
                #MLR[i,j] = MLR_MW / MLR_IMF
        else:
            for i in range(len(x_m)):
                x1 = x_m[i]
                x2 = x1
                tempvalues = np.array(midvalues)
                tempvalues[ix1] = x1

                wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                                vcjset = vcjset, MLR = True)
                MLR_IMF = np.sum(newm[whK])

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

                MLR[i] = (IMFremaining * MLR_MW) / (MWremaining * MLR_IMF)
                #MLR[i,j] = (MWremaining * MLR_MW) / (IMFremaining * MLR_IMF)
                #MLR[i,j] = MLR_MW / MLR_IMF

    plt.close('all')
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    print(samples.shape)

    fullMLR = []
    x_mbins = 0.4 + np.arange(17)/5.0
    if ('x1' in paramnames) and ('x2' in paramnames):
        x1 = samples[:,ix1]
        x2 = samples[:,ix2]
        histprint = plt.hist2d(x1,x2, bins = x_mbins)
        for i in range(len(x_m)):
            for j in range(len(x_m)):
                n_val = int(histprint[0][i,j])
                addlist = [float(MLR[i,j])] * n_val
                fullMLR.extend(addlist)
    elif 'x2' not in paramnames:
        x1 = samples[:,ix1]
        histprint = plt.hist(x1, bins = x_mbins)
        for i in range(len(x_m)):
            n_val = int(histprint[0][i])
            addlist = [float(MLR[i])] * n_val
            fullMLR.extend(addlist)

    plt.show()

    fullMLR = np.array(fullMLR)

    plt.close('all')
    percentiles = np.percentile(fullMLR, [16,50,84], axis = 0)
    print(fl, np.percentile(fullMLR, [16, 50, 84], axis=0))
    print(rtwo(percentiles[1]), rtwo(percentiles[2]-percentiles[1]),\
            rtwo(percentiles[1]-percentiles[0]))

    return MLR, MLRDict, paramnames, midvalues, histprint, mlrarr, percentiles, fullMLR

def calculate_alpha_new(fl, burnin = -1000, vcjset = None, verbose = False, limited=False,\
        linesoverride = False):
    '''
    New version of calculate_alpha that estimates alpha for every MCMC step in
    the posterior distributions specified by burnin. In addition, this function
    incorporates a new estimate for the remnant mass fraction.
    '''

    # Load the MCMC data, parameters, line names
    data, postprob, info, lastdata = mcsp.load_mcmc_file(fl, linesoverride=linesoverride)
    gal = info[2]
    names = info[3]
    high = info[4]
    low = info[5]
    paramnames = info[6]
    linenames = info[7]
    lines = info[8]
    paramdict = info[9]

    # Extract the mcmc samples used for the posterior distributions and 
    # the statistical median and sigmas
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    fitvalues = list(map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),\
            zip(*np.percentile(samples, [16, 50, 84], axis=0))))
    fitvalues = np.array(fitvalues)
    midvalues = fitvalues[:,0]

    #Generate the IMF exponent arrays and get indices for key parameters
    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)
    ix1 = np.where(paramnames == 'x1')[0][0]
    ix2 = np.where(paramnames == 'x2')[0][0]
    iage = np.where(paramnames == 'Age')[0][0]
    iz = np.where(paramnames == 'Z')[0][0]
    
    #Get interpolation function
    interps = imf.mass_ratio_prepare_isochrone()
    massarr, massremnantarr = imf.massremnant_prepare(interps[-1])
    
    MWvalues = np.array(samples[0,:]) #Generate array with same file params but w/ MW IMF
    # Create model spectrum for best-fit run params & MW-imf
    wlgMW, newmMW, basemMW = mcfi.model_spec(MWvalues, paramnames, paramdict, \
                        vcjset = vcjset, MLR = True) #Generate the MW spectrum
    whK = np.where((wlgMW >= 20300) & (wlgMW <= 23700))[0] 

    AlphaArr = [] # Array to hold alpha parameters for each MCMC step
    # Loop through every step
    for i in range(samples.shape[0]):
        if limited:
            MWvalues = np.zeros(len(samples[i,:]))
            MWvalues[iz] = samples[i,:][iz]
            MWvalues[iage] = samples[i,:][iage]
            rval = np.zeros(len(samples[i,:]))
            rval[iage] = samples[i,:][iage]
            rval[iz] = samples[i,:][iz]
            rval[ix1] = samples[i,:][ix1]
            rval[ix2] = samples[i,:][ix2]
        else:
            MWvalues = np.array(samples[i,:]) #Generate array with same file params but w/ MW IMF
            rval = np.array(samples[i,:])
        MWvalues[ix1] = 1.3 # Change IMF x1
        MWvalues[ix2] = 2.3 # Change IMF x2

        # Create model spectrum for best-fit run params & MW-imf
        wlgMW, newmMW, basemMW = mcfi.model_spec(MWvalues, paramnames, paramdict, \
                            vcjset = vcjset, MLR = True) #Generate the MW spectrum
        L_MW = np.sum(newmMW[whK])
        
        #Calculate Kroupa IMF integrals
        MW_Mass, to_mass, normconstant = imf.determine_mass_ratio_isochrone_new(1.3,2.3, \
                MWvalues[iage], interps)

        #Calculate Remnant mass at turnoff mass
        #massremnant = imf.massremnant_get(massarr, massremnantarr, to_mass)
        massremnant_mw = imf.conroy_remnants(1.3,2.3, to_mass)
        #Caluclate final remaining mass
        MW_Mass += massremnant_mw*normconstant

        wlg, newm, basem = mcfi.model_spec(rval, paramnames, paramdict, \
                        vcjset = vcjset, MLR = True)
        L_IMF = np.sum(newm[whK])

        IMF_Mass, to_massimf, normconstantimf = \
                imf.determine_mass_ratio_isochrone(rval[ix1],rval[ix2], MWvalues[iage], interps)
        massremnant_imf = imf.conroy_remnants(rval[ix1],rval[ix2], to_mass)
        IMF_Mass += massremnant_imf*normconstantimf

        alpha = (IMF_Mass * L_MW) / (MW_Mass * L_IMF)
        AlphaArr.append(alpha)
        
        if verbose and (i % 5000 == 0):
            print(i, (time()-t1)/60)
            print(IMF_Mass, L_MW, MW_Mass, L_IMF, massremnant_mw, massremnant_imf, alpha)
            print(np.percentile(AlphaArr, [16,50,84], axis = 0))

    plt.close('all')
    
    percentiles = np.percentile(AlphaArr, [16,50,84], axis = 0)
    print(fl, percentiles)

    return AlphaArr

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

def MLR_example(vcjset = None, verbose = False):

    paramnames = ['Age','Z','x1','x2']
    fitvalues = [13.5,0.0,3.0,3.0]

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
    MWvalues = np.array(fitvalues) #Generate array with same file params but w/ MW IMF
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
            tempvalues = np.array(midvalues)
            tempvalues[ix1] = x1
            tempvalues[ix2] = x2
            wlg, newm, basem = mcfi.model_spec(tempvalues, paramnames, paramdict, \
                            vcjset = vcjset, MLR = True)
            MLR_IMF = np.sum(newm[whK])

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
    plt.show()

    fullMLR = []
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            n_val = int(histprint[0][i,j])
            addlist = [float(MLR[i,j])] * n_val
            fullMLR.extend(addlist)
    fullMLR = np.array(fullMLR)

    plt.close('all')
    percentiles = np.percentile(fullMLR, [16,50,84], axis = 0)
    print(fl, np.percentile(fullMLR, [16, 50, 84], axis=0))

    return MLR, MLRDict, paramnames, midvalues, histprint, mlrarr, percentiles, fullMLR


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

