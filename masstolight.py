
def calculate_MLR(fl, instrument = 'nifs', burnin = -300, vcjset = None):

    #Load the necessary information
    data, names, gal, datatype, fitvalues, truevalues,paramnames, linenames= bestfitPrepare(fl, burnin)

    x_m = 0.5 + np.arange(16)/5.0
    paramnames = np.array(paramnames)
    ix1 = np.where(paramnames == 'x1')[0][0]
    ix2 = np.where(paramnames == 'x2')[0][0]
    
    MLR = np.zeros((len(x_m),len(x_m)))
    MWvalues = np.array(truevalues)
    MWvalues[ix1] = 1.3
    MWvalues[ix2] = 2.3
    wlgMW, newmMW = mcfs.model_spec(MWvalues, gal, paramnames, vcjset = vcjset, full = True)
    
    whK = np.where((wlgMW >= 20300) & (wlgMW <= 23700))[0] 

    MLR_MW = np.sum(newmMW[whK])
    vals = [(1.3,1.3),(1.3,2.3),(2.3,2.3),(2.9,2.9),(3.1,3.1),(3.5,3.5)]
    MLRDict = {}

    for i in range(len(x_m)):
        for j in range(len(x_m)):
            x1 = x_m[i]
            x2 = x_m[j]
            tempvalues = np.array(truevalues)
            tempvalues[ix1] = x1
            tempvalues[ix2] = x2
            wlg, newm = mcfs.model_spec(tempvalues, gal, paramnames, vcjset = vcjset, full = True)
            MLR_IMF = np.sum(newm[whK])

            if (x1,x2) in vals:
                MLRDict[(x1,x2)] = MLR_IMF

            MLR[i,j] = MLR_MW / MLR_IMF

    mpl.close('all')
    samples = data[burnin:,:,:].reshape((-1,len(names)))
    
    x1 = samples[:,ix1]
    x2 = samples[:,ix2]

    x_mbins = 0.4 + np.arange(17)/5.0
    histprint = mpl.hist2d(x1,x2, bins = x_mbins)

    fullMLR = []
    for i in range(len(x_m)):
        for j in range(len(x_m)):
            n_val = int(histprint[0][i,j])
            addlist = [float(MLR[i,j])] * n_val
            fullMLR.extend(addlist)

    mpl.close('all')

    return MLR, MLRDict, paramnames, truevalues, histprint, fullMLR
