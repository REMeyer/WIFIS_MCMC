import numpy as np
import scipy.optimize as spo

def gaussian(xs, a, sigma, x0):
    second = ((xs - x0)/sigma)**2.0
    full = a*np.exp((-1.0/2.0) * second)
    
    return full

def gaussian_os(xs, a, sigma, x0, b):
    second = ((xs - x0)/sigma)**2.0
    full = b + a*np.exp((-1.0/2.0) * second)
        
    return full

def gaussian_fit_os(xdata,ydata,p0, gaussian=gaussian_os):
    '''Performs a very simple gaussian fit using scipy.optimize.curvefit. Returns 
        the fit parameters and the covariance matrix'''

    #first = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    #second = ((x - x0)/sigma)**2.0
        
    #gaussian = lambda x,a,sigma,x0: a * np.exp((-1.0/2.0) * ((x - x0)/sigma)**2.0)
    popt, pcov = spo.curve_fit(gaussian, xdata, ydata, p0=p0)
    return [popt, pcov]

def gauss_nat(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [sigma, mean]'''
    return  (1 / (2.*np.pi*p0[0]**2.))* np.exp((-1.0/2.0) * ((xs - p0[1])/p0[0])**2.0)

