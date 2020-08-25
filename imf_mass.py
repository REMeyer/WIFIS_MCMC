import numpy as np
import matplotlib.pyplot as mpl
import scipy.optimize as spo
import scipy.interpolate as spi
#import mist_wrapper as mw
import scipy.integrate as integrate
import sys

def imffunc(x, m):

    return m**(-1*x)

def calc_imf(m1,m2,x, mass = True):

    if mass:
        #Integrating M*IMF() = M * M^(-1*x) = M^ (1 - x) which if integrated gives M^(2 - x)/(2 - x)
        exp = 2 - x
    else:
        #Integrating IMF() = M^(-1*x) which if integrated gives M^(1 - x)/(1 - x)
        exp = 1 - x

    first = m2**(exp) / exp #First integration term
    second = m1**(exp) / exp #Second integration term
    return first - second #Calculating integral

def msto(age):

    mass = np.array([25.,15., 12., 9., 7., 5., 4., 3., 2.5, 2., 1.5, 1.25, 1., 0.8])
    #ages = np.array([6.40774,11.5842,16.0176, 26.3886,43.1880,94.4591,164.734,352.503,\
    #        584.916,1115.94,2690.39,4910.11,9844.57,25027.9])
    ages = np.array([7.0591,12.7554,16.712,28.1330,46.1810,100.888,185.435,420.502,710.235,1379.94,2910.76,5588.92,12269.8,25027.9])
    ages = ages / 1000.

    turnoffinterp = spi.interp1d(ages, mass, kind=3, bounds_error=False)

    if age < 0.008:
        return 100.
    else:
        return turnoffinterp(age)
    
def normalize_imf(x1,x2, age, mass = True):
        
    to_mass = msto(age)

    if to_mass > 1.0:
        i1 = calc_imf(0.08,0.499999,x1, mass = mass)
        i2 = calc_imf(0.5,0.999999,x2, mass = mass)
        i3 = calc_imf(1.0,to_mass,2.3, mass = mass)
        sum_i = i1 + i2 + i3
        print(age, to_mass, i1, i2, i3, sum_i)
    elif (to_mass <= 1.0) and (to_mass > 0.5):
        i1 = calc_imf(0.08,0.499999,x1, mass = mass)
        i2 = calc_imf(0.5,to_mass,x2, mass = mass)
        i3 = 0
        sum_i = i1 + i2 + i3
        print(age, to_mass, i1, i2, i3, sum_i)
    elif to_mass <= 0.5:
        i1 = calc_imf(0.08,to_mass,x1, mass = mass)
        i2 = 0
        i3 = 0
        sum_i = i1 + i2 + i3
        print(age, to_mass, i1, i2, i3, sum_i)

    return sum_i, i1,i2,i3,to_mass

def determine_mass_ratio(x1,x2,age):

    fulli = normalize_imf(x1,x2, 0.0)
    age_i = normalize_imf(x1,x2, age)
    norm_constant = 1.0/fulli[0]
    #print fulli, norm_constant, age_i, age_i*norm_constant

    return age_i[0]*norm_constant

def mass_ratio_prepare_isochrone():
    remnantinfo = np.loadtxt('/home/elliot/mcmcgemini/remnants.txt')
    remnantinfo[:,2] /= 1e9

    agemassinterp = spi.interp1d(remnantinfo[:,2], remnantinfo[:,0], kind=3, bounds_error=False)
    remnantinterp = spi.interp1d(remnantinfo[:,0], remnantinfo[:,1], kind=3, bounds_error=False)
    newremnantinterp = spi.interp1d(remnantinfo[:,0], remnantinfo[:,1]/remnantinfo[:,0],\
            kind=3, bounds_error=False)

    #mpl.plot(remnantinfo[:,0], remnantinfo[:,1])
    #mpl.plot(remnantinfo[:,0], remnantinfo[:,0])
    #mpl.plot(np.arange(0.08,100, 0.01), newremnantinterp(np.arange(0.08,100,0.01)))
    #mpl.show()
    #sys.exit()

    return agemassinterp, remnantinterp, newremnantinterp

def determine_mass_ratio_isochrone(x1,x2,age, interp, bottomlight=False):
    agemassinterp, remnantinterp, newremnantinterp = interp

    fulli = normalize_imf_isochrone(x1,x2, 0.0, 100.) #Calculate full integral
    to_mass = agemassinterp(age)
    age_i = normalize_imf_isochrone(x1,x2, age, to_mass) #Calculate integral for the turnoff mass
    norm_constant = 1.0/fulli[0]

    #massabove_to = integrate.quad(remnantinterp, to_mass, 8.)
    #fullmass = massinitfinal(to_mass,8.)
    #percent_remnant = massabove_to[0] / fullmass
    #print percent_remnant, age_i[-1]*norm_constant

    #return age_i[0]*norm_constant + age_i[-1]*norm_constant*percent_remnant
    return age_i[0]*norm_constant, to_mass, norm_constant

def massremnant(newremnantinterp, to_mass):

    massarr = np.arange(to_mass, 100., 0.001)
    massremnant = 0
    for i in range(1, len(massarr)):
        mmass = np.mean(massarr[i-1:i])
        avgremratio = newremnantinterp(mmass) 
        int_val = calc_imf(massarr[i-1], massarr[i], 2.3)

        if mmass >= 6.0:
            massremnant += int_val*2.1/mmass
        else:
            massremnant += int_val*avgremratio

    return massremnant

def normalize_imf_isochrone(x1,x2, age, to_mass, mass = True):
        
    if to_mass > 1.0:
        i1 = calc_imf(0.08,0.499999,x1, mass = mass)
        i2 = calc_imf(0.5,0.999999,x2, mass = mass)
        i3 = calc_imf(1.0,to_mass,2.3, mass = mass)
        i4 = calc_imf(to_mass,8.0, 2.3, mass = mass)
        sum_i = i1 + i2 + i3
        #print(np.round(age,3), np.round(to_mass,3), np.round(i1,3), \
        #np.round(i2,3), np.round(i3,3), np.round(sum_i,3),np.round(i4,3))

    elif (to_mass <= 1.0) and (to_mass > 0.5):
        i1 = calc_imf(0.08,0.499999,x1, mass = mass)
        i2 = calc_imf(0.5,to_mass,x2, mass = mass)
        i3 = 0
        i4 = calc_imf(to_mass, 0.99999, x2, mass = mass) + calc_imf(1.0, 8.0, 2.3,mass=mass)
        sum_i = i1 + i2 + i3
        #print(np.round(age,3), np.round(to_mass,3), np.round(i1,3), \
        #np.round(i2,3), np.round(i3,3), np.round(sum_i,3), np.round(i4,3))

    elif to_mass <= 0.5:
        i1 = calc_imf(0.08,to_mass,x1, mass = mass)
        i2 = 0
        i3 = 0
        i4 = calc_imf(to_mass,0.49999,x1, mass = mass) + calc_imf(0.5,0.99999,x2, mass = mass) \
                + calc_imf(1.0,8.0,2.3, mass = mass)
        sum_i = i1 + i2 + i3
        #print(np.round(age,3), np.round(to_mass,3), np.round(i1,3),\
        #np.round(i2,3), np.round(i3,3), np.round(sum_i,3),np.round(i4,3))

    return sum_i, i1,i2,i3, to_mass, i4

def massinitfinal(m1,m2):

    return ((m2**2.)/2.) - ((m1**2.)/2.)

#####################

def inverse_return(xs, p0):
    '''Returns a gaussian function with inputs p0 over xs. p0 = [A, sigma, mean]'''
    return  p0[0] * (xs)**p0[1] + p0[2]
    #return  p0[0] * p0[1]**(xs**p0[2]) + p0[3]

def inverse(xs, a, b, c,d):

    return a * b**(xs**c) + d

def inverse2(xs, a, b,c ):

    return a * xs**b + c

def log_return(xs, p0):
    return p0[0] * np.log(xs + p0[1]) + p0[2]

def logarithm(xs,a, b, c):

    return a * np.log(xs + b) + c

def func_fit(xdata,ydata, p0, func=inverse2):
    '''Performs a very simple gaussian fit using scipy.optimize.curvefit. Returns 
    the fit parameters and the covariance matrix'''

    #first = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    #second = ((x - x0)/sigma)**2.0
    
    #gaussian = lambda x,a,sigma,x0: a * np.exp((-1.0/2.0) * ((x - x0)/sigma)**2.0)
    popt, pcov = spo.curve_fit(func, xdata, ydata, p0=p0)
    return [popt, pcov]

if __name__ == '__main__':
    interp = mass_ratio_prepare_isochrone()
    age = 5.0

    for age in [3.0,5.0,7.0,9.0,11.0,13.5]:
        x1,to_mass,norm_constant = determine_mass_ratio_isochrone(3.5,3.5, age, interp)#/0.982
        remnant = massremnant(interp[-1], to_mass, norm_constant)
        x1 += remnant * norm_constant
        
        x2,to_mass,norm_constant = determine_mass_ratio_isochrone(3.0,3.0, age, interp)#/0.982
#        remnant = massremnant(interp[-1], to_mass, norm_constant)
        x2 += remnant * norm_constant

        x3,to_mass,norm_constant = determine_mass_ratio_isochrone(2.35,2.35, age, interp)#/0.982
#        remnant = massremnant(interp[-1], to_mass, norm_constant)
        x3 += remnant * norm_constant

        x4,to_mass,norm_constant = determine_mass_ratio_isochrone(1.3,2.35, age, interp)#/0.982
#        remnant = massremnant(interp[-1], to_mass, norm_constant)
        x4 += remnant * norm_constant

        #x2 = determine_mass_ratio_isochrone(3.0,3.0, 5.0, interp)#/0.944
        #remnant = massremnant(newremnantinterp[-1], to_mass, norm_constant)
        #x3 = determine_mass_ratio_isochrone(2.3,2.3, 5.0, interp)#/0.736
        #remnant = massremnant(newremnantinterp[-1], to_mass, norm_constant)
        #x4 = determine_mass_ratio_isochrone(1.3,2.3, 5.0, interp)#/0.572
        #remnant = massremnant(newremnantinterp[-1], to_mass, norm_constant)
        #x5 = determine_mass_ratio_isochrone(0.5,0.5, 5.0, bottomlight=True)#/0.256
        #x5 = determine_mass_ratio_isochrone(0.5,0.5, 5.0)#/0.256
        if age == 5.0:
            print(x1, x1/0.985)
            print(x2, x2/0.952)
            print(x3, x3/0.752)
            print(x4, x4/0.597)
        elif age == 3.0:
            print(x1, x1/0.987)
            print(x2, x2/0.957)
            print(x3, x3/0.765)
            print(x4, x4/0.616)
        elif age == 7.0:
            print(x1, x1/0.984)
            print(x2, x2/0.949)
            print(x3, x3/0.744)
            print(x4, x4/0.585)
        elif age == 9.0:
            print(x1, x1/0.983)
            print(x2, x2/0.937)
            print(x3, x3/0.740)
            print(x4, x4/0.579)
        elif age == 11.0:
            print(x1, x1/0.982)
            print(x2, x2/0.944)
            print(x3, x3/0.736)
            print(x4, x4/0.572)
        elif age == 13.5:
            print(x1, x1/0.982)
            print(x2, x2/0.943)
            print(x3, x3/0.732)
            print(x4, x4/0.567)

            
        #print x5, 0.262/x5








