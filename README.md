# WIFIS\_MCMC

The WIFIS\_MCMC package provides MCMC 'full-index' spectral fitting for fully reduced and processed WIFIS spectra. Specifically, it is an implementation of an affine-invariant ensemble sampler MCMC (emcee, Foreman-Mackey et al. (2013)) and functions with the Conroy et al. (2018) stellar population synthesis models, which include a wide variety of differing initial mass functions (IMF) and several chemical abundance response functions. The package will optionally also fit Sauron spectra from the Atlas3D survey, specifically the HBeta, Fe5015, and MgB features. 

## Dependencies

Versions below work. Newer versions may cause issues.

- numpy (1.16.6)
- scipy (1.2.1)
- pandas (0.24.2)
- emcee (3.0.2)
- matplotlib (2.2.3)
- astropy (2.0.9)
- corner (2.0.1) (for corner plotting mcmc posterior distributions)

## Overview

The main script is,

- mcmc\_fullindex.py

and the primary supporting scripts are,

- mcmc\_support.py
- mcmc\_spectra.py
- plot\_corner.py

this script takes input parameters from a text file placed in the inputs directory and referenced in the __main__ section of mcmc\_fullindex.py. Several sets of inputs can be placed in a single input file. The input parameters are as follows,

- fl
	- The observed spectrum. This must be contained a fits file with the wavelength array and uncertainties (see mcmc\_spectra.preparespecwifis). 
- lineinclude
	- Spectral features intended for fitting. Possible feature names are: FeH, NaI, KI\_a, KI\_b, KI\_1.25, NaI123, CaI, PaB, NaI127, and CaII119
- params
	- Fitting parameters in a python dictionary style. Any static parameters can be set here, e.g. Age:3.0. Possible fit paramters: Age, Z, x1, x2, VelDisp, Na, K, Fe, Ca, Alpha (positive linear combination of Ca, C, Mg, Si). Chemical abundances are [X/H] as definited in Conroy et al. (2018). 
- target
	- Name of target galaxy
- workers
	- Number of mcmc workers. Standard of 512.
- steps
	- Number of mcmc worker steps. Standard of 4000, 3000 burn-in.
- targetz
	- Galaxy redshift. Not a fitting parameter so must be defined.
- targetsigma
	- Galaxy velocity dispersion. Set to None if VelDisp is a fit parameter.
- sfl
	- Filepath of the Atlas3D spectrum fits file. Same format as fl.
- sz
	- Redshift for the sauron spectrum (can have minor differences). Not a fit parameter.
- ssigma
	- Sauron spectrum velocity dispersion. Not a fit parameter.
- saurononly
	- Boolean flag to only fit the sauron spectrum. Default is False.
- comments
	- Text comments to include in the output data file and in the log.
- skip
	- Flag to skip this set of input parameters if including multiple sets of inputs. Default of 0 for not skipping.

