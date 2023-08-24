#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ayf

misc methods for astrophysics
"""

import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def calculate_luminosity_swift(bat_peak_flux, redshift):
    """ 
    method for calculating luminosity using flux and redshift (for SWIFT) \n
    flux ->  BAT_Peak_Flux (erg/cm^2/s)
    """

    # Load your universe (with caution ;)) and calculate luminosity 
    # distance of the GRB 
    H0 = 71 * u.km / u.s / u.Mpc
    omega0 = 0.27
    cosmo = FlatLambdaCDM(H0=H0, Om0=omega0)
    
    dL_mpc = cosmo.luminosity_distance(redshift)
    dL = dL_mpc.to(u.cm).value
    
    # luminosity = flux * dL * dL * 10**(-7)
    
    # luminosity = flux * dL * dL * 10**(-7) *  4 * np.pi * (3*1e-12)
    
    luminosity = bat_peak_flux * dL * dL * np.pi * 4    
    
    return luminosity


# flux = 3.35
# z = 9.5
# lum = 3.97E+53

# # flux = 3.86    
# # z = 0.5

# H0 = 71 * u.km / u.s / u.Mpc
# omega0 = 0.27
# cosmo = FlatLambdaCDM(H0=H0, Om0=omega0)

# dL_mpc = cosmo.luminosity_distance(z)
# dL = dL_mpc.to(u.cm).value

# print(flux * dL * dL * 4 * np.pi * 256)
# print(flux * dL * dL  * 256) 
# print(flux * dL * dL  * 256 * (3*1e-12)) 
# print(flux * dL * dL  * 256 * (3*1e-12) * np.pi*4) 
# print(flux * dL * dL  * 256 * 4.8065319e-14) 

# print(flux * dL * dL  * np.pi * 4 * 4.81e-7 * 256) 

# print(flux * dL * dL  * np.pi * 4 * 4.81e-7 * 256) 
 