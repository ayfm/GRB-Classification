#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Sinem Sasmaz
"""

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from scipy.integrate import quad

keVtoerg = 1.602E-09
h0 = 71
om0 = 0.27

def calc_dL(z, h0, om0):
# Load your universe and calculate the luminosity distance
	H0 = h0*u.km/u.s/u.Mpc
	omega0 = om0
	cosmo = FlatLambdaCDM(H0=H0, Om0=omega0)
	dL_mpc = cosmo.luminosity_distance(z)
	# convert luminosity distance from Mpc to cm
	dL = dL_mpc.to(u.cm)
	return dL.value	
	
# Power law spectral shape		
def PL_N(E, pl_ind, pl_norm):
	spec_shape = keVtoerg*pl_norm*(E/50.)**(pl_ind)
	return spec_shape

# Photon flux for a power law spectral shape	
def PL_flux(E, pl_ind, pl_norm):
	return E*PL_N(E, pl_ind, pl_norm)

# Calculate the k-correction for a power law spectral shape		
# See Bloom et al. 2001
def calc_kcorr_PL(z, pl_ind, pl_norm, e1, e2, E1, E2):			
	PL_E = quad(PL_flux, E1/(1.+z), E2/(1.+z), args=(pl_ind, pl_norm))[0]
	PL_e = quad(PL_flux, e1, e2, args=(pl_ind, pl_norm))[0]	
	kcorr_PL = PL_E/PL_e	
	return kcorr_PL, PL_e

# Calculate the k-corrected luminosity for a power law spectral shape
def calcL_PL(z, h0, om0, pl_ind, pl_norm, e1, e2, E1, E2):	
	kcorr_PL, F_PL = calc_kcorr_PL(z, pl_ind, pl_norm, e1, e2, E1, E2)
	dL = calc_dL(z, h0, om0)
	L_PL = F_PL*4.*np.pi*dL**2*kcorr_PL
	kcorr_simple = (1.+z)**(-1*pl_ind - 2.)
	return L_PL, kcorr_PL, F_PL	


# Cutoff power law spectral shape
def CPL_N(E, cpl_ind, cpl_norm, E0_cpl):
	spec_shape = keVtoerg*cpl_norm*(E/50.)**cpl_ind*np.exp(-E*(2.+cpl_ind)/E0_cpl)
	return spec_shape	

# Photon flux for a cutoff power law spectral shape 	
def CPL_flux(E, cpl_ind, cpl_norm, E0_cpl):
	return E*CPL_N(E, cpl_ind, cpl_norm, E0_cpl)

# Calculate the k-correction for a cutoff power law spectral shape	
# See Bloom et al. 2001	
def calc_kcorr_CPL(z, cpl_ind, cpl_norm, E0_cpl, e1, e2, E1, E2):		
	CPL_E = quad(CPL_flux, E1/(1.+z), E2/(1.+z), args=(cpl_ind, cpl_norm, E0_cpl))[0]
	CPL_e = quad(CPL_flux, e1, e2, args=(cpl_ind, cpl_norm, E0_cpl))[0]	
	kcorr_CPL = CPL_E/CPL_e	
	return kcorr_CPL, CPL_e

# Calculate the k-corrected luminosity for a cutoff power law spectral shape	
def calcL_CPL(z, h0, om0, cpl_ind, cpl_norm, E0_cpl, e1, e2, E1, E2):	
	kcorr_CPL, F_CPL = calc_kcorr_CPL(z, cpl_ind, cpl_norm, E0_cpl, e1, e2, E1, E2)
	dL = calc_dL(z, h0, om0)
	L_CPL = F_CPL*4.*np.pi*dL**2*kcorr_CPL
	return L_CPL, kcorr_CPL, F_CPL
	
	
def is_valid(norm):
    # return val != 0 and not np.isnan(val)
    return norm > 1e-4 and not np.isnan(norm)	
		

def calculate_luminosity(t1_data):

    grb_names = t1_data["grbname"].values
    redshifts = t1_data["z"].values
    best_models = t1_data["t1s_best_model"].values
    pl_inds = t1_data["t1s_pl_alpha"].values
    pl_norms = t1_data["t1s_pl_norm"].values
    cpl_inds = t1_data["t1s_cpl_alpha"].values
    cpl_norms = t1_data["t1s_cpl_norm"].values
    cpl_norms = t1_data["t1s_cpl_norm"].values
    cpl_epeaks = t1_data["t1s_cpl_epeak"].values

    num_files = len(grb_names)
    kcorr_lum = np.repeat(np.nan, num_files)
    kcorr = np.repeat(np.nan, num_files)
    flux = np.repeat(np.nan, num_files)
    model_used = np.repeat("", num_files)

    e1 = 15.0
    e2 = 150.0
    E1 = 15.0
    E2 = 150.0

    for i in range(0, num_files):
        z = redshifts[i]
        pl_ind = pl_inds[i]
        pl_norm = pl_norms[i]
        cpl_ind = cpl_inds[i]
        cpl_norm = cpl_norms[i]
        cpl_epeak = cpl_epeaks[i]
        best_model = best_models[i]

        if not is_valid(z):
            continue

        if best_model == "PL":
            assert is_valid(pl_norm), "Something is wrong with PL. Check %s " % i

        if best_model == "CPL":
            assert is_valid(cpl_norm), "Something is wrong with CPL. Check %s" % i

        if grb_names[i] in ["GRB070419A", "GRB060604"]:
            kcorr_lum[i], kcorr[i], flux[i] = calcL_CPL(
                z, h0, om0, cpl_ind, cpl_norm, cpl_epeak, e1, e2, E1, E2
            )
            model_used[i] = "CPL"
            continue

        if is_valid(pl_norm):
            if best_model != "CPL":
                kcorr_lum[i], kcorr[i], flux[i] = calcL_PL(
                    z, h0, om0, pl_ind, pl_norm, e1, e2, E1, E2
                )
                model_used[i] = "PL"
                continue

        if is_valid(cpl_norm):
            if best_model != "PL":
                kcorr_lum[i], kcorr[i], flux[i] = calcL_CPL(
                    z, h0, om0, cpl_ind, cpl_norm, cpl_epeak, e1, e2, E1, E2
                )
                model_used[i] = "CPL"
                continue

    t1_data["lum_kcorr"] = kcorr_lum
    t1_data["kcorr"] = kcorr
    t1_data["flux"] = flux
    t1_data["model_used"] = model_used

    return t1_data
