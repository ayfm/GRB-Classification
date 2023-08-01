import numpy as np
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from grbtools import disp

def calculate_swift_luminosity(bat_peak_flux, redshift):
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


def create_2D_gmm_figures(data, catalogue_name, feat1, feat2):
    for i in range(1,6):
        model_name = catalogue_name + "_" + feat1 + "_" + feat2 + "_N" + str(i) + "_Cfull.model"
        title = catalogue_name.upper() + " " + str(i) + "G "
        x_label = feat1
        y_label = feat2
        disp.scatter2DWithClusters(model_name, data, title=title, xlabel=x_label, ylabel=y_label, figure_save=True)
        print("Created figure: ", model_name)
            
    print("All figures for %s in feature space %s %s have been saved." %(catalogue_name, feat1, feat2))

