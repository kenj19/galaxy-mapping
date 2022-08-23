"""

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 22 2022

Description:

Code to compute the halo mass accreted over a redshift interval, 
based on the script get_mar.py written by Jordan Mirocha and using
his ARES package.

"""

import sys
sys.path.insert(0, '/Users/kennedyj/PHYS_459/Github/ares')
import ares
import numpy as np
import astropy.units as u
import matplotlib.pyplot as pl
from astropy.cosmology import FlatLambdaCDM

def calc_mass_accr(z_high, z_low, halomasses, cosmo_mod):
    '''
    Function to compute the halo mass accreted over a specific redshift interval.
    '''

    #Initialize galaxy population in ares
    pop = ares.populations.GalaxyPopulation()

    # Find redshift in lookup table
    iz = np.argmin(np.abs(pop.halos.tab_z - z_high))

    # Interpolate in halo mass to get mass accretion rate [Msun / yr]
    MAR = np.interp(halomasses, pop.halos.tab_M, pop.halos.tab_MAR[iz,:])
    #print('Mass accretion rate of Mh={:.1e} Msun at z={:.1f} is {:.1f} Msun/yr'.format(halomass,
    #       z_high, MAR))

    delta_t = ((cosmo_mod.age(z_low) - cosmo_mod.age(z_high)) / u.Gyr) * 1e9 # convert to number of years

    mass_accr = MAR * delta_t

    return mass_accr
