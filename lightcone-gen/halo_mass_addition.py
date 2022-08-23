'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 22 2022

Description:

Python code to add halo mass accretion over a redshift interval
to an existing halo mass field.

'''

import h5py
import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
from get_mar import calc_mass_accr
from utility_funcs import (L_to_MAB, get_mag_app)
from astropy.cosmology import FlatLambdaCDM

# Load in data
fname_zs = 'comoving_dist_redshift_conversion_BOX_LEN_128_zs_3.h5'
hf1 = h5py.File(fname_zs,'r')
BOX_LEN, redshifts = np.array(hf1['BOX_LEN']), np.array(hf1['redshifts'])
hf1.close()

high_z = redshifts[-1]
low_z = redshifts[1] # interpolation redshift
HII_DIM=int(np.copy(BOX_LEN))
DIM = HII_DIM*3
rseed=42142

# Load in halo mass field at high_z to add mass to
fname_cutoffs = f'/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/galaxy_cutoffs_HII_DIM_{HII_DIM}_DIM_{DIM}_BOXLEN_{BOX_LEN}_z_{high_z}_rseed_{rseed}.h5'
hf2 = h5py.File(fname_cutoffs, 'r')
halo_mass_bins, halo_mass_field = np.array(hf2['halo_mass_bins']), np.array(hf2['halo_mass_field'])
hf2.close()

print(f'\n ====== BOX_LEN: {BOX_LEN} ====== \n \n ====== (high_z, low_z): {high_z, low_z} ====== \n')

# Generate cosmological model
cosmo = FlatLambdaCDM(H0=67.32 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3158)

# Load in galaxy luminosity-halo mass relation data, parse data
Mh_ML_dat = np.loadtxt('/Users/kennedyj/PHYS_459/L1600_vs_Mh_and_z.dat').T
Mh, L_1600_z6, L_1600_z7, L_1600_z8, L_1600_z9, L_1600_z10 = Mh_ML_dat

# Define apparent magnitude cutoffs of surveys
cutoffs = {'JWST-UD': 32, 'JWST-MD': 30.6, 'JWST-WF': 29.3, 'Roman': 26.5}

# Compute the halo mass to be added for a given halo mass, redshift interval
halo_mass_field_accr = calc_mass_accr(high_z, low_z, halo_mass_field, cosmo)
interp_halo_mass_field = halo_mass_field + halo_mass_field_accr

# Convert halo masses to luminosities
Lumo = np.interp(interp_halo_mass_field, xp = Mh, fp = L_1600_z8)
MAB = L_to_MAB(Lumo)
mAB = get_mag_app(low_z, MAB, cosmo) # mAB at interpolation redshift

# Copy interp halo mass field, zero out galaxies dimmer than survey threshold
JWST_UD_gals = np.copy(interp_halo_mass_field)
JWST_UD_gals[mAB > cutoffs['JWST-UD']] = 0
JWST_MD_gals = np.copy(interp_halo_mass_field)
JWST_MD_gals[mAB > cutoffs['JWST-MD']] = 0
JWST_WF_gals = np.copy(interp_halo_mass_field)
JWST_WF_gals[mAB > cutoffs['JWST-WF']] = 0
Roman_gals = np.copy(interp_halo_mass_fields)
Roman_gals[mAB > cutoffs['Roman']] = 0

print(f"\n ====== Number of observable galaxies in each survey (JWST-UD, JWST-MD, JWST-WF, Roman): \
	({np.count_nonzero(JWST_UD_gals)},{np.count_nonzero(JWST_MD_gals)},{np.count_nonzero(JWST_WF_gals)},{np.count_nonzero(Roman_gals)}) ====== \n")

fname_save = f'/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/interp_galaxy_cutoffs_HII_DIM_{HII_DIM}_DIM_{DIM}_BOXLEN_{BOX_LEN}_z_{low_z}_rseed_{rseed}.h5'
	
hf3 = h5py.File(fname_save, 'w')
hf3.create_dataset('inter_halo_mass_field', data=interp_halo_mass_field)
hf3.create_dataset('JWST_UD_gals', data=JWST_UD_gals)
hf3.create_dataset('JWST_MD_gals', data=JWST_MD_gals)
hf3.create_dataset('JWST_WF_gals', data=JWST_WF_gals)
hf3.create_dataset('Roman_gals', data=Roman_gals)
hf3.close()
	
print(f'\n ====== File {fname_save} saved ====== \n')
