'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 22 2022

Description:

Python code to generate a field of halo masses representing galaxies 
above the apparent magnitude thresholds of surveys. Halo mass fields are
saved to a .h5 file.

'''

import h5py
import numpy as np
import py21cmfast as p21c
import astropy.units as u
import matplotlib.pyplot as plt
from utility_funcs import (L_to_MAB, get_mag_app)
from astropy.cosmology import FlatLambdaCDM

print(f"\n ============= Using 21cmFAST version {p21c.__version__} ============== \n")

# Load in data
fname_zs = 'comoving_dist_redshift_conversion_BOX_LEN_128_zs_3.h5'
hf = h5py.File(fname_zs,'r')
BOX_LEN, redshifts = np.array(hf['BOX_LEN']), np.array(hf['redshifts'])
hf.close()

print(f'\n ====== BOX_LEN: {BOX_LEN} ====== \n \n ====== redshifts: {redshifts} ====== \n') 

HII_DIM=int(np.copy(BOX_LEN)) # keep same as BOX_LEN, 1:1
DIM=HII_DIM*3 #int(np.copy(BOX_LEN))
rseed=42142
gen_field = False # generate ionization fields with halo fields at each z


# Specify initial params, generate initial conditions of coeval box
user_params = p21c.UserParams(BOX_LEN=BOX_LEN,
                              HII_DIM=HII_DIM,
                              DIM=DIM,
                              USE_INTERPOLATION_TABLES=True)

init_cond = p21c.initial_conditions(user_params=user_params,
                                    random_seed=rseed)

# Generate cosmological model
cosmo = FlatLambdaCDM(H0=67.32 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3158)

# Load in galaxy luminosity-halo mass relation data, parse data
Mh_ML_dat = np.loadtxt('/Users/kennedyj/PHYS_459/L1600_vs_Mh_and_z.dat').T
Mh, L_1600_z6, L_1600_z7, L_1600_z8, L_1600_z9, L_1600_z10 = Mh_ML_dat

# Define apparent magnitude cutoffs of surveys
cutoffs = {'JWST-UD': 32, 'JWST-MD': 30.6, 'JWST-WF': 29.3, 'Roman': 26.5}

# Loop through redshifts, generate halo fields and check if above thresholds
for redshift in redshifts:

    halo_field = p21c.determine_halo_list(redshift=redshift,
                                          init_boxes=init_cond,
                                          user_params=user_params)
            
    halo_coords = halo_field.halo_coords // int(DIM//HII_DIM) # DIM//HII_DIM gives scale
    halo_masses = halo_field.halo_masses
    halo_mass_bins = halo_field.mass_bins

    print(f"\n ====== Num Halos @ {redshift}: {len(halo_coords)} ====== \n")

    if gen_field:

        perturbed_field = p21c.perturb_field(redshift=redshift,
                                             init_boxes=init_cond)

        ionized_field = p21c.ionize_box(perturbed_field=perturbed_field)
                   
        xH_box = ionized_field.xH_box

    # Convert halo masses to luminosities
    Lumo = np.interp(halo_masses, xp = Mh, fp = L_1600_z8)
    MAB = L_to_MAB(Lumo)
    mAB = get_mag_app(redshift, MAB, cosmo)

    # Apply magnitude cutoff for surveys, get halo fields
    halo_mass_field = np.zeros(shape=(HII_DIM,HII_DIM,HII_DIM)) # to be used for mass accretion addition
    JWST_UD_gals = np.zeros(shape=(HII_DIM,HII_DIM,HII_DIM))
    JWST_MD_gals = np.zeros(shape=(HII_DIM,HII_DIM,HII_DIM))
    JWST_WF_gals = np.zeros(shape=(HII_DIM,HII_DIM,HII_DIM))
    Roman_gals = np.zeros(shape=(HII_DIM,HII_DIM,HII_DIM))

    JWST_UD_gal_coords = halo_coords[mAB<cutoffs['JWST-UD']]
    JWST_MD_gal_coords = halo_coords[mAB<cutoffs['JWST-MD']]
    JWST_WF_gal_coords = halo_coords[mAB<cutoffs['JWST-WF']]
    Roman_gal_coords = halo_coords[mAB<cutoffs['Roman']]

    JWST_UD_gal_masses = halo_masses[mAB<cutoffs['JWST-UD']]
    JWST_MD_gal_masses = halo_masses[mAB<cutoffs['JWST-MD']]
    JWST_WF_gal_masses = halo_masses[mAB<cutoffs['JWST-WF']]
    Roman_gal_masses = halo_masses[mAB<cutoffs['Roman']]

    for i in range(len(halo_masses)): 

        if i < len(Roman_gal_coords):

            halo_mass_field[tuple(halo_coords[i])] += halo_masses[i]
            JWST_UD_gals[tuple(JWST_UD_gal_coords[i])] += JWST_UD_gal_masses[i]
            JWST_MD_gals[tuple(JWST_MD_gal_coords[i])] += JWST_MD_gal_masses[i]
            JWST_WF_gals[tuple(JWST_WF_gal_coords[i])] += JWST_WF_gal_masses[i]
            Roman_gals[tuple(Roman_gal_coords[i])] += Roman_gal_masses[i]

        elif i < len(JWST_WF_gal_coords):

            halo_mass_field[tuple(halo_coords[i])] += halo_masses[i]
            JWST_UD_gals[tuple(JWST_UD_gal_coords[i])] += JWST_UD_gal_masses[i]
            JWST_MD_gals[tuple(JWST_MD_gal_coords[i])] += JWST_MD_gal_masses[i]
            JWST_WF_gals[tuple(JWST_WF_gal_coords[i])] += JWST_WF_gal_masses[i]

        elif i < len(JWST_MD_gal_coords):

            halo_mass_field[tuple(halo_coords[i])] += halo_masses[i]
            JWST_UD_gals[tuple(JWST_UD_gal_coords[i])] += JWST_UD_gal_masses[i]
            JWST_MD_gals[tuple(JWST_MD_gal_coords[i])] += JWST_MD_gal_masses[i]

        elif i < len(JWST_UD_gal_coords):

            halo_mass_field[tuple(halo_coords[i])] += halo_masses[i]
            JWST_UD_gals[tuple(JWST_UD_gal_coords[i])] += JWST_UD_gal_masses[i]

        else:

            halo_mass_field[tuple(halo_coords[i])] += halo_masses[i]

    
    fname_save = f'/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/galaxy_cutoffs_HII_DIM_{HII_DIM}_DIM_{DIM}_BOXLEN_{BOX_LEN}_z_{redshift}_rseed_{rseed}.h5'
    
    hf = h5py.File(fname_save, 'w')
    hf.create_dataset('halo_mass_bins', data=halo_mass_bins)
    hf.create_dataset('halo_mass_field', data=halo_mass_field)
    hf.create_dataset('JWST_UD_gals', data=JWST_UD_gals)
    hf.create_dataset('JWST_MD_gals', data=JWST_MD_gals)
    hf.create_dataset('JWST_WF_gals', data=JWST_WF_gals)
    hf.create_dataset('Roman_gals', data=Roman_gals)
    hf.close()
    
    print(f'\n ====== File {fname_save} saved ====== \n')
