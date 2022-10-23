'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: September 21 2022

Description:

Python script to load perturbed halo fields from 21cmFAST-cache and sort halos
based on their locations in an ionized vs neutral region of the 21cmFAST generated
ionization fields (gt) and the U-Net predicted (pred) ionization fields.

'''

import h5py
import numpy as np
from glob import glob
import py21cmfast as p21c
import matplotlib.pyplot as plt
from data_manager import DataManager
from utility_funcs import (binarize_boxes, get_n_i_halo_mass_coords)

def get_rseed(fname):

    '''Function to return rseed from .h5 file in the 21cmFAST-cache.'''

    return int(fname.split('_')[-1][1:-3])

# Coeval boxes to analyze 
coeval_boxes_dir = '/Users/kennedyj/PHYS_459/data/coeval_boxes/'
fname_coeval_boxes = 'HII_DIM_128_BOX_LEN_192_alpha_15_bar_max_2_168_boxes_new_zs_intermed_rseed_shared_shuffled_training_set_results.h5'

# Cached perturbed halo fields, sorted by increasing random seed
fname_pt_halo_fields = glob('/Users/kennedyj/21cmFAST-cache/PerturbHaloField*')
sorted_fname_pt_halo_fields = sorted(fname_pt_halo_fields, key = get_rseed)

# Load in coeval boxes
DM = DataManager(coeval_boxes_dir + fname_coeval_boxes)
redshifts = np.array(DM.data['redshifts']) 
rseeds = np.array(DM.data['random_seeds_val'])
xH_boxes_gt = binarize_boxes(DM.data["ionized_boxes"])
xH_boxes_pred = binarize_boxes(DM.data["predicted_brightness_temp_boxes"])
num_val_boxes = rseeds.shape[0]

# Load in cached perturbed halo fields, check if redshift, random seed match with validation coeval boxes
for i in range(len(fname_pt_halo_fields)):

    cached_pt_halo_field = p21c.cache_tools.readbox(fname=sorted_fname_pt_halo_fields[i])
    
    z = cached_pt_halo_field.redshift
    rseed = cached_pt_halo_field.random_seed
    index = np.where((redshifts==z) & (rseeds==rseed))[0]

    print(f'\n \n === {(z, rseed)} === \n \n')

    # If z, rseed in redshifts, rseeds --> index.size == 1
    if index.size:
        
        print(f'\n \n === {z, rseed} === \n \n')
        save_name = f"/Users/kennedyj/PHYS_459/data/halo_masses_coords/HII_DIM_128_BOX_LEN_192_alpha_15_bar_max_2_168_boxes_new_zs_intermed_UHF_False_z_{z}_rseed_{rseed}_halos.h5"
        get_n_i_halo_mass_coords(cached_pt_halo_field.halo_coords, cached_pt_halo_field.halo_masses, xH_boxes_gt[index][0], xH_boxes_pred[index][0], rseed, save_name, scale=1)
