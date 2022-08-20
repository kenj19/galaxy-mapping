''' 

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Python script designed to take as input pairs of 21cmFAST generated (gt) 
and U-Net predicted (pred) ionization fields, run the 21cmFAST
halo finder on the initial density field and generate lists of halos 
based on their location in either ionized or neutral regions of the field.

'''

import h5py
import numpy as np
import py21cmfast as p21c
import matplotlib.pyplot as plt
from data_manager import DataManager
from utils import (binarize_boxes, get_n_i_halo_mass_coords)

# Load in coeval boxes
fname_coeval_boxes='/Users/kennedyj/PHYS_459/data/coeval_boxes/_128_128_rseed_variable_Jun27_results.h5'
save_name = '/Users/kennedyj/PHYS_459/data/halos_masses_coords/HII_DIM_128_BOX_LEN_128_rseed_variable_halos.h5'
DM = DataManager(fname_coeval_boxes)
xH_boxes_pred = binarize_boxes(DM.data["predicted_brightness_temp_boxes"])
xH_boxes_gt = binarize_boxes(DM.data["ionized_boxes"])
rseeds = DM.metadata['random_seed'].split()
num_rseeds = len(rseeds)

# Break up string list into ints to process
for j in range(num_rseeds):

    if j==0:
        rseeds[0] = int(rseeds[0][1:-1])
    elif j==(num_rseeds-1):
        rseeds[-1] = int(rseeds[-1][:-2])
    else:
        rseeds[j] = int(rseeds[j][:-1])

# Coeval cube parameters
BOX_LEN = 128
HII_DIM = 128
DIM = 128*3
user_params = p21c.UserParams(BOX_LEN=BOX_LEN,
    		                  HII_DIM=HII_DIM,
    		                  USE_INTERPOLATION_TABLES=True)

# Run halo finder on each rseed coeval cube
for k in range(num_rseeds):

    init_cond = p21c.initial_conditions(user_params=user_params,
                                        random_seed=rseeds[k])

    halo_field = p21c.determine_halo_list(redshift=redshift,
                                          init_boxes=init_cond,
                                          user_params=user_params)
            
    halo_coords = halo_field.halo_coords # currently in DIM coords
    halo_masses = halo_field.halo_masses
    
    get_n_i_halo_mass_coords(halo_coords, halo_masses, xH_boxes_gt[k], 
    						xH_boxes_pred[k], rseeds[k], save_name)

