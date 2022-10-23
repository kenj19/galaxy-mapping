'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 21 2022

Description: 

Python code to organize previously generated 21cmFAST coeval cubes and 
their wedge-filtered counterparts into a single training set for the U-Net.

'''

import time
import h5py
import random
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List
from data_manager import DataManager
from utility_funcs import save_dset_to_hf

# User params, random seeds of 21cmFAST fields
HII_DIM = 128
BOX_LEN = 192
rseeds = np.arange(50,25050,50)

# Coeval boxes
fname_coeval_boxes = ['/Users/kennedyj/PHYS_459/Github/wedge-repos/outputs/coeval_boxes/correct_transpose_sep27/HII_DIM_128_BOX_LEN_192_alpha_15_bar_max_2_rseed_{}.h5'.format(i) for i in rseeds]
fname_save = '/Users/kennedyj/PHYS_459/Github/wedge-repos/outputs/coeval_boxes/correct_transpose_sep27/HII_DIM_128_BOX_LEN_192_alpha_15_bar_max_2_500_boxes_rseed_exclusive_shuffled_full_training_set.h5'

# Training set breakdown
num_train = 400
num_val = 100
num_boxes = num_val+num_train

# Arrays to cast boxes into
new_bt_boxes = np.zeros((num_boxes, HII_DIM, HII_DIM, HII_DIM)) 
new_ion_boxes = np.zeros((num_boxes, HII_DIM, HII_DIM, HII_DIM)) 
new_wedge_filtered_bt_boxes = np.zeros((num_boxes, HII_DIM, HII_DIM, HII_DIM)) 
redshifts = np.zeros(num_boxes)
random_seeds = np.zeros(num_boxes)
counter = 0

# Randomly select n redshifts per random seed
for i in range(len(fname_coeval_boxes)):

    n = 1
    DM = DataManager(fname_coeval_boxes[i])
    np.random.seed(rseeds[i])
    zs = np.random.choice(np.array(DM.data["redshifts"]), n, replace=False)
    np.random.seed(rseeds[i])
    index = np.random.choice(np.arange(np.array(DM.data["redshifts"]).shape[0]), n, replace=False)

    # Ensure redshift matches with index
    assert(zs == np.array(DM.data["redshifts"])[index])

    # Cast into arrays
    redshifts[counter:counter+n] = zs
    random_seeds[counter:counter+n] = rseeds[i]
    new_wedge_filtered_bt_boxes[counter:counter+n] = np.array(DM.data["wedge_filtered_brightness_temp_boxes"])[index]
    new_bt_boxes[counter:counter+n] = np.array(DM.data["brightness_temp_boxes"])[index]
    new_ion_boxes[counter:counter+n] = np.array(DM.data["ionized_boxes"])[index]
    
    counter += n

# Shuffle training and validation sets separately
shuffle_rseed_train = 16
shuffle_rseed_val = 91
order = np.arange(0, num_boxes)
np.random.seed(shuffle_rseed_train)
np.random.shuffle(order[:num_train])
np.random.seed(shuffle_rseed_val)
np.random.shuffle(order[num_train:])

# Save data using same DM structure
DM.dset_attrs = {'p21c_initial_conditions': "{'user_params': {'HII_DIM': 128, 'BOX_LEN': 192}}"}
DM.data = {'brightness_temp_boxes': new_bt_boxes[order], 'ionized_boxes': new_ion_boxes[order], 'redshifts': redshifts[order], 
          'wedge_filtered_brightness_temp_boxes': new_wedge_filtered_bt_boxes[order], 'random_seeds': random_seeds[order]}
DM.filepath = fname_save

# Training, validation set # of boxes per redshift
print('\n Training set breakdown: ', np.unique(DM.data['redshifts'][:num_train], return_counts=True))
print('\n Validation set breakdown: ', np.unique(DM.data['redshifts'][num_train:], return_counts=True))

# Save training set
save_dset_to_hf(DM.filepath,DM.data,DM.dset_attrs)
