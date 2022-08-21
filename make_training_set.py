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

# Random seeds of 21cmFAST fields, number of redshifts per random seed
rseeds = np.array([33255, 727602, 804489, 326134, 113527, 880855, 283828])
num_rseeds = len(rseeds)
HII_DIM = 128
BOX_LEN = 128
num_z = 8
fname_coeval_boxes=['/Users/kennedyj/PHYS_459/Github/wedge-repos/outputs/coeval_boxes/128/HII_DIM_128_BOX_LEN_128_8_zs_rseed_{}.h5'.format(i) for i in rseeds]
fname_save = f'/Users/kennedyj/PHYS_459/Github/wedge-repos/outputs/coeval_boxes/128/HII_DIM_{HII_DIM}_BOX_LEN_{BOX_LEN}_{num_rseeds*num_z}_boxes_training_set.h5'
shuffle = True # Shuffle redshift order for each random seed

# Arrays to cast boxes into
new_bt_boxes = np.zeros((num_rseeds*num_z, HII_DIM, HII_DIM, HII_DIM)) 
new_ion_boxes = np.zeros((num_rseeds*num_z, HII_DIM, HII_DIM, HII_DIM)) 
new_wedge_filtered_bt_boxes = np.zeros((num_rseeds*num_z, HII_DIM, HII_DIM, HII_DIM)) 
redshifts = np.zeros(num_rseeds*num_z)

indices = np.arange(0,num_rseeds*num_z+1,num_z)

for i in range(num_rseeds):

    order = np.arange(num_z)

    if shuffle:

        random.shuffle(order)

    DM = DataManager(fname_coeval_boxes[i])
    print(DM.data['wedge_filtered_brightness_temp_boxes'].shape)
    
    new_wedge_filtered_bt_boxes[indices[i]:indices[i+1]] = DM.data["wedge_filtered_brightness_temp_boxes"][order]
    new_bt_boxes[indices[i]:indices[i+1]] = DM.data["brightness_temp_boxes"][order]
    new_ion_boxes[indices[i]:indices[i+1]] = DM.data["ionized_boxes"][order]
    redshifts[indices[i]:indices[i+1]] = DM.data["redshifts"][order]
   
DM.dset_attrs = {'p21c_initial_conditions': "{'user_params': {'HII_DIM': 128, 'BOX_LEN': 128}, 'random_seed': variable}"}
DM.data = {'brightness_temp_boxes': new_bt_boxes, 'ionized_boxes': new_ion_boxes, 'redshifts': redshifts, 
          'wedge_filtered_brightness_temp_boxes': new_wedge_filtered_bt_boxes}
DM.filepath = fname_save

save_dset_to_hf(DM.filepath,DM.data,DM.dset_attrs)
