import h5py
import numpy as np

def binarize_boxes(xH_boxes, cutoff=0.9): # binarize ionized boxes, neutral maps to 1, ionized to 0
    
    '''
    Function to binarize ionization fields, mapping voxels above
    the cutoff value to 1 (neutral) and voxels below the cutoff
    value to 0 (ionized).
    ------------------------------------------------------------------------------
    xH_boxes:
            Ionization field(s).
    cutoff:
            Binarization cutoff value, default = 0.0.
    ------------------------------------------------------------------------------
    ''' 
    
    num_box = xH_boxes.shape[0]
    binarized_boxes = np.zeros(xH_boxes.shape)
    
    for i in range(num_box):
        
        sup_threshold_inds = (xH_boxes[i] >= cutoff) # map to 1
        sub_threshold_inds = (xH_boxes[i] < cutoff) # map to 0
        binarized_boxes[i][sup_threshold_inds] = 1 
        binarized_boxes[i][sub_threshold_inds] = 0
        
    return binarized_boxes
    
def get_n_i_halo_mass_coords(halocoords, halomasses, gt_ionizedbox, pred_ionizedbox, save_name, scale=3):
    
    '''
    Function to find the halo coordinates and masses of halos 
    found in the ionized and neutral regions of ionization field.
    ------------------------------------------------------------------------------
    halocoords: 
         Coordinates of halos in box.
    halomasses:
            Masses of halos in box.
    gt_ionizedbox:
            Ground-truth (21cmFAST output) ionization field. 
    pred_ionizedbox:
            Predicted (U-Net output) ionization field. 
    save_name:
            Name to save .h5 to.
    scale:
            DIM//HII_DIM, default = 3.
    ------------------------------------------------------------------------------
    ''' 
     
    # Need to scale down dim from high to low res
    halo_low_res_coords = halocoords // scale
    num_halos = len(halo_masses)
    
    # Maximum number of each class of halos is num_halos
    pred_neutral_halo_coords = np.zeros(shape=(num_halos, 3))
    pred_neutral_halo_masses = np.zeros(shape=num_halos)
    pred_ionized_halo_coords = np.zeros(shape=(num_halos, 3))
    pred_ionized_halo_masses = np.zeros(shape=num_halos)
    gt_neutral_halo_coords = np.zeros(shape=(num_halos, 3))
    gt_neutral_halo_masses = np.zeros(shape=num_halos)
    gt_ionized_halo_coords = np.zeros(shape=(num_halos, 3))
    gt_ionized_halo_masses = np.zeros(shape=num_halos)
    
    pred_ion_count = 0
    pred_ntl_count = 0
    gt_ion_count = 0
    gt_ntl_count = 0
    
    # Check if halo in neutral or ionized region of gt/pred fields (neutral = 1, ionized = 0)
    for i in range(num_halos): 
    
        x = halo_low_res_coords[i][0]
        y = halo_low_res_coords[i][1]
        z = halo_low_res_coords[i][2]
        
        if (pred_ionizedbox[x,y,z] == 1) and (gt_ionizedbox[x,y,z] == 1):
        
            pred_neutral_halo_coords[pred_ntl_count]=np.array([x,y,z])
            pred_neutral_halo_masses[pred_ntl_count]=halomasses[i]
            pred_ntl_count +=1 

            gt_neutral_halo_coords[gt_ntl_count]=np.array([x,y,z])
            gt_neutral_halo_masses[gt_ntl_count]=halomasses[i]
            gt_ntl_count +=1  
        
        elif (pred_ionizedbox[x,y,z] == 1) and (gt_ionizedbox[x,y,z] != 1):  
            
            pred_neutral_halo_coords[pred_ntl_count]=np.array([x,y,z])
            pred_neutral_halo_masses[pred_ntl_count]=halomasses[i]
            pred_ntl_count +=1
             
            gt_ionized_halo_coords[gt_ion_count]=np.array([x,y,z])
            gt_ionized_halo_masses[gt_ion_count]=halomasses[i]
            gt_ion_count +=1
        
        elif (pred_ionizedbox[x,y,z] != 1) and (gt_ionizedbox[x,y,z] == 1):
        
            pred_ionized_halo_coords[pred_ion_count]=np.array([x,y,z])
            pred_ionized_halo_masses[pred_ion_count]=halomasses[i]
            pred_ion_count +=1
            
            gt_neutral_halo_coords[gt_ntl_count]=np.array([x,y,z])
            gt_neutral_halo_masses[gt_ntl_count]=halomasses[i]
            gt_ntl_count +=1 
            
        else:
        
            pred_ionized_halo_coords[pred_ion_count]=np.array([x,y,z])
            pred_ionized_halo_masses[pred_ion_count]=halomasses[i]
            pred_ion_count +=1
            
            gt_ionized_halo_coords[gt_ion_count]=np.array([x,y,z])
            gt_ionized_halo_masses[gt_ion_count]=halomasses[i]
            gt_ion_count +=1
    
    # remove left-over zeros
    pred_neutral_halo_masses = np.trim_zeros(pred_neutral_halo_masses,'b')  
    pred_ionized_halo_masses = np.trim_zeros(pred_ionized_halo_masses,'b') 
    pred_neutral_halo_coords = pred_neutral_halo_coords[:len(pred_neutral_halo_masses)]
    pred_ionized_halo_coords = pred_ionized_halo_coords[:len(pred_ionized_halo_masses)]
    
    gt_neutral_halo_masses = np.trim_zeros(gt_neutral_halo_masses,'b')  
    gt_ionized_halo_masses = np.trim_zeros(gt_ionized_halo_masses,'b') 
    gt_neutral_halo_coords = gt_neutral_halo_coords[:len(gt_neutral_halo_masses)]
    gt_ionized_halo_coords = gt_ionized_halo_coords[:len(gt_ionized_halo_masses)]
    
    # save to .h5 file
    hf = h5py.File(save_name, 'w')
    
    hf.create_dataset('pred_neutral_halo_masses', data=pred_neutral_halo_masses)
    hf.create_dataset('pred_ionized_halo_masses', data=pred_ionized_halo_masses)
    hf.create_dataset('pred_neutral_halo_coords', data=pred_neutral_halo_coords)
    hf.create_dataset('pred_ionized_halo_coords', data=pred_ionized_halo_coords)
    hf.create_dataset('gt_neutral_halo_masses', data=gt_neutral_halo_masses)
    hf.create_dataset('gt_ionized_halo_masses', data=gt_ionized_halo_masses)
    hf.create_dataset('gt_neutral_halo_coords', data=gt_neutral_halo_coords)
    hf.create_dataset('gt_ionized_halo_coords', data=gt_ionized_halo_coords)
    hf.create_dataset('random_seed', data=seed)
    
    print(f"\n ======= All datasets created, saved to {save_name}. ======= \n")

    hf.close()