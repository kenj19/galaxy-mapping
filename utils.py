import numpy as np

def binarize_boxes(boxes, cutoff=0.9): # binarize ionized boxes, neutral maps to 1, ionized to 0
    num_box = boxes.shape[0]
    binarized_boxes = np.zeros(boxes.shape)
    for i in range(num_box):
        sup_threshold_inds = (boxes[i] >= cutoff) # map to 1
        sub_threshold_inds = (boxes[i] < cutoff) # map to 0
        binarized_boxes[i][sup_threshold_inds] = 1 
        binarized_boxes[i][sub_threshold_inds] = 0
    return binarized_boxes