"""
Script for saving/loading h5 datafiles
@author: j-c-carr
"""

import h5py
import numpy as np
from pprint import pprint

class DataManager():

    """
    Loads data from h5py file. Once a DataManager object is loaded, you can
    access the data like so:
         ... = DM.data["wedge_filtered_brightness_temp_boxes"]
         ... = DM.data["brightness_temp_boxes"]
         ... = DM.data["ionized_boxes"]
         ... = DM.data["redshifts"]
         ... = DM.data["predicted_brightness_temp_boxes"]
    Metadata is stored in DM.metadata
    """

    def __init__(self, filepath: str):
        assert filepath[-3:] == ".h5", "filepath must point to an h5 file."

        self.filepath = filepath
        self.metadata = {}
        self.data = {}

        self.load_data_from_h5()
        

    def load_data_from_h5(self):
        """
        Loads coeval boxes from h5py file. Assumes h5py file has 5 datasets,
        'brightness_temp_boxes' -- ground truth brightness temp boxes 
        'wedge_filtered_brightness_temp_boxes' -- brightness temp boxes minus wedge
        'predicted_brightness_temp_boxes' -- predicted brightness temp from model
        'ionized_boxes' -- ionized boxes corresponding to brightness temp box
        'redshifts' --> redshift of each brightness temp box
        """

        with h5py.File(self.filepath, "r") as hf:

            # Check we have the required datasets
            datasets = list(hf.keys())
            assert "wedge_filtered_brightness_temp_boxes" in datasets and \
                   "brightness_temp_boxes" in datasets and \
                   "predicted_brightness_temp_boxes" in datasets and \
                   "ionized_boxes" in datasets and \
                   "redshifts" in datasets, \
                   "Failed to extract datasets from h5py file."

            for k in hf.keys():
                v = np.array(hf[k][:], dtype=np.float32)
                assert np.isnan(np.sum(v)) == False
                self.data[k] = v
            self.data["redshifts"].reshape(-1) 

            # Load metadata from h5 file
            for k, v in hf.attrs.items():
                self.metadata[k] = v

        # Print success message
        print("\n----------\n")
        print(f"data loaded from {self.filepath}")
        print("Contents:")
        for k, v in self.data.items():
            print("\t{}, shape: {}".format(k, v.shape))
        print("\nMetadata:")
        pprint(self.metadata)
        print("\n----------\n")
