'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 23 2022

Description:

Python code to compare the number of galaxies identified above
survey threshold in the "interpolated" mass accretion halo mass fields
and the actual halo field obtained by running the 21cmFAST halo finder.

'''

import h5py
import numpy as np
import matplotlib.pyplot as plt

# Load in halo mass fields
interp_z = 7.997138310109906
fname_true = '/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/galaxy_cutoffs_HII_DIM_128_DIM_384_BOXLEN_128_z_7.997138310109906_rseed_42142.h5'
fname_interp = '/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/interp_galaxy_cutoffs_HII_DIM_128_DIM_384_BOXLEN_128_z_7.997138310109906_rseed_42142.h5'
hf_truth = h5py.File(fname_true, 'r')
hf_interp = h5py.File(fname_interp, 'r')

halo_mass_field_true = np.array(hf_truth['halo_mass_field'])
JWST_UD_gals_true = np.array(hf_truth['JWST_UD_gals'])
JWST_MD_gals_true = np.array(hf_truth['JWST_MD_gals'])
JWST_WF_gals_true = np.array(hf_truth['JWST_WF_gals'])
Roman_gals_true = np.array(hf_truth['Roman_gals'])

halo_mass_field_interp = np.array(hf_interp['inter_halo_mass_field'])
JWST_UD_gals_interp = np.array(hf_interp['JWST_UD_gals'])
JWST_MD_gals_interp = np.array(hf_interp['JWST_MD_gals'])
JWST_WF_gals_interp = np.array(hf_interp['JWST_WF_gals'])
Roman_gals_interp= np.array(hf_interp['Roman_gals'])

mass_diff = np.sum(halo_mass_field_true) - np.sum(halo_mass_field_interp)
print(f"\n ====== The total halo mass difference (true - interp) is: {mass_diff/1e10} 10^10 M_sol ====== \n")


# Compute number of galaxies above each survey magnitude threshold
num_JWST_UD_gals_true = len(JWST_UD_gals_true[JWST_UD_gals_true > 0])
num_JWST_MD_gals_true = len(JWST_MD_gals_true[JWST_MD_gals_true > 0])
num_JWST_WF_gals_true = len(JWST_WF_gals_true[JWST_WF_gals_true > 0])
num_Roman_gals_true = len(Roman_gals_true[Roman_gals_true > 0])

num_JWST_UD_gals_interp = len(JWST_UD_gals_interp[JWST_UD_gals_interp > 0])
num_JWST_MD_gals_interp = len(JWST_MD_gals_interp[JWST_MD_gals_interp > 0])
num_JWST_WF_gals_interp = len(JWST_WF_gals_interp[JWST_WF_gals_interp > 0])
num_Roman_gals_interp = len(Roman_gals_interp[Roman_gals_interp > 0])

# Make a bar graph with number of galaxies above threshold for true and interp halo mass fields
labels = ['JWST-UD', 'JWST-MD', 'JWST-WF', 'Roman']
true_gal_counts = [num_JWST_UD_gals_true, num_JWST_MD_gals_true, num_JWST_WF_gals_true, num_Roman_gals_true]
interp_gal_counts = [num_JWST_UD_gals_interp, num_JWST_MD_gals_interp, num_JWST_WF_gals_interp, num_Roman_gals_interp]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, true_gal_counts, width, label='True')
rects2 = ax.bar(x + width/2, interp_gal_counts, width, label='Interp')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel(r'$m_{AB, gal} > m_{AB, survey}$')
plt.xticks(x, labels)
plt.xscale('log')
plt.legend()

plt.bar_label(rects1, padding=3)
plt.bar_label(rects2, padding=3)

fig.tight_layout()
plt.savefig(f'/Users/kennedyj/PHYS_459/lc-gen-gal-cutoffs/compare_interp_@_z_{interp_z}.jpeg')
