'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 22 2022

Description:

Python code for functions used by other modules in lightcone-gen.

'''

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

def L_to_MAB(L):

	"""
	Convert luminosities to absolute magnitudes.
	"""

    cm_per_pc = 3.08568e18
    flux_AB = 3631. * 1e-23 # erg / s / cm**2 / Hz
	d10 = 10 * cm_per_pc # 10pc in cm

	return -2.5 * np.log10(L / 4. / np.pi / d10**2 / flux_AB)

def get_mag_app(z, mags, cosmo_model):

	"""
	Convert absolute magnitudes to apparent magnitudes.
	"""

	d_pc = 1e6*cosmo_model.luminosity_distance(z) / u.Mpc

	return mags + 5 * np.log10(d_pc / 10.) - 2.5 * np.log10(1. + z)

def plot_slice_gals(box, ax=None, fig=None):
	''' plot_slice(bt_boxes[0]) '''
	kwargs={'origin': 'lower', 'aspect': 'auto', 'cmap': 'Reds'}
	if ax is None:
		fig,ax=plt.subplots(1, figsize=[5,5])
	trans_slice = np.take(box, 0, axis=0) #mpc_slice_index[i]
	im=ax.imshow(trans_slice, **kwargs)
	fig.colorbar(mappable=im,ax=ax)
	#ax.set_ylabel('Mpc', fontsize=15)
	#ax.set_xlabel('Mpc', fontsize=15)
	return im

def plot_slice(box, ax=None, fig=None):
	''' plot_slice(bt_boxes[0]) '''
	kwargs={'origin': 'lower', 'aspect': 'auto', 'cmap': 'binary'}
	if ax is None:
		fig,ax=plt.subplots(1, figsize=[5,5])
	trans_slice = np.take(box, 0, axis=0) #mpc_slice_index[i]
	im=ax.imshow(trans_slice, **kwargs)
	#ax.set_ylabel('Mpc', fontsize=15)
	#ax.set_xlabel('Mpc', fontsize=15)
	return im