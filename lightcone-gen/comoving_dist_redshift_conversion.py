'''

Author: Jacob Kennedy (jacob.kennedy@mail.mcgill.ca)

Created On: August 22 2022

Description:

Python code to convert comoving distances to redshifts with the 
intent of constructing lightcones from coeval boxes with a fixed 
cMpc sidelength evaluated at the redshifts computed.

'''

import h5py
import numpy as np
import astropy.units as u
from astropy.cosmology import z_at_value, FlatLambdaCDM

H0 = 67.32 * u.km / u.s / u.Mpc
Tcmb0 = 2.725 * u.K
Om0 = 0.3158
num_z = 3
BOX_LEN = 128
cosmo = FlatLambdaCDM(H0=H0, Tcmb0 = Tcmb0, Om0=Om0)
dist = 9000 + np.arange(num_z)*BOX_LEN # in cMpc
redshifts = np.zeros(shape=num_z)
fname_save = f'comoving_dist_redshift_conversion_BOX_LEN_{BOX_LEN}_zs_{num_z}.h5'

hf = h5py.File(fname_save,'w')
hf.create_dataset('comoving_distances', data=dist)
hf.create_dataset('BOX_LEN', data=BOX_LEN)

for i in range(num_z):

    z = z_at_value(cosmo.comoving_distance, dist[i]*u.Mpc)
    redshifts[i] = z
    print(f'\n ====== Redshift: {z} ====== \n ')

hf.create_dataset('redshifts', data=redshifts)
hf.close()