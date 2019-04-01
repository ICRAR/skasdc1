import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np

import os
import os.path as osp

from multiprocessing import Pool

"""
Uses the primary beam model to correct the "apparent flux density" for each pixel 
of a cutout fits image

For B1, the size of the cutout fits image is roughly the same as the pixel resolution of the PB.

8.08729927827E-02 - 1.67847000000E-04 * 205 = 0.04646435778270001

Therefore we use a approximate "lazy" method, 
    i.e. the entire cutout fits image will be corrected using the same PB value corresponding to
    the centre pixel of that cuout fits image.

"""

NUM_PROCS = 4
output_fits_dir = 'split_B1_1000h_test_pbcorrected'

#def _setup_pb(pb_fn):
pb_fn = 'PrimaryBeam_B1.fits'
pbhdu = pyfits.open(pb_fn)
pbhead = pbhdu[0].header
pb_wcs = pywcs.WCS(pbhead)
pb_data = pbhdu[0].data[0][0]

def _apply_primary_beam(apparent_fluxes, ra, dec):
    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    pbv = pb_data[int(y)][int(x)]
    apparent_fluxes *= pbv

def pb_worker(fin):
    #print(fin)
    hdulist = pyfits.open(fin)
    data = hdulist[0].data
    wcs = pywcs.WCS(hdulist[0].header)
    width = data.shape[1]
    height = data.shape[0]
    cx = width / 2
    cy = height / 2
    ra, dec = wcs.wcs_pix2world([[cx, cy, 0, 0]], 0)[0][0:2]

    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    pbv = pb_data[int(y)][int(x)]
    hdulist[0].data *= pbv
    hdulist[0].data = np.reshape(hdulist[0].data, [1, 1, height, width])
    #print(hdulist[0].data.shape, height, width, 1, 1)
    hdulist.writeto(osp.join(output_fits_dir, osp.basename(fin)))
    
def correct_pb(fits_cutout_dir):
    pool = Pool(NUM_PROCS)
    fns = os.listdir(fits_cutout_dir)
    fns = [osp.join(fits_cutout_dir, x) for x in fns if x.endswith('.fits')]
    #print(fns)
    pool.map_async(pb_worker, fns).get(9999999)

if __name__ == '__main__':
    correct_pb('split_B1_1000h_test')
