import os
import os.path as osp

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

"""
Convert results from ClaRAN output into competition format
"""

cons = 2 ** 0.5
pixel_res_x = 1.67847000000E-04 #abs(float(fhead['CDELT1']))
pixel_res_y = 1.67847000000E-04 #abs(float(fhead['CDELT2']))
g = 2 * (pixel_res_x * 3600) # gridding kernel size as per specs
g2 = g ** 2

def restore_bmaj_bmin(b1, b2, size, clas):
    """
    Restore from image measurements into sky model parameters
    b1, b2 = bmaj, bmin
    """
    w1 = np.sqrt(b1 ** 2 - g2)
    w2 = np.sqrt(b2 ** 2 - g2)

    if (2 == size): 
        if (clas in [1, 3]): # extended source
            # don't change anything
            bmaj = w1
            bmin = w2
        else:
            # assuming core is fully covered by source
            bmaj = w1 * 2.0 #+ g / 10.0 
            bmin = bmaj #keep it circular Gaussian
    else: # compact source [1 or 3]
        # 1,2,3 for SS-AGNs, FS-AGNs, SFGs
        # 1,2,3 for LAS, Gaussian, Exponential
        if (1 == clas and 1 == size):
            bmaj = w1 * 2.0
            bmin = w2 * 2.0
        elif (3 == clas and 3 == size):
            bmaj = w1 / cons
            bmin = w2 / cons
        else:
            raise Exception('unknown combination')
    return bmaj, bmin

def parse_single(result_file, fits_dir, threshold=0.3):
    with open(result_file, 'r') as fin:
        lines = fin.read().splitlines()
    curr_fn = ''
    curr_w = None
    curr_d = None
    for line in lines:
        fds = line.split(',')
        score = float(fds[2])
        if score < threshold:
            continue
        fn = fds[0].replace('.png', '.fits')
        if (fn != curr_fn):
            curr_fn = fn
            fpath = osp.join(fits_dir, fn)
            hdulist = pyfits.open(fpath)
            fhead = hdulist[0].header
            curr_w = pywcs.WCS(fhead)
            curr_d = hdulist[0].data
            h, w = curr_d.shape[0:2]
        x1, y1, x2, y2 = [float(x) for x in fds[3].split('-')]
        box_h, box_w = y2 - y1 + 1, x2 - x1 + 1
        b1 = np.sqrt(box_h ** 2 + box_w ** 2)
        b2 = min(box_h, box_w)
        cat = fds[1].split('_')
        size = int(cat[0][0])
        clas = int(cat[1][0])
        centroid = np.array([(x1 + x2) / 2, (h - y1 + h - y2) / 2], dtype=float)
        ra, dec = curr_w.wcs_pix2world([centroid], 0)[0][0:2]
        x1, y1, x2, y2 = [int(x) for x in (x1, y1, x2, y2)]
        total_flux = np.sum(curr_d[y1:y2][x1:x2])
        bmaj, bmin = restore_bmaj_bmin(b1, b2, size, clas)
        pa = 90 if box_w > box_h else 0

