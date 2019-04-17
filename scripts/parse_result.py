import os
import os.path as osp

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

from astropy.io import fits
from scipy.special import erfinv
mad2sigma = np.sqrt(2) * erfinv(2 * 0.75 - 1)

import commands # Python 2.7 for convinience
from collections import defaultdict

"""
Convert results from ClaRAN output into competition format
"""

cons = 2 ** 0.5
pixel_res_x = 1.67847000000E-04 #abs(float(fhead['CDELT1']))
pixel_res_y = 1.67847000000E-04 #abs(float(fhead['CDELT2']))
g = 2 * (pixel_res_x * 3600) # gridding kernel size as per specs
g2 = g ** 2
BMAJ = 4.16666676756E-04
BMIN = 4.16666676756E-04
psf_bmaj_ratio = BMAJ / pixel_res_x
psf_bmin_ratio = BMIN / pixel_res_y
synth_beam_size = psf_bmaj_ratio * psf_bmin_ratio

#  object=disks allows for a "less Gaussian" and "sharper" object truncation and so it will be more flexible with non-compact source fluxes.
imfit_tpl = 'imfit in=%s "region=boxes(%d, %d, %d, %d)" object=gaussian'
histo_tpl = 'histo in=%s "region=boxes(%d, %d, %d, %d)"'

b1_sigma = 3.5789029744176247e-07
b1_median = -2.9249549e-08
b1_three_sigma = -2.9249549e-08 + 3 * b1_sigma

def obtain_sigma(fits_fn):
    with fits.open(fits_fn) as f:
        imgdata = f[0].data
    med = np.nanmedian(imgdata)
    mad = np.nanmedian(np.abs(imgdata - med))
    sigma = mad / mad2sigma
    return sigma, med

def restore_bmaj_bmin(b1, b2, size, clas):
    """
    Restore from image measurements into sky model parameters
    b1, b2 = bmaj, bmin but in number of pixels!
    """
    # degridding
    b1 *= pixel_res_x * 3600
    b2 *= pixel_res_x * 3600

    gap = (b1) ** 2 - g2
    if (gap > 0):
        w1 = np.sqrt(gap)
    else:
        w1 = b1

    gap = (b2) ** 2 - g2
    if (gap > 0):
        w2 = np.sqrt(gap)
    else:
        w2 = b2

    # if (2 == size): 
    #     if (clas in [1, 3]): # extended source
    #         # don't change anything
    #         bmaj = w1
    #         bmin = w2
    #     else:
    #         # assuming core is fully covered by source
    #         bmaj = w1 * 2.0 #+ g / 10.0 
    #         bmin = bmaj #keep it circular Gaussian
    # else: # compact source [1 or 3]
    #     # 1,2,3 for SS-AGNs, FS-AGNs, SFGs
    #     # 1,2,3 for LAS, Gaussian, Exponential
    #     if (1 == clas and 1 == size):
    #         bmaj = w1 * 2.0
    #         bmin = w2 * 2.0
    #     elif (3 == clas and 3 == size):
    #         bmaj = w1 / cons
    #         bmin = w2 / cons
    #     else:
    #         raise Exception('unknown combination')
    return w1, w2

def _derive_flux_from_msg(msg):
    for line in msg.split(os.linesep):
        if (line.find('Total integrated flux') > -1):
            fds = line.split()
            """
            for idx, fd in enumerate(fds):
                if (fd == '+/-'):
                    print(idx - 1)
                    return float(fds[idx - 1])
            """
            return float(fds[3])
    return None

incr = 1

def _get_integrated_flux(mir_file, x1, y1, x2, y2, h, w, error_codes):
    miriad_cmd = imfit_tpl % (mir_file, x1, y1, x2, y2)
    status, msg = commands.getstatusoutput(miriad_cmd)
    if (status == 0):
        #print("Can't find integrated flux from %s: %s" % (miriad_cmd, msg))
        return _derive_flux_from_msg(msg)
    else:
        if (len(error_codes) > 3):
            return None
        else:
            print(miriad_cmd, msg)
            error_codes.append(status)
            x1 = max(0, x1 - incr)
            x2 = min(w - 1, x2 + incr)
            y1 = max(0, y1 - incr)
            y2 = min(h - 1, y2 + incr)
            return _get_integrated_flux(mir_file, x1, y1, x2, y2, h, w, error_codes)

def _get_integrated_flux_from_histo(miriad_cmd):
    status, msg = commands.getstatusoutput(miriad_cmd)
    if (status == 0):
        for line in msg.split(os.linesep):
            if (line.find('Flux') > -1):
                fds = line.split()
                """
                for idx, fd in enumerate(fds):
                    if (fd == 'Flux'):
                        print(idx + 1)
                        return float(fds[idx + 1])
                """
                return float(fds[5])
        print("Can't find integrated flux from %s: %s" % (miriad_cmd, msg))
        return None
    else:
        print('Fail to execute %s: %d' % (miriad_cmd, status))
        return None

def _primary_beam_correction(total_flux, ra, dec, pb_wcs, pb_data):
    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #print(pb_data.shape)
    pbv = pb_data[int(y)][int(x)]
    #print(x, y, pbv)
    return total_flux / pbv

def visual_result(result_file, fits_dir, out_dir, threshold=0.8, show_class=True):
    img_dict = dict()
    box_dict = defaultdict(list)
    text_dict = defaultdict(list)
    bp_fmt = 'physical; box %fp %fp %dp %dp 0 # color=white'
    cls_fmt = 'physical; text %fp %fp {%d} # color=white'
    with open(result_file, 'r') as fin:
        lines = fin.read().splitlines()
    for idx, line in enumerate(lines):
        if (idx % 1000 == 0):
            print('Done %d' % (idx + 1))

        fds = line.split(',')
        score = float(fds[2])
        if score < threshold:
            continue
        fn = fds[0].replace('.png', '.fits')
        if (fn not in img_dict):
            hdulist = pyfits.open(osp.join(fits_dir, fn))
            #print(hdulist[0].data.shape)
            img_h, w = hdulist[0].data.shape[2:]
            img_dict[fn] = (img_h, w)
        else:
            img_h, w = img_dict[fn]
        x1, y1, x2, y2 = [float(x) for x in fds[3].split('-')]
        bh, bw = y2 - y1 + 1, x2 - x1 + 1
        cx = (x1 + x2) / 2 + 1
        cy = (y1 + y2) / 2 + 1
        #print(cy, img_h)
        cy = img_h - cy
        box_dict[fn].append(bp_fmt % (cx, cy, bw + 1, bh + 1))
        if (show_class):
            cls_lbl = int(fds[1])
            text_dict[fn].append(cls_fmt % (x1 + 1, cy + (bh + 1) / 2, cls_lbl))
        
    for k, v in box_dict.items():
        out_fn = osp.join(out_dir, k.replace('.fits', '.reg'))
        with open(out_fn, 'w') as fout:
            fout.write(os.linesep.join(v))
            if (show_class):
                fout.write(os.linesep)
                fout.write(os.linesep.join(text_dict[k]))

def _setup_pb(pb_fn):
    pbhdu = pyfits.open(pb_fn)
    pbhead = pbhdu[0].header
    pb_wcs = pywcs.WCS(pbhead)
    pb_data = pbhdu[0].data[0][0]
    return pb_wcs, pb_data

def parse_single(result_file, fits_dir, mir_dir, pb_fn, start_id=1, threshold=0.8):
    """
    For B1
    SIZE_CLASS, count
    1_1,    4034
    2_1,    380
    2_2,    3728
    2_3,    179946
    3_3,    372

    Q. How is the core position defined?
    A. For the resolved steep-spectrum AGNs, this is the position of the central spot, where the active nucleus is. 
    Depending on the morphology of the AGN it could not be very bright and therefore difficult to localise. 
    For all the other sources this is the position of the peak, which, given the central symmetry, 
    also coincides with the centroid position.

    core_fract for class 1 - central spot value / total flux
    core_fract for class 2,3 - peak spot value / total flux

    use miriad histo instead of imfit to get the total flux
    but if central spot value is negative, then use peak spot value

    Q. How is the size defined? A. it depends on the source population.
    SS-AGNs (drawn as real AGN images): bmaj is the Largest Angular Size (LAS), 
        defined as the diameter of the smallest circle that encloses the whole source. 
        Due to the complex morphology of these sources and the absence of a uniquely defined axis, 
        bmin and PA are not well defined and will not be scored.  
    Flat-spectrum AGNs + unresolved sources (draw as elliptical Gaussians): 
        bmaj/bmin are the fwhm of the two Gaussians along the axes of the ellipse.
    Resolved SFGs (drawn as elliptical exponentials): 
        bmaj/bmin are the exponential scale lengths along the axes of the ellipse.

    """
    pb_wcs, pb_data = _setup_pb(pb_fn)
    with open(result_file, 'r') as fin:
        lines = fin.read().splitlines()
    curr_fn = ''
    curr_w = None
    curr_d = None
    img_dict = dict()
    clas_size_dict = {1:1, 2:2, 3:2}
    outlines = ['ID      RA (core)     DEC (core)  RA (centroid) DEC (centroid)           FLUX      Core frac           BMAJ           BMIN             PA      SIZE     CLASS']
    for idx, line in enumerate(lines):
        if (idx % 1000 == 0):
            print('Done %d' % (idx + 1))

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
            h, w = curr_d.shape
            #print(h, w)
        x1, y1, x2, y2 = [float(x) for x in fds[3].split('-')]
        box_h, box_w = y2 - y1 + 1, x2 - x1 + 1
        b1 = np.sqrt(box_h ** 2 + box_w ** 2)
        b2 = min(box_h, box_w)
        clas = int(fds[1])#.split('_')
        size = 1
        cx = (x1 + x2) / 2 + 1
        cy = (y1 + y2) / 2 + 1
        #print(cy, img_h)
        cy = h - cy
        centroid = np.array([cx, cy, 0, 0], dtype=float)
        ra, dec = curr_w.wcs_pix2world([centroid], 0)[0][0:2]
        x1, y1, x2, y2 = [int(x) for x in (x1, y1, x2, y2)]
        
        #source_of_flux = 'imfit'
        failed_attempts = []
        mir_file = osp.join(mir_dir, fn.replace('.fits', '.mir'))
        total_flux = _get_integrated_flux(mir_file, x1, h - y2, x2, h - y1, h, w, failed_attempts)
        # if (total_flux is None or np.isnan(total_flux)):
        #     #source_of_flux = 'histo'
        #     miriad_cmd = histo_tpl % (mir_file, x1, h - y2, x2, h - y1)
        #     total_flux = _get_integrated_flux_from_histo(miriad_cmd)
        #     if (total_flux is None or np.isnan(total_flux)):
        #         #source_of_flux = 'np.sum'
        #         total_flux = np.sum(sli)
        sli = None
        if (total_flux is None or total_flux <= 0):
            print('Failed miriad - ', failed_attempts)
            sli = curr_d[(h - y2):min(h - y1 + 1, h), x1:min(x2 + 1, w)]
            sli = sli[np.where(sli >= 0)]
            total_flux = np.sum(sli) / synth_beam_size
            if (total_flux <= 0):
                print('still non-positive integrated flux at %s' % fn)
                continue # ignore this source
        if (clas == 3): # 1 component
            core_flux = 0.0
        else:
            if (clas == 1): # 3 component
                core_flux = curr_d[(h - y1 + h - y2) // 2][(x1 + x2) // 2]
            else: # 2 component
                #ind = np.unravel_index(np.argmax(sli, axis=None), sli.shape)
                #core_flux = sli[ind]
                if (sli is None):
                    sli = curr_d[(h - y2):min(h - y1 + 1, h), x1:min(x2 + 1, w)]
                core_flux = np.max(sli)
        core_flux = max(0.0, core_flux)
        core_frac = core_flux / total_flux
        #origin_flux = total_flux
        total_flux = _primary_beam_correction(total_flux, ra, dec, pb_wcs, pb_data)
        print(total_flux)
        bmaj, bmin = restore_bmaj_bmin(b1, b2, size, clas)
        #bmaj, bmin = max(b1, b2), min(b1, b2)
        pa = 90 if box_w > box_h else 0
        out_line = [start_id, ra, dec, ra, dec, total_flux, core_frac, bmaj, bmin, pa, size, clas]#, origin_flux, core_flux, source_of_flux]
        out_line = [str(x) for x in out_line]
        out_line = '     '.join(out_line)
        start_id += 1
        outlines.append(out_line)
    
    with open('icrar_560MHz_1000h_v5_st_%d.txt' % start_id, 'w') as fout:
        fc = os.linesep.join(outlines)
        fout.write(fc)

if __name__ == '__main__':
    result_file = '21592081.result'
    fits_dir = 'split_B1_1000h_test'
    mir_dir = 'split_B1_1000h_test_mir'
    pb = 'PrimaryBeam_B1.fits'
    parse_single(result_file, fits_dir, mir_dir, pb, start_id=0, threshold=0.8)
    #visual_result(result_file, fits_dir, 'reg_out')

