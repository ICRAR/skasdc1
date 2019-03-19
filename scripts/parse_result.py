import os
import os.path as osp

import numpy as np

import astropy.io.fits as pyfits
import astropy.wcs as pywcs

import commands # Python 2.7 for convinience

"""
Convert results from ClaRAN output into competition format
"""

cons = 2 ** 0.5
pixel_res_x = 1.67847000000E-04 #abs(float(fhead['CDELT1']))
pixel_res_y = 1.67847000000E-04 #abs(float(fhead['CDELT2']))
g = 2 * (pixel_res_x * 3600) # gridding kernel size as per specs
g2 = g ** 2

imfit_tpl = 'imfit in=%s "region=boxes(%d, %d, %d, %d)" object=gaussian'
histo_tpl = 'histo in=%s "region=boxes(%d, %d, %d, %d)"'

def restore_bmaj_bmin(b1, b2, size, clas):
    """
    Restore from image measurements into sky model parameters
    b1, b2 = bmaj, bmin
    """
    if (b1 ** 2 > g2):
        w1 = np.sqrt(b1 ** 2 - g2)
    else:
        w1 = b1
    if (b2 ** 2 > g2):
        w2 = np.sqrt(b2 ** 2 - g2)
    else:
        w2 = b2

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

incr = 2

def _get_integrated_flux(mir_file, x1, y1, x2, y2, h, w, error_codes):
    miriad_cmd = imfit_tpl % (mir_file, x1, y1, x2, y2)
    status, msg = commands.getstatusoutput(miriad_cmd)
    if (status == 0):
        #print("Can't find integrated flux from %s: %s" % (miriad_cmd, msg))
        return _derive_flux_from_msg(msg)
    else:
        if (len(error_codes) > 10):
            return None
        else:
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

def parse_single(result_file, fits_dir, mir_dir, pb, start_id=1, threshold=0.3):
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
    with open(result_file, 'r') as fin:
        lines = fin.read().splitlines()
    curr_fn = ''
    curr_w = None
    curr_d = None
    pbhdu = pyfits.open(pb)
    pbhead = pbhdu[0].header
    pb_wcs = pywcs.WCS(pbhead)
    pb_data = pbhdu[0].data[0][0]
    outlines = ['ID      RA (core)     DEC (core)  RA (centroid) DEC (centroid)           FLUX      Core frac           BMAJ           BMIN             PA      SIZE     CLASS']
    for idx, line in enumerate(lines):
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
        centroid = np.array([(x1 + x2) / 2, (h - y1 + h - y2) / 2, 0, 0], dtype=float)
        ra, dec = curr_w.wcs_pix2world([centroid], 0)[0][0:2]
        x1, y1, x2, y2 = [int(x) for x in (x1, y1, x2, y2)]
        sli = curr_d[(h - y2):min(h - y1 + 1, h), x1:min(x2 + 1, w)]
        mir_file = osp.join(mir_dir, fn.replace('.fits', '.mir'))
        
        #source_of_flux = 'imfit'
        failed_attempts = []
        total_flux = _get_integrated_flux(mir_file, x1, h - y2, x2, h - y1, h, w, failed_attempts)
        if (total_flux is None or np.isnan(total_flux)):
            #source_of_flux = 'histo'
            miriad_cmd = histo_tpl % (mir_file, x1, h - y2, x2, h - y1)
            total_flux = _get_integrated_flux_from_histo(miriad_cmd)
            if (total_flux is None or np.isnan(total_flux)):
                #source_of_flux = 'np.sum'
                total_flux = np.sum(sli)
        if (total_flux < 0):
            continue # ignore this source
        if (clas == 3):
            core_flux = 0.0
        else:
            if (clas == 1):
                core_flux = curr_d[(h - y1 + h - y2) // 2][(x1 + x2) // 2]
            else:
                #ind = np.unravel_index(np.argmax(sli, axis=None), sli.shape)
                #core_flux = sli[ind]
                core_flux = np.max(sli)
        core_flux = max(0.0, core_flux)
        core_frac = core_flux / total_flux
        #origin_flux = total_flux
        total_flux = _primary_beam_correction(total_flux, ra, dec, pb_wcs, pb_data)
        bmaj, bmin = restore_bmaj_bmin(b1, b2, size, clas)
        pa = 90 if box_w > box_h else 0
        out_line = [start_id, ra, dec, ra, dec, total_flux, core_frac, bmaj, bmin, pa, size, clas]#, origin_flux, core_flux, source_of_flux]
        out_line = [str(x) for x in out_line]
        out_line = '     '.join(out_line)
        start_id += 1
        outlines.append(out_line)
        if (idx % 1000 == 0):
            print('Done %d' % (idx + 1))
        # if (idx == 2000):
        #     break
    
    with open('icrar_560MHz_1000h_v3_st_%d.txt' % start_id, 'w') as fout:
        fc = os.linesep.join(outlines)
        fout.write(fc)

if __name__ == '__main__':
    result_file = '19327573.result'
    fits_dir = 'split_B1_1000h_test'
    mir_dir = 'split_B1_1000h_test_mir'
    pb = 'PrimaryBeam_B1.fits'
    parse_single(result_file, fits_dir, mir_dir, pb, start_id=0, threshold=0.0)

