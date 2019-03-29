import commands
import os.path as osp
import numpy as np
import os

from astropy.io import fits
from scipy.special import erfinv
mad2sigma = np.sqrt(2) * erfinv(2 * 0.75 - 1)
NUM_SIGMA = 1

def obtain_sigma(fits_fn):
    with fits.open(fits_fn) as f:
        imgdata = f[0].data
    med = np.nanmedian(imgdata)
    mad = np.nanmedian(np.abs(imgdata - med))
    sigma = mad / mad2sigma
    return sigma, med, imgdata

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

imfit_tpl = 'imfit in=%s "region=boxes(%d, %d, %d, %d)" object=gaussian clip=%f'
incr = 2

def _get_integrated_flux(mir_file, clip_level, x1, y1, x2, y2, h, w):
    miriad_cmd = imfit_tpl % (mir_file, x1, y1, x2, y2, clip_level)
    status, msg = commands.getstatusoutput(miriad_cmd)
    if (status == 0):
        #print("Can't find integrated flux from %s: %s" % (miriad_cmd, msg))
        return _derive_flux_from_msg(msg)
    else:
        return np.nan
        # if (len(error_codes) > 1):
        #     return None
        # else:
        #     error_codes.append(status)
        #     x1 = max(0, x1 - incr)
        #     x2 = min(w - 1, x2 + incr)
        #     y1 = max(0, y1 - incr)
        #     y2 = min(h - 1, y2 + incr)
        #     return _get_integrated_flux(mir_file, x1, y1, x2, y2, h, w, error_codes)

if __name__ == '__main__':
    mir_dir = 'split_B1_1000h_mir'
    fits_dir = 'split_B1_1000h'

    with open('source_entry.csv') as fin:
        lines = fin.read().splitlines()
    ret = []
    sigma_dict = dict()
    for line in lines:
        fds = line.split(',')
        fid = fds[0]
        fits_img = osp.join(fits_dir, fid)
        if (fid not in sigma_dict):
            sigma, med, _ = obtain_sigma(fits_img)
            clip_level = med + NUM_SIGMA * sigma
            sigma_dict[fid] = clip_level
        else:
            clip_level = sigma_dict[fid]
        mir_file = osp.join(mir_dir, fds[0].replace('.fits', '.mir'))
        flux = _get_integrated_flux(mir_file, clip_level, *[int(x) for x in fds[2:]])
        #print(flux)
        ret.append('%s,%f' % (fds[1], flux))
    
    with open('miriad_flux_clipped.csv', 'w') as fout:
        fout.write(os.linesep.join(ret))