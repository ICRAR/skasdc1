import os
import os.path as osp

import numpy as np
import astropy.io.fits as pyfits
import astropy.wcs as pywcs

montage_path = '/home/ngas/software/Montage_v3.3/bin'
subimg_exec = '%s/mSubimage' % montage_path
regrid_exec = '%s/mProject' % montage_path
imgtbl_exec = '%s/mImgtbl' % montage_path
coadd_exec = '%s/mAdd' % montage_path
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)

SDC1_HOME = '/Users/chen/gitrepos/ml/skasdc1'

# these are for all map
b1_sigma = 3.5789029744176247e-07
b1_median = -2.9249549e-08

b2_sigma = 6.800226440872264e-08
b2_median = -2.3471074e-09

b5_sigma = 3.8465818166348711e-08
b5_median = -5.2342429e-11

# these are for training map only
#b1_sigma = 3.8185009938219004e-07 #
#b1_median = -1.9233363e-07 #

num_sigma = 1

b1_three_sigma = b1_median + num_sigma * b1_sigma
b2_three_sigma = b2_median + num_sigma * b2_sigma
b5_three_sigma = b5_median + num_sigma * b5_sigma

def convert_to_csv(fname):
    csv_list = []
    with open(fname, 'r') as fin:
        lc = fin.read().splitlines()
        hdr_done = False
        for line in lc:
            if (line.startswith('COLUMN')):
                continue
            if (not hdr_done and line.find('RA (core)') > -1):
                line = line.replace('RA (core)', 'RA_core')\
                .replace('DEC (core)', 'DEC_core')\
                .replace('RA (centroid)', 'RA_centroid')\
                .replace('DEC (centroid)', 'DEC_centroid')\
                .replace('Core frac', 'Core_frac')
                hdr_done = True
            fds = line.split()
            csv_line = ','.join(fds)
            csv_list.append(csv_line)
    
    outfname = fname.replace('.txt', '.csv')
    with open(outfname, 'w') as fou:
        fou.write(os.linesep.join(csv_list))

def cutout_training_image(cat_csv, field_fits):
    """
    Cutout a single training image that contains all the objects 
    on the training catalog from the "field_fits"
    """
    hdulist = pyfits.open(field_fits)
    fhead = hdulist[0].header
    pixel_res = abs(float(fhead['CDELT1'])) * 3600
    w = pywcs.WCS(fhead)
    xs = []
    ys = []
    szs = []
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
        for line in lines[1:]: #skip the header
            fds = line.split(',')
            ra, dec = fds[3:5]
            ra, dec = float(ra), float(dec)
            size = float(fds[-5]) / 2.0 / pixel_res
            x, y = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
            xs.append(x)
            ys.append(y)
            szs.append(size)
    # mesz = max(szs) / (abs(float(fhead['CDELT1'])) * 3600) # deg to arcsec
    # need to add the potential source edge LAS or exponential scale length
    # https://astronomy.swin.edu.au/cosmos/S/Scale+Length
    mesz = np.mean(szs)
    x1, y1, x2, y2 = int(np.floor(min(xs) - mesz)), int(np.floor(min(ys) - mesz)),\
                     int(np.ceil(max(xs) + mesz)), int(np.ceil(max(ys) + mesz))
    print(x1, y1, x2, y2, mesz)
    out_fname = field_fits.replace('.fits', '_train_image.fits')
    print(splitimg_cmd % (field_fits, out_fname, x1, y1, (x2 - x1), (y2 - y1)))

def _primary_beam_gridding(total_flux, ra, dec, pb_wcs, pb_data):
    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #print(pb_data.shape)
    pbv = pb_data[int(y)][int(x)]
    #print(x, y, pbv)
    return total_flux * pbv

def gen_ds9_region(cat_csv, fits_img, pb, consider_psf=True, fancy=True):
    """
    For B1
    SIZE_CLASS, count
    1_1,    4034
    2_1,    380
    2_2,    3728
    2_3,    179946
    3_3,    372

    But after sigma clipping
    {'1_1': 117, '2_1': 37, '2_2': 278, '3_3': 250, '2_3': 19585}

    COLUMN11:    SIZE    [none]    1,2,3 for LAS, Gaussian, Exponential
    COLUMN12:    CLASS    [none]   1,2,3 for SS-AGNs, FS-AGNs,SFGs

    """
    pbhdu = pyfits.open(pb)
    pbhead = pbhdu[0].header
    pb_wcs = pywcs.WCS(pbhead)
    pb_data = pbhdu[0].data[0][0]

    cons = 2 ** 0.5
    hdulist = pyfits.open(fits_img)
    fhead = hdulist[0].header
    pixel_res_x = abs(float(fhead['CDELT1']))
    pixel_res_y = abs(float(fhead['CDELT2']))
    g = 2 * (pixel_res_x * 3600) # gridding kernel size as per specs
    g2 = g ** 2
    psf_bmaj_ratio = float(fhead['BMAJ']) / pixel_res_x if consider_psf else 1.0
    psf_bmin_ratio = float(fhead['BMIN']) / pixel_res_y if consider_psf else 1.0    
    ellipses = []
    cores = []
    e_fmt = 'fk5; ellipse %sd %sd %f" %f" %fd'
    c_fmt = 'fk5; point %sd %sd #point=cross 12'
    flux = []
    faint = 0
    tt_selected = 0
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
        for line in lines[1:]: #skip the header, test the first few only
            fds = line.split(',')
            selected = int(fds[-3])
            if (0 == selected):
                continue
            tt_selected += 1
            size, clas = int(fds[-5]), int(fds[-4])
            pos_list = [float(x) for x in fds[1:5]]
            ra_core, dec_core, ra_centroid, dec_centroid = pos_list
            if (ra_core < 0):
                ra_core += 360.0
            bmaj = float(fds[7])
            bmin = float(fds[8])
            opa = float(fds[9])
            # ds9 angle starts from due east, sdc1 angles starts from due west (as per specs)
            pa = 180 - opa
            if (fancy):
                #TODO use w1, w2 to replace bmaj, bmin
                if (2 == size): 
                    if (clas in [1, 3]): # extended source
                        # don't change anything
                        w1 = bmaj
                        w2 = bmin
                    else:
                        # assuming core is fully covered by source
                        w1 = bmaj / 2.0 #+ g / 10.0 
                        w2 = w1 #keep it circular Gaussian
                else: # compact source [1 or 3]
                    # 1,2,3 for SS-AGNs, FS-AGNs, SFGs
                    # 1,2,3 for LAS, Gaussian, Exponential
                    if (1 == clas and 1 == size):
                        w1 = bmaj / 2.0
                        w2 = bmin / 2.0
                    elif (3 == clas and 3 == size):
                        w1 = bmaj * cons
                        w2 = bmin * cons
                    else:
                        raise Exception('unknown combination')
                
                #TODO calculate b1 and b2 from w1 and w2 for gridded sky model
                b1 = np.sqrt(w1 ** 2 + g2) * psf_bmaj_ratio
                b2 = np.sqrt(w2 ** 2 + g2) * psf_bmin_ratio
            else:
                b1 = bmaj
                b2 = bmin
            area_pixel = ((b1 / 3600 / pixel_res_x) * (b2 / 3600 / pixel_res_x)) / 1.1
            #area_pixel = (b1 * b2)
            #print(b1, b2, area_pixel)
            total_flux = float(fds[5])
            total_flux = _primary_beam_gridding(total_flux, ra_centroid, dec_centroid, pb_wcs, pb_data)
            total_flux /= area_pixel
            if (total_flux < b2_three_sigma):
                faint += 1
                continue
            ellipses.append(e_fmt % (ra_centroid, dec_centroid, b1 / 2, b2 / 2, pa))
            cores.append(c_fmt % (ra_core, dec_core))
            flux.append(float(fds[5]))
    
    region_fn = cat_csv.replace('.csv', '_1sigma_global.reg')
    with open(region_fn, 'w') as fout:
        for el, co in zip(ellipses, cores):
            fout.write(el)
            fout.write(os.linesep)
            fout.write(co)
            fout.write(os.linesep)
    print(np.mean(flux), np.std(flux))
    print('total selected sources %d, faint source %d' % (tt_selected, faint))
    

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '..', 'data')
    train_file = 'TrainingSet_B2_v2.txt'
    train_csv = 'TrainingSet_B2_v2.csv'
    whole_field = 'SKAMid_B2_1000h_v3.fits'
    train_field = 'SKAMid_B2_1000h_v3_train_image.fits'
    #convert_to_csv(os.path.join(data_dir, train_file))
    #cutout_training_image(osp.join(data_dir, train_csv), osp.join(data_dir, whole_field))
    pb = 'PrimaryBeam_B2.fits'
    gen_ds9_region(osp.join(data_dir, train_csv), osp.join(data_dir, train_field), osp.join(data_dir, pb))