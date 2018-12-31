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
            size = float(fds[-5]) / 2.0
            x, y = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
            xs.append(x)
            ys.append(y)
            szs.append(size)
    mesz = max(szs) / (float(fhead['CDELT1']) * 3600) # deg to arcsec
    # need to add the potential source edge LAS or exponential scale length
    # https://astronomy.swin.edu.au/cosmos/S/Scale+Length
    x1, y1, x2, y2 = int(np.floor(min(xs) - mesz)), int(np.floor(min(ys) - mesz)),\
                     int(np.ceil(max(xs) + mesz)), int(np.ceil(max(ys) + mesz))
    print(x1, y1, x2, y2, mesz)
    out_fname = field_fits.replace('.fits', '_train_image.fits')
    print(splitimg_cmd % (field_fits, out_fname, x1, y1, (x2 - x1), (y2 - y1)))

def gen_ds9_region(cat_csv, fits_img):
    """
    For B1
    SIZE_CLASS, count
    1_1,    4034
    2_1,    380
    2_2,    3728
    2_3,    179946
    3_3,    372
    """
    cons = 2 ** 0.5
    hdulist = pyfits.open(fits_img)
    fhead = hdulist[0].header
    g = 2 * (float(fhead['CDELT1']) * 3600)
    g2 = g ** 2
    #TODO add psf factor properly (i.e. read bmaj/bmin from fits header)
    ellipses = []
    cores = []
    e_fmt = 'fk5; ellipse %sd %sd %f" %f" %fd'
    c_fmt = 'fk5; point %sd %sd #point=cross 12'
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
        for line in lines[1:4000]: #skip the header, test the first 200 only
            fds = line.split(',')
            size, clas = int(fds[-2]), int(fds[-1])
            
            ra_core, dec_core, ra_centroid, dec_centroid = fds[1:5]
            if (ra_core < 0):
                ra_core += 360.0
            bmaj = float(fds[7])
            bmin = float(fds[8])
            opa = float(fds[9])
            pa = 180 - opa
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
            b1 = np.sqrt(w1 ** 2 + g2) * (1.5 / (g / 2))
            b2 = np.sqrt(w2 ** 2 + g2) * (1.5 / (g / 2))
            ellipses.append(e_fmt % (ra_centroid, dec_centroid, b1 / 2, b2 / 2, pa))
            cores.append(c_fmt % (ra_core, dec_core))
    
    region_fn = cat_csv.replace('.csv', '.reg')
    with open(region_fn, 'w') as fout:
        for el, co in zip(ellipses, cores):
            fout.write(el)
            fout.write(os.linesep)
            fout.write(co)
            fout.write(os.linesep)
    

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '..', 'data')
    train_file = 'TrainingSet_B5.txt'
    train_csv = 'TrainingSet_B1.csv'
    whole_field = 'SKAMid_B1_1000h_v2.fits'
    train_field = 'SKAMid_B1_1000h_v2_train_image.fits'
    #convert_to_csv(os.path.join(data_dir, train_file))
    #cutout_training_image(osp.join(data_dir, train_csv), osp.join(data_dir, whole_field))
    gen_ds9_region(osp.join(data_dir, train_csv), osp.join(data_dir, train_field))