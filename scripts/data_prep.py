import os, math
import os.path as osp

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import cv2
import json

from collections import defaultdict

cons = 2 ** 0.5
pixel_res_x = 1.67847000000E-04 #abs(float(fhead['CDELT1']))
pixel_res_y = 1.67847000000E-04 #abs(float(fhead['CDELT2']))
g = 2 * (pixel_res_x * 3600) # gridding kernel size as per specs
g2 = g ** 2

flux_mean, flux_std = 3.995465965009945e-06, 0.00012644451470476082

"""
For B1
SIZE_CLASS, count
1_1,    4034
2_1,    380
2_2,    3728
2_3,    179946
3_3,    372
"""

CAT_XML_COCO_DICT = {'1_1': 1, '2_1': 2, '2_2': 3, '2_3': 4, '3_3': 5}

def _get_fits_mbr(fin, row_ignore_factor=10):
    hdulist = pyfits.open(fin)
    data = hdulist[0].data
    wcs = pywcs.WCS(hdulist[0].header)
    width = data.shape[1]
    height = data.shape[0]

    bottom_left = [0, 0, 0, 0]
    top_left = [0, height - 1, 0, 0]
    top_right = [width - 1, height - 1, 0, 0]
    bottom_right = [width - 1, 0, 0, 0]

    def pix2sky(pix_coord):
        return wcs.wcs_pix2world([pix_coord], 0)[0][0:2]

    ret = np.zeros([4, 2])
    ret[0, :] = pix2sky(bottom_left)
    ret[1, :] = pix2sky(top_left)
    ret[2, :] = pix2sky(top_right)
    ret[3, :] = pix2sky(bottom_right)
    RA_min, DEC_min, RA_max, DEC_max = np.min(ret[:, 0]),   np.min(ret[:, 1]),  np.max(ret[:, 0]),  np.max(ret[:, 1])
    
    # http://pgsphere.projects.pgfoundry.org/types.html
    sqlStr = "SELECT sbox '((%10fd, %10fd), (%10fd, %10fd))'" % (RA_min, DEC_min, RA_max, DEC_max)
    return sqlStr

def _setup_db_pool():
    from psycopg2.pool import ThreadedConnectionPool
    return ThreadedConnectionPool(1, 3, database='chen', user='chen')

def build_fits_cutout_index(fits_cutout_dir,
                            tablename):
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    for fn in os.listdir(fits_cutout_dir):
        # if (not fn.startswith(prefix)):
        #     continue
        # if (not fn.endswith('.fits')):
        #     continue
        # if (fn.find('-') < 0):
        #     continue
        fits_fn = osp.join(fits_cutout_dir, fn)
        sqlStr = _get_fits_mbr(fits_fn)
        cur = conn.cursor()
        cur.execute(sqlStr)
        res = cur.fetchall()
        if (not res or len(res) == 0):
            errMsg = "fail to calculate sbox {0}".format(sqlStr)
            print(errMsg)
            raise Exception(errMsg)
        coverage = res[0][0]
        sqlStr = """INSERT INTO {0}(coverage,fileid) VALUES('{1}','{2}')"""
        sqlStr = sqlStr.format(tablename, coverage, fn)
        print(sqlStr)
        cur.execute(sqlStr)
        conn.commit()
    g_db_pool.putconn(conn)

def derive_pos_from_cat(line, fancy):
    fds = line.split(',')
    selected = int(fds[-3])
    if (0 == selected):
        return []
    size, clas = int(fds[-5]), int(fds[-4])
    pos_list = [float(x) for x in fds[1:5]]
    ra_core, dec_core, ra_centroid, dec_centroid = pos_list
    bmaj = float(fds[7])
    bmin = float(fds[8])
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
        b1 = np.sqrt(w1 ** 2 + g2)
        b2 = np.sqrt(w2 ** 2 + g2)
    else:
        b1 = bmaj
        b2 = bmin
    combo = '%s_%s' % (fds[10], fds[11])
    return [float(fds[3]), float(fds[4]), b1, b2, float(fds[9]), combo]

def _gen_single_bbox(fits_fn, ra, dec, major, minor, pa, major_scale=1.0):
    """
    Form the bbox BEFORE converting wcs to the pixel coordinates
    major and mior are in arcsec
    """
    ra = float(ra)
    dec = float(dec)

    hdulist = pyfits.open(fits_fn)
    height, width = hdulist[0].data.shape[0:2]
    w = pywcs.WCS(hdulist[0].header)
    cx, cy = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    if (cx < 0 or cx > width):
        return []
    if (cy < 0 or cy > height):
        return []
    #cx, cy = w.wcs_world2pix(ra, dec, 0)
    cy = height - cy
    ang = major * major_scale / 3600.0 / 2 #actually semi-major
    res_x = pixel_res_x #abs(hdulist[0].header['CDELT1'])
    angp = ang / res_x
    paa = np.radians(pa)
    angpx = angp * abs(np.sin(paa))
    angpy = angp * abs(np.cos(paa))
    #print('\n---- %s' % fits_fn)
    #print(ang, angp, cx, cy, ra, dec)
    xmin = cx - angpx
    ymin = cy - angpy
    xmax = cx + angpx
    ymax = cy + angpy
    
    # crop it around the border
    xp_min = max(xmin, 0)
    yp_min = max(ymin, 0)
    xp_max = min(xmax, width - 1)
    if (xp_max <= xp_min):
        return []
    yp_max = min(ymax, height - 1)
    if (yp_max <= yp_min):
        return []
    #print(xp_min, yp_min, xp_max, yp_max, height, width)
    return (xp_min, yp_min, xp_max, yp_max, height, width)

def fits2png(fits_dir, png_dir):
    """
    Convert fits to png files based on the D1 method
    """
    cmd_tpl = '%s -cmap Heat'\
        ' -zoom to fit -scale squared -scale mode zscale -export %s -exit'
    # cmd_tpl = '%s -cmap gist_heat -cmap value 0.684039 0'\
    #     ' -zoom to fit -scale log -scale mode minmax -export %s -exit'
    from sh import Command
    ds9_path = '/Applications/SAOImageDS9.app/Contents/MacOS/ds9'
    ds9 = Command(ds9_path)
    #fits = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23/split_fits/30arcmin/gama_linmos_corrected_clipped0-0.fits'
    #png = '/Users/chen/gitrepos/ml/rgz_rcnn/data/EMU_GAMA23/split_png/30arcmin/gama_linmos_corrected_clipped0-0.png'
    for fits in os.listdir(fits_dir):
        if (fits.endswith('.fits')):
            png = fits.replace('.fits', '.png')
            cmd = cmd_tpl % (osp.join(fits_dir, fits), osp.join(png_dir, png))
            ds9(*(cmd.split()))

def prepare_json(cat_csv, split_fits_dir, split_png_dir, table_name, fancy=True):
    images = []
    annolist = []
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()

    fid_dict = dict() #mapping from fits file id to json file id
    #TODO go thrugh split png dir, and build its json id, save that to dict
    for idx, png_fn in enumerate(os.listdir(split_png_dir)):
        im = cv2.imread(osp.join(split_png_dir, png_fn))
        h, w = im.shape[0:2]
        img_d = {'id': idx, 'license': 1, 'file_name': '%s.png' % png_fn, 'height':h, 'width': w}
        fid_dict[png_fn] = idx
        images.append(img_d)
    valid_source_cc = 0
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
        for idx, line in enumerate(lines[1:]): #skip the header, test the first few only
            ret = derive_pos_from_cat(line, fancy)
            lll = line.split(',')
            source_id = int(lll[0])
            if len(ret) == 0:
                continue
            ra, dec, major, minor, pa, combo = ret

            sqlStr = "select fileid from %s where coverage ~ scircle " % table_name +\
                    "'<(%fd, %fd), %fd>'" % (ra, dec, max(pixel_res_x, pixel_res_y))
            cur = conn.cursor(sqlStr)
            cur = conn.cursor()
            cur.execute(sqlStr)
            res = cur.fetchall()
            if (not res or len(res) == 0):
                print("Fail to find fits for source %d".format(source_id))
                continue
            for fd in res:
                fid = fd[0]
                fits_img = osp.join(split_fits_dir, fid)
                rrr = _gen_single_bbox(fits_img, ra, dec, major, minor, pa)
                if (len(rrr) == 0):
                    continue
                x1, y1, x2, y2, h, w = rrr
                anno = dict()
                valid_source_cc += 1
                anno['category_id'] = CAT_XML_COCO_DICT[combo]
                bw = x2 - x1
                bh = y2 - y1
                anno['bbox'] = [x1, y1, bw, bh]
                anno['area'] = bh * bw
                anno['id'] = valid_source_cc#source_id
                anno['source_id'] = source_id
                anno['flux'] = (float(lll[5]) - flux_mean) / flux_std
                anno['image_id'] = fid_dict[fid.replace('.fits', '.png')]
                anno['iscrowd'] = 0
                annolist.append(anno)
                #print(anno)
            if (idx % 100 == 0):
                print('Done %d' % idx)
                #return None, None
    
    return images, annolist

def create_coco_anno():
    anno = dict()
    anno['info'] = {"description": "RGZ data release 1", "year": 2018}
    anno['licenses'] = [{"url": r"http://creativecommons.org/licenses/by-nc-sa/2.0/", 
                         "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
    anno['images'] = []
    anno['annotations'] = []
    anno['categories'] = create_categories()
    return anno

def create_categories():
    """
    For B1
    SIZE_CLASS, count
    1_1,    4034
    2_1,    380
    2_2,    3728
    2_3,    179946
    3_3,    372
    """
    catlist = []
    catlist.append({"supercategory": "galaxy", "id": 1, "name": "1S_1C"})
    catlist.append({"supercategory": "galaxy", "id": 2, "name": "2S_1C"})
    catlist.append({"supercategory": "galaxy", "id": 3, "name": "2S_2C"})
    catlist.append({"supercategory": "galaxy", "id": 4, "name": "3S_3C"})
    catlist.append({"supercategory": "galaxy", "id": 5, "name": "3S_3C"})
   
    return catlist

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '..', 'data')
    train_csv =  osp.join(data_dir, 'TrainingSet_B1_v2.csv')
    tablename = 'b1_1000h_train'
    fits_cutout_dir = osp.join(data_dir, 'split_B1_1000h')
    #build_fits_cutout_index(fits_cutout_dir, tablename)
    png_dir = osp.join(data_dir, 'split_B1_1000h_png')
    #fits2png(fits_cutout_dir, png_dir)
    images, annolist = prepare_json(train_csv, fits_cutout_dir, png_dir, tablename)
    anno = create_coco_anno()
    anno['images'].extend(images)
    with open(osp.join(data_dir, 'train_B1_1000h.json')):
        pass
