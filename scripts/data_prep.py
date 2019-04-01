import os, math
import os.path as osp

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np
import cv2
import json

import os.path as osp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from collections import defaultdict, namedtuple

from astropy.io import fits
from scipy.special import erfinv
mad2sigma = np.sqrt(2) * erfinv(2 * 0.75 - 1)


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
NUM_SIGMA = 1

b1_sigma = 3.5789029744176247e-07
b1_median = -2.9249549e-08
b1_three_sigma = -2.9249549e-08 + 3 * b1_sigma

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

GTSource = namedtuple('GTSource', 'box clas size')

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

    if (width * abs(float(hdulist[0].header['CDELT1'])) * 2 < (RA_max - RA_min)):
        #print(fin, width * abs(float(hdulist[0].header['CDELT1'])), (RA_max - RA_min))
        RA_max, RA_min = RA_min, RA_max
        #print(fin)
    
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
        #print(sqlStr)
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
        b1 = np.sqrt(w1 ** 2 + g2) * psf_bmaj_ratio
        b2 = np.sqrt(w2 ** 2 + g2) * psf_bmin_ratio
    else:
        b1 = bmaj
        b2 = bmin
    combo = '%s_%s' % (fds[10], fds[11])
    return [float(fds[3]), float(fds[4]), b1, b2, float(fds[9]), combo]

half_pi = np.pi / 2
def _get_bbox_from_ellipse(phi, r1, r2, cx, cy, h, w):
    """
    https://stackoverflow.com/questions/87734/
    how-do-you-calculate-the-axis-aligned-bounding-box-of-an-ellipse

    angle in degrees
    r1, r2 in number of pixels (half major/minor)
    cx and cy is pixel coordinated
    """
    ux = r1 * np.cos(phi)
    uy = r1 * np.sin(phi)
    vx = r2 * np.cos(phi + half_pi)
    vy = r2 * np.sin(phi + half_pi)

    hw = np.sqrt(ux * ux + vx * vx)
    hh = np.sqrt(uy * uy + vy * vy)
    x1, y1, x2, y2 = cx - hw, cy - hh, cx + hw, cy + hh
    return (x1, y1, x2, y2)

fits_fn_dict = dict()
def _gen_single_bbox(fits_fn, ra, dec, major, minor, pa, for_png=False):
    """
    Form the bbox BEFORE converting wcs to the pixel coordinates
    major and mior are in arcsec
    """
    ra = float(ra)
    dec = float(dec)
    #print(ra, dec)
    if (fits_fn not in fits_fn_dict):
        hdulist = pyfits.open(fits_fn)
        height, width = hdulist[0].data.shape[0:2]
        w = pywcs.WCS(hdulist[0].header).deepcopy()
        fits_fn_dict[fits_fn] = (w, height, width)
    else:
        w, height, width = fits_fn_dict[fits_fn]

    cx, cy = w.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #cx = np.ceil(cx)
    if (not for_png):
        cx += 1
    #cy = np.ceil(cy)
    cy += 1
    if (cx < 0 or cx > width):
        print('got it cx {0}, {1}'.format(cx, fits_fn))
        return []
    if (cy < 0 or cy > height):
        print('got it cy {0}'.format(cy))
        return []
    if (for_png):
        cy = height - cy
    majorp = major / 3600.0 / pixel_res_x / 2 #actually semi-major 
    minorp = minor / 3600.0 / pixel_res_x / 2
    paa = np.radians(pa)
    x1, y1, x2, y2 = _get_bbox_from_ellipse(paa, majorp, minorp, cx, cy, height, width)
    # return x1, y1, x2, y2, height, width
    origin_area = (y2 - y1) * (x2 - x1)

    # crop it around the border
    xp_min = max(x1, 0)
    yp_min = max(y1, 0)
    xp_max = min(x2, width - 1)
    if (xp_max <= xp_min):
        return []
    yp_max = min(y2, height - 1)
    if (yp_max <= yp_min):
        return []
    new_area = (yp_max - yp_min) * (xp_max - xp_min)

    if (origin_area / new_area > 2):
        print('cropped box is too small, discarding...')
        return []
    return (xp_min, yp_min, xp_max, yp_max, height, width, cx, cy)

def fits2png(fits_dir, png_dir):
    """
    Convert fits to png files based on the D1 method
    """
    cmd_tpl = '%s -cmap Heat'\
        ' -zoom to fit -scale asinh -scale mode minmax -export %s -exit'
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
            #print(cmd)
            ds9(*(cmd.split()))

def _primary_beam_correction(total_flux, ra, dec, pb_wcs, pb_data):
    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    #print(pb_data.shape)
    pbv = pb_data[int(y)][int(x)]
    #print(x, y, pbv)
    return total_flux / pbv

def _apply_primary_beam(model_flux, ra, dec, pb_wcs, pb_data):
    x, y = pb_wcs.wcs_world2pix([[ra, dec, 0, 0]], 0)[0][0:2]
    pbv = pb_data[int(y)][int(x)]
    return model_flux * pbv

def region_sources(el_dict, box_dict, region_dir):
    for k, v in el_dict.items():
        box_list = box_dict[k]
        #print(len(box_list), len(v))
        k = k.replace('.fits', '.reg')
        region_fn = osp.join(region_dir, k)
        with open(region_fn, 'w') as fout:
            for el, bbox in zip(v, box_list):
                fout.write(el)
                fout.write(os.linesep)
                fout.write(bbox)
                fout.write(os.linesep)

def draw_sources(claran_result, png_dir, target_dir):
    for k, v in claran_result.items():
        k = k.replace('.fits', '.png')
        png_fn = osp.join(png_dir, k)
        im = cv2.imread(png_fn)
        h, w, _ = im.shape
        my_dpi = 96
        fig = plt.figure(figsize=(h / my_dpi, w / my_dpi), dpi=my_dpi)
        #fig.set_size_inches(h / my_dpi, w / my_dpi)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        ax.set_xlim([0, w])
        ax.set_ylim([h, 0])
        im = im[:, :, (2, 1, 0)]
        ax.imshow(im, aspect='equal')

        for cs in v:
            bbox = cs.box
            ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='white', linewidth=0.2)
            )
            # ax.text(bbox[0], bbox[1] - 2,
            #     '{0}C_{1}S'.format(cs.clas, cs.size),
            #     bbox=dict(facecolor='None', alpha=1.0, edgecolor='None'),
            #     fontsize=10, color='white')
        
        plt.axis('off')
        plt.draw()
        out_fn = osp.join(target_dir, k.replace('.png', '_pred.png'))
        plt.savefig(out_fn, dpi=my_dpi * 2)
        plt.close()

def obtain_sigma(fits_fn):
    with fits.open(fits_fn) as f:
        imgdata = f[0].data
    med = np.nanmedian(imgdata)
    mad = np.nanmedian(np.abs(imgdata - med))
    sigma = mad / mad2sigma
    return sigma, med, imgdata


def draw_gt(cat_csv, split_fits_dir, split_png_dir, table_name, pb, target_dir, fancy=True):
    ret_dict = defaultdict(list)
    elip_dict = defaultdict(list)
    box_dict = defaultdict(list)
    #core_dict = defaultdict(list)

    e_fmt = 'fk5; ellipse %sd %sd %f" %f" %fd'
    c_fmt = 'fk5; point %sd %sd #point=cross 12'
    b_fmt = 'fk5; box %fd %fd %dp %dp 0'
    bp_fmt = 'physical; box %fp %fp %dp %dp 0'

    pb_wcs, pb_data = _setup_pb(pb)
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()

    sigma_dict = dict()

    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()

    for idx, line in enumerate(lines[1:]): #skip the header, test the first few only
        # if (idx == 15000):
        #     break
        if (idx % 1000 == 0):
            print("scanned %d" % idx)
        ret = derive_pos_from_cat(line, fancy)
        #print(ret)
        if len(ret) == 0:
            continue
        ra, dec, major, minor, pa, combo = ret
        lll = line.split(',')
        size, clas = int(lll[-5]), int(lll[-4])
        area_pixel = ((major / 3600 / pixel_res_x) * (minor / 3600 / pixel_res_x)) / 1.1
        total_flux = float(lll[5]) / area_pixel
        total_flux = _primary_beam_correction(total_flux, ra, dec, pb_wcs, pb_data)

        sqlStr = "select fileid from %s where coverage ~ scircle " % table_name +\
                    "'<(%fd, %fd), %fd>'" % (ra, dec, max(pixel_res_x, pixel_res_y))
        cur = conn.cursor()
        cur.execute(sqlStr)
        res = cur.fetchall()
        if (not res or len(res) == 0):
            #print("Fail to find fits for source {0}".format(source_id))
            continue
        
        for fd in res:
            fid = fd[0]
            fits_img = osp.join(split_fits_dir, fid)
            if (fid not in sigma_dict):
                sigma, med, _ = obtain_sigma(fits_img)
                clip_level = med + NUM_SIGMA * sigma
                sigma_dict[fid] = clip_level
            else:
                clip_level = sigma_dict[fid]
            if (total_flux) < clip_level:
                continue
            
            rrr = _gen_single_bbox(fits_img, ra, dec, major, minor, pa, for_png=True)
            if (len(rrr) == 0):
               continue
            x1, y1, x2, y2, h, w, cx, cy = rrr
            gs = GTSource(box=(x1, y1, x2, y2), clas=clas, size=size)
            ret_dict[fid].append(gs)
            box_dict[fid].append(bp_fmt % (cx, cy, int(x2 - x1), int(y2 - y1)))
            #box_dict[fid].append(b_fmt % (ra, dec, int(x2 - x1 + 1), int(y2 - y1 + 1)))
            elip_dict[fid].append(e_fmt % (ra, dec, major / 2, minor / 2, 180 - pa))
        
    draw_sources(ret_dict, split_png_dir, target_dir)
    print(len(elip_dict), len(box_dict))
    #region_sources(elip_dict, box_dict, target_dir.replace('png_gt', 'regions'))

def prepare_json(cat_csv, split_fits_dir, split_png_dir, table_name, pb, fancy=True):
    pb_wcs, pb_data = _setup_pb(pb)

    images = []
    annolist = []
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()

    sigma_dict = dict()

    fid_dict = dict() #mapping from fits file id to json file id
    #TODO go thrugh split png dir, and build its json id, save that to dict
    for idx, png_fn in enumerate(os.listdir(split_png_dir)):
        if (not png_fn.endswith('.png')):
            continue
        im = cv2.imread(osp.join(split_png_dir, png_fn))
        if (im is None):
            raise Exception(osp.join(split_png_dir, png_fn))
        h, w = im.shape[0:2]
        img_d = {'id': idx, 'license': 1, 'file_name': '%s' % png_fn, 'height':h, 'width': w}
        fid_dict[png_fn] = idx
        images.append(img_d)
    valid_source_cc = 0
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
    for idx, line in enumerate(lines[1:]): #skip the header, test the first few only
        if (idx % 1000 == 0):
            print("scanned %d" % idx)
        ret = derive_pos_from_cat(line, fancy)           
        if len(ret) == 0:
            continue
        lll = line.split(',')
        source_id = int(lll[0])
        ra, dec, major, minor, pa, combo = ret
        avg_image_flux = _calc_image_flux(float(lll[5]), major, minor, ra, dec, pb_wcs, pb_data)
        res = _find_fid_from_db(conn, ra, dec, table_name)
        if (not res or len(res) == 0):
            #print("Fail to find fits for source {0}".format(source_id))
            continue
        for fd in res:
            fid = fd[0]
            fits_img = osp.join(split_fits_dir, fid)
            if (fid not in sigma_dict):
                sigma, med, _ = obtain_sigma(fits_img)
                clip_level = med + NUM_SIGMA * sigma
                sigma_dict[fid] = clip_level
            else:
                clip_level = sigma_dict[fid]
            if (avg_image_flux) < clip_level:
                continue
            #print(fid)
            png_fid = fid.replace('.fits', '.png')
            if (png_fid not in fid_dict):
                # corresponding image is not in the training/testing set.
                # print("Image {1} not in the current set for source {0}".format(source_id, png_fid))
                continue
            
            rrr = _gen_single_bbox(fits_img, ra, dec, major, minor, pa, for_png=True)
            if (len(rrr) == 0):
                continue
            x1, y1, x2, y2, h, w, cx, cy = rrr
            anno = dict()
            valid_source_cc += 1
            #anno['category_id'] = CAT_XML_COCO_DICT[combo]
            anno['category_id'] = int(combo.split('_')[1])
            bw = x2 - x1
            bh = y2 - y1
            anno['bbox'] = [int(x) for x in [x1, y1, bw, bh]]
            anno['area'] = int(bh * bw)
            anno['id'] = valid_source_cc#source_id
            anno['source_id'] = source_id
            #anno['flux'] = (float(lll[5]) - flux_mean) / flux_std
            anno['image_id'] = fid_dict[png_fid]
            anno['iscrowd'] = 0
            annolist.append(anno)
            #print(anno)
    print('%d valid sources on %d images' % (valid_source_cc, len(fid_dict)))
    
    return images, annolist

def _setup_pb(pb_fn):
    pbhdu = pyfits.open(pb_fn)
    pbhead = pbhdu[0].header
    pb_wcs = pywcs.WCS(pbhead)
    pb_data = pbhdu[0].data[0][0]
    return pb_wcs, pb_data

def _calc_image_flux(model_flux, major, minor, ra, dec, pb_wcs, pb_data):
    area_pixel = ((major / 3600 / pixel_res_x) * (minor / 3600 / pixel_res_x)) / 1.1
    return model_flux / area_pixel
    #return _apply_primary_beam(model_flux, ra, dec, pb_wcs, pb_data)

def _find_fid_from_db(conn, ra, dec, table_name):
    sqlStr = "select fileid from %s where coverage ~ scircle " % table_name +\
                "'<(%fd, %fd), %fd>'" % (ra, dec, max(pixel_res_x, pixel_res_y))
    cur = conn.cursor()
    cur.execute(sqlStr)
    res = cur.fetchall()
    return res

def verify_flux(cat_csv, split_fits_dir, split_png_dir, table_name, pb, target_fn, fancy=True):
    pb_wcs, pb_data = _setup_pb(pb)
    g_db_pool = _setup_db_pool()
    conn = g_db_pool.getconn()
    sigma_dict = dict()
    entries = []
    source_entries = []
    with open(cat_csv, 'r') as fin:
        lines = fin.read().splitlines()
    for idx, line in enumerate(lines[1:]): #skip the header, test the first few only
        if (idx % 1000 == 0):
            print("scanned %d" % idx)
        ret = derive_pos_from_cat(line, fancy)           
        if len(ret) == 0:
            continue
        lll = line.split(',')
        source_id = int(lll[0])
        model_flux = float(lll[5])
        ra, dec, major, minor, pa, combo = ret
        avg_image_flux = _calc_image_flux(model_flux, major, minor, ra, dec, pb_wcs, pb_data)
        res = _find_fid_from_db(conn, ra, dec, table_name)
        if (not res or len(res) == 0):
            print("Fail to find fits for source {0}".format(source_id))
            continue
        for fd in res:
            fid = fd[0]
            fits_img = osp.join(split_fits_dir, fid)
            if (fid not in sigma_dict):
                sigma, med, curr_d = obtain_sigma(fits_img)
                clip_level = med + NUM_SIGMA * sigma
                sigma_dict[fid] = clip_level
            else:
                clip_level = sigma_dict[fid]
            if avg_image_flux < clip_level:
                continue
            rrr = _gen_single_bbox(fits_img, ra, dec, major, minor, pa, for_png=True)
            if (len(rrr) == 0):
                print('fail to produce valid box for %s at %f, %f' % (fits_img, ra, dec))
                continue
            x1, y1, x2, y2, h, w, cx, cy = rrr
            x1, y1, x2, y2 = [int(x) for x in (x1, y1, x2, y2)]
            # since for_png was set to true, the coordinates are based on PNG 
            # just to simulate ClaRAN outout. This means we need to get the inverse for y
            # in the numpy array converted from FITS file
            sli = curr_d[(h - y2):min(h - y1 + 1, h), x1:min(x2 + 1, w)]
            #sli = curr_d[y1:min(y2 + 1, h), x1:min(x2 + 1, w)]
            sli = sli[np.where(sli > 0)]
            int_flux01 = np.sum(sli)
            int_flux02 = int_flux01 / synth_beam_size

            #_get_integrated_flux(mir_file, x1, h - y2, x2, h - y1, h, w, failed_attempts)
            source_entry = '%s,%d,%d,%d,%d,%d,%d,%d' % (fid, source_id, x1, h - y2, x2, h - y1, h, w)
            source_entries.append(source_entry)
            entry = '%d,%f,%f,%f' % (source_id, 
                                        model_flux, 
                                        int_flux01, int_flux02)
            entries.append(entry)

    
    with open(target_fn, 'w') as fout:
        fout.write(os.linesep.join(entries))
    
    with open('source_entry.csv', 'w') as fout:
        fout.write(os.linesep.join(source_entries))

def create_coco_anno():
    anno = dict()
    anno['info'] = {"description": "RGZ data release 1", "year": 2018}
    anno['licenses'] = [{"url": r"http://creativecommons.org/licenses/by-nc-sa/2.0/", 
                         "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}]
    #anno['images'] = []
    #anno['annotations'] = []
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
    catlist.append({"supercategory": "galaxy", "id": 1, "name": "1"})
    catlist.append({"supercategory": "galaxy", "id": 2, "name": "2"})
    catlist.append({"supercategory": "galaxy", "id": 3, "name": "3"})
    #catlist.append({"supercategory": "galaxy", "id": 4, "name": "2S_3C"})
    #catlist.append({"supercategory": "galaxy", "id": 5, "name": "3S_3C"})
    return catlist

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(cur_dir, '..', 'data')
    train_csv =  osp.join(data_dir, 'TrainingSet_B1_v2.csv')
    tablename = 'b1_1000h_train'
    fits_cutout_dir = osp.join(data_dir, 'split_B1_1000h')
    #build_fits_cutout_index(fits_cutout_dir, tablename)
    png_dir = osp.join(data_dir, 'split_B1_1000h_png_val')
    tgt_png_dir = osp.join(data_dir, 'split_B1_1000h_png_gt')
    #png_dir = osp.join(data_dir, 'split_B1_1000h_png_val')
    #fits2png(fits_cutout_dir, png_dir)
    pb = osp.join(data_dir, 'PrimaryBeam_B1.fits')
    #draw_gt(train_csv, fits_cutout_dir, png_dir, tablename, pb, tgt_png_dir, fancy=True)
    # images, annolist = prepare_json(train_csv, fits_cutout_dir, png_dir, tablename, pb)
    # anno = create_coco_anno()
    # anno['images'] = images
    # anno['annotations'] = annolist
    # with open(osp.join(data_dir, 'instances_val_B1_1000h.json'), 'w') as fout:
    #     json.dump(anno, fout)
    output_fn = osp.join(data_dir, 'flux_compare')
    verify_flux(train_csv, fits_cutout_dir, png_dir, tablename, pb, output_fn)
