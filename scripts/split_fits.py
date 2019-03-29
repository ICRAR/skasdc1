import os
import os.path as osp

import astropy.io.fits as pyfits
import astropy.wcs as pywcs
import numpy as np

montage_path = '/home/ngas/software/Montage_v3.3/bin'
subimg_exec = '%s/mSubimage' % montage_path
regrid_exec = '%s/mProject' % montage_path
imgtbl_exec = '%s/mImgtbl' % montage_path
coadd_exec = '%s/mAdd' % montage_path
subimg_cmd = '{0} %s %s %.4f %.4f %.4f %.4f'.format(subimg_exec)
splitimg_cmd = '{0} -p %s %s %d %d %d %d'.format(subimg_exec)

def split_file(fname, width_ratio, height_ratio, halo_ratio=50,
               show_split_scheme=False, work_dir='/tmp', 
               equal_aspect=False):
    """
    width_ratio = current_width / new_width, integer
    height_ratio = current_height / new_height, integer
    halo in pixel
    """
    if (show_split_scheme):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
    d = pyfits.open(fname)[0].data
    #fhead = file[0].header
    h = d.shape[-2] #y
    w = d.shape[-1] #x
    print(h, w)
    if (equal_aspect):
        size = int(min(h / height_ratio, w / width_ratio))
        halo_h, halo_w = (size / halo_ratio,) * 2
        height_ratio, width_ratio = h / size, w / size # update the ratio just in case
        ny = np.arange(height_ratio) * size
        nx = np.arange(width_ratio) * size
        if (ny[-1] + size + halo_h < h):
            ny = np.hstack([ny, h - size])
        if (nx[-1] + size + halo_w < w):
            nx = np.hstack([nx, w - size])
        new_w, new_h = (size,) * 2
    else:
        new_h = int(h / height_ratio)
        halo_h = new_h / halo_ratio
        ny = np.arange(height_ratio) * new_h
        new_w = int(w / width_ratio)
        halo_w = new_w / halo_ratio
        nx = np.arange(width_ratio) * new_w
    print(new_h, new_w)
    print(ny)
    print(nx)

    if (show_split_scheme):
        _, ax = plt.subplots(1)
        ax.imshow(np.reshape(d, [d.shape[-2], d.shape[-1]]))

    for i, x in enumerate(nx):
        for j, y in enumerate(ny):
            x1 = max(x - halo_w, 0)
            y1 = max(y - halo_w, 0)
            wd = new_w
            hd = new_h
            
            x2_c = x1 + wd + halo_w
            x2 = min(x2_c, w - 1)
            x1 -= max(0, x2_c - (w - 1))
                
            y2_c = y1 + hd + halo_h
            y2 = min(y2_c, h - 1)
            y1 -= max(0, y2_c - (h - 1))
            fid = osp.basename(fname).replace('.fits', '%d-%d.fits' % (i, j))
            out_fname = osp.join(work_dir, fid)
            #print(out_fname)
            print(splitimg_cmd % (fname, out_fname, x1, y1, (x2 - x1), (y2 - y1)))
            if (show_split_scheme):
                rect = patches.Rectangle((x1, y1), (x2 - x1), (y2 - y1),
                                        linewidth=0.5, edgecolor='r',
                                        facecolor='none')
                # Add the patch to the Axes
                ax.add_patch(rect)
    if (show_split_scheme):
        plt.tight_layout()
        plt.show()
        plt.savefig('test.pdf')

if __name__ == '__main__':
    fname = 'SKAMid_B1_1000h_v3.fits'
    fname = 'VLACOSMOS_1400MHz.fits'
    workdir = './split_B1_1000h'
    workdir = './split_1400mhz'
    split_file(fname, 20, 20, show_split_scheme=False, 
               equal_aspect=True, work_dir=workdir, halo_ratio=20)