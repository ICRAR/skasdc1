import os
import os.path as osp

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
from collections import defaultdict, namedtuple

ClaranSource = namedtuple('ClaranSource', 'box clas score')

def parse_claran_result(result_file, threshold=0.8):
    ret = defaultdict(list)

    with open(result_file, 'r') as fin:
        lines = fin.read().splitlines()
    
    for idx, line in enumerate(lines):
        fds = line.split(',')
        score = float(fds[2])
        if score < threshold:
            continue
        png_fid = fds[0]
        # if (not png_fid.startswith('SKAMid_B1_1000h_v354')):
        #     continue
        clas = fds[1]
        x1, y1, x2, y2 = [float(x) for x in fds[3].split('-')]
        cs = ClaranSource(box=(x1, y1, x2, y2), clas=clas, score=score)
        ret[png_fid].append(cs)
    return ret

def draw_sources(claran_result, png_dir, target_dir):
    for k, v in claran_result.items():
        png_fn = osp.join(png_dir, k)
        im = cv2.imread(png_fn)
        h, w, _ = im.shape
        my_dpi = 100.0
        fig = plt.figure()
        fig.set_size_inches(h / my_dpi, w / my_dpi)
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
                          edgecolor='magenta', linewidth=1.0)
            )
            ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.2f}'.format(cs.clas, cs.score),
                bbox=dict(facecolor='None', alpha=0.4, edgecolor='None'),
                fontsize=10, color='white')
        
        plt.axis('off')
        plt.draw()
        out_fn = osp.join(target_dir, k.replace('.png', '_pred.png'))
        plt.savefig(out_fn)
        plt.close()

if __name__ == '__main__':
    data_dir = '/mnt/gleam3/ngas_data_volume/sdc1/data'
    data_dir = '/Users/chen/gitrepos/ml/skasdc1/data'
    data_dir = '/scratch/cwu'
    png_dir = 'split_B5_1000h_test_png'
    target_dir = 'split_B5_1000h_test_png_pred_v14'
    result_file = 'B5_v3.result'

    claran_result = parse_claran_result(osp.join(data_dir, result_file))
    draw_sources(claran_result, osp.join(data_dir, png_dir), osp.join(data_dir, target_dir))