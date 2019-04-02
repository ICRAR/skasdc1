import json
from collections import defaultdict
import os.path as osp
import os
"""
evaluate based on two json files

val_B1_1000h-outputs35000.json
val_B1_1000h.json

produce DS9 region files and flux for evaluation
"""

bp_fmt = 'physical; box %fp %fp %dp %dp 0 # color=white'
cls_fmt = 'physical; text %fp %fp {%d} # color=white'

show_class = True

def parse_val_images(fn):
    ret = dict()
    with open(fn, 'r') as fin:
        images = json.load(fin)['images']
    for img in images:
        ret[img['id']] = (img['file_name'], img['height'])
    return ret

def parse_val_result(fn, img_dict, threshold=0.8):
    box_dict = defaultdict(list)
    text_dict = defaultdict(list)
    with open(fn, 'r') as fin:
        jo = json.load(fin) 
    for det in jo:
        if (det['score'] < threshold):
            continue
        img_id = det['image_id']
        img_fn, img_h = img_dict[img_id]
        x1, y1, bw, bh = det['bbox']
        cx = (x1 + x1 + bw) / 2 + 1
        cy = (y1 + y1 + bh) / 2 + 1
        cy = img_h - cy
        box_dict[img_fn].append(bp_fmt % (cx, cy, bw + 1, bh + 1))
        if (show_class):
            cls_lbl = det['category_id']
            if (cls_lbl != 3):
                print(cls_fmt % (x1 + 1, cy + (bh + 1) / 2, cls_lbl))
            text_dict[img_fn].append(cls_fmt % (x1 + 1, cy + (bh + 1) / 2, cls_lbl))
    return box_dict, text_dict

if __name__ == '__main__':
    cur_dir = osp.dirname(osp.abspath(__file__))
    data_dir = osp.join(cur_dir, '..', 'data')
    #png_dir = osp.join(data_dir, 'split_B1_1000h_png_val')
    eval_dir = osp.join(data_dir, 'evaluate')
    img_json = osp.join(eval_dir, 'instances_val_B1_1000h.json')
    val_json = osp.join(eval_dir, 'val_B1_1000h_more-outputs100000.json')
    out_dir = val_json.replace('.json', '')
    if (not osp.exists(out_dir)):
        os.mkdir(out_dir)

    box_dict, text_dict = parse_val_result(val_json, parse_val_images(img_json))
    for k, v in box_dict.items():
        out_fn = osp.join(out_dir, k.replace('.png', '.reg'))
        with open(out_fn, 'w') as fout:
            fout.write(os.linesep.join(v))
            if (show_class):
                fout.write(os.linesep)
                fout.write(os.linesep.join(text_dict[k]))




