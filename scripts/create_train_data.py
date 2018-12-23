import os

import astropy.wcs as pywcs

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
    pass

if __name__ == '__main__':
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    train_file = 'TrainingSet_B5.txt'
    convert_to_csv(os.path.join(cur_dir, '..', 'data', train_file))