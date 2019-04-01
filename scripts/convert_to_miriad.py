import os

"""
Convert fits file into miriad file for fitting
"""

cmd_t = 'fits in=%s/%s op=xyin  out=%s/%s'

def convert(fits_dir, mir_dir):
    for fn in os.listdir(fits_dir):
        if (fn.endswith('.fits')):
            cmd = cmd_t % (fits_dir, fn, mir_dir, fn.replace('.fits', '.mir'))
            print(cmd)

if __name__ == '__main__':
    fits_dir = 'split_B1_1000h_test'
    mir_dir = 'split_B1_1000h_test_mir'
    print('source ~/miriad_env.sh')
    convert(fits_dir, mir_dir)