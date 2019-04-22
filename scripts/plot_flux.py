import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def comparison_plot(x_fn, y_fn, idx, title=None):
    with open(x_fn, 'r') as fin:
        xlines = fin.read().splitlines()[1:]

    with open(y_fn, 'r') as fin:
        ylines = fin.read().splitlines()[1:]
    
    assert len(xlines) == len(ylines), 'line number does not match'

    xp, yp = [], []
    for x, y in zip(xlines, ylines):
        xv = float(x.split()[idx])
        xp.append(xv)
        yv = float(y.split()[idx])
        yp.append(yv)
    
    plt.scatter(np.log10(xp), np.log10(yp), marker='+')
    plt.xlabel('Truth')
    plt.ylabel('Estimated')
    if (title is not None):
        plt.suptitle(title)
    #plt.plot([0, ])
    plt.show()

def dist_plot(x_fn):
    with open(x_fn, 'r') as fin:
        xlines = fin.read().splitlines()[1:]
    #xp = []
    #print(xlines[2])
    fluxes_raw = [float(x.split()[-7]) for x in xlines]
    ff_raw = np.array(fluxes_raw, dtype=np.float32)
    print('ff_raw <= 0', np.sum(ff_raw <= 0))

    fluxes = [np.log10(float(x.split()[-7])) for x in xlines]
    #print(fluxes[0:30])
    print(max(fluxes), min(fluxes))
    ff = np.array(fluxes, dtype=np.float32)
    nan_ind = np.isnan(ff)
    print(ff_raw[nan_ind])
    print(np.sum(nan_ind))
    ff = ff[np.where(~nan_ind)]
    print(np.sum(ff > 1))
    #ff = ff[np.where(ff < 1)]
    plt.hist(ff, bins=40)
    plt.show()


if __name__ == "__main__":
    x_fn = '../data/icrar_560MHz_1000h_v8.txt'
    y_fn = '../data/icrar_560MHz_1000h_v4.txt'
    #comparison_plot(x_fn, y_fn, -5, title='Size comparison, truth is V5')
    #comparison_plot(x_fn, y_fn, -7, title='Flux comparison, truth is V5')
    dist_plot(x_fn)