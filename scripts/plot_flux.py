import matplotlib.pyplot as plt
import numpy as np


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
    print(xlines[2])
    fluxes = [np.log10(float(x.split()[-7])) for x in xlines]
    print(fluxes[0:30])
    plt.hist(fluxes, bins=10)
    plt.show()


if __name__ == "__main__":
    x_fn = '../data/icrar_560MHz_1000h_v5.txt'
    y_fn = '../data/icrar_560MHz_1000h_v4.txt'
    #comparison_plot(x_fn, y_fn, -5, title='Size comparison, truth is V5')
    #comparison_plot(x_fn, y_fn, -7, title='Flux comparison, truth is V5')
    dist_plot(x_fn)