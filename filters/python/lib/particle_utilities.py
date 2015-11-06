"""
particle_utilities.py
Contains utility functions (like resampling) for particle filters
"""

import numpy as np

def resample(XK,WI):
    # number of states
    nx = XK.shape[0]
    # number of particles
    Ns = XK.shape[1]
    #inverse of the number of particles
    Nsinv = 1.0/float(Ns)
    # initialize output arrays
    XR = np.zeros(XK.shape)
    WR = np.zeros(WI.shape)
    ipl = []

    # initialize the CDF:
    cd = np.zeros(Ns)
    for k in range(1,Ns):
        cd[k] = cd[k-1] + WI[k]
    # initialize parent index
    idx = 0
    # draw a starting point
    u1 = np.random.uniform(high=Nsinv)
    for j in range(Ns):
        # move along the cdf
        uj = u1 + Nsinv*j
        while (uj > cd[idx]) and (idx < (Ns-1)):
            idx = idx+1
        # assign sample
        XR[:,j] = XK[:,idx]
        WR[j] = Nsinv
        ipl.append(idx)
    return(XR,WR,ipl)
