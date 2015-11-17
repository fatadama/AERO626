"""@package modality_test
Module the performs the test for multi-modality in data output from prelim_analysis.py
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import math

def etaCalc(k,kmax,dt):
    dtest = dt*float(kmax-1)/float(k)
    eta = dtest-dt
    if eta < 120:
        print("Estimated time remaining: %f s" % (eta) )
    elif eta < 1800:
        print("Estimated time remaining: %f min" % (eta*0.01666666666) )
    else:
        print("Estimated time remaining: %f hrs" % (eta*0.00027777777) )

def pdf2d(x,mu,P):
    sqrtterm = 1.0/math.sqrt(2.0*math.pi*np.linalg.det(P))
    return ( sqrtterm*math.exp(-0.5*np.dot(x-mu,np.dot(np.linalg.inv(P),x-mu))) )

def modality_test(nt=2,mt=100):
    # load from file
    inarray = np.genfromtxt("prelim.txt",'float',delimiter=',')
    t = inarray[:,0]
    Np = (inarray.shape[1]-1)/2
    x1 = inarray[:,np.arange(1,2*Np+1,2)]
    x2 = inarray[:,np.arange(2,2*Np+2,2)]

    # number of points to test
    ntry = nt
    # index of points to test
    indtry = np.linspace(0,len(t)-1,ntry).astype(int)

    hctrit_v = np.zeros(ntry)
    P_unimodal = np.zeros(ntry)

    print("Starting iterations over time data")
    t1 = time.time()
    for nouter in range(ntry):
        # data to use
        X1u = x1[indtry[nouter],:]
        X2u = x2[indtry[nouter],:]
        # binary search to find the critical value of h
        hug = [0.1,10.0]
        # nominal value
        hu = (hug[0]+hug[1])*0.5
        # maximum number of iterations
        itermax = 50
        # iteration tolerance
        itertol = 0.01
        # length of data
        n = Np

        # grid over which to evaluate
        x1grid = np.linspace(-15,15,20)
        x2grid = np.linspace(-10,10,20)
        for i in range(itermax):
            hu = (hug[0]+hug[1])*0.5

            # compute the kernel density estimate (KDE)
            kdegrid = np.zeros((x1grid.shape[0],x2grid.shape[0]))
            for k in range(x1grid.shape[0]):
                for j in range(x2grid.shape[0]):
                    for h in range(n):
                        xterm = np.array([ x1grid[k]-X1u[h],x2grid[j]-X2u[h] ])/hu
                        fc = pdf2d( xterm,np.zeros(2),np.diag([1.0,1.0]) )
                        kdegrid[k,j] = kdegrid[k,j]+1.0/(n*hu)*fc
            # count the number of local maxima in the gridded KDE
            next = 0
            for k in range(1,x1grid.shape[0]-1):
                for j in range(1,x2grid.shape[0]-1):
                    if kdegrid[k,j] > kdegrid[k,j-1] and kdegrid[k,j] > kdegrid[k,j+1] and kdegrid[k,j] > kdegrid[k-1,j] and kdegrid[k,j] > kdegrid[k+1,j]:
                        next = next + 1
            if next == 1:
                hug[1] = hu
            else:
                hug[0] = hu

            print(hug)
            if hug[1]-hug[0] < itertol:
                break
        # critical smoothing parameter
        hcrit = hu
        # evaluate the significance level
        miter = mt#100
        next = np.zeros(miter)
        for mi in range(miter):
            nsamp = np.random.uniform(low=0,high=n-1,size=(n,)).astype(int)
            # bootstrap sample
            XKK = np.vstack((X1u[nsamp],X2u[nsamp])).transpose()
            # smooth bootstrap sample
            ei = np.random.normal(size=(n,2))
            sigmat = np.array([X1u.std(),X2u.std()])
            f1 = 1.0/math.sqrt( 1.0+hcrit*hcrit/(sigmat[0]*sigmat[0]) )
            f2 = 1.0/math.sqrt( 1.0+hcrit*hcrit/(sigmat[1]*sigmat[1]) )
            XKK = XKK + hu*ei
            XKK[:,0] = f1*XKK[:,0]
            XKK[:,1] = f2*XKK[:,1]
            # evaluate the number of modes, using the critical smoothing factor
            kdegrid = np.zeros((x1grid.shape[0],x2grid.shape[0]))
            for k in range(x1grid.shape[0]):
                for j in range(x2grid.shape[0]):
                    for h in range(n):
                        xterm = np.array([ x1grid[k]-XKK[h,0],x2grid[j]-XKK[h,1] ])/hu
                        #fc = pdf2d( xterm,np.zeros(2),np.diag([1.0,1.0]) )
                        kdegrid[k,j] = kdegrid[k,j]+1.0/(n*hu)*pdf2d( xterm,np.zeros(2),np.diag([1.0,1.0]) )
            # count the number of local maxima in the gridded KDE
            next[mi] = 0
            for k in range(1,x1grid.shape[0]-1):
                for j in range(1,x2grid.shape[0]-1):
                    if kdegrid[k,j] > kdegrid[k,j-1] and kdegrid[k,j] > kdegrid[k,j+1] and kdegrid[k,j] > kdegrid[k-1,j] and kdegrid[k,j] > kdegrid[k+1,j]:
                        next[mi] = next[mi] + 1
        n1 = np.nonzero( next <= 1 )
        P_unimodal[nouter] = float(len(n1))/float(miter)
        print("Approximate significance level: %f at t = %f" % (P_unimodal[nouter],t[indtry[nouter]]) )

        #fig = plt.figure()
        #ax = fig.add_subplot(111)
        #ax.contour(x1grid,x2grid,kdegrid)
        #fig.show()

        # do stuff
        t2 = time.time()
        etaCalc(k,ntry,t2-t1)



def main():
    modality_test(2,2)
    print("Completed run")
    raw_input("return to exit")
    return

if __name__ == "__main__":
    main()
