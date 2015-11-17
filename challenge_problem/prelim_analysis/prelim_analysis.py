"""@package prelim_analysis
Run Monte Carlo for points distributed normally about the origin in x1-x2 space for the challenge problem. Write results to a file for MATLAB analysis.
Dependencies: numpy, matplotlib, python2.x, package cp_dynamics
"""

import numpy as np
import math
import sys
import matplotlib.pyplot as plt
import time

sys.path.append('../')# path to the dynamics module

import cp_dynamics as cpd

def convergence_test(Npi = 8):
    # initial state uncertainty
    sigma_x1 = 5.0
    sigma_x2 = 1.0
    # generate some random initial points; simulate for some seconds
    Np = Npi

    x0 = np.array([ 0.0,0.0 ])

    X0 = np.kron(np.ones((Np,1)),x0)
    # add noise
    print(X0)
    X0[:,0] = X0[:,0] + np.random.normal(scale=sigma_x1,size=(Np,))
    X0[:,1] = X0[:,1] + np.random.normal(scale=sigma_x2,size=(Np,))
    #X0 = np.array([ [-5.22661728,-1.26191424],[1.56423089,0.99178616] ])
    print(X0)

    # final sim time
    tf = 10.0

    # number of integration steps
    nsteps = 1000

    # vector of times for plotting
    tsp = np.linspace(0.0,tf,nsteps)

    # time the problem
    t1 = time.time()
    yout = np.zeros((nsteps,2*Np))
    for k in range(X0.shape[0]):
        yp = cpd.ode_wrapper(cpd.eqom_stoch,X0[k,:],tsp)
        yout[:,(k*2):(k*2+2)] = yp.copy()
    t2 = time.time()
    print("Elapsed time is %f sec for %d sims" % (t2-t1,Np))

    fig = plt.figure()

    ax = [fig.add_subplot(131,ylabel='x_1'),fig.add_subplot(132,ylabel='x_2'),fig.add_subplot(133,ylabel='phase')]
    ax[0].plot(tsp,yout[:,np.arange(0,2*Np,2)])
    ax[1].plot(tsp,yout[:,np.arange(1,2*Np+1,2)])
    ax[2].plot(yout[:,np.arange(0,2*Np,2)],yout[:,np.arange(1,2*Np+1,2)])

    fig.show()

    return (tsp,yout)

def main():
    Np = 512
    (t1,y1) = convergence_test(Np)

    outarray = np.column_stack((t1,y1))

    # export data
    np.savetxt('prelim.txt',outarray,'%f',delimiter=',')
    print("Wrote to file prelim.txt")

    print("Completed")

    raw_input("Return to exit")

if __name__ == "__main__":
    main()
