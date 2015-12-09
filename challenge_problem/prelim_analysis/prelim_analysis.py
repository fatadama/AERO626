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

def plot_animate(tsp,yout,dest='animate/'):
    nSteps = yout.shape[0]
    Np = yout.shape[1]/2

    fig = plt.figure()

    for k in range(1,nSteps,50):
        # clear figure
        fig.clf()
        tilt = 'Phase portrait at t = %f' % tsp[k]
        ax = [fig.add_subplot(111,ylabel='X2',xlabel='X1',title=tilt)]
        ax[0].plot(yout[0:k,np.arange(0,2*Np,2)],yout[0:k,np.arange(1,2*Np+1,2)])
        ax[0].plot(yout[k,np.arange(0,2*Np,2)],yout[k,np.arange(1,2*Np+1,2)],'x')

        #fig.show()
        fig.savefig(dest+'anim_f%d.png' % (k))
        time.sleep(0.1)
        print(k,tsp[k])
    '''
    for k in range(1,nSteps,50):
        # clear figure
        fig.clf()
        ax = [fig.add_subplot(131,ylabel='x_1'),fig.add_subplot(132,ylabel='x_2'),fig.add_subplot(133,ylabel='phase')]
        ax[0].plot(tsp[0:k],yout[0:k,np.arange(0,2*Np,2)])
        ax[1].plot(tsp[0:k],yout[0:k,np.arange(1,2*Np+1,2)])
        ax[2].plot(yout[0:k,np.arange(0,2*Np,2)],yout[0:k,np.arange(1,2*Np+1,2)])
        ax[2].plot(yout[k,np.arange(0,2*Np,2)],yout[k,np.arange(1,2*Np+1,2)],'x')

        #fig.show()
        fig.savefig(dest+'anim_f%d_t%f.png' % (k,tsp[k]))
        time.sleep(0.1)
        print(k,tsp[k])
    '''

def convergence_test(Npi = 8):
    # initial state uncertainty
    sigma_x1 = 5.0
    sigma_x2 = 1.0
    # generate some random initial points; simulate for some seconds
    Np = Npi

    x0 = np.array([ 0.0,0.0 ])

    X0 = np.kron(np.ones((Np,1)),x0)
    # add noise
    X0[:,0] = X0[:,0] + np.random.normal(scale=sigma_x1,size=(Np,))
    X0[:,1] = X0[:,1] + np.random.normal(scale=sigma_x2,size=(Np,))

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
        #yp = cpd.ode_wrapper(cpd.eqom_stoch,X0[k,:],tsp) # white noise forcing
        yp = cpd.ode_wrapper(cpd.eqom_det_f,X0[k,:],tsp) # cosine forcing
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
    Np = 64
    '''
    (t1,y1) = convergence_test(Np)

    outarray = np.column_stack((t1,y1))

    # export data
    np.savetxt('prelim.txt',outarray,'%f',delimiter=',')
    print("Wrote to file prelim.txt")
    '''
    outarray = np.genfromtxt('prelim.txt','float',delimiter =',')
    t1 = outarray[:,0]
    y1 = outarray[:,1:]
    plot_animate(t1,y1)

    print("Completed")

    raw_input("Return to exit")

if __name__ == "__main__":
    main()
