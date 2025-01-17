import sis
import numpy as np
import math # for exp()
import matplotlib.pyplot as plt
import scipy.integrate as sp

## measurement noise
sigma_y = 0.01
## initial value of state
X0 = np.array([1.0,0.0])
## uncertainty in the initial state
SIGMA_X0 = np.array([0.1,0.1])

def eqom(x,t):
    dx = np.zeros(2)
    dx[0] = x[1]
    dx[1] = -2.0*x[0]
    return dx

def measurement(x):
    return np.array([x[0] + np.random.normal()*sigma_y])

## Function for the filter to propagate a particle. Assumes dynamics: /ddot{x} = -1.5*x
#
#   Uses Euler first order itnegration
#   @param xk Particle
#   @param dt time through which to integrate
#   @param vk process noise
def propagateFunction(xk,dt,vk):
    # Euler first order approximation
    dxk = np.zeros(2)
    dxk[0] = xk[1]
    dxk[1] = -1.5*xk[0] + vk[0]
    xk = xk + dt*dxk
    return xk

## Function for the filter to compute the PDF of a measurement given a prior
#
#   @param yt: the measurement
#   @param xk: the prior (a particle)
#   @output py_x: the PDF of yt, given xk
def measurementPdf(yt,xk):
    # we're measureing position, so the expectation yk is simply xk[0]:
    yk = xk[0]
    # compute the error w.r.t. the measurement
    nk = yt[0] - yk
    # evaluate the Gaussian PDF with mean 0.0 and standard deviation sigma_y
    py_x = math.exp(-0.5*(nk/sigma_y)*(nk/sigma_y))/(math.sqrt(2.0*math.pi)*sigma_y)
    return py_x

## Function for the filter that returns an appropriate process noise sample
#
# Our process noise is based 100% on measurement error. We cannot know the error exactly IRL, but we might know that it's linear in the state
#   @param xk the current state
def processNoise(xk):
    #draw from the normal distribution
    vk = np.array([ np.random.uniform(low=-0.8*xk[0],high=0.8*xk[0]) ])
    return vk

def initialParticle():
    err = np.zeros(2)
    err[0] = np.random.normal()*SIGMA_X0[0]
    err[1] = np.random.normal()*SIGMA_X0[1]
    return (X0 + err)

def main():
    # number of particles
    Nsu = 250

    SIS = sis.sis(2,Nsu,propagateFunction,processNoise,measurementPdf)

    print("SIS.is_init: %d" % (SIS.initFlag))
    SIS.init(initialParticle)
    print("SIS.is_init: %d" % (SIS.initFlag))

    tf = 5.0
    dt = 0.01
    nSteps = int(tf/dt)
    xsim = X0.copy()
    tsim = 0.0

    px1 = np.zeros((nSteps,SIS.Ns))
    px2 = np.zeros((nSteps,SIS.Ns))
    weights = np.zeros((nSteps,SIS.Ns))
    xksim = np.zeros((nSteps,2))

    for k in range(nSteps):
        # ode integration
        simout = sp.odeint(eqom,xsim,np.array([tsim,tsim+dt]))
        # store new sim state
        xsim = simout[-1,:].transpose()
        # update time
        tsim = tsim + dt
        # generate a measurement
        yt = measurement(xsim)
        # call SIS
        SIS.update(dt,yt)
        print("%f,%f,%f,%f" % (tsim,yt[0],xsim[0],xsim[1]))
        #print(tsim,SIS.WI)
        # store
        px1[k,:] = SIS.XK[0,:].copy()
        px2[k,:] = SIS.XK[1,:].copy()
        weights[k,:] = SIS.WI.copy()
        xksim[k,:] = (xsim.transpose()).copy()
    tplot = np.arange(0.0,tf,dt)


    # len(tplot) x Ns matrix of times
    tMesh = np.kron(np.ones((SIS.Ns,1)),tplot).transpose()
    x1Mesh = px1.copy()
    x2Mesh = px2.copy()
    # sort out the most likely particle at each time
    xml = np.zeros((nSteps,2))
    for k in range(nSteps):
        idxk = np.argmax(weights[k,:])
        xml[k,0] = px1[k,idxk]
        xml[k,1] = px2[k,idxk]

    fig = plt.figure()

    ax = []
    for k in range(4):
        if k < 2:
            nam = 'x' + str(k+1)
        else:
            nam = 'pdf' + str(k-1)
        ax.append( fig.add_subplot(2,2,k+1,ylabel=nam) )
        if k < 2:
            ax[k].plot(tplot,xksim[:,k],'b-')
            if k == 0:
                #ax[k].plot(tplot,px1,'.')
                ax[k].plot(tplot,xml[:,k],'r.')
            elif k == 1:
                #ax[k].plot(tplot,px2,'.')
                ax[k].plot(tplot,xml[:,k],'r.')
        elif k < 4:
            if k == 2:
                # plot the discrete PDF as a function of time
                mex = tMesh.reshape((len(tplot)*SIS.Ns,))
                mey = x1Mesh.reshape((len(tplot)*SIS.Ns,))
                mez = weights.reshape((len(tplot)*SIS.Ns,))
            elif k == 3:
                # plot the discrete PDF as a function of time
                mex = tMesh.reshape((len(tplot)*SIS.Ns,))
                mey = x2Mesh.reshape((len(tplot)*SIS.Ns,))
                mez = weights.reshape((len(tplot)*SIS.Ns,))
            idx = mez.argsort()
            mexx,meyy,mezz = mex[idx],mey[idx],mez[idx]

            cc = ax[k].scatter(mexx,meyy,c=mezz,s=20,edgecolor='')
            fig.colorbar(cc,ax=ax[k])
            # plot the truth
            ax[k].plot(tplot,xksim[:,k-2],'b-')

        ax[k].grid()
    fig.show()

    raw_input('Return to continue')

    print("Exiting sis_test.py")
    return

if __name__ == '__main__':
    main()
