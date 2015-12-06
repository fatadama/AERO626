
import sys
import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

sys.path.append('../lib')
import enkf

## stateDerivative(x,t,u) - returns the derivative of the filter state at a given time, for particular control values
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u "control" given by (accelerometer measurement, gyro measurement, gravity constant)
def stateDerivative(x,t,u):
    f = np.zeros((2,))
    f[0] = x[1]
    f[1] = -2.0*(1.5/1.0)*(x[0]*x[0]-1.0)*x[1]-(1.2/1.0)*x[0]
    return f

## stateDerivativeEKF - function used by the filter for propagation
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u control term, not used here
#   @param vk process noise vector, used for filter
def stateDerivativeEKF(x,t,u,vk):
    f = np.zeros((2,))
    f[0] = x[1]
    f[1] = -2.0*(1.0/1.0)*(x[0]*x[0]-1.0)*x[1]-(1.0/1.0)*x[0] + vk[0]
    return f

def simMeasurementFunction(xk,t):
    ymeas = np.array([ xk[0]+ np.random.normal(0.0,0.01) ])
    return ymeas

def main(argin='./',adaptFlag = False):
    # output file
    FOUT = open('python_enkf_test.csv','w')
    FOUT.write('t,x1,x2,ymeas,x1hat,x2hat,P11,P22\n');

    # initialize EKF
    #Qkin = np.array([[20.0]])#continuous-time integration value
    Qkin = np.array([[20.0]])#Euler integration value
    Hkin = np.array([[1.0,0.0]])
    Rkin = np.array([ [0.0001] ])
    if not adaptFlag:
        EnKF = enkf.enkf(2,1,stateDerivativeEKF,Hk=Hkin,Qk=Qkin,Rk=Rkin,Ns=50)
    else:
        EnKF = enkf.adaptive_enkf(2,1,stateDerivativeEKF,Hk=Hkin,Qk=Qkin,Rk=Rkin)

    dt = 0.01
    tfin = 10.0#10.0
    nSteps = int(tfin/dt)
    tsim = 0.0

    xk = np.array([1.0,0.0])
    yk = simMeasurementFunction(xk,tsim)

    # initial covariance
    P0 = np.diag([Rkin[0,0],1.0])
    EnKF.init(xk,P0)

    Enkfx = np.mean(EnKF.xk,axis=1)

    xt = np.zeros((nSteps,2))
    xf = np.zeros((nSteps,2))
    XF = np.zeros((nSteps,2,EnKF._N))
    yt = np.zeros(nSteps)
    Npts = np.zeros(nSteps)
    Pxd = np.zeros((nSteps,2))
    Pxx = np.zeros((2,2))
    for k in range(nSteps):
        # log
        xt[k,:] = xk.copy()
        xf[k,:] = Enkfx.copy()
        if not adaptFlag:
            XF[k,:,:] = EnKF.xk.copy()
        Pxd[k,0] = Pxx[0,0]
        Pxd[k,1] = Pxx[1,1]
        # propagate filter
        EnKF.propagate(dt)
        #EnKF.propagateOde(dt)
        # simulate
        y = sp.odeint(stateDerivative,xk,np.array([tsim,tsim+dt]),args=([],) )
        xk = y[-1,:].copy()

        # update time
        tsim = tsim + dt
        # measurement
        ymeas = simMeasurementFunction(xk,tsim)
        # store measurement
        yt[k] = ymeas[0]
        # update EKF
        EnKF.update(ymeas)
        EnKF.resample()
        # get the mean and covariance estimate out
        Enkfx = np.mean(EnKF.xk,axis=1)

        # store number of points
        Npts[k] = EnKF.get_N()
        Pxx = np.zeros((2,2))
        for k in range(EnKF.get_N()):
            Pxx = Pxx + 1.0/(1.0+float(EnKF._N))*np.outer(EnKF.xk[:,k]-Enkfx,EnKF.xk[:,k]-Enkfx)
        # log to file
        FOUT.write('%f,%f,%f,%f,%f,%f,%f,%f\n' % (tsim,xk[0],xk[1],ymeas[0],Enkfx[0],Enkfx[1],Pxx[0,0],Pxx[1,1]) )

    FOUT.close()
    print('Completed simulation')

    # plot
    tplot = np.arange(0.0,tfin,dt)
    fig = plt.figure()

    ax = []
    for k in range(4):
        if k < 2:
            nam = 'x' + str(k+1)
        else:
            nam = 'e' + str(k-1)
        ax.append( fig.add_subplot(2,2,k+1,ylabel=nam) )
        if k < 2:
            ax[k].plot(tplot,xt[:,k],'b-')
            ax[k].plot(tplot,xf[:,k],'r--')
            if k == 0:
                ax[k].plot(tplot,yt,'y-')
        elif k < 4:
            ax[k].plot(tplot,xt[:,k-2]-xf[:,k-2],'b-')
            ax[k].plot(tplot,3.0*np.sqrt(Pxd[:,k-2]),'r--')
            ax[k].plot(tplot,-3.0*np.sqrt(Pxd[:,k-2]),'r--')
            if k == 2:
                ax[k].plot(tplot,xt[:,k-2]-yt,'y-')

        ax[k].grid()
    fig.show()

    if not adaptFlag:
        fig2 = plt.figure()
        ax = fig2.add_subplot(121,ylabel='x1')
        ax.plot(tplot,xt[:,0],'b-')
        ax.plot(tplot,XF[:,0,:],'d')
        ax.grid()

        ax = fig2.add_subplot(122,ylabel='x2')
        ax.plot(tplot,xt[:,1],'b-')
        ax.plot(tplot,XF[:,1,:],'d')
        ax.grid()        

        fig2.show()
        pass
    else:
        fig2 = plt.figure()
        ax = fig2.add_subplot(111,ylabel='N')
        ax.plot(tplot,Npts)
        ax.grid()

        fig2.show()

    raw_input("Return to exit")

    print("Completed test_enky.py")
    return

if __name__ == '__main__':
    adapt = False
    main(adaptFlag = adapt)
