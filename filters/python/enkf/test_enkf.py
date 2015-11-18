
import sys
import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt

import enkf

## vicon2xhatInitial(yk) - returns initial state and covariance estimate based on a single initial vicon measurement
#   @param yk [x,y,z,b0,b1,b2,b3] vicon measurement of position and attitude (quaternion)
def vicon2xhatInitial(yk):
    # xhat: [(x,y,z)_inertial (v1,v2,v3)_body, quat_body]
    xhat = np.zeros(10)
    # initialize position to vicon
    xhat[0:3] = yk[0:3].copy()
    # initialize attitude to vicon
    xhat[6:10] = yk[3:7].copy()
    # velocity is already zero
    var_x = 0.001*0.001#meters^2
    var_vel = 0.1*0.1
    var_quat = 0.01*0.01
    # covariance estimate
    Pk = np.diag([var_x, var_x, var_x, var_vel, var_vel, var_vel, var_quat, var_quat, var_quat, var_quat])
    Pk = Pk + 1e-12*np.ones((10,10))
    return (xhat,Pk)

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

## stateGradient(x,t,u) - returns the gradient of the derivative of the filter state w.r.t. the filter state
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u "control" given by (accelerometer measurement, gyro measurement, gravity constant)
def stateGradient(x,t,u):
    Fk = np.zeros((2,2))
    Fk = np.array([ [0.0,1.0],[-4.0*x[0]*x[1]-1.0,-2.0*(x[0]*x[0]-1.0)] ])
    return Fk

## stateProcessInfluence(x,t,u) - returns the process noise matrix on the derivative of the filter state
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u "control" given by (accelerometer measurement, gyro measurement, gravity constant)
def stateProcessInfluence(x,t,u):
    # numerically validated for one case
    Gk = np.array([ [0.0],[1.0] ])
    return Gk

## viconFunction(x,t) - returns the expectation of a vicon measurement, given the state and time
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
def measFunction(x,t):
    yexp = np.array([ x[0] ])
    return yexp

## viconGradient(x,t) - returns the gradient of the expectation given the state and time
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
def measGradient(x,t):
    Hk = np.array([ [1.0,0.0] ])
    return Hk

def initFunction(yinit):
    return (np.array([ yinit[0],0.0 ]),np.identity(2)*1000.0+1e-6*np.ones((2,2)))

def simMeasurementFunction(xk,t):
    ymeas = np.array([ xk[0]+ np.random.normal(0.0,0.01) ])
    return ymeas

def main(argin='./'):
    # output file
    FOUT = open('python_enkf_test.csv','w')
    FOUT.write('t,x1,x2,ymeas,x1hat,x2hat,P11,P22\n');

    # initialize EKF
    Qkin = np.array([[20.0]])#continuous-time integration value
    #Qkin = np.array([[20.0]])#Euler integration value
    Hkin = np.array([[1.0,0.0]])
    Rkin = np.array([ [0.0001] ])
    EnKF = enkf.enkf(2,1,stateDerivativeEKF,Hk=Hkin,Qk=Qkin,Rk=Rkin,Ns=50)

    dt = 0.01
    tfin = 10.0
    nSteps = int(tfin/dt)
    tsim = 0.0

    xk = np.array([1.0,0.0])
    yk = measFunction(xk,tsim)

    # initial covariance
    P0 = np.diag([Rkin[0,0],1.0])
    EnKF.init(xk,P0)

    Enkfx = np.mean(EnKF.xk,axis=1)

    xt = np.zeros((nSteps,2))
    xf = np.zeros((nSteps,2))
    Pxd = np.zeros((nSteps,2))
    Pxx = np.zeros((2,2))
    for k in range(nSteps):
        # log
        xt[k,:] = xk.copy()
        xf[k,:] = Enkfx.copy()
        Pxd[k,0] = Pxx[0,0]
        Pxd[k,1] = Pxx[1,1]
        # propagate filter
        #EnKF.propagate(dt)
        EnKF.propagateOde(dt)
        # simulate
        y = sp.odeint(stateDerivative,xk,np.array([tsim,tsim+dt]),args=([],) )
        xk = y[-1,:].copy()

        # update time
        tsim = tsim + dt
        # measurement
        ymeas = simMeasurementFunction(xk,tsim)
        # update EKF
        EnKF.update(ymeas)
        # get the mean and covariance estimate out
        Enkfx = np.mean(EnKF.xk,axis=1)
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
        elif k < 4:
            ax[k].plot(tplot,xt[:,k-2]-xf[:,k-2],'b-')
            ax[k].plot(tplot,3.0*np.sqrt(Pxd[:,k-2]),'r--')
            ax[k].plot(tplot,-3.0*np.sqrt(Pxd[:,k-2]),'r--')

        ax[k].grid()
    fig.show()

    raw_input("Return to exit")

    print("Completed test_enky.py")
    return

if __name__ == '__main__':
    main()
