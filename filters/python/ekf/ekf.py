'''
offboard_filter.py
accepts vicon, gyro, accel, baro? and propagates in between
'''

# data format: [timestamp, accx, accy, accz]
import sys
import time
import numpy as np
#import matplotlib.pyplot as plt
# math for pi
import math

import scipy.integrate as sp

sys.path.append('../lib')

#import kalman

class ekf():
    def __init__(self,n=1,m=1,propagateFunction=None,propagateGradient=None,processNoiseMatrix=None,Qk=None):
        ## initialization flag
        self.initFlag = False
        ## state estimate vector
        self.xhat = np.zeros(n)
        ## control vector
        self.u = np.zeros(m)
        ## current time, set to system time on init
        self.t = 0.0
        ## covariance matrix
        self.Pk = np.zeros((n,n))
        ## process noise covariance matrix
        self.Qk = Qk
        ## number of states
        self.n = n
        ## number of controls
        self.m = m
        ## propagateFunction: returns the derivative of the state for propagation. arguments as: propagateFunction(x,t,u)
        self.propagateFunction = propagateFunction
        ## propagateGradient: returns the gradient of the state derivative with respect to the state arguments: propagateGradient(x,t,u)
        self.propagateGradient = propagateGradient
        ## processNoiseMatrix: returns the process noise linear influence matrix as a function of time, state, and control
        self.processNoiseMatrix = processNoiseMatrix
        return
    ## Initialization function for ekf class
    #   @param yk the measurement used to initialize the function
    #   @param y2xInitFunction function handle, returns xinitial = y2xInitFunction(yk)
    def init(self,yk,y2xInitFunction,tInit):
        # initialize state using the passed function
        (self.xhat,self.Pk) = y2xInitFunction(yk)
        # initialize time
        self.t = tInit
        # set flag to initialized
        self.initFlag = True
    def propagateRK4(self,dt):
        #sp.odeint(self.propagateFunction,self.xhat,np.array([self.t,self.t+dt]),args=([self.u],) )
        Pcol = np.reshape(self.Pk,(self.n*self.n,))
        xaug = np.concatenate((self.xhat,Pcol))

        # one step RK4
        #h = dt
        h6 = 1.0/6.0

        k1 = dt*self.derivatives(xaug,self.t,self.u)
        k2 = dt*self.derivatives(xaug+0.5*k1,self.t+0.5*dt,self.u)
        k3 = dt*self.derivatives(xaug+0.5*k2,self.t+0.5*dt,self.u)
        k4 = dt*self.derivatives(xaug+k3,self.t+dt,self.u)
        # update the state
        xaug = xaug + h6*(k1+2.0*k2+2.0*k3+k4)
        # store
        self.xhat = xaug[0:self.n].copy()
        self.Pk = xaug[self.n:].reshape((self.n,self.n))
        self.t = self.t + dt
        return
    def derivatives(self,xarg,ts,uarg):
        xk = xarg[0:self.n]
        Pk = np.reshape(xarg[self.n:],(self.n,self.n) )
        # state derivative
        dxstate = self.propagateFunction(xk,ts,uarg)
        # Jacobian
        Fk = self.propagateGradient(xarg,ts,uarg)
        # process noise matrix
        Gk = self.processNoiseMatrix(xarg,ts,uarg)
        # Pdot = F*P+P*F'+G*Q*G'
        Pdot = np.dot(Fk,Pk)+np.dot(Pk,Fk.transpose()) + np.dot(Gk,np.dot(self.Qk,Gk.transpose() ))
        Pdotcol = Pdot.reshape((self.n*self.n,))
        dx = np.concatenate((dxstate,Pdotcol))
        return dx

    ## propagate(dt) - propagate state for dt seconds using Euler integration
    #   @param dt the time for which to propagate
    def propagate(self,dt):
        # first order Euler approximation to propagate, for now
        xkprior = self.xhat + dt*self.propagateFunction(self.xhat,self.t,self.u)
        # discretized function gradient
        Fk = self.propagateGradient(self.xhat,self.t,self.u)*dt + np.eye(self.n)
        # discretized process noise matrix
        Gk = self.processNoiseMatrix(self.xhat,self.t,self.u)*dt
        # update the state
        self.xhat = xkprior.copy()

        #print("Fk:",Fk)
        #print("Gk:",Gk)
        #print("Pk:",self.Pk)
        #print("G*Q*G':",np.dot(np.dot(Gk,self.Qk),Gk.transpose()) )
        #print("F*P*F':",np.dot(np.dot(Fk,self.Pk),Fk.transpose()) )

        # propagate the covariance
        self.Pk = np.dot(np.dot(Fk,self.Pk),Fk.transpose() ) + np.dot(np.dot(Gk,self.Qk),Gk.transpose() )
        # propagate the time
        self.t = self.t + dt
        #print(self.t,self.xhat,self.Pk)
    ## update(self,xprior,Pkprior,tstamp,yk,measurementFunction,measurementGradient,Rk)
    #
    #   update the a priori state using a measurement
    def update(self,tstamp,yk,measurementFunction,measurementGradient,Rk):
        # expected output, is independent of the control!
        yexp = measurementFunction(self.xhat,tstamp)
        # Kalman gain
        Hk = measurementGradient(self.xhat,tstamp)
        invTerm = np.linalg.inv( np.dot(np.dot(Hk,self.Pk),Hk.transpose()) + Rk )

        #print("invterm:",invTerm)

        #print("H'*invTerm:",np.dot(Hk.transpose(),invTerm))
        #print("Pk:",self.Pk)

        Kk = np.dot(self.Pk,np.dot(Hk.transpose(),invTerm))

        #print("Kk:",Kk)

        # update the prior state
        self.xhat = self.xhat + np.dot( Kk,yk-1.0*yexp )
        # update the prior covariance

        #print("Kk*Hk*Pk:",np.dot(np.dot(Kk,Hk),self.Pk))

        self.Pk = np.dot((np.identity(self.n) - np.dot(Kk,Hk)),self.Pk)
        #print(self.xhat,self.Pk)

