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

class ukf():
    def __init__(self,n=1,m=1,nv=1,propagateFunction=None,Qk=None):
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
        ## number of process noise members
        self.nv = nv
        ## propagateFunction: returns the derivative of the state for propagation. arguments as: propagateFunction(x,t,u)
        self.propagateFunction = propagateFunction
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
    ## Propagate by time dt and then perform Kalman update, using Euler integration
    #   @param dt time increment
    #   @param ymeas measurement
    #   @param measurementFunction function pointer of the form: yexp = measurementFunction(x,t,g) where g is the measurement noise vector, of the same length as y
    #   @param Rk measurement noise covariance matrix, of size ny x ny, ny is length of the measurement vector
    def sync(self,dt,ymeas,measurementFunction,Rk):
        # constants
        # TODO move computation of weights and filter parameters to the initialization step
        ny = len(ymeas)
        L = self.n+self.nv+ny

        alpha = 1.0e-3
        Kappa = 0.0
        beta = 2.0
        lambd = alpha*alpha*(L+Kappa)-L
        gamm = math.sqrt(L+lambd)
        wm = 0.5/(L+lambd)*np.ones(2*L+1)
        wm[0] = lambd/(L+lambd)
        wc = 0.5/(L+lambd)*np.ones(2*L+1)
        wc[0] = wm[0] + 1.0-alpha*alpha+beta

        # augmented state
        xaug = np.concatenate((self.xhat,np.zeros(ny+self.nv)))
        # augmented covariance
        Paug = np.zeros((self.n+self.nv+ny,self.n+self.nv+ny))
        Paug[0:self.n,0:self.n] = self.Pk.copy()
        Paug[self.n:(self.n+self.nv),self.n:(self.n+self.nv)] = self.Qk.copy()
        Paug[(self.n+self.nv):(self.n+self.nv+ny),(self.n+self.nv):(self.n+self.nv+ny)] = Rk.copy()

        # compute cholesky decomposition of Paug
        Psq = np.linalg.cholesky(Paug)
        # compute the sigma points
        XAUG = np.zeros((L,2*L+1))
        XAUG[:,0] = xaug.copy()
        for k in range(L):
            XAUG[:,k+1] = xaug + gamm*Psq[:,k]
            XAUG[:,k+1+L] = xaug - gamm*Psq[:,k]
        # propagate each sigma point

        self.xhat = np.zeros(self.n)
        for k in range(2*L+1):
            dx = self.propagateFunction(XAUG[0:self.n,k],self.t,self.u,XAUG[self.n:(self.n+self.nv),k])
            XAUG[0:self.n,k] = XAUG[0:self.n,k]+dt*dx
            # compute the apriori state
            self.xhat = self.xhat + wm[k]*XAUG[0:self.n,k]
        # compute the apriori covariance
        self.Pk = np.zeros((self.n,self.n))
        for k in range(2*L+1):
            self.Pk = self.Pk + wc[k]*np.outer( XAUG[0:self.n,k]-self.xhat,(XAUG[0:self.n,k]-self.xhat) )
        # pass the propagated state through the measurement function
        YAUG = np.zeros((ny,2*L+1))
        yhat = np.zeros(ny)
        for k in range(2*L+1):
            YAUG[:,k] = measurementFunction(XAUG[0:self.n,k],self.t,XAUG[(self.n+self.nv):L,k])
            yhat = yhat + wm[k]*YAUG[:,k]
        # compute covariance and cross covariance
        Pyy = np.zeros((ny,ny))
        Pxy = np.zeros((self.n,ny))
        for k in range(2*L+1):
            Pyy = Pyy + wc[k]*np.outer(YAUG[:,k]-yhat,YAUG[:,k]-yhat)
            Pxy = Pxy + wc[k]*np.outer(XAUG[0:self.n,k]-self.xhat,YAUG[:,k]-yhat)
        # Kalman gain
        Kk = np.dot(Pxy,np.linalg.inv(Pyy))
        # state update
        self.xhat = self.xhat + np.dot(Kk,ymeas-yhat)
        # covariance update
        self.Pk = self.Pk-np.dot(Kk,Pxy.transpose())

    def propagateRK4(self,dt,xk):
        # one step RK4
        #h = dt
        h6 = 1.0/6.0

        k1 = dt*self.propagateFunction(xk,self.t,self.u)
        k2 = dt*self.propagateFunction(xk+0.5*k1,self.t+0.5*dt,self.u)
        k3 = dt*self.propagateFunction(xk+0.5*k2,self.t+0.5*dt,self.u)
        k4 = dt*self.propagateFunction(xk+k3,self.t+dt,self.u)
        # update the state
        xk = xk + h6*(k1+2.0*k2+2.0*k3+k4)
        # store
        return xk
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

