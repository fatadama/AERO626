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
    ## Propagate by time dt and then perform Kalman update
    def sync(self,dt,ymeas,measurementFunction,Rk):
        # constants
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
        #print(XAUG.transpose())
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

        print(self.Pk)
        print(self.xhat)

        # pass the propagated state through the measurement function
        YAUG = np.zeros((ny,2*L+1))
        yhat = np.zeros(ny)
        for k in range(2*L+1):
            YAUG[:,k] = measurementFunction(XAUG[0:self.n,k],self.t,XAUG[(self.n+self.nv):L,k])
            yhat = yhat + wm[k]*YAUG[:,k]
        print(YAUG.transpose())
        # compute covariance and cross covariance
        Pyy = np.zeros((ny,ny))
        Pxy = np.zeros((self.n,ny))
        for k in range(2*L+1):
            Pyy = Pyy + wc[k]*np.outer(YAUG[:,k]-yhat,YAUG[:,k]-yhat)
            Pxy = Pxy + wc[k]*np.outer(XAUG[0:self.n,k]-self.xhat,YAUG[:,k]-yhat)
        print(Pyy)
        print(Pxy)


    def propagateRK4(self,dt):
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

