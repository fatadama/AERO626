'''
ukf.py
generic UKF class object definition
dependencies: Numpy, Scipy, base Python. Tested in 2.7, should work in 3.4 as well?
'''

import numpy as np
# math for sqrt
import math

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
    ## Initialization function for ukf class
    #   @param yk the measurement used to initialize the function
    #   @param y2xInitFunction function handle, returns xinitial = y2xInitFunction(yk)
    def init(self,yk,y2xInitFunction,tInit):
        # initialize state using the passed function
        (self.xhat,self.Pk) = y2xInitFunction(yk)
        # initialize time
        self.t = tInit
        # set flag to initialized
        self.initFlag = True
    ## Alternate initialization function for the UKF: initial x, time, and covariance are specified
    def init_P(self,x0,Pk,tInit):
        self.xhat = x0.copy()
        self.Pk = Pk.copy()
        self.t = tInit
        self.initFlag = True
    ## Propagate by time dt and then perform Kalman update, using Euler integration
    #   @param dt time increment
    #   @param ymeas measurement
    #   @param measurementFunction function pointer of the form: yexp = measurementFunction(x,t,g) where g is the measurement noise vector, of the same length as y
    #   @param Rk measurement noise covariance matrix, of size ny x ny, ny is length of the measurement vector
    #   @param flag_rk4 (True) Set to True to use runge-kutta 4th order propagation for sigma points. Uses first-order Euler integration otherwise. Default: RK4
    def sync(self,dt,ymeas,measurementFunction,Rk,flag_rk4=True):
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
        try:
            Psq = np.linalg.cholesky(Paug)
        except np.linalg.linalg.LinAlgError:
            raise np.linalg.linalg.LinAlgError('Matrix square root failed, singular covariance?')
        # compute the sigma points
        XAUG = np.zeros((L,2*L+1))
        XAUG[:,0] = xaug.copy()
        for k in range(L):
            XAUG[:,k+1] = xaug + gamm*Psq[:,k]
            XAUG[:,k+1+L] = xaug - gamm*Psq[:,k]
        # propagate each sigma point and compute the apriori state
        self.xhat = np.zeros(self.n)
        for k in range(2*L+1):
            if not flag_rk4:
                dx = self.propagateFunction(XAUG[0:self.n,k],self.t,self.u,XAUG[self.n:(self.n+self.nv),k])
                XAUG[0:self.n,k] = XAUG[0:self.n,k]+dt*dx
            else:
                XAUG[0:self.n,k] = self.propagateRK4(dt,XAUG[0:self.n,k],XAUG[self.n:(self.n+self.nv),k])
            # compute the apriori state
            self.xhat = self.xhat + wm[k]*XAUG[0:self.n,k]
        self.Pk = np.zeros((self.n,self.n))
        for k in range(2*L+1):
            self.Pk = self.Pk + wc[k]*np.outer( XAUG[0:self.n,k]-self.xhat,(XAUG[0:self.n,k]-self.xhat) )
        # time update
        self.t = self.t + dt
        # compute the apriori covariance
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
    ## propagateRK4(self,dt,xk,vk) Propagate a given state xk with constant process noise vk over an interval dt.
    #
    # Uses a runge-kutta timestep with fixed time step. No error checking, may not converge
    #   @param dt interval over which to propagate
    #   @param xk current state
    #   @param vk current process noise for propagation
    #   @param dtout minimum step size for integration - defaults to 0.01
    def propagateRK4(self,dt,xk,vk,dtout=None):
        if dtout == None:
            dtout = 0.01
        # vector of times over which to integrate
        nu = int(dt/dtout)
        dts = [dtout]
        for k in range(1,nu):
            dts.append(dtout)
        rem = dt - dtout*nu
        if rem > dt*1.0e-4:
            dts.append(rem)
            nu = nu + 1
        for kouter in range(nu):
            # one step RK4
            h6 = 1.0/6.0

            k1 = dts[kouter]*self.propagateFunction(xk,self.t,self.u,vk)
            k2 = dts[kouter]*self.propagateFunction(xk+0.5*k1,self.t+0.5*dts[kouter],self.u,vk)
            k3 = dts[kouter]*self.propagateFunction(xk+0.5*k2,self.t+0.5*dts[kouter],self.u,vk)
            k4 = dts[kouter]*self.propagateFunction(xk+k3,self.t+dts[kouter],self.u,vk)
            # update the state
            xk = xk + h6*(k1+2.0*k2+2.0*k3+k4)
            # update the time
            self.t = self.t + dts[kouter]
        # store
        return xk

