"""@package cp_dynamics
Module with the governing equations for the challenge problem
"""

import numpy as np
import math
import scipy.integrate as sp

## epsilon_eqom nonlinear parameter in governing equation
epsilon_eqom = 1.0e-2
## a_0 amplitude of forcing costine term
a_0 = 0.5
## omega_t frequency of forcing function
omega_t = 1.25
## DT discretization time for dynamic uncertainty
DT = 1e-3
## q_w standard deviation of the stochastic forcing term in the nonlinear governing dynamics
q_w = 1.0*DT
## r_w standard deviation of measurement noise
r_w = 1.0
## r_w_cluster standard deviation of measurement noise to use for cluster trials: reduces the uncertainty
r_w_cluster = 0.01
## q_w_cluster standard deviation of process noise to use for cluster trials: reduces the uncertainty
q_w_cluster = 0.01*DT

## ode_wrapper Ensures good integrator convergence by enforcing piecewise-constant noise
#
# Wrapper function that integrates over discrete ranges, for which the process noise in the underlying functions is maintained as constant
# @param[in] fun function of the form (dx) = fun(x,t) or (dx) = fun(x,t,v=None)
# @param[in] x0 initial state
# @param[in] tsp numpy array of times at which the state history should be evaluated
# @param[out] ysp simulated system state at the supplied times in tsp
def ode_wrapper(fun,x0,tsp):
    # array of times at which dynamic uncertainty happens
    tdisc = np.arange(tsp[0],tsp[-1]+DT,DT)

    # determine (ad hoc) whether the called function takes 2 or 3 args
    flag_stoch = False
    try:
        dy = fun(x0,tsp[0],0.0)
        flag_stoch = True
    except TypeError:
        dy = fun(x0,tsp[0])
        flag_stoch = False

    yt = np.zeros((len(tdisc),len(x0)))
    y0 = x0.copy()
    for k in range(len(tdisc)-1):
        if flag_stoch:
            # compute the noise
            v = np.random.normal(scale=q_w)
            yp = sp.odeint(fun,y0,tdisc[k:k+2],args=(v,))
        else:
            yp = sp.odeint(fun,y0,tdisc[k:k+2])
        yt[k,:] = y0.copy()
        y0 = yp[-1,:].copy()
    yt[-1,:] = y0.copy()
    # interpolate in the new history to match the passed-in history tsp
    ysp = np.zeros((len(tsp),len(x0)))
    for k in range(len(x0)):
        ysp[:,k] = np.interp(tsp,tdisc,yt[:,k])
    return ysp

## eqom_det Deterministic equation of motion with no forcing
#
#   \ddot{x} = x-epsilon_eqom*x^3+a_0*cos(omega_t*t)
#   @param[in] t time
#   @param[in] x state (position, velocity)
def eqom_det(x,t):
    dx = np.zeros((2,))
    dx[0] = x[1]
    try:
        dx[1] = x[0]-epsilon_eqom*math.pow(x[0],3.0)
    except OverflowError:
        print("Overflow warning in eqom_det: x = (%f,%f)" % (x[0],x[1]))
    return dx

## eqom_det_jac Jacobian of eqom_det at a given state and time
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term; dummy here, only for consistent calling form
def eqom_det_jac(x,t,v=None):
    Fk = np.zeros((2,2))
    Fk[0,1] = 1.0;
    Fk[1,0] = 1.0-epsilon_eqom*3.0*x[0]*x[0]
    return Fk

## eqom_det_Gk process noise influence matrix of eqom_det at a given state and time
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term; dummy here, only for consistent calling form
def eqom_det_Gk(x,t,v=None):
    return np.array([ [0.0],[1.0] ])

## eqom_det_f Deterministic equation of motion with cosine forcing and no white noise
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
def eqom_det_f(x,t):
    dx = eqom_det(x,t)
    dx[1] = dx[1] + a_0*math.cos(omega_t*t)
    return dx

## eqom Stochastic equation of motion with forcing and uncertainty
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term
def eqom(x,t,v=None):
    dx = eqom_det_f(x,t)
    if v is None:
        dx[1] = dx[1] + np.random.normal(scale=q_w)
    else:
        dx[1] = dx[1] + v
    return dx

## eqom_stoch Stochastic equation of motion without forcing
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term
def eqom_stoch(x,t,v=None):
    dx = eqom_det(x,t)
    if v is None:
        dx[1] = dx[1] + np.random.normal(scale=q_w)
    else:
        dx[1] = dx[1] + v
    return dx

## Stochastic equation of motion without forcing, with reduced process noise for the clustering case
def eqom_stoch_cluster(x,t,v=None):
    dx = eqom_det(x,t)
    if v is None:
        dx[1] = dx[1] + np.random.normal(scale=q_w_cluster)
    else:
        dx[1] = dx[1] + v
    return dx

## eqom_stoch_jac Jacobian of eqom_stoch at a given state and time
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term; dummy here, only for consistent calling form
def eqom_stoch_jac(x,t,v=None):
    Fk = np.zeros((2,2))
    Fk[0,1] = 1.0;
    Fk[1,0] = 1.0-epsilon_eqom*3.0*x[0]*x[0]
    return Fk

## eqom_stoch_Gk process noise influence matrix of eqom_stoch at a given state and time
#
#   @param[in] t time
#   @param[in] x state (position, velocity)
#   @param[in] v process noise term; dummy here, only for consistent calling form
def eqom_stoch_Gk(x,t,v=None):
    return np.array([ [0.0],[1.0] ])

class cp_simObject:
    def __init__(self,funarg=eqom,x0=np.array([0.0,0.0]),Tsin=0.01):
        ## underlying equation of motion that drives the system dynamics for simulation
        self._eqomf = funarg
        ## current state
        self._xk = x0.copy()
        ## measurement sample period
        self._Ts = Tsin
        ## current simulation time
        self._t = 0.0
    ## step propagate the system by ony measurement time step (self._Ts)
    #
    # @param[out] yk scalar, most recent measurement with noise
    # @param[out] xk 2 x 1 numpy array most recent truth state
    def step(self):
        # propagate to time t = t + Ts
        xs = ode_wrapper(self._eqomf,self._xk,np.array([self._t,self._t+self._Ts]))
        # update time
        self._t = self._t + self._Ts
        # update state
        self._xk = xs[-1,:].copy()
        # take measurement
        yk = self.measureFunction()
        return(yk,self._xk.copy())
    ## measureFunction take a measurement at the current state, with a process noise defined globally
    #
    #@param[out] yk scalar measurement of the position with measurement noise defined by global r_w
    def measureFunction(self):
        return np.array([ self._xk[0] + np.random.normal(scale=r_w) ])
    ## simFull simulate the system for Tf-self._t seconds, taking measurements of the resulting time histories.
    #
    #@param[in] Tf final time; the total simulation time is Tf - self._t
    #@param[out] yk N-length numpy vector defining the measurements at each discrete interval
    #@param[out] xk [N x 2] numpy array defining the true state history
    #@param[out] tk N-length numpy vector defining the output times
    def simFull(self,Tf=1.0):
        # tk: vector of measurement times
        tk = np.arange(self._t,Tf+self._Ts,self._Ts)
        # simulate, getting back the state at the measurement times
        xk = ode_wrapper(self._eqomf,self._xk,tk)
        # measure at each time in xk
        yk = np.zeros(len(tk))
        for k in range(len(tk)):
            self._xk = xk[k,:].copy()
            self._t = tk[k]
            yk[k] = self.measureFunction()
        return(yk,xk,tk)

## cp_simObjectNonInformative
#
# simulation object for propagating system with measurements of position-squared only
# Other functionality should all be inherited from the original simObject
class cp_simObjectNonInformative(cp_simObject):
    ## Returns the position squared measurement with sensor noise as a 1-length numpy vector
    def measureFunction(self):
        return np.array([ self._xk[0]*self._xk[0] + np.random.normal(scale=r_w_cluster) ])

## Derived simulation class object for the specific case of multiple state clusters - uses a smaller measurement variable to try to encourage these bifurcated solutions in the output
class cp_simObjectCluster(cp_simObject):
    ## Returns the position measurement with less noise, to exacerbate the difference in the several cases
    def measureFunction(self):
        return np.array([ self._xk[0] + np.random.normal(scale=r_w_cluster) ])