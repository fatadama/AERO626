"""@package enkf
Ensemble Kalman Filter for systems with linear updates and (possibly) nonlinear propagation functions
"""

import numpy as np
import math
import scipy.integrate as sp

class enkf():
	## __init__
	#
	# @param[in] n Length of state vector
	# @param[in] m length of control vector
	# @param[in] propagateFunction Function that returns the state derivative as follows: dx = propagateFunction(x,t,u,v). u is the system control, v is the noise
	# @param[in] Hk measurement function input matrix
	# @param[in] Qk process noise covariance matrix
	# @param[in] Rk measurement noise covariance matrix 
	# @param[in] Ns number of emsemble samples to use (fixed here)
	def __init__(self,n=1,m=1,propagateFunction=None,Hk=None,Qk=None,Rk=None,Ns=None):
		## state vector length
		self._n = n
		## control length
		self._m = m
		## control vector
		self.u = np.zeros(m)
		## current time
		self.t = 0.0
		## propagation function
		self._propagateFunction = propagateFunction
		self._Hk = Hk
		self._Qk = Qk
		self._Rk = Rk
		## number of ensemble members (samples)
		if Ns is not None:
			self._N = Ns
		else:
			self._N = 2*n+1
		## initialization flag
		self._initFlag = False
		## ensemble state vector
		self.xk = np.zeros((self._n,self._N))
	## get_N get the number of ensemble members
	#
	# @param[out] No number of ensemble members
	def get_N(self):
		return self._N
	## set_control set the control to a specified value
	#
	# @param[in] ui m-length numpy vector to set the control to
	def set_control(self,ui):
		self.u = ui.copy()
	## init Initialize the ensemble estimates with the apriori mean and covariance
	#
	#	@param[in] mux n-length numpy vector
	#	@param[in] Pxx n x n numpy array
	def init(self,mux,Pxx):
		self.xk = np.random.multivariate_normal(mux,Pxx,size=(self._N,)).transpose()
		return
	## propagateOde propagate the system using the scipy integrator
	def propagateOde(self,dt):
		# generate process noise for each sample, size is nv x N where nv is length of process noise vector
		vk = np.random.multivariate_normal(np.zeros(self._Qk.shape[0]), self._Qk, size=(self._N,)).transpose()
		# propagate each sample
		for k in range(self._N):
			y = sp.odeint(self._propagateFunction,self.xk[:,k],np.array([self.t,self.t+dt]),args=(self.u,vk[:,k],))
			self.xk[:,k] = y[-1,:].copy()
		# update the time
		self.t = self.t + dt
	## propagate the system for dt seconds using a first-order Euler approximation
	def propagate(self,dt):
		# generate process noise for each sample, size is nv x N where nv is length of process noise vector
		vk = np.random.multivariate_normal(np.zeros(self._Qk.shape[0]), self._Qk, size=(self._N,)).transpose()
		# propagate each sample
		for k in range(self._N):
			self.xk[:,k] = self.xk[:,k] + dt*self._propagateFunction(self.xk[:,k],self.t,self.u,vk[:,k])
		# update the time
		self.t = self.t + dt
		return
	## update update the system in response to a measurement
	#
	# @param[in] ymeas measurement at current time
	def update(self,ymeas):
		# generate sample measurement noises from the covariance _Rk
		# dy is N x h where h is the number of outputs
		dy = np.random.multivariate_normal( np.zeros(self._Rk.shape[0]) ,self._Rk,size=(self._N,)).transpose()
		# compute the predicted mean
		xhatm = np.mean(self.xk,axis=1)
		# compute the state covariance Pxx
		coef = 1.0/(1.0+float(self._N))
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		# inverse term for the Kalman gian
		invTerm = np.linalg.inv(self._Rk + np.dot(self._Hk,np.dot(Pxx,self._Hk.transpose())))
		# Compute the Kalman gain
		Kk = np.dot(Pxx,np.dot(self._Hk.transpose(),invTerm))
		# update each state
		for k in range(self._N):
			innov = Kk*(ymeas + dy[:,k] - np.dot(self._Hk,self.xk[:,k]))
			for j in range(self._n):
				self.xk[j,k] = self.xk[j,k] + innov[j]
