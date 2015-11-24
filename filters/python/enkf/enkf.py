"""@package enkf
Ensemble Kalman Filter for systems with linear updates and (possibly) nonlinear propagation functions
"""

import numpy as np
import math
import scipy.integrate as sp

def convergenceCheck(XK,PXX,tol):
		# remove one sample, then check to see if the covariance changes more than the tolerancve
		mux = np.mean(XK[:,0:-1],axis=1)
		N = XK.shape[1]
		n = XK.shape[0]
		Pxx = np.zeros((n,n))
		coef = 1.0/float(N)
		for k in range(N-1):
			Pxx = Pxx + coef*np.outer(XK[:,k]-mux,XK[:,k]-mux)
		# convergence metric
		metric = np.linalg.norm(Pxx-PXX)/np.linalg.norm(PXX)
		if metric < tol:
			while metric < tol:
				# it converged, remove last point and keep removing
				XK = XK[:,0:-1]
				N = N-1
				# repeat
				mux = np.mean(XK[:,0:-1],axis=1)
				N = XK.shape[1]
				n = XK.shape[0]
				Pxx = np.zeros((n,n))
				coef = 1.0/float(N)
				for k in range(N-1):
					Pxx = Pxx + coef*np.outer(XK[:,k]-mux,XK[:,k]-mux)
				# convergence metric
				metric = np.linalg.norm(Pxx-PXX)/np.linalg.norm(PXX)
				print("Decreased to %d pts, metric = %f" % (N,metric))
		else:
			while metric > tol:
				# update covariance to the newer value
				PXX = Pxx.copy()
				# didn't converge, add a point
				XK = np.concatenate((XK,np.random.multivariate_normal(mux,Pxx,size=(1,)).transpose()),axis=1)
				N = N+1
				# repeat
				mux = np.mean(XK[:,0:-1],axis=1)
				N = XK.shape[1]
				n = XK.shape[0]
				Pxx = np.zeros((n,n))
				coef = 1.0/float(N)
				for k in range(N-1):
					Pxx = Pxx + coef*np.outer(XK[:,k]-mux,XK[:,k]-mux)
				# convergence metric
				metric = np.linalg.norm(Pxx-PXX)/np.linalg.norm(PXX)
				print("Increased to %d pts, metric = %f" % (N,metric))
		return (XK,PXX)

def convergenceCheckProp(XK,PXX,tol,propFunction):
	# check convergence, add new points if needed and propagate forward
	return

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
	def init(self,mux,Pxx,ts=0.0):
		self.xk = np.random.multivariate_normal(mux,Pxx,size=(self._N,)).transpose()
		self.t = ts
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

## adaptive_enkf
#
# version of the Ensemble Kalman Filter that adapts the size of the ensemble based on the convergence properties of the computed mean and covariance
class adaptive_enkf(enkf):
	## __init__
	#
	# @param[in] n Length of state vector
	# @param[in] m length of control vector
	# @param[in] propagateFunction Function that returns the state derivative as follows: dx = propagateFunction(x,t,u,v). u is the system control, v is the noise
	# @param[in] Hk measurement function input matrix
	# @param[in] Qk process noise covariance matrix
	# @param[in] Rk measurement noise covariance matrix 
	# @param[in] Ns number of particles/ensemble members to use
	# @param[in] tolIn (1.0e-2) tolerance metric used for convergence of the covariance estimate. Aggressive values may fail to converge in Python
	def __init__(self,n=1,m=1,propagateFunction=None,Hk=None,Qk=None,Rk=None,Ns=None,tolIn = 0.01):
		# initialization is the same as for enkf, with some additional class members
		## convergence value for the mean and covariance
		self._convTol= tolIn
		if Ns is None:
			Ns = 2*n+1
		# perform normal initialization
		enkf.__init__(self,n,m,propagateFunction,Hk,Qk,Rk,Ns)
	## init Initialize the ensemble estimates with the apriori mean and covariance
	#
	#	@param[in] mux n-length numpy vector
	#	@param[in] Pxk n x n numpy array
	def init(self,mux,Pxk,ts=0.0):
		# initialize
		self.xk = np.random.multivariate_normal(mux,Pxk,size=(self._N,)).transpose()
		# add more points until we converge

		# compute the predicted mean
		xhatm = np.mean(self.xk,axis=1)
		# compute the state covariance Pxx
		coef = 1.0/(1.0+float(self._N))
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		# call the convergence function
		(self.xk,Pxx) = convergenceCheck(self.xk,Pxx,self._convTol)
		self._N = self.xk.shape[1]
		print("Initialized adaptive EnKF with %d pts" % (self._N))
		'''
		# mean
		xhatm = np.mean(self.xk,axis=1)
		# covariance
		coef = 1.0/(1.0+float(self._N))
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		# error metrics
		err2 = np.linalg.norm(Pxk-Pxx)/np.linalg.norm(Pxx)
		print("e2 = %f" % (err2))
		while err2 > self._convTol:
			# add a point
			self.xk = np.concatenate((self.xk,np.random.multivariate_normal(mux,Pxk,size=(1,)).transpose()),axis=1)
			self._N  = self._N+1
			# mean
			xhatm = np.mean(self.xk,axis=1)
			# covariance
			coef = 1.0/(1.0+float(self._N))
			Pxx = np.zeros((self._n,self._n))
			for k in range(self._N):
				Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
			# error metrics
			err2 = np.linalg.norm(Pxk-Pxx)/np.linalg.norm(Pxx)
			print("Increased to %d points,e2 = %f,mu = %f,%f" % (self._N,err2,xhatm[0],xhatm[1]))
		'''
		self.t = ts
		return
	## propagate the system for dt seconds using a first-order Euler approximation. Adjust the ensemble size as computation proceeds.
	def propagate(self,dt):
		# generate process noise for each sample, size is nv x N where nv is length of process noise vector
		vk = np.random.multivariate_normal(np.zeros(self._Qk.shape[0]), self._Qk, size=(self._N,)).transpose()
		# propagate each sample
		for k in range(self._N):
			self.xk[:,k] = self.xk[:,k] + dt*self._propagateFunction(self.xk[:,k],self.t,self.u,vk[:,k])
		# analyze the convergence
		# update the time
		self.t = self.t + dt
		return
	## update update the system in response to a measurement. Adjust ensemble size to eliminate unused/add needed samples until covariance stops changing
	#
	# @param[in] ymeas measurement at current time
	def update(self,ymeas):
		# compute the predicted mean
		xhatm = np.mean(self.xk,axis=1)
		# compute the state covariance Pxx
		coef = 1.0/(1.0+float(self._N))
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		# call the convergence function
		(self.xk,Pxx) = convergenceCheck(self.xk,Pxx,self._convTol)
		self._N = self.xk.shape[1]
		# inverse term for the Kalman gian
		invTerm = np.linalg.inv(self._Rk + np.dot(self._Hk,np.dot(Pxx,self._Hk.transpose())))
		# Compute the Kalman gain
		Kk = np.dot(Pxx,np.dot(self._Hk.transpose(),invTerm))
		# generate sample measurement noises from the covariance _Rk
		# dy is N x h where h is the number of outputs
		dy = np.random.multivariate_normal( np.zeros(self._Rk.shape[0]) ,self._Rk,size=(self._N,)).transpose()
		# update each state
		for k in range(self._N):
			innov = Kk*(ymeas + dy[:,k] - np.dot(self._Hk,self.xk[:,k]))
			for j in range(self._n):
				self.xk[j,k] = self.xk[j,k] + innov[j]


