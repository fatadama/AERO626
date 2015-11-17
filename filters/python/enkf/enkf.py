"""@package enkf
Ensemble Kalman Filter for systems with linear updates and (possibly) nonlinear propagation functions
"""

import numpy as numpy
import math

class enkf():
	## __init__
	#
	# @param[in] n Length of state vector
	# @param[in] m length of control vector
	# @param[in] propagateFunction Function that returns the state derivative as follows: dx = propagateFunction(x,t,u)
	# @param[in] Hk measurement function input matrix
	# @param[in] Qk process noise covariance matrix
	# @param[in] Rk measurement noise covariance matrix 
	def __init__(self,n=1,m=1,propagateFunction=None,Hk=None,Qk=None,Rk=None):
		## state vector length
		self_n = n
		## control length
		self._m = m
		## propagation function
		self._propagateFunction = propagateFunction
		self._Hk = Hk
		self._Qk = Qk
		self._Rk = Rk
		## number of ensemble members (samples)
		self._N = 2*n+1
		## initialization flag
		self._initFlag = False
		## ensemble state vector
		self.xk = np.zeros((self._n,self._N)
	## init Initialize the ensemble estimates with the apriori mean and covariance
	#
	#	@param[in] mux n-length numpy vector
	#	@param[in] Pxx n x n numpy array
	def init(self,mux,Pxx):
		self.xk = np.random.multivariate_normal(mux,Pxx,size=(self._n,))
		print(self.xk.shape)
		return