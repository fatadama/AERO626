"""@package gmm
Gaussian mixture model class
"""

## Returns the value of the Gaussian multivariate distribution at point x with mean mu and covariance P
#
# @param[in] x point at which to evaluate PDF
# @param[in] mu mean of underlying distribution
# @param[in] P covariance of distribution
# @param[out] p the pdf evaluated at x
def gaussianNormalPdf(x,mu,P):
	Pinv = np.linalg.inv(P)
	p = math.exp(-0.5*np.dot(x-mu,np.dot(Pinv,x-mu)))/math.sqrt(math.pow(2.0*math.pi,x.shape[0])*np.linalg.det(P))
	return p

class gmm():
	## __init__ Gauss mixture model estimator creation function
	#
	#@param[in] n dimension of the state vector
	#@param[in] Ns number of means to use in propagation
	#@param[in] Qk process nosie covariance
	#@param[in] Rk measurement noise covariance
	#@param[in] propagateFunction returns the state derivative: dy = eqom(x,t,u)
	#@param[in] propagationJacobian returns the jacobian at a given state
	#@param[in] processInfluence returns the process noise infuence matrix Gk as Gk = Jac(x,t,u)
	#@param[in] measurementFunction returns the value of the expectation for a given time and state y = exp(x,t)
	#@param[in] measurementJacobian gradient of the optionally nonlinear output function
	def __init__(self,n,Ns,Qk,Rk,propagateFunction,propagationJacobian,processInfluence,measurementFunction,measurementJacobian):
		## n x Ns array of means
		self.aki = np.zeros((n,Ns))
		## n x n x Ns array of associated covariances
		self.Pki = np.zeros((n,n,Ns))
		## propagation function: dy = eqom(x,t,u)
		self.propagateFunction = propagateFunction
		## propagation Jacobian: F = gradient(x,t,u)
		self.propagationJacobian = propagationJacobian
		## process noise influence matrix
		self.processInfluence = processInfluence
		## measurement equation
		self.measurementFunction = measurementFunction
		## measurement Jacobian .. Hk = measurement(x,t)
		self.measurementJacobian = measurementJacobian
		## process noise covariance
		self.Qk = Qk.copy()
		## measurement noise covariance
		self.Rk = Rk.copy()
		## Ns-length vector of scalar weights that sum to zero
		self.alphai = no.zeros(Ns)
		## time (scalar)
		self.ts = 0.0
		## control vector: current unknown size, initialize to zero
		self.u = 0.0
	## propagation function for the case where the order of the process noise and state uncertainty is comparable
	#
	# @param[in] dt time increment; first-order Euler integration used
	def propagate_normal(self,dt):
		# update the means: evaluate the propagation function, Jacobian, and process noise influence for each means
		dxi = np.zeros(self.aki.shape)
		Fki = np.zeros((self.aki.shape[0],self.aki.shape[0],self.aki.shape[1]))
		Gki = np.zeros((self.aki.shape[0],self.Qk.shape[0],self.aki.shape[1]))
		for k in range(self.aki.shape[1]):
			# eval the derivative
			dxi[:,k] = self.propagateFunction(self.aki[:,k],self.ts,self.u)
			# eval the Jacobian
			Fki[:,:,k] = self.propagationJacobian(self.aki[:,k],self.ts,self.u)
			# eval the process noise influence
			Gki[:,:,k] = self.processInfluence(self.aki[:,k],self.ts,self.u)
			# propagate the aki
			self.aki[:,k] = self.aki[:,k] + dt*dxi[:,k]
			Fki[:,:,k] = np.identity(self.aki.shape[0]) + dt*Fki[:,:,k]
			Gki[:,:,k] = Gki[:,:,k]*dt
			# propagate the covariance
			self.Pki[:,:,k] = np.dot(Fki[:,:,k],np.dot(self.Pki[:,:,k],Fki[:,:,k].transpose())) + np.dot(Gki[:,:,k],np.dot(self.Qk,Gki[:,:,k].transpose()))
		#update the time
		self.ts = self.ts + dt
	def update(self,ymeas):
		Bkj = np.zeros(self.aki.shape[1])
		# compute the normalizing factor
		normalizingFactor = 0.0
		for j in range(self.aki.shape[1]):
			# evaluate Bkj
			Hk = self.measurementJacobian(self.aki[:,j],self.ts)
			Pkj = np.dot(Hk,np.dot(self.Pki[:,:,j],Hk.transpose())) + self.Rk
			Bkj[j] = gaussianNormalPdf(ymeas-self.measurementFunction(self.axi[:,j],self.ts),0.0,Pkj)
			normalizingFactor = normalizingFactor + self.alphai[j]*Bkj[j]
		normalizingFactor = 1.0/normalizingFactor
		# now update everything else
		for j in range(self.aki.shape[1]):
			# measurement gradient
			Hk = self.measurementJacobian(self.aki[:,j],self.ts)
			# Kalman gain
			invTerm = np.dot(Hk,np.dot(self.Pki[:,:,j],Hk.transpose())) + self.Rk
			Ki = np.dot(self.Pki[:,:,j],np.dot(Hk.transpose(),np.linalg.inv(invTerm)))
			# expectation
			yexp = self.measurementFunction(self.aki[:,j],self.ts)
			# update the mean
			self.aki[:,j] = self.aki[:,j] + np.dot(Ki,ymeas-yexp)
			# update the covariance
			self.Pki[:,:,j] = self.Pki[:,:,j] - np.dot(Ki,np.dot(Hk,self.Pki[:,:,j]))
			# update the weights
			self.alphai[j] = self.alphai[j]*Bkj[j]*normalizingFactor