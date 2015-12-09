"""@package gmm
Gaussian mixture model class
"""

import numpy as np
import math
import scipy.stats as stats

## Returns the value of the Gaussian multivariate distribution at point x with mean mu and covariance P
#
# @param[in] x point at which to evaluate PDF
# @param[in] mu mean of underlying distribution
# @param[in] P covariance of distribution
# @param[out] p the pdf evaluated at x
def gaussianNormalPdf(x,mu,P):
	Pinv = np.linalg.inv(P)
	d = x.shape[0]
	dp = np.dot(x-mu,np.dot(Pinv,x-mu))
	dt = 1.0/math.sqrt(math.pow(2.0*math.pi,float(d))*np.linalg.det(P))
	p = math.exp(-0.5*dp)*dt
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
		self.alphai = np.zeros(Ns)
		## time (scalar)
		self.ts = 0.0
		## control vector: current unknown size, initialize to zero
		self.u = 0.0
	## init_monte
	#
	# initialize the Gaussian terms based on an initial (Gaussian) proposed distribution
	def init_monte(self,mu,P0,ts=None):
		if ts is not None:
			self.ts = ts
		# draw random points for the initial means
		self.aki = np.random.multivariate_normal(mu,P0,size=(self.aki.shape[1],)).transpose()
		# uniform initial weights
		self.alphai = 1.0/float(self.aki.shape[1])*np.ones(self.aki.shape[1])
		# initial covariances are all equal
		Pkk0 = P0.copy()
		for k in range(self.aki.shape[1]):
			Pkk0 = Pkk0 - self.alphai[k]*np.outer(self.aki[:,k]-mu,self.aki[:,k]-mu)
		# force the covariane to be positive definite
		Pkk0 = np.absolute(Pkk0)
		# force the covariance to be diagonal
		Pkk0 = np.diag(np.diag(Pkk0))
		for k in range(self.aki.shape[1]):
			self.Pki[:,:,k] = Pkk0.copy()
	## propagation function for the case where the order of the process noise and state uncertainty is comparable
	#
	# handles integration using a maximum step size of 0.01 seconds
	# @param[in] dt time increment; first-order Euler integration used
	def propagate_normal(self,dt):
		if dt > 0.01:
			# determine the number of steps 
			stepsU = int(dt/0.01)
			dtsize = []
			for k in range(stepsU):
				dtsize.append(0.01)
			rem = (dt % 0.01)
			if rem > 1.0e-4:
				# append the remainder to the steps
				dtsize.append(rem)
				stepsU = stepsU + 1
			for k in range(stepsU):
				self.propagate_normal(dtsize[k])
				#print("%d,%f,%f" % (k,self.ts,dtsize[k]))
			return
		else:
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
			return
	def update(self,ymeas):
		Bkj = np.zeros(self.aki.shape[1])
		# compute the normalizing factor
		normalizingFactor = 0.0
		for j in range(self.aki.shape[1]):
			# evaluate Bkj
			Hk = self.measurementJacobian(self.aki[:,j],self.ts)
			Pkj = np.dot(Hk,np.dot(self.Pki[:,:,j],Hk.transpose())) + self.Rk
			Bkj[j] = gaussianNormalPdf(ymeas,self.measurementFunction(self.aki[:,j],self.ts),Pkj)
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
	##eval_pdf_x
	#
	# evaluate the PDF using the current Gaussian mixture model at a specified point x
	# @ param[in] xk [d-length] numpy vector at which to evaluate PDF
	def eval_pdf_x(self,xk):
		pk = 0.0
		for j in range(self.aki.shape[1]):
			pk = pk + self.alphai[j]*gaussianNormalPdf(xk,self.aki[:,j],self.Pki[:,:,j])
		return pk
	## get_pdf
	#
	# Return the value of the mixture PDF evaluated at each of the means
	# @ param[out] XK n x Ns numpy array of the current means in the mixture
	# @ param[out] pk Ns-length vector of the PDF value at each entry in XK
	def get_pdf(self):
		XK = self.aki.copy()
		pk = np.zeros(self.aki.shape[1])
		for k in range(self.aki.shape[1]):
			xk = XK[:,k].copy()
			pk[k] = self.eval_pdf_x(xk)
		return(XK,pk)
	## get_max_likelihood
	#
	# Find the current mean with the maximum PDF value associated
	# @param[out] Mean with the maximum likelihood
	def get_max_likelihood(self):
		(XK,pk) = self.get_pdf()
		idx = np.argmax(pk)
		return self.aki[:,idx]
	## resample from the kth Gaussian with probability alphai[k], and set the new weights to equal
	def resample(self):
		Aki = np.zeros(self.aki.shape)
		Ppki = np.zeros(self.Pki.shape)
		for k in range(self.aki.shape[1]):
			u1 = np.random.uniform(0,1)
			s = 0.0
			ju = 0
			for j in range(self.aki.shape[1]):
				s = s + self.alphai[j]
				ju = j
				if s > u1:
					break
			Aki[:,k] = np.random.multivariate_normal(self.aki[:,ju],self.Pki[:,:,ju])
			Ppki[:,:,k] = self.Pki[:,:,ju].copy()
		# resample the covariances also
		self.aki = Aki.copy()
		self.Pki = Ppki.copy()
		# equalize the weights
		self.alphai = 1.0/float(self.aki.shape[1])*np.ones(self.aki.shape[1])
	# resample using Markov chain Monte Carlo according to a normal proposal distribution
	def resample_mcmc(self):
		(XK,pk) = self.get_pdf()
		ix = np.argmax(pk)
		x0 = XK[:,ix]
		# eval the PDF at x0
		fx0 = self.eval_pdf_x(x0)
		for k in range(self.aki.shape[1]):
			# sample the next point
			xp = np.random.multivariate_normal(x0,np.identity(self.aki.shape[0]))
			# evaluate pdf at xp
			fxp = self.eval_pdf_x(xp)
			if fxp > fx0:
				x0 = xp.copy()
				fx0 = fxp
			else:
				# draw uniform on 0,1
				u1 = np.random.uniform(0,1)
				# go to xp if u1 < fxp/fx0
				if u1 < (fxp/fx0):
					x0 = xp.copy()
					fx0 = fxp
			self.aki[:,k] = x0.copy()