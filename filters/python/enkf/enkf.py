"""@package enkf
Ensemble Kalman Filter for systems with linear updates and (possibly) nonlinear propagation functions
"""

import numpy as np
import math
import scipy.integrate as sp
import scipy.linalg
# requires kmeans from filters/python/lib
import kmeans
# import scipy stats
import scipy.stats as stats

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

# clusterConvergence2Modes
#
# clusterConvergence algorithm for the case when there are either one or two modes
#
# @param[in] xk [p x N] numpy array of N points in p dimensions
# @param[in] activeMeans current number of means in the data set
def clusterConvergence2Modes(xk,activeMeans):
	# evaluate the fit of two means to the data
	testMeans = 2
	(idxk,mui) = kmeans.kmeans(xk.transpose(),testMeans)
	# dimension of the data
	p = xk.shape[0]
	# numnber of particles
	N = xk.shape[1]
	# evaluate the mean and covariance of each cluster
	meansk = np.zeros((p,testMeans))
	Pkkk = np.zeros((p,p,testMeans))

	# evaluate the mean and covariance of each cluster
	for k in range(testMeans):
		# compute the mean of all members where meansIdx == k
		idx = np.nonzero(idxk==k)
		idx = idx[0]
		meansk[:,k] = mui[k,:].transpose()
		# compute the covariance
		Pkk = np.zeros((p,p))
		coef = 1.0/(float(N)-1.0)
		for j in idx:
			Pkk = Pkk + coef*np.outer(xk[:,j]-meansk[:,k],xk[:,j]-meansk[:,k])
		Pkkk[:,:,k] = Pkk.copy()
	# evaluate the likelihood for each point, under the bimodal assumption
	L2 = np.zeros(N)
	for k in range(N):
		# assume the liklihood is proportional to the PDF
		pxk = gaussianNormalPdf(xk[:,j],meansk[:,idxk[k]],Pkkk[:,:,idxk[k]])
		L2[k] = pxk
	#Akaike information criterion for bimodal case
	AIC2 = 2.0*4 - 2.0*math.log(np.max(L2))
	# evaluate the likelihood under the monomodal assumption
	P11 = np.zeros((p,p))
	mux = np.mean(xk,axis=1)
	for k in range(N):
		P11 = P11 + (1.0)/(float(N)-1.0)*np.outer(xk[:,k]-mux,xk[:,k]-mux)
	L = np.zeros(N)
	for k in range(N):
		L[k] = gaussianNormalPdf(xk[:,j],mux,P11)
	# information criterion for unimodal case
	AIC = 2.0*2 - 2.0*math.log(np.max(L))
	# smaller AIC value is better
	print("AIC1 = %f,AIC2 = %f, L1 = %f, L2 = %f" % (AIC,AIC2,np.max(L),np.max(L2)))
	#if AIC < AIC2:
	if np.max(L) > np.max(L2):
		(idxk,mui) = kmeans.kmeans(xk.transpose(),1)
	return(idxk,mui)

# @param[in] xk [p x N] numpy array of N points in p dimensions
# @param[in] activeMeans current number of means in the data set
# @param[in] recursing boolean; set to 'True' by this function, when it calls itself to evaluate the goodness of fit with 1 more and 1 fewer mean. Do not set this argument when calling manually.
def clusterConvergenceCheck(xk,activeMeans,recursing=False):
	# evaluate the fit of the current number of means to the data
	(idxk,mui) = kmeans.kmeans(xk.transpose(),activeMeans)
	# dimension of the data
	p = xk.shape[0]
	# numnber of particles
	N = xk.shape[1]
	# evaluate the mean and covariance of each cluster
	meansk = np.zeros((p,activeMeans))
	Pkkk = np.zeros((p,p,activeMeans))
	# flag; set to True if we get a bad flag failure
	badFit = 0
	for k in range(activeMeans):
		# compute the mean of all members where meansIdx == k
		idx = np.nonzero(idxk==k)
		idx = idx[0]
		meansk[:,k] = np.mean(xk[:,idx],axis=1)
		# compute the covariance
		Pkk = np.zeros((p,p))
		coef = 1.0/(float(N)-1.0)
		for j in idx:
			Pkk = Pkk + coef*np.outer(xk[:,j]-meansk[:,k],xk[:,j]-meansk[:,k])
		Pkkk[:,:,k] = Pkk.copy()
		# index of points not in current cluster
		idnotx = np.setdiff1d(range(0,N),idx)
		# transform the points NOT in the current mean by Pkk^(-1/2)*(x_i - meansk[:,k])
		# L = Pkk(-1/2)
		L = np.real( scipy.linalg.sqrtm(np.linalg.inv(Pkk) ) )
		# evaluate how well points INSIDE the current mean fit the cluster
		badPts = 0
		for j in idx:
			# transform 
			yk = np.dot(L,xk[:,j]-meansk[:,k])
			#print(np.linalg.norm(yk))
			if np.linalg.norm(yk) > 3.0:
				badPts = badPts + 1
		if float(badPts)/len(idx) > 0.05:
			print("Too many member pts outside this cluster's 3sigma bounds: %f%%" % (float(badPts)/len(idx)) )
			badFit = 1
			break
		# evaluate how well points OUTSIDE the current cluster fit it
		for j in idnotx:
			# transform 
			yk = np.dot(L,xk[:,j]-meansk[:,k])
			#print(np.linalg.norm(yk))
			if np.linalg.norm(yk) < 3.0:
				badPts = badPts + 1
		if float(badPts)/len(idx) > 0.05:
			print("Too many outside points within this cluster's 3sigma bounds: %f%%" % (float(badPts)/len(idnotx)) )
			badFit = 2
			break
	print("%d means gets %d flag" % (activeMeans,badFit))
	# if called recursively, return immediately with the badFit flag
	if recursing:
		return(meansk,Pkkk,badFit)
	if badFit == 0:
		# the current fit is good; return the mean and covariance
		return(meansk,Pkkk,badFit)
	while badFit == 2:
		while activeMeans > 1:
			# try fewer clusters
			(meanskf,Pkkkf,badFitf) = clusterConvergenceCheck(xk,activeMeans-1,recursing=True)
			if badFitf == 0:
				#return immediately
				return(meansf,Pkkkf,badFitf)
			else:
				activeMeans = activeMeans - 1
				meansk = meanskf.copy()
				Pkkk = Pkkkf.copy()
				badFit = badFitf
		return(meansk,Pkkk,badFit)
	while badFit == 1:
		# try more clusters
		(meanskf,Pkkkf,badFitf) = clusterConvergenceCheck(xk,activeMeans+1,recursing=True)
		if badFitf == 0:
			#return immediately
			return(meansf,Pkkkf,badFitf)
		else:
			activeMeans = activeMeans + 1
			if activeMeans > 2:
				return(meansk,Pkkk,badFit)
			meansk = meanskf.copy()
			Pkkk = Pkkkf.copy()
			badFit = badFitf	

	# badFit == 1: the current fit failed with too many member points outside a cluster's 3sigma. This suggests fewer clusters??
	# badFit == 2: the current fit failed with too many external points inside a cluster's 3sigma. This suggests more clusters.

def convergenceCheck(XK,PXX,tol):
		# remove one sample, then check to see if the covariance changes more than the tolerancve
		mux = np.mean(XK[:,0:-1],axis=1)
		N = XK.shape[1]
		n = XK.shape[0]
		Pxx = np.zeros((n,n))
		coef = 1.0/(float(N)-2.0)
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
				coef = 1.0/(float(N)-2.0)
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
				coef = 1.0/(float(N)-2.0)
				for k in range(N-1):
					Pxx = Pxx + coef*np.outer(XK[:,k]-mux,XK[:,k]-mux)
				# convergence metric
				metric = np.linalg.norm(Pxx-PXX)/np.linalg.norm(PXX)
				print("Increased to %d pts, metric = %f" % (N,metric))
		return (XK,PXX)

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
	#
	# @param[in] dt delta-time over which to integrate
	# @param[in] dtout delta-time at which to return propagation values, for smoother plotting with long-period measurements
	# @param[out] XPROP value of the system ensemble members during simulation, discretized at dtout
	def propagateOde(self,dt,dtout = None):
		if dtout == None:
			dtout = dt
		# generate process noise for each sample, size is nv x N where nv is length of process noise vector
		vk = np.random.multivariate_normal(np.zeros(self._Qk.shape[0]), self._Qk, size=(self._N,)).transpose()
		# time vector for simulation
		tsim = np.arange(self.t,self.t+dt+dtout,dtout)
		# propagate each sample
		XPROP = np.zeros((tsim.shape[0],2,self._N))
		for k in range(self._N):
			y = sp.odeint(self._propagateFunction,self.xk[:,k],tsim,args=(self.u,vk[:,k],))
			XPROP[:,:,k] = y.copy()
			#y = sp.odeint(self._propagateFunction,self.xk[:,k],np.array([self.t,self.t+dt]),args=(self.u,vk[:,k],))
			self.xk[:,k] = y[-1,:].copy()
		# update the time
		self.t = self.t + dt
		return(XPROP)
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
		coef = 1.0/(float(self._N)-1.0)
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

## Ensemble Kalman filter derived class for the case where the measurement function is nonlinear. Uses the Unscented form of the update equations.
class nonlinear_enkf(enkf):
	def __init__(self,n=1,m=1,propagateFunction=None,measurementFunction=None,Qk=None,Rk=None,Ns=None):
		self.measurementFunction = measurementFunction
		enkf.__init__(self,n,m,propagateFunction,None,Qk,Rk,Ns)
	## update update the system in response to a measurement. Use the nonlinear measurement function in the innovation computation
	#
	# @param[in] ymeas measurement at current time
	def update(self,ymeas):
		# generate sample measurement noises from the covariance _Rk
		# dy is N x h where h is the number of outputs
		dy = np.random.multivariate_normal( np.zeros(self._Rk.shape[0]) ,self._Rk,size=(self._N,)).transpose()
		# compute the predicted mean
		xhatm = np.mean(self.xk,axis=1)
		# compute the state covariance Pxx
		coef = 1.0/(float(self._N)-1.0)
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		yexp = np.zeros((self._Rk.shape[0],self._N))
		for k in range(self._N):
			# compute the measurement expectation
			yexp[:,k] = self.measurementFunction(self.xk[:,k],self.t,self.u,dy[:,k])
		# compute the mean expectation
		yhat = np.mean(yexp,axis=1)
		# compute the output covariance and cross-covariance
		Pyy = np.zeros(self._Rk.shape)
		Pxy = np.zeros((self._n,self._Rk.shape[0]))
		coef = 1.0/(float(self._N)-1.0)
		for k in range(self._N):
			Pyy = Pyy + coef*np.outer(yexp[:,k]-yhat,yexp[:,k]-yhat)
			Pxy = Pxy + coef*np.outer(self.xk[:,k]-xhatm,yexp[:,k]-yhat)
		# Compute the Kalman gain
		Kk = np.dot(Pxy,np.linalg.inv(Pyy))
		# update each state
		for k in range(self._N):
			innov = Kk*(ymeas - yexp[:,k])
			for j in range(self._n):
				self.xk[j,k] = self.xk[j,k] + innov[j]

## clustering_enkf
#
# version of the ensemble Kalman filter that's modified for spatially clustered data and non-informative measurements
# Primary change: basic algorithms for handling covariance are modified to run kmeans clustering, and return the mean and covariance of multiple clusters if applicable
#
# @param[in] maxMeans maximum number of clusters into which to attempt partitioning data
class clustering_enkf(nonlinear_enkf):
	def __init__(self,n=1,m=1,propagateFunction=None,measurementFunction=None,Qk=None,Rk=None,Ns=None,maxMeans = 3):
		## number of clusters in use currently. Always initialize to 1 currently b/c of how the init() function works
		self.activeMeans = 1
		## max number of clusters allowed
		self.maxMeans = maxMeans
		## array for the system means
		self.means = np.zeros((n,maxMeans))
		## index of which system members (particles) are in which mean: all in mean 0 initially
		self.meansIdx = np.zeros(Ns).astype(int)
		## array for the system covariances
		self.covariances = np.zeros((n,n,maxMeans))
		# run normal initialization
		nonlinear_enkf.__init__(self,n,m,propagateFunction,measurementFunction,Qk,Rk,Ns)
	## initialization function for clustering EnKF
	#
	# initialize the means and covariances members after running standard init function
	def init(self,mux,Pxx,ts=0.0):
		nonlinear_enkf.init(self,mux,Pxx,ts)
		# initialize means
		for k in range(self.activeMeans):
			# compute the mean of all members where meansIdx == k
			idx = np.nonzero(self.meansIdx==k)
			idx = idx[0]
			self.means[:,k] = np.mean(self.xk[:,idx],axis=1)
			# compute the covariance
			Pkk = np.zeros((self._n,self._n))
			coef = 1.0/(float(self._N)-1.0)
			for j in idx:
				Pkk = Pkk + coef*np.outer(self.xk[:,j]-self.means[:,k],self.xk[:,j]-self.means[:,k])
			self.covariances[:,:,k] = Pkk.copy()
	## propagate forward the system by dt, then perform clustering on the updated states
	def propagateOde(self,dt,dtout = None):
		# use standard propagation
		nonlinear_enkf.propagateOde(self,dt,dtout)
		# call the convergenceCheck function
		(idxk,mui) = clusterConvergence2Modes(self.xk,self.activeMeans)
		print(mui)
		self.meansIdx = idxk
		self.activeMeans = mui.shape[0]
		self.means[:,0:self.activeMeans] = mui.transpose().copy()
		# update covariances
		for k in range(self.activeMeans):
			# compute the mean of all members where meansIdx == k
			idx = np.nonzero(self.meansIdx==k)
			idx = idx[0]
			# compute the covariance
			Pkk = np.zeros((self._n,self._n))
			coef = 1.0/(float(len(idx))-1.0)
			for j in idx:
				Pkk = Pkk + coef*np.outer(self.xk[:,j]-self.means[:,k],self.xk[:,j]-self.means[:,k])
			self.covariances[:,:,k] = Pkk.copy()
	## update update the system in response to a measurement
	#
	# @param[in] ymeas measurement at current time
	def update(self,ymeas):
		# generate sample measurement noises from the covariance _Rk
		# dy is N x h where h is the number of outputs
		dy = np.random.multivariate_normal( np.zeros(self._Rk.shape[0]) ,self._Rk,size=(self._N,)).transpose()
		yexp = np.zeros((self._Rk.shape[0],self._N))
		# individually update each cluster, based on the measurement
		for jk in range(self.activeMeans):
			idx = np.nonzero(self.meansIdx==jk)
			idx = idx[0]
			L = len(idx)
			# compute the predicted mean
			xhatm = np.mean(self.xk[:,idx],axis=1)
			# compute the state covariance Pxx
			coef = 1.0/(float(L)-1.0)
			Pxx = np.zeros((self._n,self._n))
			for k in idx:
				Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
			for k in idx:
				# compute the measurement expectation
				yexp[:,k] = self.measurementFunction(self.xk[:,k],self.t,self.u,dy[:,k])
			# compute the mean expectation
			yhat = np.mean(yexp[:,idx],axis=1)
			# compute the output covariance and cross-covariance
			Pyy = np.zeros(self._Rk.shape)
			Pxy = np.zeros((self._n,self._Rk.shape[0]))
			coef = 1.0/(float(L)-1.0)
			for k in idx:
				Pyy = Pyy + coef*np.outer(yexp[:,k]-yhat,yexp[:,k]-yhat)
				Pxy = Pxy + coef*np.outer(self.xk[:,k]-xhatm,yexp[:,k]-yhat)
			# Compute the Kalman gain
			Kk = np.dot(Pxy,np.linalg.inv(Pyy))
			# update each state
			for k in idx:
				innov = Kk*(ymeas - yexp[:,k])
				for j in range(self._n):
					self.xk[j,k] = self.xk[j,k] + innov[j]
			# store the means
			self.means[:,jk] = np.mean(self.xk[:,idx],axis=1)
			# compute the covariance
			Pkk = np.zeros((self._n,self._n))
			coef = 1.0/(float(L)-1.0)
			for j in idx:
				Pkk = Pkk + coef*np.outer(self.xk[:,j]-self.means[:,jk],self.xk[:,j]-self.means[:,jk])
			self.covariances[:,:,jk] = Pkk.copy()

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
		coef = 1.0/(float(self._N)-1.0)
		Pxx = np.zeros((self._n,self._n))
		for k in range(self._N):
			Pxx = Pxx + coef*np.outer(self.xk[:,k]-xhatm,self.xk[:,k]-xhatm)
		# call the convergence function
		(self.xk,Pxx) = convergenceCheck(self.xk,Pxx,self._convTol)
		self._N = self.xk.shape[1]
		print("Initialized adaptive EnKF with %d pts" % (self._N))
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
		coef = 1.0/(float(self._N)-1.0)
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


