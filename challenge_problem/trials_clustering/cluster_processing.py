"""@package cluster_processing
parses cluster data into NEES and MSE statistics
"""

import numpy as np
import math
import scipy.stats as stats # for chi-sqaured functions
import scipy.linalg # for sqrtm() function
import sys

sys.path.append('../../filters/python/lib')

import kmeans

## Returns the value of the Gaussian multivariate distribution at point x with mean mu and covariance P
#
# @param[in] x point at which to evaluate PDF
# @param[in] mu mean of underlying distribution
# @param[in] P covariance of distribution
# @param[out] p the pdf evaluated at x
def gaussianNormalPdf(x,mu,P):
	mr = np.linalg.matrix_rank(P)
	dtP = np.linalg.det(P)
	if mr == P.shape[0] and ( dtP > 0.0 ) :
		Pinv = np.linalg.inv(P)
		d = x.shape[0]
		dp = np.dot(x-mu,np.dot(Pinv,x-mu))
		dt = 1.0/math.sqrt(math.pow(2.0*math.pi,float(d))*dtP)
		p = math.exp(-0.5*dp)*dt
		return p
	else:
		#matrix is singular; if x == mu, then the probability is 1
		if np.linalg.norm(x-mu) > 1.0e-4:
			return 0.0
		else:
			return 1.0

## Compute the NEES and MSE associated with the aposteriori updates
# works for the Ensemble Cluster case
# @param[in] Xf [nSteps * 2 * Np] numpy array where nSteps is the simulation length and Np is the number of ensemble particles
# @param[in] Idx [nSteps * Np] numpy array that identifies the cluster membership of each point in the ensemble at each time in the simulation
# @param[in] x [nSteps * 2] array of the truth state for each simulation
# @param[in] y [nSteps * 1] array of the measurements for each point in time
# @param[out] e1 [nSteps * 2] array of the estimation errors. When the system has bifurcated, a random cluster is chosen to to compute the error
# @param[out] chi2 [nSteps]-length vector of the NEES values at every time in the simulation. When 2 clusters are present one is chosen randomly to compute the NEES.
# @param[out] mx [nSteps * 2] array of the mean at each time
# @param[out] Pk [nSteps * 2 * 2] array of covariance estimates, logs the same one used in NEES computation
def singleSimErrors(Xf,Idx,x,y):
	nSteps = Xf.shape[0]
	e1 = np.zeros((nSteps,2))
	chi2 = np.zeros(nSteps)
	Pk = np.zeros((nSteps,2,2))
	mx = np.zeros((nSteps,2))
	for k in range(nSteps):
		#compute the number of active means
		meansIdx = Idx[k,:].copy()
		activeMeans = 1
		if np.any(meansIdx > 0):
			activeMeans = 2
		# index of which mean to use, 0 is default (unimodal)
		idu = 0
		'''
		# if activeMeans == 2, pick randomly which cluster to use to compute the error
		idu = 0
		if activeMeans == 2:
			u1 = np.random.uniform()
			if u1 > 0.5:
				idu = 1
		'''
		# if activeMeans == 2, evaluate the likelihood of the measurement, given the cluster.
		if activeMeans == 2:
			# evaluate the expectation of each cluster
			mu2 = np.zeros((2,2))
			yexp = np.zeros((2,1))
			for jk in range(activeMeans):
				idx = np.nonzero(meansIdx==jk)
				idx = idx[0]
				print(np.mean(Xf[k,:,idx],axis=0))
				mu2[jk,:] = np.mean(Xf[k,:,idx],axis=0).transpose()
				yexp[jk,0] = mu2[jk,0]*mu2[jk,0]
			#print("E1: %g, E2: %g" % (yexp[0,0]-y[k],yexp[1,0]-y[k]))
			if np.fabs(yexp[0,0]-y[k]) > np.fabs(yexp[1,0]-y[k]):
				idu = 1
		#get index of points in the current cluster
		axu = np.nonzero(meansIdx==idu)
		axu = axu[0]
		mux = np.mean(Xf[k,:,axu],axis=0)
		# compute the error
		e1[k,:] = mux - x[k,:]
		# compute the covariance
		coef = 1.0/(float(len(axu))-1.0)
		Pxx = np.zeros((2,2))
		for kj in axu:
			Pxx = Pxx + coef*np.outer(Xf[k,:,kj]-mux,Xf[k,:,kj]-mux)
		# log the mean
		mx[k,:] = mux.copy()
		# log the covariance that we chose
		Pk[k,:,:] = Pxx.copy()
		# compute NEES
		Pinv = np.linalg.inv(Pxx)
		chi2[k] = np.dot(e1[k,:],np.dot(Pinv,e1[k,:]))
	return(e1,chi2,mx,Pk)

## Compute the MSE and NEES metrics for a single particle filter run
# Use kmeans to cluster and decide on 1 or 2 clusters based on maximum likelihood criterion.
# @param[in] Xf [nSteps * 2 * Ns] array of the propagated (but pre-resampled) particles
# @param[in] weights [nSteps * Ns] array of particle weights
# @param[in] xk [nSteps * 2] simulation truth states
# @param[out] e1 [nSteps * 2] array of the estimation errors. When the system has bifurcated, a random cluster is chosen to to compute the error
# @param[out] chi2 [nSteps]-length vector of the NEES values at every time in the simulation. When 2 clusters are present one is chosen randomly to compute the NEES.
# @param[out] mx [nSteps * 2] array of the mean at each time
# @param[out] Pk [nSteps * 2 * 2] array of covariance estimates, logs the same one used in NEES computation
def singleSimErrorsPf(Xf,weights,xk):
	nSteps = Xf.shape[0]
	Ns = Xf.shape[2]
	e1 = np.zeros((nSteps,2))
	chi2 = np.zeros(nSteps)
	Pk = np.zeros((nSteps,2,2))
	mx = np.zeros((nSteps,2))
	for k in range(nSteps):
		# cluster the current data
		(idx2,mu2) = kmeans.kmeans(Xf[k,:,:].transpose(),2)
		# compute the covariance of each cluster
		L2max = 0.0
		L2x = np.zeros(2)
		L2P = np.zeros((2,2))
		for jk in range(2):
			idx = np.nonzero(idx2==jk)
			idx = idx[0]
			muk = 0.0
			for j in idx:
				muk = muk + weights[k,j]*Xf[k,:,j]
			Pxk = np.zeros((2,2))
			for j in idx:
				Pxk = Pxk + weights[k,j]*np.outer(Xf[k,:,j]-muk,Xf[k,:,j]-muk)
			#evaluate the likelihood of each point
			for j in idx:
				likelihoodv = gaussianNormalPdf(Xf[k,:,j],muk,Pxk)
				if likelihoodv > L2max:
					L2max = likelihoodv
					L2x = Xf[k,:,j].copy()
					L2P = Pxk.copy()
		# evaluate the single-mode likelihood
		mu1 = 0.0
		for j in range(Ns):
			mu1 = mu1 + weights[k,j]*Xf[k,:,j]
		Px1 = np.zeros((2,2))
		for j in range(Ns):
			Px1 = Px1 + weights[k,j]*np.outer(Xf[k,:,j]-mu1,Xf[k,:,j]-mu1)
		#evaluate the likelihood
		L1max = 0.0
		L1x = np.zeros(2)
		L1P = Pxk.copy()
		for j in range(Ns):
			likelihoodv = gaussianNormalPdf(Xf[k,:,j],mu1,Px1)
			if likelihoodv > L1max:
				L1max = likelihoodv
				L1x = Xf[k,:,j].copy()
		# determine if 1 or 2 modes is more likely
		if L1max > L2max:
			Lmax = L1max
			Lx = L1x.copy()
			LP = L1P.copy()
		else:
			Lmax = L2max
			Lx = L2x.copy()
			LP = L2P.copy()
		# log the mean
		mx[k,:] = Lx.copy()
		# log the covariance that we chose
		Pk[k,:,:] = LP.copy()
		# evaluate the error and NEES using the maximum likelihood point
		e1[k,:] = Lx-xk[k,:]
		# compute NEES
		mr = np.linalg.matrix_rank(LP)
		if mr == 2:
			Pinv = np.linalg.inv(LP)
			chi2[k] = np.dot(e1[k,:],np.dot(Pinv,e1[k,:]))
		else:
			print("Singular covariance matrix at index %d" % (k))
			chi2[k] = np.nan
	return(e1,chi2,mx,Pk)

## Compute the MSE and NEES metrics for a single gauss mixture model cluster simulation
# @param[in] Xf [nSteps * 2 * Ns] array of the propagated (but pre-resampled) particles
# @param[in] weights [nSteps * Ns] array of particle weights
# @param[in] xk [nSteps * 2] simulation truth states
# @param[out] e1 [nSteps * 2] array of the estimation errors. When the system has bifurcated, a random cluster is chosen to to compute the error
# @param[out] chi2 [nSteps]-length vector of the NEES values at every time in the simulation. When 2 clusters are present one is chosen randomly to compute the NEES.
# @param[out] mx [nSteps * 2] array of the mean at each time
# @param[out] Pk [nSteps * 2 * 2] array of covariance estimates, logs the same one used in NEES computation
def singleSimErrorsGmm(Xf,weights,xk):
	nSteps = Xf.shape[0]
	Ns = Xf.shape[2]
	e1 = np.zeros((nSteps,2))
	chi2 = np.zeros(nSteps)
	Pk = np.zeros((nSteps,2,2))
	mx = np.zeros((nSteps,2))
	for k in range(nSteps):
		# find the largest weight
		im = np.argmax(weights[k,:])
		return


		# cluster the current data
		(idx2,mu2) = kmeans.kmeans(Xf[k,:,:].transpose(),2)
		# compute the covariance of each cluster
		L2max = 0.0
		L2x = np.zeros(2)
		L2P = np.zeros((2,2))
		for jk in range(2):
			idx = np.nonzero(idx2==jk)
			idx = idx[0]
			muk = 0.0
			for j in idx:
				muk = muk + weights[k,j]*Xf[k,:,j]
			Pxk = np.zeros((2,2))
			for j in idx:
				Pxk = Pxk + weights[k,j]*np.outer(Xf[k,:,j]-muk,Xf[k,:,j]-muk)
			#evaluate the likelihood of each point
			for j in idx:
				likelihoodv = gaussianNormalPdf(Xf[k,:,j],muk,Pxk)
				if likelihoodv > L2max:
					L2max = likelihoodv
					L2x = Xf[k,:,j].copy()
					L2P = Pxk.copy()
		# evaluate the single-mode likelihood
		mu1 = 0.0
		for j in range(Ns):
			mu1 = mu1 + weights[k,j]*Xf[k,:,j]
		Px1 = np.zeros((2,2))
		for j in range(Ns):
			Px1 = Px1 + weights[k,j]*np.outer(Xf[k,:,j]-mu1,Xf[k,:,j]-mu1)
		#evaluate the likelihood
		L1max = 0.0
		L1x = np.zeros(2)
		L1P = Pxk.copy()
		for j in range(Ns):
			likelihoodv = gaussianNormalPdf(Xf[k,:,j],mu1,Px1)
			if likelihoodv > L1max:
				L1max = likelihoodv
				L1x = Xf[k,:,j].copy()
		# determine if 1 or 2 modes is more likely
		if L1max > L2max:
			Lmax = L1max
			Lx = L1x.copy()
			LP = L1P.copy()
		else:
			Lmax = L2max
			Lx = L2x.copy()
			LP = L2P.copy()
		# log the mean
		mx[k,:] = Lx.copy()
		# log the covariance that we chose
		Pk[k,:,:] = LP.copy()
		# evaluate the error and NEES using the maximum likelihood point
		e1[k,:] = Lx-xk[k,:]
		# compute NEES
		mr = np.linalg.matrix_rank(LP)
		if mr == 2:
			Pinv = np.linalg.inv(LP)
			chi2[k] = np.dot(e1[k,:],np.dot(Pinv,e1[k,:]))
		else:
			print("Singular covariance matrix at index %d" % (k))
			chi2[k] = np.nan
	return(e1,chi2,mx,Pk)