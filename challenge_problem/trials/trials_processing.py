"""@package trials_processing.py
parse error and NEES histories into output formats, and saves
"""

import numpy as np
import math
import scipy.stats as stats # for chi-sqaured functions
import matplotlib.pyplot as plt
import scipy

## simOutput class used by the ****_trials.py files to indicate the simulation performance of a particular run.
class simOutput():
	def __init__(self):
		self.completed = False
		self.singular_covariance = False
		## index of the last successful simulation step
		self.last_index = -1
	def fail_singular_covariance(self,k):
		self.singular_covariance = True
		self.last_index = k
	def complete(self,k):
		self.completed = True
		self.last_index = k

# everything gets parsed into e_sims, a nSteps*2 array
# and nees_history, a nSteps*nSims array of the NEES values

## compute Errors - given a filter's best estimate, compute its error and NEES statistics
# @param[in] Xf - [nSteps x 2] estimated state at each time
# @param[in] Pf - [nSteps x 2 x 2] covariance history for plotting the error covariance
# @param[in] xk - [nSteps x 2] truth states
# @param[out] e1 -[nSteps x 2] error time history
# @param[out] chi2 - nSteps-length vector of NEES (normalized expectation of error squared) time history
def computeErrors(Xf,Pf,xk):
	nSteps = Xf.shape[0]
	e1 = Xf-xk
	chi2 = np.zeros(nSteps)
	for k in range(nSteps):
		# compute the covariance
		Pxx = Pf[k,:,:].copy()
		# compute NEES
		Pinv = np.linalg.inv(Pxx)
		chi2[k] = np.dot(e1[k,:],np.dot(Pinv,e1[k,:]))
	return(e1,chi2)

## Parses standard xxx_trials.py output into NEES statistics, MSE
# logs raw error statistics
# 
# Writes to files of the format "<fileid>_<filtername>(_<mods>)_<nameNow>.xxx",
# 	where <fileid> is "metrics", "nees", "raw", "Nf", and <mods> is optional
#
# @param[in] e_sims [(nSteps*nSims) x 2] numpy array containing the error in at each time in each simulation;
# @param[in] nees_history [nSteps x nSims] numpy array of the NEES statistic in each simulation at each time; simulations have length nSteps
# @param[in] filtername identifying acronym for the current filter, used to name output files
# @param[in] nameNow name for the data loaded to produce this simulation; used to name output files
# @param[in] mods (optional) string that gets added to the output file; e.g. the number of particles for a particular run for a filter
def errorParsing(e_sims,nees_history,filtername,nameNow,mods=None):
	# number of steps
	nSteps = nees_history.shape[0]
	# number of simulations
	Ns = nees_history.shape[1]
	# compute the MSE across all sims
	mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
	# compute the normally distributed, unit variance chi-sqaure statistic
	chi_statistic = 1.0/math.sqrt(Ns)*np.sum((nees_history-2.0)/2.0,axis=1)
	# find where chi_statistic < 3 == 3sigma bound
	idBounded = np.nonzero(np.fabs(chi_statistic) <= 3.0)
	idBounded = idBounded[0]
	nCons = np.nonzero(chi_statistic < -3.0)
	nOpt = np.nonzero(chi_statistic > 3.0)

	nGood = len(idBounded)
	nBad = nSteps-nGood

	print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
	print("Fraction of bad points: %f, %d sims" % (float(nBad)/float(nSteps),Ns))

	# write raw to file
	fname = 'raw_' + filtername + "_"
	if mods is not None:
		fname = fname + str(mods) + "_"
	fname = fname + nameNow + ".txt"
	FID = open(fname,'w')
	# write the number of steps and number of sims
	FID.write("%d,%d\n" % (nSteps,Ns))
	for k in range(nSteps):
		for j in range(Ns):
			FID.write("%g," % nees_history[k,j])
		FID.write("\n")
	FID.close()

	plotting = False
	if plotting:
		# plot the chi-sqaured variables
		fig = plt.figure()
		ax = fig.add_subplot(111,xlabel='Chi-squared statistic')
		ax.hist(chi_statistic,bins=25)
		fig.show()
		raw_input("Return to close")
		plt.close(fig)
	return

def covarianceEllipse(mux,Pxx):
	# draw points on a unit circle
	thetap = np.linspace(0,2*math.pi,20)
	circlP = np.zeros((20,2))
	circlP[:,0] = 3.0*np.cos(thetap)
	circlP[:,1] = 3.0*np.sin(thetap)
	# transform the points circlP through P^(1/2)*circlP + mu
	Phalf = np.real(scipy.linalg.sqrtm(Pxx))
	ellipsP = np.zeros(circlP.shape)
	for kj in range(circlP.shape[0]):
		ellipsP[kj,:] = np.dot(Phalf,circlP[kj,:])+mux
	return ellipsP

## printSingleSim - generate plots of a simulation at sequential time steps, with options for passing in particle swarm estimates
# @param[in] tf - nSteps-length time array, for labels
# @param[in] Xf - [nSteps x 2] estimated state at each time
# @param[in] Pf - [nSteps x 2 x 2] covariance history for plotting the error covariance
# @param[in] xp - (optional) [nSteps x 2 x nParticles] array of particle time histories
# @param[in] idx - (optional) [nSteps x nParticles] array of cluster membership... not used in this version
# @param[in] save_flag - (optional) path to write output figures to. If None don't save
# @param[in] history_lines - trace the path history of the truth and estimate in the snapshot plots
# @param[in] draw_snapshots - set to False to only draw the error history plot
def printSingleSim(tf,Xf,Pf,xk,xp=None,idx=None,name='filter',save_flag=None,history_lines=False,draw_snapshots=True):
	nSteps = Xf.shape[0]
	if draw_snapshots:
		fig = []
		# generate sequential phase portraits
		for k in range(nSteps):
			fig.append(plt.figure())
			ax = fig[k].add_subplot(1,1,1,title="t = %f" % (tf[k]),ylabel='x2',xlabel='x1')#,xlim=(-25,25),ylim=(-20,20),ylabel='x2',xlabel='x1')
			mux = Xf[k,:].copy()
			Pxx = Pf[k,:,:].copy()
			ellipsP = covarianceEllipse(mux,Pxx)
			# plot estimate
			ax.plot(Xf[k,0],Xf[k,1],'bo')
			if history_lines:
				ax.plot(Xf[1:(k+1),0],Xf[1:(k+1),1],'b-')
				ax.plot(xk[1:(k+1),0],xk[1:(k+1),1],'c-')
			# plot covariance
			ax.plot(ellipsP[:,0],ellipsP[:,1],'b--')
			# plot the truth state
			ax.plot(xk[k,0],xk[k,1],'cs')
			ax.grid()
			fig[k].show()

	# generate sequential time histories
	fig2 = plt.figure()
	ax = []

	'''
	enorm = np.zeros((nSteps,2))
	for k in range(nSteps):
		# compute P^(-1/2) * (x-xhat)
		Pxx = Pf[k,:,:].copy()
		Phalf = np.real(scipy.linalg.sqrtm(Pxx))
		enorm[k,:] = np.dot(np.linalg.inv(Phalf),Xf[k,:]-xk[k,:])
	'''

	for j in range(2):
		lab = 'e' + str(j+1)
		ax.append( fig2.add_subplot(1,2,j+1,title="Error histories for %s" % (name.upper()),ylabel=lab,xlabel='t'))
		# plot estimate
		ax[j].plot(tf,Xf[:,j]-xk[:,j],'b-')
		# plot covariance
		ax[j].plot(tf,3.0*np.sqrt(Pf[:,j,j]),'r--')
		ax[j].plot(tf,-3.0*np.sqrt(Pf[:,j,j]),'r--')
		#ax[j].plot(tf,enorm[:,j],'b-')
		#ax[j].plot(tf,np.ones(nSteps),'r--')
		#ax[j].plot(tf,-1.0*np.ones(nSteps),'r--')
		# plot grid
		ax[j].grid()
	fig2.show()

	raw_input("Return to close")

	if draw_snapshots:
		for k in range(nSteps):
			if save_flag is not None:
				fig[k].savefig(name + '_' + str(k) + '.png')
			plt.close(fig[k])
	plt.close(fig2)