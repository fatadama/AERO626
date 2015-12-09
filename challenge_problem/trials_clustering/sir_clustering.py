"""@package sir_trials
loads data, passes through SIR (sampling importance resampling filter)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats
import scipy.linalg # for sqrtm() function
import scipy.integrate as sp

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/sis_particle')
sys.path.append('../../filters/python/sir_particle')
sys.path.append('../../filters/python/lib')
import sir
import kmeans

sys.path.append('../sim_data')
import data_loader

import cluster_processing

## Function for the filter to propagate a particle.
#
#@param[in] x initial state
#@param[in] dt time interval over which to propagate
#@param[in] v process noise (piecewise-constant) over which to integrate
#@param[out] xk the propagated state at time t+dt
#
# We're treating the process noise as piecewise constant - let's try to get it to work that way, and if it won't we'll use or modify ode_wrapper to sample at cp_dynamics.DT while integrating
def eqom_use(x,dt,v=None):
	# continuous-time integration
    ysim = sp.odeint(cp_dynamics.eqom_stoch_cluster,x,np.array([0,dt]),args=(v,))
    # return new state
    xk = ysim[-1,:].transpose()
    return xk

## Function for the filter that returns an appropriate process noise sample
#
#@param[global] Qk the process noise used in this file, set locally depending on measurement sample period
def processNoise(xk):
	global Qu
	muk = np.array([0.0])
	# draw from the normal distribution	
	vk = np.random.multivariate_normal(muk,Qu)
	return vk

## Function for the filter to compute the pdf value associated with a measurement, given a prior
#
#   @param[in] yt: the measurement
#   @param[in] xk: the prior (a particle)
#	@param[global] Rk: the measurement noise, set outside this function as a global
#   @param[out] py_x: the PDF of yt, given xk
def measurementPdf(yt,xk):
	global Ru
	Rkinv = np.linalg.inv(Ru)
	# we're measuring position, so the expectation yk is simply xk[0]:
	yk = xk[0]
	# compute the error w.r.t. the measurement
	nk = yt[0] - yk
	# evaluate the Gaussian PDF with mean 0.0 and standard deviation Rk
	py_x = math.exp(-0.5*np.dot(nk,np.dot(Rkinv,nk)))/(math.sqrt(2.0*math.pi*np.linalg.det(Ru)))
	return py_x

## Function that returns the value of the pdf assocaited with non-informative measurements of position squared
def measurementPdfNoninformative(yt,xk):
	global Ru
	Rkinv = np.linalg.inv(Ru)
	# we're measuring position squared, so the expectation yk is simply xk[0]:
	yk = xk[0]*xk[0]
	# compute the error w.r.t. the measurement
	nk = yt[0] - yk
	# evaluate the Gaussian PDF with mean 0.0 and standard deviation Rk
	py_x = math.exp(-0.5*np.dot(nk,np.dot(Rkinv,nk)))/(math.sqrt(2.0*math.pi*np.linalg.det(Ru)))
	return py_x

## Function that samples from a Gaussian initial distribution to get a single initial particle
def initialParticle():
	global mux_sample
	global P_sample
	xr = np.random.multivariate_normal(mux_sample,P_sample)
	return xr

def sir_test(dt,tf,mux0,P0,YK,Qk,Rk,Nparticles = 100,informative=True):
	global nameBit
	global mux_sample
	global P_sample
	global Ru
	global Qu
	Ru = Rk.copy()
	Qu = Qk.copy()
	mux_sample = mux0.copy()
	P_sample = P0.copy()

	# number of particles
	Nsu = Nparticles

	if informative:
		SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdf)
	else:
		SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdfNoninformative)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize the particle filter
	SIR.init(initialParticle)
	
	# propagated particles
	Xp = np.zeros((nSteps,2,Nsu))
	# the weights
	weights = np.zeros((nSteps,SIR.Ns))
	# weights after resampling
	weightss = np.zeros((nSteps,SIR.Ns))

	weights[0,:] = SIR.WI.copy()
	Xp[0,:,:] = SIR.XK.copy()
	## particles after resampling
	Xs = np.zeros((nSteps,2,Nsu))

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# call SIR
		SIR.update(dt,ym,1.0e-2)
		print("Propagate to t = %f in %f sec" % (ts,time.time()-t1))
		# store
		weights[k,:] = SIR.WI.copy()
		Xp[k,:,:] = SIR.XK.copy()
		# resample
		SIR.sample()
		## store resampled points
		Xs[k,:,:] = SIR.XK.copy()
		weightss[k,:] = SIR.WI.copy()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	# compute the mean and covariance over time - this is the propagated state
	mux = np.zeros((nSteps,2))
	Pxx = np.zeros((nSteps,4))
	for k in range(nSteps):
		mux[k,0] = np.sum( np.multiply(Xp[k,0,:],weights[k,:]) )
		mux[k,1] = np.sum( np.multiply(Xp[k,1,:],weights[k,:]) )
		Pxk = np.zeros((2,2))
		for j in range(Nsu):
			iv = np.array([ Xp[k,0,j]-mux[k,0],Xp[k,1,j]-mux[k,1] ])
			Pxk = Pxk + weights[k,j]*np.outer(iv,iv)
			Pxx[k,:] = Pxk.reshape((4,))
	# compute the aposteriori mean
	muxs = np.zeros((nSteps,2))
	for k in range(nSteps):
		muxs[k,0] = np.sum( np.multiply(Xs[k,0,:],weightss[k,:]) )
		muxs[k,1] = np.sum( np.multiply(Xs[k,1,:],weightss[k,:]) )

	return(mux,Pxx,Xp,weights,Xs,muxs)

def main():
	# number of particles, 300 seems necessary for pretty good consistent performance
	Nsu = 300
	global nameBit
	names = ['sims_01_bifurcation_noninformative']
	flag_informative=False
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		'''
		tsim = tsim[0:2]
		XK = XK[0:2,:]
		YK = YK[0:2,:]
		tf = tsim[1]
		'''
		Ns = 1

		nameBit = int(nameNow[5:7],2)
		# parse the name
		if nameBit == 1:
			# noise levels for the SIR with white noise forcing
			Qk = np.array([[3.16]])
			Rk = np.array([[0.1]])
		'''
		if nameBit == 2:
			# noise levels for the SIR with cosine forcing
			Qk = np.array([[31.6]])
			Rk = np.array([[0.01]])
		'''
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,Pf,Xp,weights,Xs,xs) = sir_test(dt,tf,mu0,P0,yk,Qk,Rk,Nsu,flag_informative)
			# call PF cluster processing function
			weightsEqual = np.ones((nSteps,Xs.shape[2]))*1.0/float(Xs.shape[2])
			(e1,chi2,mxnu,Pxnu) = cluster_processing.singleSimErrorsPf(Xp,weights,xk)

			# chi2 is the NEES statistic. Take the mean
			nees_history[:,counter] = chi2.copy()
			#mean_nees = np.sum(chi2)/float(nSteps)
			mean_nees = np.mean(chi2)
			print(mean_nees)
			# mean NEES
			mse = np.sum(np.power(e1,2.0),axis=0)/float(nSteps)
			e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = e1.copy()
			print("sir_clustering case %d" % counter)
			print("MSE: %f,%f" % (mse[0],mse[1]))
		if Ns < 2:
			# loop over the particles after sampling and cluster using kmeans
			# errors for the bifurcated case
			'''
			e2case = np.zeros((2,nSteps,2))
			for k in range(nSteps):
				# cluster into two means
				(idxk,mui) = kmeans.kmeans(Xs[k,:,:].transpose(),2)
				# compute the errors for the two means
				for jk in range(2):
					e2case[jk,k,:] = mui[jk,:]-xk[k,:]
			'''
			# plot of the discrete PDF and maximum likelihood estimate
			# len(tsim) x Ns matrix of times
			tMesh = np.kron(np.ones((Nsu,1)),tsim).transpose()
			fig1 = plt.figure()
			ax = []
			for k in range(6):
				if k < 2:
					nam = 'pdf' + str(k+1)
				elif k < 4:
					nam = 'xf' + str(k-1)
				else:
					nam = 'ef' + str(k-3)
				ax.append( fig1.add_subplot(3,2,k+1,ylabel=nam) )
				if k < 2:
					if k == 0:
						# plot the discrete PDF as a function of time
						mex = tMesh.reshape((len(tsim)*Nsu,))
						mey = Xp[:,0,:].reshape((len(tsim)*Nsu,))
						mez = weights.reshape((len(tsim)*Nsu,))
					elif k == 1:
						# plot the discrete PDF as a function of time
						mex = tMesh.reshape((len(tsim)*Nsu,))
						mey = Xp[:,1,:].reshape((len(tsim)*Nsu,))
						mez = weights.reshape((len(tsim)*Nsu,))
					idx = mez.argsort()
					mexx,meyy,mezz = mex[idx],mey[idx],mez[idx]

					cc = ax[k].scatter(mexx,meyy,c=mezz,s=20,edgecolor='')
					fig1.colorbar(cc,ax=ax[k])
					# plot the truth
					ax[k].plot(tsim,xk[:,k],'b-')
				elif k < 4:
					ax[k].plot(tsim,xf[:,k-2],'m--')
					ax[k].plot(tsim,xk[:,k-2],'b-')
				else:
					#ax[k].plot(tsim,xf[:,k-4]-xk[:,k-4],'b-')
					# plot the error based on maximum likelihood
					ax[k].plot(tsim,e1[:,k-4],'y-')
					ax[k].plot(tsim,3.0*np.sqrt(Pxnu[:,k-4,k-4]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pxnu[:,k-4,k-4]),'r--')
					#ax[k].plot(tsim,e2case[0,:,k-4],'y-')
					#ax[k].plot(tsim,e2case[1,:,k-4],'y-')
					#ax[k].plot(tsim,3.0*np.sqrt(Pf[:,k-4 + 2*(k-4)]),'r--')
					#ax[k].plot(tsim,-3.0*np.sqrt(Pf[:,k-4 + 2*(k-4)]),'r--')
				ax[k].grid()

			fig1.show()

		mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
		print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))

		if Ns == 1:
				fig = []
				for k in range(nSteps):
					fig.append(plt.figure())
					ax = fig[k].add_subplot(1,1,1,title="t = %f" % (tsim[k]),xlim=(-25,25),ylim=(-20,20),ylabel='x2',xlabel='x1')
					ax.plot(Xp[k,0,:],Xp[k,1,:],'bd')#propagated values
					ax.plot(Xs[k,0,:],Xs[k,1,:],'ys')#re-sampled values
					#compute the number of active means
					# plot the truth state
					ax.plot(xk[k,0],xk[k,1],'cs')
					# plot the maximum likelihood cluster mean and covariance ellipse
					# plot the single-mean covariance ellipsoid
					# draw points on a unit circle
					'''
					thetap = np.linspace(0,2*math.pi,20)
					circlP = np.zeros((20,2))
					circlP[:,0] = 3.0*np.cos(thetap)
					circlP[:,1] = 3.0*np.sin(thetap)
					# transform the points circlP through P^(1/2)*circlP + mu
					Phalf = np.real(scipy.linalg.sqrtm(Pxnu[k,:,:]))
					ellipsP = np.zeros(circlP.shape)
					for kj in range(circlP.shape[0]):
						ellipsP[kj,:] = np.dot(Phalf,circlP[kj,:])+mxnu[k,:]
					ax.plot(ellipsP[:,0],ellipsP[:,1],'r--')
					'''
					ax.plot(mxnu[k,0],mxnu[k,1],'rd')
					'''
					meansIdx = Idx[k,:].copy()
					activeMeans = 1
					if np.any(meansIdx > 0):
						activeMeans = 2
					for jk in range(activeMeans):
						idx = np.nonzero(meansIdx==jk)
						idx = idx[0]
						mux = np.mean(Xf[k,:,idx],axis=0)
						Pxx = np.zeros((2,2))
						for kj in idx:
							Pxx = Pxx + 1.0/(float(len(idx))-1.0)*np.outer(Xf[k,:,kj]-mux,Xf[k,:,kj]-mux)
						mux0 = np.mean(Xp[k,:,idx],axis=0)
						Pxx0 = np.zeros((2,2))
						for kj in idx:
							Pxx0 = Pxx0 + 1.0/(float(len(idx))-1.0)*np.outer(Xp[k,:,kj]-mux0,Xp[k,:,kj]-mux0)
						if jk == 0:
							ax.plot(Xf[k,0,idx],Xf[k,1,idx],'mo')
							ax.plot(Xp[k,0,idx],Xp[k,1,idx],'bd')
						else:
							ax.plot(Xf[k,0,idx],Xf[k,1,idx],'co')
							ax.plot(Xp[k,0,idx],Xp[k,1,idx],'rd')
						# plot the single-mean covariance ellipsoid
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
						if jk == 0:
							ax.plot(ellipsP[:,0],ellipsP[:,1],'m--')
						else:
							ax.plot(ellipsP[:,0],ellipsP[:,1],'c--')
						# transform the points circlP through P^(1/2)*circlP + mu
						Phalf = np.real(scipy.linalg.sqrtm(Pxx0))
						ellipsP = np.zeros(circlP.shape)
						for kj in range(circlP.shape[0]):
							ellipsP[kj,:] = np.dot(Phalf,circlP[kj,:])+mux
						if jk == 0:
							ax.plot(ellipsP[:,0],ellipsP[:,1],'b--')
						else:
							ax.plot(ellipsP[:,0],ellipsP[:,1],'r--')
						# plot the truth state
						ax.plot(xk[k,0],xk[k,1],'ks')
					'''

					fig[k].show()
				raw_input("Return to quit")
				for k in range(nSteps):
					fig[k].savefig('stepByStep/sir_' + str(Nsu) + "_" + str(k) + '.png')
					plt.close(fig[k])
		else:
			# get the mean NEES value versus simulation time across all sims
			nees_mean = np.sum(nees_history,axis=1)/Ns
			# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
			chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
			chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)

			# plot the mean NEES with the 95% confidence bounds
			fig2 = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
			tilt = "SIR, Ts = %.2f, %d sims, %d particles, " % (dt, Ns, Nsu)
			if nameBit == 0:
				tilt = tilt + 'unforced'
			if nameBit == 1:
				#white-noise only
				tilt = tilt + 'white-noise forcing'
			if nameBit == 2:
				tilt = tilt + 'cosine forcing'
			if nameBit == 3:
				#white-noise and cosine forcing
				tilt = tilt + 'white-noise and cosine forcing'
			ax = fig2.add_subplot(111,ylabel='mean NEES')#,title=tilt)
			ax.set_title(tilt,fontsize = 12)
			ax.plot(tsim,chiUpper*np.ones(nSteps),'r--')
			ax.plot(tsim,chiLower*np.ones(nSteps),'r--')
			ax.plot(tsim,nees_mean,'b-')
			ax.grid()
			fig2.show()
			# save the figure
			fig2.savefig('nees_sir_' + str(Nsu) + "_" + nameNow + '.png')
			# find fraction of inliers
			l1 = (nees_mean < chiUpper).nonzero()[0]
			l2 = (nees_mean > chiLower).nonzero()[0]
			# get number of inliers
			len_in = len(set(l1).intersection(l2))
			# get number of super (above) liers (sic)
			len_super = len((nees_mean > chiUpper).nonzero()[0])
			# get number of sub-liers (below)
			len_sub = len((nees_mean < chiLower).nonzero()[0])

			print("Conservative (below 95%% bounds): %f" % (float(len_sub)/float(nSteps)))
			print("Optimistic (above 95%% bounds): %f" % (float(len_super)/float(nSteps)))

			# save metrics
			FID = open('metrics_sir_' + str(Nsu) + "_" + nameNow + '.txt','w')
			FID.write("mse1,mse2,nees_below95,nees_above95\n")
			FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
			FID.close()

	print("Leaving sir_clustering")

	return

if __name__ == "__main__":
    main()
