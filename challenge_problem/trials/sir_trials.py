"""@package sir_trials
loads data, passes through SIR (sampling importance resampling filter)
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats
import scipy.integrate as sp

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/sis_particle')
#import sis
sys.path.append('../../filters/python/sir_particle')
sys.path.append('../../filters/python/lib')
import sir

sys.path.append('../sim_data')
import data_loader

## Function for the filter to propagate a particle.
#
#@param[in] x initial state
#@param[in] dt time interval over which to propagate
#@param[in] v process noise (piecewise-constant) over which to integrate
#@param[out] xk the propagated state at time t+dt
#
# We're treating the process noise as piecewise constant - let's try to get it to work that way, and if it won't we'll use or modify ode_wrapper to sample at cp_dynamics.DT while integrating
def eqom_use(x,dt,v):
	# continuous-time integration
    ysim = sp.odeint(cp_dynamics.eqom_stoch,x,np.array([0,dt]),args=(v,))
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

## Function that samples from a Gaussian initial distribution to get a single initial particle
def initialParticle():
	global mux_sample
	global P_sample
	xr = np.random.multivariate_normal(mux_sample,P_sample)
	return xr

def sir_test(dt,tf,mux0,P0,YK,Qk,Rk,Nparticles = 100):
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

	# add in this functionality so we can change the propagation function dependent on the nameBit ... may or may not be needed
	if nameBit == 1:
		# create SIR object
		SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdf)
	elif nameBit == 2:
		# create SIR object
		SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdf)
	elif nameBit == 3:
		# create SIR object
		SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdf)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize the particle filter
	SIR.init(initialParticle)
	
	# the estimate (weighted mean)
	#xf = np.zeros((nSteps,2))
	#tk = np.arange(0.0,tf,dt)
	px1 = np.zeros((nSteps,SIR.Ns))
	px2 = np.zeros((nSteps,SIR.Ns))
	weights = np.zeros((nSteps,SIR.Ns))

	px1[0,:] = SIR.XK[0,:].copy()
	px2[0,:] = SIR.XK[1,:].copy()
	weights[0,:] = SIR.WI.copy()

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# call SIR
		SIR.update(dt,ym)
		# store
		px1[k,:] = SIR.XK[0,:].copy()
		px2[k,:] = SIR.XK[1,:].copy()
		weights[k,:] = SIR.WI.copy()
		# resample
		SIR.sample()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	# sort out the most likely particle at each time
	xml = np.zeros((nSteps,2))
	for k in range(nSteps):
		idxk = np.argmax(weights[k,:])
		xml[k,0] = px1[k,idxk]
		xml[k,1] = px2[k,idxk]
	# compute the mean and covariance over time
	mux = np.zeros((nSteps,2))
	Pxx = np.zeros((nSteps,4))
	for k in range(nSteps):
		mux[k,0] = np.sum( np.multiply(px1[k,:],weights[k,:]) )
		mux[k,1] = np.sum( np.multiply(px2[k,:],weights[k,:]) )
		Pxk = np.zeros((2,2))
		for j in range(Nsu):
			iv = np.array([ px1[k,j]-mux[k,0],px2[k,j]-mux[k,1] ])
			Pxk = Pxk + weights[k,j]*np.outer(iv,iv)
			Pxx[k,:] = Pxk.reshape((4,))

	return(mux,Pxx,px1,px2,weights)

def main():
	# number of particles
	Nsu = 100
	global nameBit
	names = ['sims_01_fast']# test case
	#names = ['sims_01_slow','sims_01_medium','sims_01_fast']
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		Ns = 1

		nameBit = int(nameNow[5:7])
		# parse the name
		if nameBit == 1:
			# tuned noise levels for the SIR with white noise forcing
			Qk = np.array([[1.0*dt]])
			if dt < 0.09:
				Qk = np.array([[10.0/dt]])
			Rk = np.array([[1.0]])
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,Pf,px1,px2,weights) = sir_test(dt,tf,mu0,P0,yk,Qk,Rk,Nsu)
			# TODO compute the unit variance transformation of the error
			e1 = np.zeros((nSteps,2))
			chi2 = np.zeros(nSteps)
			for k in range(nSteps):
				P = Pf[k,:].reshape((2,2))
				Pinv = np.linalg.inv(P)
				chi2[k] = np.dot(xk[k,:]-xf[k,:],np.dot(Pinv,xk[k,:]-xf[k,:]))
			# chi2 is the NEES statistic. Take the mean
			nees_history[:,counter] = chi2.copy()
			mean_nees = np.sum(chi2)/float(nSteps)
			print(mean_nees)
			# mean NEES
			mse = np.sum(np.power(xk-xf,2.0),axis=0)/float(nSteps)
			e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = xk-xf

			print("MSE: %f,%f" % (mse[0],mse[1]))
		if Ns < 2:
			# plot of the discrete PDF and maximum likelihood estimate
			# len(tsim) x Ns matrix of times
			tMesh = np.kron(np.ones((Nsu,1)),tsim).transpose()
			fig = plt.figure()
			ax = []
			for k in range(6):
				if k < 2:
					nam = 'pdf' + str(k+1)
				elif k < 4:
					nam = 'xf' + str(k-1)
				else:
					nam = 'ef' + str(k-3)
				ax.append( fig.add_subplot(3,2,k+1,ylabel=nam) )
				if k < 2:
					if k == 0:
						# plot the discrete PDF as a function of time
						mex = tMesh.reshape((len(tsim)*Nsu,))
						mey = px1.reshape((len(tsim)*Nsu,))
						mez = weights.reshape((len(tsim)*Nsu,))
					elif k == 1:
						# plot the discrete PDF as a function of time
						mex = tMesh.reshape((len(tsim)*Nsu,))
						mey = px2.reshape((len(tsim)*Nsu,))
						mez = weights.reshape((len(tsim)*Nsu,))
					idx = mez.argsort()
					mexx,meyy,mezz = mex[idx],mey[idx],mez[idx]

					cc = ax[k].scatter(mexx,meyy,c=mezz,s=20,edgecolor='')
					fig.colorbar(cc,ax=ax[k])
					# plot the truth
					ax[k].plot(tsim,xk[:,k],'b-')
				elif k < 4:
					ax[k].plot(tsim,xf[:,k-2],'m--')
					ax[k].plot(tsim,xk[:,k-2],'b-')
				else:
					ax[k].plot(tsim,xf[:,k-4]-xk[:,k-4],'b-')
					ax[k].plot(tsim,3.0*np.sqrt(Pf[:,k-4 + 2*(k-4)]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pf[:,k-4 + 2*(k-4)]),'r--')
				ax[k].grid()

			fig.show()

		mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
		print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
		
		# get the mean NEES value versus simulation time across all sims
		nees_mean = np.sum(nees_history,axis=1)/Ns
		# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
		chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
		chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)

		# plot the mean NEES with the 95% confidence bounds
		fig2 = plt.figure(figsize=(5.0,2.81)) #figsize tuple is width, height
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

	raw_input("Return to quit")

	print("Leaving sir_trials")

	return

if __name__ == "__main__":
    main()
