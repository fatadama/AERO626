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
import generate_data

sys.path.append('../trials')
import trials_processing

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
	# create SIR object
	SIR = sir.sir(2,Nsu,eqom_use,processNoise,measurementPdf)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize the particle filter
	SIR.init(initialParticle)
	# initialize performance object
	simOut = trials_processing.simOutput()
	
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
	simOut.complete(nSteps)

	# sort out the most likely particle at each time
	xml = np.zeros((nSteps,2))
	for k in range(nSteps):
		idxk = np.argmax(weights[k,:])
		xml[k,0] = px1[k,idxk]
		xml[k,1] = px2[k,idxk]
	# compute the mean and covariance over time
	mux = np.zeros((nSteps,2))
	Pxx = np.zeros((nSteps,2,2))
	for k in range(nSteps):
		mux[k,0] = np.sum( np.multiply(px1[k,:],weights[k,:]) )
		mux[k,1] = np.sum( np.multiply(px2[k,:],weights[k,:]) )
		Pxk = np.zeros((2,2))
		for j in range(Nsu):
			iv = np.array([ px1[k,j]-mux[k,0],px2[k,j]-mux[k,1] ])
			Pxk = Pxk + weights[k,j]*np.outer(iv,iv)
			Pxx[k,:,:] = Pxk.copy()

	return(mux,Pxx,px1,px2,weights,simOut)

def main():
	P0 = np.array([[2.0,0.0],[0.0,1.0]])
	Ns = 100
	Ts = 5.0
	(tsim,XK,YK,mu0,dt,tf) = generate_data.execute_sim(cp_dynamics.eqom_stoch_cluster,Ts,30*Ts,Ns,P0,cluster=True,informative=True)

	# number of particles
	Nsu = 200
		
	Qk = np.array([[0.1]])
	Rk = np.array([[0.01]])
	# number of steps in each simulation
	nSteps = len(tsim)
	nees_history = np.zeros((nSteps,Ns))
	e_sims = np.zeros((Ns*nSteps,2))
	count_good = 0
	count_singular_covariance = 0
	count_large_errors = 0
	for counter in range(Ns):
		xk = XK[:,(2*counter):(2*counter+2)]
		yk = YK[:,counter]

		(xf,Pf,px1,px2,weights,simOut) = sir_test(dt,tf,mu0,P0,yk,Qk,Rk,Nsu)

		if simOut.singular_covariance:
			print("Simulation exited with singular covariance at index %d" % (simOut.last_index))
			count_singular_covariance = count_singular_covariance + 1
			continue

		# errors
		e1 = xk-xf
		# mean NEES
		mse = np.sum(np.power(xk-xf,2.0),axis=0)/float(nSteps)
		e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = xk-xf

		print("MSE: %f,%f" % (mse[0],mse[1]))
		if (mse[0] > 1.0) or (mse[1] > 1.0):
			count_large_errors = count_large_errors + 1
			continue
		count_good = count_good + 1
	if Ns < 2:
		trials_processing.printSingleSim(tsim,xf,Pf,xk,name='sir',save_flag=None,history_lines=True,draw_snapshots=False)
	
	fname = 'sir_data.txt'
	FID = open(fname,'a')
	FID.write("%d,%d,%d,%d,%g,%g\n" % (Ns,count_good,count_singular_covariance,count_large_errors,Qk[0,0],Ts))
	FID.close()

	raw_input("Return to quit")

	print("Leaving sir_trials")

	return

if __name__ == "__main__":
    main()
