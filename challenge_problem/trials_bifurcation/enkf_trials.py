"""@package enkf_trials
loads data, passes through ensemble Kalman Filter
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/enkf')
sys.path.append('../../filters/python/lib')
import enkf

sys.path.append('../sim_data')
import generate_data

sys.path.append('../trials')
import trials_processing

def eqom_enkf(x,t,u,v):
	return cp_dynamics.eqom_stoch(x,t,v)

def enkf_test(dt,tf,mux0,P0,YK,Qk,Rk,flag_adapt=False):
	global nameBit

	# measurement influence matrix
	Hk = np.array([ [1.0,0.0] ])

	eqom_use = eqom_enkf
	
	if flag_adapt:
		ENKF = enkf.adaptive_enkf(2,0,eqom_use,Hk,Qk,Rk,Ns=100)
	else:
		# create nonadaptive EnKF object
		ENKF = enkf.enkf(2,0,eqom_use,Hk,Qk,Rk,Ns=100)

	nSteps = int(tf/dt)+1
	ts = 0.0

	#initialize EnKF
	ENKF.init(mux0,P0,ts)
	# initialize performance object
	simOut = trials_processing.simOutput()

	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,2,2))
	Nf = np.zeros(nSteps)
	XK = np.zeros((nSteps,2,ENKF._N))
	tk = np.arange(0.0,tf,dt)

	#get the mean and covariance estimates
	Nf[0] = ENKF.get_N()
	xf[0,:] = np.mean(ENKF.xk,axis=1)
	Pxx = np.zeros((2,2))
	for k in range(ENKF.get_N()):
		Pxx = Pxx + 1.0/(1.0+float(ENKF._N))*np.outer(ENKF.xk[:,k]-xf[0,:],ENKF.xk[:,k]-xf[0,:])
	Pf[0,:,:] = Pxx.copy()

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# sync the ENKF, with continuous-time integration
		# propagate filter
		ENKF.propagateOde(dt)
		#ENKF.propagate(dt)
		# update
		ENKF.update(ym)
		# log
		xf[k,:] = np.mean(ENKF.xk,axis=1)
		Pxx = np.zeros((2,2))
		for kj in range(ENKF.get_N()):
			Pxx = Pxx + 1.0/(float(ENKF._N)-1.0)*np.outer(ENKF.xk[:,kj]-xf[k,:],ENKF.xk[:,kj]-xf[k,:])
		Pf[k,:,:] = Pxx.copy()
		Nf[k] = ENKF.get_N()
		# check that the eigenvalukes are reasonably bounded
		w = np.linalg.eigvalsh(Pf[k,:,:].copy())
		for jj in range(len(w)):
			if math.fabs(w[jj]) > 1.0e6:
				simOut.fail_singular_covariance(k)
				print("Covariance eigenvalue too large, t = %f" % (ts))
				return(xf,Pf,Nf,XK,simOut)
		if not flag_adapt:
			XK[k,:,:] = ENKF.xk.copy()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))
	simOut.complete(nSteps)

	return(xf,Pf,Nf,XK,simOut)

def main():
	P0 = np.array([[2.0,0.0],[0.0,1.0]])
	Ns = 100
	Ts = 3.5
	(tsim,XK,YK,mu0,dt,tf) = generate_data.execute_sim(cp_dynamics.eqom_stoch_cluster,Ts,30*Ts,Ns,P0,cluster=True,informative=True)
	flag_adapt = False

	Qk = np.array([[0.01]])
	#Qk = np.array([[0.01*Ts*0.1]])
	Rk = np.array([[0.01]])
	# number of steps in each simulation
	nSteps = len(tsim)
	nees_history = np.zeros((nSteps,Ns))
	Nf_history = np.zeros((nSteps,Ns))
	e_sims = np.zeros((Ns*nSteps,2))
	count_good = 0
	count_singular_covariance = 0
	count_large_errors = 0
	for counter in range(Ns):
		xk = XK[:,(2*counter):(2*counter+2)]
		yk = YK[:,counter]

		(xf,Pf,Nf,XKO,simOut) = enkf_test(dt,tf,mu0,P0,yk,Qk,Rk,flag_adapt)

		if simOut.singular_covariance:
			print("Simulation exited with singular covariance at index %d" % (simOut.last_index))
			count_singular_covariance = count_singular_covariance + 1
			continue

		# store the number of particles, relevant if adaptive
		Nf_history[:,counter] = Nf.copy()
		# errors
		e1 = np.zeros((nSteps,2))
		chi2 = np.zeros(nSteps)
		(e1[0:simOut.last_index,:],chi2[0:simOut.last_index]) = trials_processing.computeErrors(xf[0:simOut.last_index,:],Pf[0:simOut.last_index,:,:],xk[0:simOut.last_index,:])
		nees_history[:,counter] = chi2.copy()
		mean_nees = np.sum(chi2)/float(nSteps)
		print(mean_nees)
		# mean NEES
		mse = np.sum(np.power(e1,2.0),axis=0)/float(nSteps)
		# get the mean number of particles in time
		Nf_mean = np.sum(Nf_history,axis=1)/Ns
		e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = e1.copy()

		if (mse[0] > 1.0) or (mse[1] > 1.0):
			count_large_errors = count_large_errors + 1
			continue
		count_good = count_good + 1
		print("MSE: %f,%f" % (mse[0],mse[1]))
	if Ns < 2:
		trials_processing.printSingleSim(tsim,xf,Pf,xk,name='enkf',save_flag=None,history_lines=True,draw_snapshots=False)
	if flag_adapt:
		fig = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
		tilt = "AENKF, Ts = %.2f, %d sims, " % (dt, Ns)
		ax = fig.add_subplot(111,ylabel='mean particles',title=tilt)
		ax.plot(tsim,Nf_mean,'b-')
		ax.grid()
		fig.show()
		raw_input("Return to close")

	#trials_processing.errorParsing(e_sims,nees_history,'enkf','sims_01_bifurcation')
	
	# write to file
	# Ns, count_good, count_singular_covariance, count_large_errors, Qk[0,0], Ts
	if not flag_adapt:
		fname = 'enkf_data.txt'
		FID = open(fname,'a')
		FID.write("%d,%d,%d,%d,%g,%g\n" % (Ns,count_good,count_singular_covariance,count_large_errors,Qk[0,0],Ts))
		FID.close()
	else:
		fname = 'aenkf_data.txt'
		FID = open(fname,'a')
		FID.write("%d,%d,%d,%d,%g,%g\n" % (Ns,count_good,count_singular_covariance,count_large_errors,Qk[0,0],Ts))
		FID.close()

	print("Leaving enkf_trials")

	return


if __name__ == "__main__":
	main()