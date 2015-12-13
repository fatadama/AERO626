"""@package ekf_trials
loads data, passes through EKF
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/ekf')
import ekf

sys.path.append('../sim_data')
import generate_data

sys.path.append('../trials')
import trials_processing

def eqom_ekf(x,t,u):
	return cp_dynamics.eqom_det(x,t)

def eqom_jacobian_ekf(x,t,u):
	return cp_dynamics.eqom_det_jac(x,t)

def eqom_gk_ekf(x,t,u):
	return cp_dynamics.eqom_det_Gk(x,t)

def measurement_ekf(x,t):
	return np.array([ x[0] ])

def measurement_gradient(x,t):
	return np.array([ [1.0,0.0] ])

def ekf_test(dt,tf,mux0,P0,YK,Qk,Rk):

	#Qk = np.array([[1.0*dt]])
	#Rk = np.array([[1.0]])

	# create EKF object
	EKF = ekf.ekf(2,0,eqom_ekf,eqom_jacobian_ekf,eqom_gk_ekf,Qk)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize EKF
	EKF.init_P(mux0,P0,ts)
	# initialize performance object
	simOut = trials_processing.simOutput()

	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,2,2))
	tk = np.arange(0.0,tf,dt)

	xf[0,:] = EKF.xhat.copy()
	Pf[0,:,:] = EKF.Pk.copy()

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# sync the EKF, with continuous-time integration
		EKF.propagateOde(dt)
		#EKF.propagateRK4(dt)
		EKF.update(ts,ym,measurement_ekf,measurement_gradient,Rk)
		# check that the eigenvalukes are reasonably bounded
		w = np.linalg.eigvalsh(EKF.Pk.copy())
		for jj in range(len(w)):
			if math.fabs(w[jj]) > 1.0e6:
				simOut.fail_singular_covariance(k)
				print("Covariance eigenvalue too large, t = %f" % (ts))
				return(xf,Pf,simOut)
		# copy
		xf[k,:] = EKF.xhat.copy()
		Pf[k,:,:] = EKF.Pk.copy()
		t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))
	simOut.complete(nSteps)

	return(xf,Pf,simOut)

def main():
	P0 = np.array([[2.0,0.0],[0.0,1.0]])
	Ns = 100
	Ts = 4.0
	(tsim,XK,YK,mu0,dt,tf) = generate_data.execute_sim(cp_dynamics.eqom_stoch_cluster,Ts,30*Ts,Ns,P0,cluster=True,informative=True)

	Qk = np.array([[0.005]])
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

		(xf,Pf,simOut) = ekf_test(dt,tf,mu0,P0,yk,Qk,Rk)

		if simOut.singular_covariance:
			print("Simulation exited with singular covariance at index %d" % (simOut.last_index))
			count_singular_covariance = count_singular_covariance + 1
			continue

		# compute the unit variance transformation of the error
		e1 = np.zeros((nSteps,2))
		chi2 = np.zeros(nSteps)
		(e1[0:simOut.last_index,:],chi2[0:simOut.last_index]) = trials_processing.computeErrors(xf[0:simOut.last_index,:],Pf[0:simOut.last_index,:,:],xk[0:simOut.last_index,:])
		nees_history[:,counter] = chi2.copy()
		mean_nees = np.sum(chi2)/float(nSteps)
		print(mean_nees)
		# mean NEES
		mse = np.sum(np.power(e1,2.0),axis=0)/float(nSteps)
		e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = e1.copy()

		if (mse[0] > 1.0) or (mse[1] > 1.0):
			count_large_errors = count_large_errors + 1
			continue
		count_good = count_good + 1
		print("MSE: %f,%f" % (mse[0],mse[1]))

		# chi-square test statistics
		# (alpha) probability of being less than the returned value: stats.chi2.ppf(alpha,df=Nsims)
	if Ns < 2:
		trials_processing.printSingleSim(tsim,xf,Pf,xk,name='ekf',save_flag=None,history_lines=True,draw_snapshots=False)
	#trials_processing.errorParsing(e_sims,nees_history,'ekf','sims_01_bifurcation')
	
	# write to file
	# Ns, count_good, count_singular_covariance, count_large_errors, Qk[0,0], Ts
	fname = 'ekf_data.txt'
	FID = open(fname,'a')
	FID.write("%d,%d,%d,%d,%g,%g\n" % (Ns,count_good,count_singular_covariance,count_large_errors,Qk[0,0],Ts))
	FID.close()

	print("Leaving ekf_trials")

	return

if __name__ == "__main__":
    main()
