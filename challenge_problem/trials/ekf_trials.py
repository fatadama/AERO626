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
import data_loader

def eqom_ekf(x,t,u):
	return cp_dynamics.eqom_stoch(x,t)

def eqom_jacobian_ekf(x,t,u):
	return cp_dynamics.eqom_stoch_jac(x,t)

def eqom_gk_ekf(x,t,u):
	return cp_dynamics.eqom_stoch_Gk(x,t)

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

	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,4))
	tk = np.arange(0.0,tf,dt)

	xf[0,:] = EKF.xhat.copy()
	Pf[0,:] = EKF.Pk.reshape((4,))

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# sync the EKF, with continuous-time integration
		EKF.propagate(dt)
		#EKF.propagateRK4(dt)
		EKF.update(ts,ym,measurement_ekf,measurement_gradient,Rk)
		# copy
		xf[k,:] = EKF.xhat.copy()
		Pf[k,:] = EKF.Pk.reshape((4,))
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(xf,Pf)

def main():
	#names = ['sims_01_slow']# test case
	names = ['sims_01_slow','sims_01_medium','sims_01_fast']
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')
		namebit = int(nameNow[5:7])
		# parse the name
		if namebit == 1:
			# this heuristic produces a reasonable balance between conservative and optimistic at all three sample rates, but the performance at the slow rate still sucks. It is stable, though.
			Qk = np.array([[1.0 + 50.0*(dt-0.01)-40.0*(dt-0.01)*(dt-0.01)]])
			Rk = np.array([[1.0]])
		print(Qk[0,0])
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,Pf) = ekf_test(dt,tf,mu0,P0,yk,Qk,Rk)

			# compute the unit variance transformation of the error
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

			# chi-square test statistics
			# (alpha) probability of being less than the returned value: stats.chi2.ppf(alpha,df=Nsims)
		if Ns < 2:
			fig1 = plt.figure()
			ax = []
			for k in range(4):
				if k < 2:
					nam = 'x' + str(k+1)
				else:
					nam = 'e' + str(k-1)
				ax.append(fig1.add_subplot(2,2,k+1,ylabel=nam))
				if k < 2:
					ax[k].plot(tsim,xk[:,k],'b-')
					ax[k].plot(tsim,xf[:,k],'m--')
					if k == 0:
						ax[k].plot(tsim,yk,'r--')
				else:
					ax[k].plot(tsim,xk[:,k-2]-xf[:,k-2])
					ax[k].plot(tsim,3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
				ax[k].grid()
			fig1.show()

		mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
		print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
		
		# get the mean NEES value versus simulation time across all sims
		nees_mean = np.sum(nees_history,axis=1)/Ns
		# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
		chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
		chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)

		# plot the mean NEES with the 95% confidence bounds
		fig2 = plt.figure(figsize=(5.0,2.81)) #figsize tuple is width, height
		tilt = "EKF, Ts = %.2f, %d sims, " % (dt, Ns)
		if namebit == 0:
			tilt = tilt + 'unforced'
		if namebit == 1:
			#white-noise only
			tilt = tilt + 'white-noise forcing'
		if namebit == 2:
			tilt = tilt + 'cosine forcing'
		if namebit == 3:
			#white-noise and cosine forcing
			tilt = tilt + 'white-noise and cosine forcing'
		ax = fig2.add_subplot(111,ylabel='mean NEES',title=tilt)
		ax.plot(tsim,chiUpper*np.ones(nSteps),'r--')
		ax.plot(tsim,chiLower*np.ones(nSteps),'r--')
		ax.plot(tsim,nees_mean,'b-')
		ax.grid()
		fig2.show()
		# save the figure
		fig2.savefig('nees_ekf_' + nameNow + '.png')

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
		FID = open('metrics_ekf_' + nameNow + '.txt','w')
		FID.write("mse1,mse2,nees_below95,nees_above95\n")
		FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
		FID.close()

		# plot all NEES
		'''
		fig = plt.figure(figsize=(5.0,2.81))
		ax = fig.add_subplot(111,ylabel='NEES')
		ax.plot(tsim,nees_history,'b-')
		ax.grid()
		fig.show()
		'''

	raw_input("Return to quit")

	print("Leaving ekf_trials")

	return

if __name__ == "__main__":
    main()
