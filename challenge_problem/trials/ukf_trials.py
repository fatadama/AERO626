"""@package ukf_trials
loads data, passes through UKF
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/ukf')
import ukf

sys.path.append('../sim_data')
import data_loader

import trials_processing

def eqom_ukf(x,t,u,v):
	return cp_dynamics.eqom_stoch(x,t,v)

def measurement_ukf(x,t,n):
	return np.array([[ x[0] + n[0] ]])

def ukf_test(dt,tf,mux0,P0,YK,Qk,Rk):
	global nameBit

	# add in this functionality so we can change the propagation function dependent on the nameBit ... may or may not be needed
	if nameBit == 1:
		# create UKF object
		UKF = ukf.ukf(2,0,1,eqom_ukf,Qk)
	elif nameBit == 2:
		# create UKF object
		UKF = ukf.ukf(2,0,1,eqom_ukf,Qk)
	elif nameBit == 3:
		# create UKF object
		UKF = ukf.ukf(2,0,1,eqom_ukf,Qk)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize UKF
	UKF.init_P(mux0,P0,ts)

	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,4))
	tk = np.arange(0.0,tf,dt)

	xf[0,:] = UKF.xhat.copy()
	Pf[0,:] = UKF.Pk.reshape((4,))

	t1 = time.time()
	for k in range(1,nSteps):
		# get the new measurement
		ym = np.array([YK[k]])
		ts = ts + dt
		# sync the UKF, with continuous-time integration
		UKF.sync(dt,ym,measurement_ukf,Rk,True)
		# copy
		#if k < nSteps-1:
		xf[k,:] = UKF.xhat.copy()
		Pf[k,:] = UKF.Pk.reshape((4,))
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(xf,Pf)

def main():
	global nameBit
	#names = ['sims_11_slow']# test case
	names = ['sims_10_medium','sims_01_medium','sims_11_medium']
	#names = ['sims_01_slow','sims_01_medium','sims_01_fast','sims_10_slow','sims_10_medium','sims_10_fast','sims_11_slow','sims_11_medium','sims_11_fast']
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		nameBit = int(nameNow[5:7],2)
		# parse the name
		if nameBit == 1:
			Rk = np.array([[1.0]])
			# tuned UKF with white noise forcing
			if dt > .9:# slow sampling
				Qk = np.array([[0.00316]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[0.01]])
			else:# fast sampling
				Qk = np.array([[0.00316]])
		if nameBit == 2:
			# tuned noise levels for the UKF with cosine forcing
			if dt > .9:# slow sampling
				Qk = np.array([[0.25]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[1.5]])
			else:# fast sampling
				Qk = np.array([[20.0]])
			Rk = np.array([[1.0]])
		if nameBit == 3:
			# noise levels for the UKF with cosine forcing and white noise
			if dt > .9:# slow sampling
				Qk = np.array([[0.25]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[1.5]])
			else:# fast sampling
				Qk = np.array([[16.0]])
			Rk = np.array([[1.0]])
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,Pf) = ukf_test(dt,tf,mu0,P0,yk,Qk,Rk)

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

			fig2 = plt.figure()
			ax = []
			ax.append(fig2.add_subplot(111,ylabel = 'nees metric'))
			ax[0].plot(tsim,chi2)
			ax[0].grid()
			fig2.show()
		else:
			trials_processing.errorParsing(e_sims,nees_history,'ukf',nameNow)

			mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
			print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
			
			# get the mean NEES value versus simulation time across all sims
			nees_mean = np.sum(nees_history,axis=1)/Ns
			# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
			chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
			chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)

			# plot the mean NEES with the 95% confidence bounds
			fig2 = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
			tilt = "UKF, Ts = %.2f, %d sims, " % (dt, Ns)
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
			ax = fig2.add_subplot(111,ylabel='mean NEES',title=tilt)
			ax.plot(tsim,chiUpper*np.ones(nSteps),'r--')
			ax.plot(tsim,chiLower*np.ones(nSteps),'r--')
			ax.plot(tsim,nees_mean,'b-')
			ax.grid()
			fig2.show()
			# save the figure
			fig2.savefig('nees_ukf_' + nameNow + '.png')
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
			FID = open('metrics_ukf_' + nameNow + '.txt','w')
			FID.write("mse1,mse2,nees_below95,nees_above95\n")
			FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
			FID.close()

			# plot all NEES
			fig = plt.figure(figsize=(6.0,3.37))
			ax = fig.add_subplot(111,ylabel='NEES')
			ax.plot(tsim,nees_history,'b-')
			ax.grid()
			fig.show()

	raw_input("Return to quit")

	print("Leaving ukf_trials")
	return

if __name__ == "__main__":
    main()
