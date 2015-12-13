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
import data_loader

import trials_processing

def eqom_enkf(x,t,u,v):
	return cp_dynamics.eqom_stoch(x,t,v)

def enkf_test(dt,tf,mux0,P0,YK,Qk,Rk,flag_adapt=False):
	global nameBit

	# measurement influence matrix
	Hk = np.array([ [1.0,0.0] ])

	if nameBit == 1:
		eqom_use = eqom_enkf
	if nameBit == 2:
		eqom_use = eqom_enkf
	if nameBit == 3:
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

	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,4))
	Nf = np.zeros(nSteps)
	XK = np.zeros((nSteps,2,ENKF._N))
	tk = np.arange(0.0,tf,dt)

	#get the mean and covariance estimates
	Nf[0] = ENKF.get_N()
	xf[0,:] = np.mean(ENKF.xk,axis=1)
	Pxx = np.zeros((2,2))
	for k in range(ENKF.get_N()):
		Pxx = Pxx + 1.0/(1.0+float(ENKF._N))*np.outer(ENKF.xk[:,k]-xf[0,:],ENKF.xk[:,k]-xf[0,:])
	Pf[0,:] = Pxx.reshape((4,))

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
		# resample ??
		#ENKF.resample()
		# log
		xf[k,:] = np.mean(ENKF.xk,axis=1)
		Pxx = np.zeros((2,2))
		for kj in range(ENKF.get_N()):
			Pxx = Pxx + 1.0/(float(ENKF._N)-1.0)*np.outer(ENKF.xk[:,kj]-xf[k,:],ENKF.xk[:,kj]-xf[k,:])
		Pf[k,:] = Pxx.reshape((4,))
		Nf[k] = ENKF.get_N()
		if not flag_adapt:
			XK[k,:,:] = ENKF.xk.copy()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(xf,Pf,Nf,XK)

def main():
	global nameBit
	names = ['sims_10_fast']
	#names = ['sims_01_slow','sims_01_medium','sims_10_slow','sims_10_medium','sims_11_slow','sims_11_medium']# test case
	flag_adapt = False
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		Ns = 1

		nameBit = int(nameNow[5:7],2)
		# parse the name
		if nameBit == 1:
			# tuned noise levels for the ENKF with white noise forcing
			if dt > 0.9:# slow sampling
				Qk = np.array([[1.0]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[0.1]])
			else:# fast sampling
				Qk = np.array([[0.001]])
			Rk = np.array([[1.0]])
		if nameBit == 2:
			# tuned noise levels for the ENKF with cosine forcing
			if dt > 0.9:# slow sampling
				Qk = np.array([[6.0]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[30.0]])
			else:# fast sampling
				Qk = np.array([[100.0]])
			Rk = np.array([[1.0]])
		if nameBit == 3:
			# tuned noise levels for the ENKF with cosine forcing and white noise
			if dt > 0.9:# slow sampling
				Qk = np.array([[5.0]])
			elif dt > 0.09:# medium sampling
				Qk = np.array([[20.0]])
			else:# fast sampling
				Qk = np.array([[160.0]])
			Rk = np.array([[1.0]])
		# number of steps in each simulation
		print(Qk[0,0])

		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		Nf_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,Pf,Nf,XKO) = enkf_test(dt,tf,mu0,P0,yk,Qk,Rk,flag_adapt)

			# store the number of particles, relevant if adaptive
			Nf_history[:,counter] = Nf.copy()
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
			print("ENKF sim %d/%d case %d/%d" % (counter+1,Ns,namecounter+1,len(names)))
		if Ns < 2:
			fig1 = plt.figure()
			ax = []
			for k in range(6):
				if k < 2:
					nam = 'x' + str(k+1)
				elif k < 4:
					nam = 'e' + str(k-1)
				else:
					nam = 'xp' + str(k-3)
				ax.append(fig1.add_subplot(3,2,k+1,ylabel=nam))
				if k < 2:
					ax[k].plot(tsim,xk[:,k],'b-')
					ax[k].plot(tsim,xf[:,k],'m--')
					if k == 0:
						ax[k].plot(tsim,yk,'r--')
				elif k < 4:
					ax[k].plot(tsim,xk[:,k-2]-xf[:,k-2])
					ax[k].plot(tsim,3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
				else:
					ax[k].plot(tsim,xk[:,k-4],'b-')
					ax[k].plot(tsim,XKO[:,k-4,:],'d')
				ax[k].grid()
			fig1.show()

		if flag_adapt:
			trials_processing.errorParsing(e_sims,nees_history,'aenkf',nameNow)
		else:
			trials_processing.errorParsing(e_sims,nees_history,'enkf',nameNow)

		mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
		print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
		
		# get the mean NEES value versus simulation time across all sims
		nees_mean = np.sum(nees_history,axis=1)/Ns
		# get the mean number of particles in time
		Nf_mean = np.sum(Nf_history,axis=1)/Ns
		# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
		chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
		chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)

		# plot the mean NEES with the 95% confidence bounds
		fig2 = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
		if flag_adapt:
			tilt = "AENKF, Ts = %.2f, %d sims, " % (dt, Ns)
		else:
			tilt = "ENKF, Ts = %.2f, %d sims, " % (dt, Ns)
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
		if flag_adapt:
			fig2.savefig('nees_aenkf_' + nameNow + '.png')
		else:
			fig2.savefig('nees_enkf_' + nameNow + '.png')
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
		if flag_adapt:
			FID = open('metrics_aenkf_' + nameNow + '.txt','w')
		else:
			FID = open('metrics_enkf_' + nameNow + '.txt','w')
		FID.write("mse1,mse2,nees_below95,nees_above95\n")
		FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
		FID.close()

		# plot the mean number of particles
		if flag_adapt:
			fig = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
			tilt = "AENKF, Ts = %.2f, %d sims, " % (dt, Ns)
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
			ax = fig.add_subplot(111,ylabel='mean particles',title=tilt)
			ax.plot(tsim,Nf_mean,'b-')
			ax.grid()
			fig.show()
			# save the figure
			fig.savefig('Nf_aenkf_' + nameNow + '.png')


	raw_input("Return to exit")
	return


if __name__ == "__main__":
	main()