"""@package enkf_clustering
loads data, runs the ensemble KF algorithm with clustering tacked on
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats # for chi-sqaured functions
import scipy.linalg # for sqrtm() function
import cluster_processing

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/lib')
sys.path.append('../../filters/python/enkf')
import enkf

sys.path.append('../sim_data')
import data_loader

def eqom_enkf(x,t,u,v):
	return cp_dynamics.eqom_stoch(x,t,v)

## default measurement function for the case with linear position measurement
def measurement_enkf(x,t,u,n):
	return np.array([x[0]+n[0]])

## measurement function for the case with measurement of position squared with linear measurement noise
def measurement_uninformative(x,t,u,n):
	return np.array([x[0]*x[0]+n[0]])

## Driver for the clustering 
# @param[out] Xf aposteriori estimates, all points
# @param[out] Pf single-mode covariance, probably not used
# @param[out] Xp apriori estimates, all points
# @param[out] Idx index of which points were classified into which means
def enkf_test(dt,tf,mux0,P0,YK,Qk,Rk,flag_adapt=False,flag_informative=True):
	global nameBit

	Nsu = 200

	# measurement influence matrix
	Hk = np.array([ [1.0,0.0] ])

	# add in this functionality so we can change the propagation function dependent on the nameBit ... may or may not be needed
	if not flag_informative:
		measure_argument = measurement_uninformative
	else:
		measure_argument = measurement_enkf
	if nameBit == 1:
		# create EnKF object
		ENKF = enkf.clustering_enkf2(2,0,eqom_enkf,measure_argument,Qk,Rk,Ns=Nsu,maxMeans=2)
	elif nameBit == 2:
		# create EnKF object
		ENKF = enkf.clustering_enkf2(2,0,eqom_enkf,measure_argument,Qk,Rk,Ns=Nsu,maxMeans=2)
	elif nameBit == 3:
		# create EnKF object
		ENKF = enkf.clustering_enkf2(2,0,eqom_enkf,measure_argument,Qk,Rk,Ns=Nsu,maxMeans=2)

	nSteps = int(tf/dt)+1
	ts = 0.0

	#initialize EnKF
	ENKF.init(mux0,P0,ts)

	xf = np.zeros((nSteps,2))
	# aposteriori state values
	Xf = np.zeros((nSteps,2,Nsu))
	# propagated state values
	Xp = np.zeros((nSteps,2,Nsu))
	Pf = np.zeros((nSteps,4))
	Nf = np.zeros(nSteps)
	tk = np.arange(0.0,tf,dt)
	# index of cluster membership
	Idx = np.zeros((nSteps,Nsu))

	t1 = time.time()
	fig = []
	for k in range(0,nSteps):
		if k > 0:
			# get the new measurement
			ym = np.array([YK[k]])
			ts = ts + dt
			# sync the ENKF, with continuous-time integration
			print("Propagate to t = %f" % (ts))
			# propagate filter
			ENKF.propagateOde(dt,dtout=0.1)
		# log
		Xp[k,:,:] = ENKF.xk.copy()
		# leave IDX log here for the old clustering_enkf,move for clustering_enkf2
		#Idx[k,:] = ENKF.meansIdx.copy().astype(int)

		if k > 0:
			# update
			ENKF.update(ym)
		# log
		Xf[k,:,:] = ENKF.xk.copy()
		###
		Idx[k,:] = ENKF.meansIdx.copy().astype(int)
		###
		xf[k,:] = np.mean(ENKF.xk,axis=1)
		Pxx = np.zeros((2,2))
		for kj in range(ENKF.get_N()):
			Pxx = Pxx + 1.0/(1.0+float(ENKF._N))*np.outer(ENKF.xk[:,kj]-xf[k,:],ENKF.xk[:,kj]-xf[k,:])
		Pf[k,:] = Pxx.reshape((4,))
		Nf[k] = ENKF.get_N()

		# add the aposteriori state to the plot in black

	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(Xf,Pf,Idx,Xp)

def main():
	global nameBit
	names = ['sims_01_bifurcation_noninformative']
	flag_informative = False
	flag_adapt = False
	best_error = True
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		'''
		tsim = tsim[0:5]
		XK = XK[0:5,:]
		YK = YK[0:5,:]
		tf = tsim[4]
		'''
		Ns = 100

		nameBit = int(nameNow[5:7],2)
		# parse the name
		if nameBit == 1:
			# noise levels for the ENKF with white noise forcing
			Qk = np.array([[10.0]])
			Rk = np.array([[0.01]])
		if nameBit == 2:
			# noise levels for the UKF with cosine forcing
			Qk = np.array([[3.16/dt]])
			Rk = np.array([[0.1]])
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		Nf_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(Xf,Pf,Idx,Xp) = enkf_test(dt,tf,mu0,P0,yk,Qk,Rk,flag_adapt,flag_informative)
			print("enkf_clustering case %d/%d" % (counter+1,Ns))

			if Ns == 1:
				fig = []
				for k in range(nSteps):
					fig.append(plt.figure())
					ax = fig[k].add_subplot(1,1,1,title="t = %f" % (tsim[k]),xlim=(-25,25),ylim=(-20,20),ylabel='x2',xlabel='x1')
					#compute the number of active means
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
							ax.plot(Xf[k,0,idx],Xf[k,1,idx],'yo')
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
							ax.plot(ellipsP[:,0],ellipsP[:,1],'y--')
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
						ax.plot(xk[k,0],xk[k,1],'cs')
					ax.grid()
					fig[k].show()
				raw_input("Return to quit")
				for k in range(nSteps):
					fig[k].savefig('stepByStep/enkf_' + str(Xf.shape[2]) + "_" + str(k) + '.png')
					plt.close(fig[k])

			(e1,chi2,mx,Pk) = cluster_processing.singleSimErrors(Xf,Idx,xk,yk,best_error)
			nees_history[:,counter] = chi2.copy()
			mean_nees = np.sum(chi2)/float(nSteps)
			print(mean_nees)
			# mean NEES
			mse = np.sum(np.power(e1,2.0),axis=0)/float(nSteps)
			e_sims[(counter*nSteps):(counter*nSteps+nSteps),:] = e1.copy()

			print("MSE: %f,%f" % (mse[0],mse[1]))

		if Ns < 2:
			# plot the mean trajectories and error
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
					ax[k].plot(tsim,mx[:,k],'m--')
					'''
					if k == 0:
						ax[k].plot(tsim,yk,'r--')
					'''
				else:
					ax[k].plot(tsim,e1[:,k-2])
					ax[k].plot(tsim,3.0*np.sqrt(Pk[:,k-2,k-2]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pk[:,k-2,k-2]),'r--')
				ax[k].grid()
			fig1.show()
		else:
			if best_error:
				mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
				print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
				
				# get the mean NEES value versus simulation time across all sims
				nees_mean = np.sum(nees_history,axis=1)/Ns
				# get the mean number of particles in time
				Nf_mean = np.sum(Nf_history,axis=1)/Ns
				# get 95% confidence bounds for chi-sqaured... the df is the number of sims times the dimension of the state
				chiUpper = stats.chi2.ppf(.975,2.0*Ns)/float(Ns)
				chiLower = stats.chi2.ppf(.025,2.0*Ns)/float(Ns)
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
				FID = open('bestErrors_enkf_' + str(Xf.shape[2]) + '_' + nameNow + '.txt','w')
				FID.write("mse1,mse2,nees_below95,nees_above95\n")
				FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
				FID.close()
			else:
				print("Passing to exit")
				pass
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
				fig2.savefig('nees_enkf2_' + str(Xf.shape[2]) + '_' + nameNow + '.png')
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
				FID = open('metrics_enkf2_' + str(Xf.shape[2]) + '_' + nameNow + '.txt','w')
				FID.write("mse1,mse2,nees_below95,nees_above95\n")
				FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
				FID.close()
	
	raw_input("Return to exit")
	return


if __name__ == "__main__":
	main()