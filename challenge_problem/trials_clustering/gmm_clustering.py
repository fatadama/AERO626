"""@package gmm_clustering
loads data, runs the gaussian mixture model filter with clustering tacked on
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
sys.path.append('../../filters/python/gmm')
import gmm

sys.path.append('../sim_data')
import data_loader

def eqom_gmm(x,t,u=None):
	return cp_dynamics.eqom_det(x,t)

def jac_gmm(x,t,u=None):
	return cp_dynamics.eqom_det_jac(x,t)

def process_influence(x,t,u=None):
	return cp_dynamics.eqom_det_Gk(x,t)

## default measurement function for the case with linear position measurement
def measurement_gmm(x,t,u=None):
	return np.array([x[0]])

def measurement_jac_gmm(x,t,u=None):
	return np.array([[1.0, 0.0]])

def measurement_jac_uninformative(x,t,u=None):
	return np.array([[2.0*x[0], 0.0]])

## measurement function for the case with measurement of position squared with linear measurement noise
def measurement_uninformative(x,t,u=None):
	return np.array([x[0]*x[0]])

## Driver for the clustering 
#@param[out] xml the maximum likelihood estimate based on the PDF
#@param[out] pdf nSteps x Np numpy array; the value of the pdf at discrete points in pdfPts
#@param[out] pdfPts nSteps x 2 x Np numpy array; the points at which the PDF is evaluated at each time. Also corresponds to the aposteriori means in the GMM
#@param[out] alphai the weights associated with each mean in the GMM at each time in the output
#@param[out] Pki nSteps x 2 x 2 x Np numpy array; the covariance assocaited with each mean
def gmm_test(dt,tf,mux0,P0,YK,Qk,Rk,Nsu=20,flag_informative=True):
	global nameBit

	# add in this functionality so we can change the propagation function dependent on the nameBit ... may or may not be needed
	if not flag_informative:
		measure_argument = measurement_uninformative
		measure_jacobian = measurement_jac_uninformative
	else:
		measure_argument = measurement_enkf
		measure_jacobian = measurement_jac_gmm
	if nameBit == 1:
		# create EnKF object
		GMM = gmm.gmm(2,Nsu,Qk,Rk,eqom_gmm,jac_gmm,process_influence,measure_argument,measure_jacobian)
	elif nameBit == 2:
		# create EnKF object
		GMM = gmm.gmm(2,Nsu,Qk,Rk,eqom_gmm,jac_gmm,process_influence,measure_argument,measure_jacobian)
	elif nameBit == 3:
		# create EnKF object
		GMM = gmm.gmm(2,Nsu,Qk,Rk,eqom_gmm,jac_gmm,process_influence,measure_argument,measure_jacobian)

	nSteps = int(tf/dt)+1
	ts = 0.0

	#initialize EnKF
	GMM.init_monte(mux0,P0,ts)

	xml = np.zeros((nSteps,2))
	pdf = np.zeros((nSteps,GMM.aki.shape[1]))
	pdfPts = np.zeros((nSteps,2,GMM.aki.shape[1]))
	alphai = np.zeros((nSteps,GMM.aki.shape[1]))
	Pki = np.zeros((nSteps,2,2,GMM.aki.shape[1]))
	tk = np.arange(0.0,tf,dt)

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
			GMM.propagate_normal(dt)
			GMM.update(ym)
		# log
		alphai[k,:] = GMM.alphai.copy()
		xml[k,:] = GMM.get_max_likelihood()
		Pki[k,:,:,:] = GMM.Pki.copy()
		(pdfPts[k,:,:],pdf[k,:]) = GMM.get_pdf()
		if k > 0:
			GMM.resample()

	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(xml,pdf,pdfPts,alphai,Pki)

def main():
	global nameBit
	## number of particles to use
	Nsu = 100
	names = ['sims_01_bifurcation_noninformative']
	flag_informative = False
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		'''
		tsim = tsim[0:5]
		XK = XK[0:5,:]
		YK = YK[0:5,:]
		tf = tsim[4]
		'''
		
		Ns = 1

		nameBit = int(nameNow[5:7],2)
		# parse the name
		if nameBit == 1:
			# noise levels for the ENKF with white noise forcing
			Qk = np.array([[1.0]])
			Rk = np.array([[0.1]])
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

			#(Xf,Pf,Idx,Xp) = gmm_test(dt,tf,mu0,P0,yk,Qk,Rk,flag_informative)
			(Xf,pdf,pdfPts,alphai,Pki) = gmm_test(dt,tf,mu0,P0,yk,Qk,Rk,Nsu,flag_informative)
			print("gmm_clustering case %d/%d" % (counter+1,Ns))

			if Ns == 1:
				fig = []
				for k in range(nSteps):
					fig.append(plt.figure())
					ax = fig[k].add_subplot(1,1,1,title="t = %f" % (tsim[k]))

					# the number of active means
					activeMeans = pdf.shape[1]
					for jk in range(activeMeans):
						mux = pdfPts[k,:,jk]
						ax.plot(pdfPts[k,0,jk],pdfPts[k,1,jk],'o')
						# plot the single-mean covariance ellipsoid
						# draw points on a unit circle
						thetap = np.linspace(0,2*math.pi,20)
						circlP = np.zeros((20,2))
						circlP[:,0] = 3.0*np.cos(thetap)
						circlP[:,1] = 3.0*np.sin(thetap)
						# transform the points circlP through P^(1/2)*circlP + mu
						Phalf = np.real(scipy.linalg.sqrtm(Pki[k,:,:,jk]))
						ellipsP = np.zeros(circlP.shape)
						for kj in range(circlP.shape[0]):
							ellipsP[kj,:] = np.dot(Phalf,circlP[kj,:])+mux
						ax.plot(ellipsP[:,0],ellipsP[:,1],'--')
						# plot the truth state
						ax.plot(xk[k,0],xk[k,1],'ks')
					ax.grid()
					fig[k].show()
				raw_input("Return to quit")
				for k in range(nSteps):
					plt.close(fig[k])
			'''
			(e1,chi2,mx,Pk) = cluster_processing.singleSimErrors(Xf,Idx,xk,yk)
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
				else:
					ax[k].plot(tsim,e1[:,k-2])
					ax[k].plot(tsim,3.0*np.sqrt(Pk[:,k-2,k-2]),'r--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pk[:,k-2,k-2]),'r--')
				ax[k].grid()
			fig1.show()
		
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
		fig2.savefig('nees_enkf2_' + str(Ns) + '_' + nameNow + '.png')
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
		FID = open('metrics_enkf2_' + str(Ns) + '_' + nameNow + '.txt','w')
		FID.write("mse1,mse2,nees_below95,nees_above95\n")
		FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
		FID.close()
	
	raw_input("Return to exit")
	'''
	return


if __name__ == "__main__":
	main()