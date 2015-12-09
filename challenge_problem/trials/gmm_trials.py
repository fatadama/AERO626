"""@package gmm_trials
loads data, passes through GMM
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import time
import scipy.stats as stats

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/gmm')
import gmm

sys.path.append('../sim_data')
import data_loader

def eqom_use(x,t,u):
	return cp_dynamics.eqom_det(x,t)

def eqom_jacobian_use(x,t,u):
	return cp_dynamics.eqom_det_jac(x,t)

def eqom_gk_use(x,t,u):
	return cp_dynamics.eqom_det_Gk(x,t)

def measurement_use(x,t):
	return np.array([ x[0] ])

def measurement_gradient(x,t):
	return np.array([ [1.0,0.0] ])

#@param[out] xml the maximum likelihood estimate based on the PDF
#@param[out] pdf nSteps x Np numpy array; the value of the pdf at discrete points in pdfPts
#@param[out] pdfPts nSteps x 2 x Np numpy array; the points at which the PDF is evaluated at each time. Also corresponds to the aposteriori means in the GMM
#@param[out] alphai the weights associated with each mean in the GMM at each time in the output
#@param[out] Pki nSteps x 2 x 2 x Np numpy array; the covariance assocaited with each mean
def gmm_test(dt,tf,mux0,P0,YK,Qk,Rk):

	# create gmm object
	GMM = gmm.gmm(2,10,Qk,Rk,eqom_use,eqom_jacobian_use,eqom_gk_use,measurement_use,measurement_gradient)

	nSteps = int(tf/dt)+1
	ts = 0.0

	# initialize gmm
	GMM.init_monte(mux0,P0,ts)

	xml = np.zeros((nSteps,2))
	pdf = np.zeros((nSteps,GMM.aki.shape[1]))
	pdfPts = np.zeros((nSteps,2,GMM.aki.shape[1]))
	alphai = np.zeros((nSteps,GMM.aki.shape[1]))
	Pki = np.zeros((nSteps,2,2,GMM.aki.shape[1]))
	tk = np.arange(0.0,tf,dt)

	t1 = time.time()
	for k in range(0,nSteps):
		if k > 0:
			# get the new measurement
			ym = np.array([YK[k]])
			ts = ts + dt
			# sync the gmm, with continuous-time integration
			GMM.propagate_normal(dt)
			GMM.update(ym)
			# resample
			#GMM.resample_mcmc()
		alphai[k,:] = GMM.alphai.copy()
		xml[k,:] = GMM.get_max_likelihood()
		Pki[k,:,:,:] = GMM.Pki.copy()
		(pdfPts[k,:,:],pdf[k,:]) = GMM.get_pdf()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	return(xml,pdf,pdfPts,alphai,Pki)

def main():
	#names = ['sims_11_fast']# test case
	names = ['sims_01_slow','sims_01_medium','sims_01_fast','sims_10_slow','sims_10_medium','sims_10_fast','sims_11_slow','sims_11_medium','sims_11_fast']
	for namecounter in range(len(names)):
		nameNow = names[namecounter]
		(tsim,XK,YK,mu0,P0,Ns,dt,tf) = data_loader.load_data(nameNow,'../sim_data/')

		namebit = int(nameNow[5:7],2)
		# parse the name
		if namebit == 1:
			# tuned white noise
			if dt > 0.9:#slow sampling
				Qk = np.array([[30.0]])
			elif dt > 0.09:#medium sampling
				Qk = np.array([[3.0]])
			else:# fast sampling
				Qk = np.array([[1.0]])
			Rk = np.array([[1.0]])
		elif namebit == 2:
			# tuned cosine forcing
			if dt > 0.9:#slow sampling
				Qk = np.array([[80.0]])
			elif dt > 0.09:#medium sampling
				Qk = np.array([[20.0]])
			else:# fast sampling
				Qk = np.array([[15.0]])
			Rk = np.array([[1.0]])
		elif namebit == 3:
			# tuned GMM with cosine forcing and white noise
			if dt > 0.9:#slow sampling
				Qk = np.array([[100.0]])
			elif dt > 0.09:#medium sampling
				Qk = np.array([[20.0]])
			else:# fast sampling
				Qk = np.array([[20.0]])
			Rk = np.array([[1.0]])
		print(Qk[0,0])
		# number of steps in each simulation
		nSteps = len(tsim)
		nees_history = np.zeros((nSteps,Ns))
		e_sims = np.zeros((Ns*nSteps,2))
		for counter in range(Ns):
			xk = XK[:,(2*counter):(2*counter+2)]
			yk = YK[:,counter]

			(xf,pdf,pdfPts,weights,Pki) = gmm_test(dt,tf,mu0,P0,yk,Qk,Rk)

			# find the (single mode) covariance
			Pf = np.zeros((xk.shape[0],4))
			# find the covariance associated with the maximum likelihood estimate
			Pm = np.zeros((xk.shape[0],4))
			for k in range(xk.shape[0]):
				# find the covariance of the maximum likelihood estimate
				idx = np.argmax(pdf[k,:])
				Pm[k,:] = Pki[k,:,:,idx].reshape((4,))
				# find the covariance of the whole distribution
				Pxx = np.zeros((2,2))
				'''
				mu = np.zeros(2)
				for j in range(pdfPts.shape[1]):
					mu = mu + weights[k,j]*pdfPts[k,:,j]
				'''
				mu = xf[k,:].copy()
				for j in range(pdfPts.shape[2]):
					Pxx = Pxx + weights[k,j]*(Pki[k,:,:,j] + np.outer(pdfPts[k,:,j]-mu,pdfPts[k,:,j]-mu))
				Pf[k,:] = Pxx.reshape((4,))

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
			for k in range(6):
				if k < 2:
					nam = 'x' + str(k+1)
				elif k < 4:
					nam = 'e' + str(k-1)
				else:
					nam = 'pdf' + str(k-3)
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
					ax[k].plot(tsim,3.0*np.sqrt(Pm[:,3*(k-2)]),'c--')
					ax[k].plot(tsim,-3.0*np.sqrt(Pm[:,3*(k-2)]),'c--')
				else:
					# len(tplot) x Ns matrix of times
					tMesh = np.kron(np.ones((pdf.shape[1],1)),tsim).transpose()
					mex = tMesh.reshape((len(tsim)*pdf.shape[1],))
					mey = pdfPts[:,k-4,:].reshape((len(tsim)*pdf.shape[1],))
					mez = pdf.reshape((len(tsim)*pdf.shape[1],))
					idx = mez.argsort()
					mexx,meyy,mezz = mex[idx],mey[idx],mez[idx]

					cc = ax[k].scatter(mexx,meyy,c=mezz,s=20,edgecolor='')
					fig1.colorbar(cc,ax=ax[k])
					# plot the truth
					ax[k].plot(tsim,xk[:,k-4],'b-')
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
		fig2 = plt.figure(figsize=(6.0,3.37)) #figsize tuple is width, height
		tilt = "GMM, Ts = %.2f, %d sims, " % (dt, Ns)
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
		fig2.savefig('nees_gmm_' + nameNow + '.png')

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
		FID = open('metrics_gmm_' + nameNow + '.txt','w')
		FID.write("mse1,mse2,nees_below95,nees_above95\n")
		FID.write("%f,%f,%f,%f\n" % (mse_tot[0],mse_tot[1],float(len_sub)/float(nSteps),float(len_super)/float(nSteps)))
		FID.close()

	raw_input("Return to quit")

	print("Leaving gmm_trials")

	return

if __name__ == "__main__":
    main()
