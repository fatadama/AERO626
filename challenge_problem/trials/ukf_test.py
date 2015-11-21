"""@package ukf_test
Module that instantiates a UKF filter and simulates for some seconds
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import time

sys.path.append('../')
import cp_dynamics

sys.path.append('../../filters/python/ukf')
import ukf

def eqom_ukf(x,t,u,v):
	return cp_dynamics.eqom_stoch(x,t,v)

def measurement_ukf(x,t,n):
	return np.array([[ x[0] + n[0] ]])

def ukf_test(dt = 0.01):

	Qk = np.array([[1.0e-2]])
	Rk = np.array([[1.0]])

	# create UKF object
	UKF = ukf.ukf(2,0,1,eqom_ukf,Qk)

	P0 = np.array([ [0.1, 1.0e-6],[1.0e-6, 1.0] ])
	mux0 = np.array([0.0,0.0])

	x0 = np.random.multivariate_normal(mux0,P0)
	sim = cp_dynamics.cp_simObject(cp_dynamics.eqom_stoch,x0,dt)

	tf = 30.0
	nSteps = int(tf/dt)
	ts = 0.0

	# initialize UKF
	UKF.init_P(mux0,P0,ts)

	xk = np.zeros((nSteps,2))
	xf = np.zeros((nSteps,2))
	Pf = np.zeros((nSteps,4))
	tk = np.arange(0.0,tf,dt)

	xk[0,:] = x0.copy()
	xf[0,:] = UKF.xhat.copy()
	Pf[0,:] = UKF.Pk.reshape((4,))

	t1 = time.time()
	for k in range(nSteps):
		# step the simulation and take a measurement
		(ym,x) = sim.step()
		ts = ts + dt
		# sync the UKF, with continuous-time integration
		UKF.sync(dt,ym,measurement_ukf,Rk,True)
		# copy
		if k < nSteps-1:
			xf[k+1,:] = UKF.xhat.copy()
			Pf[k+1,:] = UKF.Pk.reshape((4,))
			xk[k+1,:] = x.copy()
	t2 = time.time()
	print("Elapsed time: %f sec" % (t2-t1))

	fig1 = plt.figure()

	print(tk.shape)
	print(xk.shape)

	ax = []
	for k in range(4):
		if k < 2:
			nam = 'x' + str(k+1)
		else:
			nam = 'e' + str(k-1)
		ax.append(fig1.add_subplot(2,2,k+1,ylabel=nam))
		if k < 2:
			ax[k].plot(tk,xk[:,k])
			ax[k].plot(tk,xf[:,k],'m--')
		else:
			ax[k].plot(tk,xk[:,k-2]-xf[:,k-2])
			ax[k].plot(tk,3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
			ax[k].plot(tk,-3.0*np.sqrt(Pf[:,3*(k-2)]),'r--')
		ax[k].grid()
	fig1.show()

	# compute the unit variance transformation of the error
	e1 = np.zeros((nSteps,2))
	chi2 = np.zeros(nSteps)
	for k in range(nSteps):
		P = Pf[k,:].reshape((2,2))
		R = np.linalg.cholesky(P)
		e1[k,:] = np.dot(R,(xk[k,:]-xf[k,:]))
		chi2[k] = np.dot(e1[k,:],e1[k,:])

	(W,p) = stats.shapiro(e1.reshape((2*nSteps,)))
	print("Shapiro-Wilk output for all residuals: W = %f, p = %g" % (W,p) )
	for k in range(2):
		(W,p) = stats.shapiro(e1[:,k])
		print("Shapiro-Wilk output for axis %d: W = %f, p = %g" % (k,W,p) )
	
	fig2 = plt.figure()
	ax = []
	for k in range(2):
		nam = 'et' + str(k+1)
		ax.append(fig2.add_subplot(1,2,k+1,ylabel = nam))
		ax[k].plot(tk,e1[:,k])
		ax[k].grid()
	fig2.show()

	raw_input("Return to quit")

	print("Leaving ukf_test")

	return

def main():
	ukf_test(0.01)
	return

if __name__ == "__main__":
    main()
