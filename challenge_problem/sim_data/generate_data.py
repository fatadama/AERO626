"""@package generate_data
Script that generates Monte Carlo sim output for many cases at once, for specified initial distributions, sample rates, & other variables
"""

import sys
sys.path.append('../')
import cp_dynamics

import numpy as np
import matplotlib.pyplot as plt
import time

def etaCalc(k,kmax,dt):
    dtest = dt*float(kmax)/float(k+1)
    eta = dtest-dt
    if eta < 120:
        print("Estimated time remaining: %f s" % (eta) )
    elif eta < 1800:
        print("Estimated time remaining: %f min" % (eta*0.01666666666) )
    else:
        print("Estimated time remaining: %f hrs" % (eta*0.00027777777) )

## initial covariance
P0 = np.array([ [0.1, 1.0e-6],[1.0e-6, 1.0] ])
# use [2.0 1.0e-6],[1.0e-6, 1.0] for bifurcation case to really exaggerate
Pcluster = np.array([ [2.0, 1.0e-6],[1.0e-6, 1.0] ])
## initial mean
mux0 = np.array([0.0,0.0])
## number of simulations to run per case
Ns = 100
#Ns = 100
## simulation stop time
tf = 30.0
#tf = 30.0

# @param[in] Nsims number of simulations to run
# @param[in] P0 initial covariance of points
# TODO modify generate_sim to call this function
def execute_sim(function,Ts,tf,Nsims,Pi,name=None,cluster=False,informative=True):
	nSteps = int(tf/Ts)+1
	## matrix of initial conditions, size 2 x N
	X0 = np.random.multivariate_normal(mux0,Pi,size=(Nsims,)).transpose()
	## simulation output/measurement times
	tsim = np.arange(0.0,tf+Ts,Ts)
	## simulation output measurements
	YK = np.zeros((nSteps,Nsims))
	## simulation state history
	XK = np.zeros((nSteps,2*Nsims))

	t1 = time.time()
	for k in range(Nsims):
		if not cluster:
			sim = cp_dynamics.cp_simObject(function,X0[:,k],Ts)
		if cluster:
			if informative:
				sim = cp_dynamics.cp_simObjectCluster(function,X0[:,k],Ts)
			else:
				sim = cp_dynamics.cp_simObjectNonInformative(function,X0[:,k],Ts)
		# simulate
		(YK[:,k],XK[:,(2*k):(2*k+2)],tk) = sim.simFull(Tf=tf)
		t2 = time.time()
		etaCalc(k,Nsims,t2-t1)
	t2 = time.time()
	print("Completed %d sims in %g secs" % (Nsims,t2-t1))

	return(tsim,XK,YK,mux0,Ts,tf)

# @param[in] informative Set to False to use non-informative position measurements (y = position^2)
def generate_sim(function,Ts,tf,name=None,cluster=False,informative=True):
	nSteps = int(tf/Ts)+1
	## matrix of initial conditions, size 2 x N
	if not cluster:
		X0 = np.random.multivariate_normal(mux0,P0,size=(Ns,)).transpose()
	else:
		X0 = np.random.multivariate_normal(mux0,Pcluster,size=(Ns,)).transpose()
	## simulation output/measurement times
	tsim = np.arange(0.0,tf+Ts,Ts)
	## simulation output measurements
	YK = np.zeros((nSteps,Ns))
	## simulation state history
	XK = np.zeros((nSteps,2*Ns))

	t1 = time.time()
	for k in range(Ns):
		if not cluster:
			sim = cp_dynamics.cp_simObject(function,X0[:,k],Ts)
		if cluster:
			if informative:
				sim = cp_dynamics.cp_simObjectCluster(function,X0[:,k],Ts)
			else:
				sim = cp_dynamics.cp_simObjectNonInformative(function,X0[:,k],Ts)
		# simulate
		(YK[:,k],XK[:,(2*k):(2*k+2)],tk) = sim.simFull(Tf=tf)
		t2 = time.time()
		etaCalc(k,Ns,t2-t1)
	t2 = time.time()
	print("Completed %d sims in %g secs" % (Ns,t2-t1))
	
	if name is not None:
		# write to file
		# write settings
		FID = open(name + "_settings.ini",'w')
		FID.write("[%s]\n" % (name))
		FID.write("Function: " + function.__name__ + "\n")
		FID.write("Ts: %g\n" % (Ts))
		FID.write("tf: %g\n" % tf)
		FID.write("Ns: %d\n" % Ns)
		if not cluster:
			FID.write("P0_11: %f\n" %  P0[0,0])
			FID.write("P0_12: %f\n" %  P0[0,1])
			FID.write("P0_21: %f\n" %  P0[1,0])
			FID.write("P0_22: %f\n" %  P0[1,1])
		else:
			FID.write("P0_11: %f\n" %  Pcluster[0,0])
			FID.write("P0_12: %f\n" %  Pcluster[0,1])
			FID.write("P0_21: %f\n" %  Pcluster[1,0])
			FID.write("P0_22: %f\n" %  Pcluster[1,1])
		FID.write("mux_1: %f\n" % mux0[0])
		FID.write("mux_2: %f\n" % mux0[1])
		FID.close()
		print("Wrote settings file")
		# write data
		datafilename = name + "_data.csv"
		FID = open(datafilename,'w')
		for k in range(nSteps):
			FID.write("%f," % tsim[k])
			for j in range(Ns):
				FID.write("%f,%f,%f," % (XK[k,2*j],XK[k,2*j+1],YK[k,j]))
			FID.write("\n")
		FID.close()
		print("Wrote data file")
		pass
	else:
		# plot histories and measurements
		fig = plt.figure()
		ax = []
		for k in range(2):
			nam = 'x' + str(k+1)
			ax.append(fig.add_subplot(1,2,k+1,ylabel=nam))
			inplot1 = range(k,2*Ns,2)
			print(inplot1)
			ax[k].plot(tsim,XK[:,inplot1])
			if k == 0:
				ax[k].plot(tsim,YK)
		fig.show()
		raw_input("Return to continue")
	return
	
def main():
	# run test sims
	# sims_xx_fast/medium/slow format: xx is a bitstring, where the lowest bit indicates if the Gaussian process noise is on, and highest bit indicates if the forcing function is on
	# fast/medium/slow refers to the sample rate: (0.01,0.1,1.0)
	'''
	generate_sim(cp_dynamics.eqom_stoch,0.01,tf,name='sims_01_fast')
	generate_sim(cp_dynamics.eqom_stoch,0.1,tf,name='sims_01_medium')
	generate_sim(cp_dynamics.eqom_stoch,1.0,tf,name='sims_01_slow')

	generate_sim(cp_dynamics.eqom_det_f,0.01,tf,name='sims_10_fast')
	generate_sim(cp_dynamics.eqom_det_f,0.1,tf,name='sims_10_medium')
	generate_sim(cp_dynamics.eqom_det_f,1.0,tf,name='sims_10_slow')

	generate_sim(cp_dynamics.eqom,0.01,tf,name='sims_11_fast')
	generate_sim(cp_dynamics.eqom,0.1,tf,name='sims_11_medium')
	generate_sim(cp_dynamics.eqom,1.0,tf,name='sims_11_slow')
	'''
	# very long period observations - bifurcation case
	generate_sim(cp_dynamics.eqom_stoch_cluster,5.0,60.0,name='sims_01_bifurcation',cluster=True,informative=True)
	# simulation bifurcation case with uninformative measurements
	generate_sim(cp_dynamics.eqom_stoch_cluster,5.0,60.0,name='sims_01_bifurcation_noninformative',cluster=True,informative=False)


if __name__ == "__main__":
	main()