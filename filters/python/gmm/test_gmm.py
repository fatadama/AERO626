
import sys
import numpy as np
import scipy.integrate as sp
import matplotlib.pyplot as plt
import time

import gmm

## stateDerivative(x,t,u) - returns the derivative of the filter state at a given time, for particular control values
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u "control" given by (accelerometer measurement, gyro measurement, gravity constant)
def stateDerivative(x,t,u):
    f = np.zeros((2,))
    f[0] = x[1]
    f[1] = -2.0*(1.5/1.0)*(x[0]*x[0]-1.0)*x[1]-(1.2/1.0)*x[0]
    return f

## stateDerivativeGMM - function used by the filter for propagation
#
#   @param x state given by ( position(inertial), velocity(body frame), quaternion )
#   @param t current time
#   @param u control term, not used here
def stateDerivativeGMM(x,t,u):
    f = np.zeros((2,))
    f[0] = x[1]
    f[1] = -2.0*(1.0/1.0)*(x[0]*x[0]-1.0)*x[1]-(1.0/1.0)*x[0]
    return f

def stateJacobian(x,t,u):
	Fk = np.zeros((2,2))
	Fk[0,1] = 1.0
	Fk[1,0] = -4.0*x[0]*x[1]-1.0
	Fk[1,1] = -2.0*(x[0]*x[0]-1.0)
	return Fk

def stateProcessInfluence(x,t,u):
	Gk = np.zeros((2,1))
	Gk[1,0] = 1.0
	return Gk

def measurementFunction(x,t):
	return np.array([x[0]])

def measurementJacobian(x,t):
	Hk = np.zeros((1,2))
	Hk[0,0] = 1.0
	return Hk

def simMeasurementFunction(xk,t):
    ymeas = np.array([ xk[0]+ np.random.normal(0.0,0.01) ])
    return ymeas

def main(argin='./',adaptFlag = False):
    # output file

    # initialize EKF
    #Qkin = np.array([[20.0]])#continuous-time integration value
    Qkin = np.array([[20.0]])#Euler integration value
    Rkin = np.array([ [0.0001] ])
    GMM = gmm.gmm(2,25,Qkin,Rkin,stateDerivativeGMM,stateJacobian,stateProcessInfluence,measurementFunction,measurementJacobian)

    dt = 0.01
    tfin = 10.0#10.0
    nSteps = int(tfin/dt)
    tsim = 0.0

    muk = np.array([1.0,0.0])
    Pk = np.array([[0.1,0.0],[0.0,1.0]])
    xk = np.random.multivariate_normal(muk,Pk)
    yk = simMeasurementFunction(xk,tsim)

    # initial covariance
    GMM.init_monte(xk,Pk)

    ## true state
    xt = np.zeros((nSteps,2))
    ## discretized PDF value
    XK = np.zeros((nSteps,2,GMM.aki.shape[1]))
    pk = np.zeros((nSteps,GMM.aki.shape[1]))
    alphai = np.zeros((nSteps,GMM.aki.shape[1]))
    Pki = np.zeros((nSteps,2,2,GMM.aki.shape[1]))
    yt = np.zeros(nSteps)

    t1 = time.time()
    for k in range(nSteps):
        # log
        xt[k,:] = xk.copy()
        (XK[k,:,:],pk[k,:]) = GMM.get_pdf()
        alphai[k,:] = GMM.alphai.copy()
        Pki[k,:,:,:] = GMM.Pki.copy()
        # propagate filter
        GMM.propagate_normal(dt)
        # simulate
        y = sp.odeint(stateDerivative,xk,np.array([tsim,tsim+dt]),args=([],) )
        xk = y[-1,:].copy()
        # update time
        tsim = tsim + dt
        # measurement
        ymeas = simMeasurementFunction(xk,tsim)
        # store measurement
        yt[k] = ymeas[0]
        # update EKF
        GMM.update(ymeas)
        print("%f,%f" % (tsim,ymeas[0]))
        # resample?
        GMM.resample()
    t2 = time.time()
    tplot = np.arange(0.0,tfin,dt)

    print('Completed simulation in %f seconds' % (t2-t1))

    # len(tplot) x Ns matrix of times
    tMesh = np.kron(np.ones((GMM.aki.shape[1],1)),tplot).transpose()

    # find the max of the PDF for the maximum liklihood estimate
    xml = np.zeros((nSteps,2))
    Pkk = np.zeros((nSteps,2))
    for k in range(nSteps):
    	idmax = np.argmax(pk[k,:])
    	xml[k,:] = XK[k,:,idmax].transpose()
    	# compute the covariance
    	mu = np.zeros(2)
    	for j in range(GMM.aki.shape[1]):
    		mu = mu + alphai[k,j]*XK[k,:,j]
    	Pxx = np.zeros((2,2))
    	for j in range(GMM.aki.shape[1]):
    		Pxx = Pxx + alphai[k,j]*(Pki[k,:,:,j] + np.outer(XK[k,:,j]-mu,XK[k,:,j]-mu))
		#print("%f,%f|%f,%f,%f,%f" % (mu[0],mu[1],Pxx[0,0],Pxx[0,1],Pxx[1,0],Pxx[1,1]))
		Pkk[k,0] = Pxx[0,0]
		Pkk[k,1] = Pxx[1,1]

    fig = plt.figure()

    ax = []
    for k in range(4):
		if k < 2:
			nam = 'x' + str(k+1)
		else:
			nam = 'e' + str(k-1)
		ax.append( fig.add_subplot(2,2,k+1,ylabel=nam) )
		if k < 2:
			if k == 0:
				# plot the discrete PDF as a function of time
				mex = tMesh.reshape((len(tplot)*GMM.aki.shape[1],))
				mey = XK[:,0,:].reshape((len(tplot)*GMM.aki.shape[1],))
				mez = pk.reshape((len(tplot)*GMM.aki.shape[1],))
			elif k == 1:
				# plot the discrete PDF as a function of time
				mex = tMesh.reshape((len(tplot)*GMM.aki.shape[1],))
				mey = XK[:,1,:].reshape((len(tplot)*GMM.aki.shape[1],))
				mez = pk.reshape((len(tplot)*GMM.aki.shape[1],))

			idx = mez.argsort()
			mexx,meyy,mezz = mex[idx],mey[idx],mez[idx]

			cc = ax[k].scatter(mexx,meyy,c=mezz,s=20,edgecolor='')
			fig.colorbar(cc,ax=ax[k])
			# plot the truth
			ax[k].plot(tplot,xt[:,k],'b-')
		elif k < 4:
			ax[k].plot(tplot,xt[:,k-2]-xml[:,k-2],'b-')
			ax[k].plot(tplot,3.0*np.sqrt(Pkk[:,k-2]),'r--')
			ax[k].plot(tplot,-3.0*np.sqrt(Pkk[:,k-2]),'r--')
		ax[k].grid()
    fig.show()

    raw_input("Return to exit")

    print("Completed test_enky.py")
    return

if __name__ == '__main__':
    adapt = True
    main(adaptFlag = adapt)
