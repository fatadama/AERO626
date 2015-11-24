import numpy as np
import matplotlib.pyplot as plt

## kmeans
#
# @param[in] X [N x p] numpy array of p-dimensional points
# @param[in] kargin number of means into which to partition the data
def kmeans(X,kargin):
	#randomly initialize
	# make sure all unique
	
	resample = True
	while resample == True:
		inr = np.random.uniform(low=0,high=X.shape[0],size=kargin).astype(int)
		resample = False
		for k in range(kargin):
			for j in range(0,k)+range(k+1,kargin):
				if inr[k]==inr[j]:
					#resample
					resample = True
	# initial means
	mui = X[inr,:].copy()
	for outer in range(100):
		idx = np.zeros(X.shape[0])
		means = np.zeros((kargin,X.shape[1]))
		nmembers = np.zeros(kargin)
		for k in range(X.shape[0]):
			jmin = -1
			dmin = 1.0e10
			for j in range(kargin):
				D = np.sqrt(np.sum(np.power(X[k,:]-mui[j,:],2.0)))
				if D < dmin:
					dmin = D
					jmin = j
			#print("%f to %d at %f" % (X[k,0],jmin,mui[jmin,0]))
			idx[k] = jmin
			means[jmin,:] = means[jmin,:] + X[k,:]
			nmembers[jmin] = nmembers[jmin] + 1.0
		# calculate new means to be the geometric centers of the current clusters
		for j in range(X.shape[1]):
			# new means
			means[:,j] = np.divide(means[:,j],nmembers)
		# calculate he change in mean for convergence
		meanShift = np.sqrt(np.sum(np.power(np.add(mui,-1.0*means),2.0),axis=1))
		print(meanShift)
		mui = means.copy()
		if not np.any(meanShift > 1.0e-2):
			break
	return (idx,mui)

def test():
	# generate data from bimodal distribution
	mu1 = np.array([0.0])
	P1 = np.array([[1.0]])
	mu2 = np.array([-1.0])
	P2 = np.array([[1.0]])

	x1 = np.random.multivariate_normal(mu1,P1,size=(25,))
	x2 = np.random.multivariate_normal(mu1,P2,size=(25,))
	X = np.concatenate((x1,x2),axis=0)

	(idx,mui) = kmeans(X,2)
	print(mui)

	# assign values to means
	Xi = []
	for k in range(2):
		Xi.append(np.array([]))
		for j in range(X.shape[0]):
			if idx[j] == k:
				Xi[k] = np.append(Xi[k],X[j,:],axis=0)
	print(Xi)

	'''
	fig = plt.figure()
	ax = fig.add_subplot(111,ylabel='pts')
	ax.plot(Xi[0])
	'''

	return