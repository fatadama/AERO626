import numpy as np
import matplotlib.pyplot as plt

## kmeans
#
# sorts input data into 'kargin' clusters. Converges to local minimum, repeat if results not satisfactory.
#
# @param[in] X [N x p] numpy array of p-dimensional points
# @param[in] kargin number of means into which to partition the data
# @param[out] idx index of which rows of X belong to which means
# @param[out] mui [kargin x p] the mean associated with each cluster.
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
		# calculate the change in mean for convergence
		meanShift = np.sqrt(np.sum(np.power(np.add(mui,-1.0*means),2.0),axis=1))
		#print(meanShift)
		mui = means.copy()
		if not np.any(meanShift > 1.0e-2):
			break
	return (idx,mui)

## test function for kmeans() algorithm
#
#
def test():
	# generate data from bimodal distribution
	mu1 = np.array([0.0,1.0])
	P1 = np.array([[1.0,0.0],[0.0,0.1]])
	mu2 = np.array([-1.0,0.0,])
	P2 = np.array([[1.0,0.25],[0.25,1.0]])
	mu3 = np.array([-0.5,0.5,])
	P3 = np.array([[10.0,0.0],[0.0,0.1]])

	x1 = np.random.multivariate_normal(mu1,P1,size=(50,))
	x2 = np.random.multivariate_normal(mu2,P2,size=(50,))
	x3 = np.random.multivariate_normal(mu3,P3,size=(50,))
	X = np.concatenate((x1,x2,x3),axis=0)

	(idx,mui) = kmeans(X,3)
	print(mui)

	# assign values to means
	Xi = []
	for k in range(3):
		first = True
		#Xi.append(np.array([]))
		for j in range(X.shape[0]):
			if idx[j] == k:
				if first:
					first = False
					Xi.append( np.array([ [0.0,0.0] ]) )
					Xi[k][0,:] = X[j,:].copy()
					#Xi.append(X[j,:].copy())
				else:
					print(Xi[k])
					print(X[j,:])
					Xi[k] = np.vstack((Xi[k],X[j,:]))
				#Xi[k] = np.append(Xi[k],X[j,:],axis=0)
	print(Xi)

	
	fig = plt.figure()
	ax = fig.add_subplot(121,ylabel='pts')
	ax.plot(Xi[0][:,0],Xi[0][:,1],'bd')
	ax.plot(Xi[1][:,0],Xi[1][:,1],'rd')
	ax.plot(Xi[2][:,0],Xi[2][:,1],'cd')
	# draw the estimated and true means
	ax.plot(mui[0,0],mui[0,1],'bx')
	ax.plot(mui[1,0],mui[1,1],'rx')
	ax.plot(mui[2,0],mui[2,1],'cx')
	ax.plot(mu1[0],mu1[1],'ko')
	ax.plot(mu2[0],mu2[1],'ko')
	ax.plot(mu3[0],mu3[1],'ko')

	ax = fig.add_subplot(122,ylabel='truth')
	ax.plot(x1[:,0],x1[:,1],'bd')
	ax.plot(x2[:,0],x2[:,1],'rd')
	ax.plot(x3[:,0],x3[:,1],'cd')
	fig.show()

	raw_input("Return to quit")
	plt.close(fig)	

	return