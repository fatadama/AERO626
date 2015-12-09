"""@package trials_processing.py
parse error and NEES histories into output formats, and saves
"""

import numpy as np
import math
import scipy.stats as stats # for chi-sqaured functions
import matplotlib.pyplot as plt


# everything gets parsed into e_sims, a nSteps*2 array
# and nees_history, a nSteps*nSims array of the NEES values

## Parses standard xxx_trials.py output into NEES statistics, MSE
# logs raw error statistics
# 
# Writes to files of the format "<fileid>_<filtername>(_<mods>)_<nameNow>.xxx",
# 	where <fileid> is "metrics", "nees", "raw", "Nf", and <mods> is optional
#
# @param[in] e_sims [(nSteps*nSims) x 2] numpy array containing the error in at each time in each simulation;
# @param[in] nees_history [nSteps x nSims] numpy array of the NEES statistic in each simulation at each time; simulations have length nSteps
# @param[in] filtername identifying acronym for the current filter, used to name output files
# @param[in] nameNow name for the data loaded to produce this simulation; used to name output files
# @param[in] mods (optional) string that gets added to the output file; e.g. the number of particles for a particular run for a filter
def errorParsing(e_sims,nees_history,filtername,nameNow,mods=None):
	# number of steps
	nSteps = nees_history.shape[0]
	# number of simulations
	Ns = nees_history.shape[1]
	# compute the MSE across all sims
	mse_tot = np.mean(np.power(e_sims,2.0),axis=0)
	# compute the normally distributed, unit variance chi-sqaure statistic
	chi_statistic = 1.0/math.sqrt(Ns)*np.sum((nees_history-2.0)/2.0,axis=1)
	# find where chi_statistic < 3 == 3sigma bound
	idBounded = np.nonzero(np.fabs(chi_statistic) <= 3.0)
	idBounded = idBounded[0]
	nCons = np.nonzero(chi_statistic < -3.0)
	nOpt = np.nonzero(chi_statistic > 3.0)

	nGood = len(idBounded)
	nBad = nSteps-nGood

	print("mse_tot: %f,%f" % (mse_tot[0],mse_tot[1]))
	print("Fraction of bad points: %f, %d sims" % (float(nBad)/float(Ns),Ns))

	# write raw to file
	fname = 'raw_' + filtername + "_"
	if mods is not None:
		fname = fname + str(mods) + "_"
	fname = fname + nameNow + ".txt"
	FID = open(fname,'w')
	# write the number of steps and number of sims
	FID.write("%d,%d\n" % (nSteps,Ns))
	for k in range(nSteps):
		for j in range(Ns):
			FID.write("%g," % nees_history[k,j])
		FID.write("\n")
	FID.close()

	plotting = False
	if plotting:
		# plot the chi-sqaured variables
		fig = plt.figure()
		ax = fig.add_subplot(111,xlabel='Chi-squared statistic')
		ax.hist(chi_statistic,bins=25)
		fig.show()
		raw_input("Return to close")
		plt.close(fig)
	return

