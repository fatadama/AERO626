"""@package data_loader
Module with functions for loading data from an output file generated by generate_data.py
"""

import ConfigParser
import numpy as np
import matplotlib.pyplot as plt

def main():
	name = 'sims_01_fast'
	(tsim,XK,YK,mu0,P0,Ns,dt,tf) = load_data(name)
	print(tsim,XK,YK)
	print("Loaded simulation with %d runs, initial mean = %f,%f,\n and initial covariance P = [[%f,%f],[%f,%f]]" % (Ns,mu0[0],mu0[1],P0[0,0],P0[0,1],P0[1,0],P0[1,1]))

## load simulation data from a particular case
#
#@param[in] name name of simulation batch to load; "<name>_settings.ini" and "<name>_data.csv" must exist
#@param[in] fpath path to data files
#@param[out] tsim simulation time vector
#@param[out] XK len(tsim) x 2*Ns matrix of true system states; even columns are position history, odd are velocity
#@param[out] YK len(tsim) x Ns matrix of system measurements for Ns Monte Carlo runs
#@param[out] mu0 mean initial state
#@param[out] P0 initial covariance
#@param[out] Ns number of simulations
#@param[out] dt sample period of measurements
#@param[out] tf final simulation time
def load_data(name,fpath='./'):
	config = ConfigParser.ConfigParser()
	config.read(fpath + name + "_settings.ini")
	# sample time
	dt = float(config.get(name,'ts'))
	# number of data points
	Ns = int(config.get(name,'ns'))
	# final time
	tf = float(config.get(name,'tf'))
	# initial covariance
	P0 = np.zeros((2,2))
	P0[0,0] = float(config.get(name,'p0_11'))
	P0[0,1] = float(config.get(name,'p0_12'))
	P0[1,0] = float(config.get(name,'p0_21'))
	P0[1,1] = float(config.get(name,'p0_22'))
	# initial state mean
	mu0 = np.zeros(2)
	mu0[0] = float(config.get(name,'mux_1'))
	mu0[1] = float(config.get(name,'mux_2'))
	# load data
	datain = np.genfromtxt(fpath + name+'_data.csv','float',delimiter=',')
	## tsim: simulation time
	tsim = datain[:,0]
	inx = sorted( range(1,3*Ns+1,3) + range(2,3*Ns+1,3) )
	## XK: nSteps  x 2*Nsims array of state histories
	XK = datain[:,inx]
	## YK: nSteps x Nsims array of measurement of position
	YK = datain[:,range(3,3*Ns+1,3)]
	return (tsim,XK,YK,mu0,P0,Ns,dt,tf)

if __name__ == "__main__":
    main()