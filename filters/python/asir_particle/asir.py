"""
asir.py
Auxiliary sampling importance resampling filter
"""

import numpy as np
import sys
sys.path.append('../sis_particle')
import sis
sys.path.append('../lib')
import particle_utilities as pu

class asir(sis.sis):
    def __init__(self,nx=1,Ns=10,propagateFunction=None,processNoiseSampleFunction=None,measurementPdfFunction=None,meanPropagateFunction=None):
        # initialize base class
        sis.sis.__init__(self,nx,Ns,propagateFunction,processNoiseSampleFunction,measurementPdfFunction)
        # set mean propagate function
        self.meanPropagateFunction = meanPropagateFunction
    def update(self,dt,ymeas):
        if self.initFlag:
            MUK = np.zeros(self.XK.shape)
            XKI = np.zeros(self.XK.shape)
            for k in range(self.Ns):
                # compute the expectation of the current particle when propagated to the next state
                # this is probably just the propagateFunction with no process noise applied, but we make no assumptions about this in this class
                MUK[:,k] = self.meanPropagateFunction(self.XK[:,k],dt)
                # also?? propagate the particles through dt?? This step is unclear, but the propagated particles are used in the resample algorithm
                # draw process noise according to the process noise model
                vki = self.processNoiseSample(self.XK[:,k])
                XKI[:,k] = self.propagateParticle(self.XK[:,k],dt,vki)
                # update the current particle's weight based on the measurement PDF
                self.WI[k] = self.WI[k]*self.measurementNoisePdf(ymeas,MUK[:,k])
            # normalize the weights
            weightFactor = 1.0/np.sum(self.WI)
            for k in range(self.Ns):
                self.WI[k] = self.WI[k]*weightFactor
            # use the standard resample function, which depends on the propagated particles?
            (gar1,gar2,ipl) = pu.resample(XKI,self.WI)
            # copies of the prior state and weights
            XKc = np.zeros(self.XK.shape)
            WIc = np.zeros(self.WI.shape)
            for j in range(self.Ns):
                # draw x(j) from the prior
                # draw process noise according to the process noise model
                vki = self.processNoiseSample(self.XK[:,ipl[j]])
                # propagate the particle
                XKc[:,j] = self.propagateParticle(self.XK[:,ipl[j]],dt,vki)
                # assign weights
                try:
                    WIc[k] = self.measurementNoisePdf(ymeas,XKc[:,j])/self.measurementNoisePdf(ymeas,MUK[:,ipl[j]])
                except ZeroDivisionError:
                    print("Error: divide by zero in particle weight computation. Probably particle impoverishment; try more particles or new characterization function")
                    return
            # replace the prior by the updated particles and weights
            self.XK = XKc.copy()
            self.WI = WIc.copy()
            # normalize the weights
            weightFactor = 1.0/np.sum(self.WI)
            for k in range(self.Ns):
                self.WI[k] = self.WI[k]*weightFactor
        else:
            print("Error: uninitialized filter called")
