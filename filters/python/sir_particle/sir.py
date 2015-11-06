"""
sir.py
Sampling importance resampling (SIR) filter
Inherits from the SIS class
"""

import numpy as np
import sys
sys.path.append('../sis_particle')
import sis
sys.path.append('../lib')
import particle_utilities as pu

class sir(sis.sis):
    '''
    def update(self,dt,ymeas):
        if self.initFlag:
            # for each particle
            for k in range(self.Ns):
                # draw process noise according to the process noise model
                vki = self.processNoiseSample(self.XK[:,k])
                # propagate the particle
                self.XK[:,k] = self.propagateParticle(self.XK[:,k],dt,vki)
            # compute the weights
            for k in range(self.Ns):
                # using the prior for the weight update, each updated weight is simpy the current value times the probability of the measurement given the prior state
                self.WI[k] = self.WI[k]*self.measurementNoisePdf(ymeas,self.XK[:,k])
            # normalize the weights
            weightSum = 1.0/np.sum(self.WI)
            for k in range(self.Ns):
                self.WI[k] = self.WI[k]*weightSum
        else:
            print("Error: uninitialized filter called")
    '''
    def sample(self):
        # resample
        (self.XK,self.WI,unusedv) = pu.resample(self.XK,self.WI)
