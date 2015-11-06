"""
sis.py
Sequential important sampling class object (basic particle filter)
The prior is used as the importance density
"""

import numpy as np
import math

class sis():
    def __init__(self,nx=1,Ns=10,propagateFunction=None,processNoiseSampleFunction=None,measurementPdfFunction=None):
        ## initialization flag, set to 1 only when the initial estimate and weights are given
        self.initFlag = False
        ## number of states
        self.nx = nx
        ## number of particles
        self.Ns = Ns
        ## matrix of particles; each column is a particle
        self.XK = np.zeros((self.nx,self.Ns))
        ## vector of weights
        self.WI = np.zeros(self.Ns)
        ## function that returns a sample of the process noise; this is most likely Gaussian. template: vk = processNoiseSample(xk)
        self.processNoiseSample = processNoiseSampleFunction
        ## function that propagates a state xk with a process noise vk through time dt. template: xk <-- propagateParticle(xk,dt,vk)
        self.propagateParticle = propagateFunction
        ## function that computes the PDF of a measurement y given a prior xki. template: p(yk|xk) = measurementNoisePdf(yk,xk)
        #
        # Commonly our measurement is linear in Gaussian noise and the pdf(y,x) is simply gaussian_pdf(yk-h(xk)), where h(xk) is the nonlinear output function
        self.measurementNoisePdf = measurementPdfFunction
    def init(self,priorSampleFunction):
        for k in range(self.Ns):
            # sample from the sample function
            self.XK[:,k] = priorSampleFunction()
            # initialize all weights equal
            self.WI[k] = 1.0/float(self.Ns)
        # initialize if we got this far and the various function handles were assigned
        if (self.processNoiseSample is not None) and (self.propagateParticle is not None) and (self.measurementNoisePdf is not None):
            self.initFlag = True
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
                self.WI[k] = self.WI[k]*measurementNoisePdf(ymeas,self.XK[:,k])
        else:
            print("Error: uninitialized filter called")
