import sis
import numpy as np
import math # for exp()

## measurement noise
sigma_y = 0.01
## initial value of state
X0 = np.array([1.0,0.0])
## uncertainty in the initial state
SIGMA_X0 = np.array([0.1,0.1])

def eqom(x,t):
    dx = np.zeros(2)
    dx[0] = x[1]
    dx[1] = -2.0*x[0]
    return dx

def measurement(x):
    return np.array([x[0] + np.random.normal()*sigma_y])

## Function for the filter to propagate a particle. Assumes dynamics: /ddot{x} = -1.5*x
#
#   Uses Euler first order itnegration
#   @param xk Particle
#   @param dt time through which to integrate
#   @param vk process noise
def propagateFunction(xk,dt,vk):
    # Euler first order approximation
    dxk = np.zeros(2)
    dxk[0] = xk[1]
    dxk[1] = -1.5*x[0] + vk[0]
    xk = xk + dt*xdk
    return xk

## Function for the filter to compute the PDF of a measurement given a prior
#
#   @param yt: the measurement
#   @param xk: the prior (a particle)
#   @output py_x: the PDF of yt, given xk
def measurementPdf(yt,xk):
    # we're measureing position, so the expectation yk is simply xk[0]:
    yk = x[0]
    # compute the error w.r.t. the measurement
    nk = yt[0] - yk
    # evaluate the Gaussian PDF with mean 0.0 and standard deviation sigma_y
    py_x = math.exp(0.5*(nk/sigma_y)*(nk/sigma_y))/(math.sqrt(2.0*math.pi)*sigma_y)
    return py_x

## Function for the filter that returns an appropriate process noise sample
#
# Our process noise is based 100% on measurement error. We cannot know the error exactly IRL, but we might know that it's linear in the state
#   @param xk the current state
def processNoise(xk):
    #draw from the normal distribution
    vk = np.random.uniform(low=-0.5*xk[0],high=0.5*xk[0])
    return vk

def initialParticle():
    err = np.zeros(2)
    err[0] = np.random.normal()*SIGMA_X0[0]
    err[1] = np.random.normal()*SIGMA_X0[1]
    return (X0 + err)

def main():
    SIS = sis.sis(2,10,propagateFunction,processNoise,measurementPdf)

    print("SIS.is_init: %d" % (SIS.initFlag))
    SIS.init(initialParticle)
    print("SIS.is_init: %d" % (SIS.initFlag))

    tf = 1.0
    dt = 0.1
    nSteps = int(tf/dt)
    xsim = X0.copy()
    tsim = 0.0

    for k in range(nSteps):
        # propagate the simulation
        dx = eqom(xsim,tsim)
        # Euler integrate
        xsim = xsim + dt*dx
        # update time
        tsim = tsim + dt
        # generate a measurement
        yt = measurement(xsim)
        # call SIS
        SIS.update(dt,yt)
        print("%f,%f,%f,%f" % (tsim,yt[0],xsim[0],xsim[1]))

    print("Exiting sis_test.py")
    return

if __name__ == '__main__':
    main()
