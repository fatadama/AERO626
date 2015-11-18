"""@package rk4
Contains simple RK4 integrator
"""

import numpy as np

## rk4Simple Simple fixed-accuracy Runge-Kutta fourth-order propagator for a fixed time step 'h'
#
#   @param[in] fun function arguments that acts as follows: dx = fun(x,t,u)
#   @param[in] x initial state
#   @param[in] t initial time
#   @param[in] h step-size for integration
#   @param[in] u additional arguments to derivative function
#   @param[out] xf state at (t0 + h)
def rk4Simple(fun,x,t,h,u=None):
    h6 = 1.0/6.0
    if u is not None:
        # evaluate at the first point
        k1 = h*fun(x,t,u)
        # second point
        k2 = h*fun(x+0.5*k1,t+0.5*h,u)
        # third point
        k3 = h*fun(x+0.5*k2,t+0.5*h,u)
        # final point
        k4 = h*fun(x+k3,t+h,u)
    else:
        # evaluate at the first point
        k1 = h*fun(x,t)
        # second point
        k2 = h*fun(x+0.5*k1,t+0.5*h)
        # third point
        k3 = h*fun(x+0.5*k2,t+0.5*h)
        # final point
        k4 = h*fun(x+k3,t+h)
    # update x
    return (x + h6*(k1+2.0*k2+2.0*k3+k4))
