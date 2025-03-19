import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from matplotlib.pylab import *

a = 22e-3 # Radius, r
d = 86e-3 # height

m = 0
n = 1
p = 2


c = 299792458 # speed of light, m/s

print('TE%i%i%i'%(m,n,p))

bessel_root = jnp_zeros(m, n)[-1]

print('Bessel root:', bessel_root)
term1 = ((c * bessel_root)/ np.pi)**2.0
term2 = ((c * p / 2.0)**2.0) * ((2.0*a/d)**2.0)

rhs = term1 + term2

f = np.sqrt(rhs) / (2.0*a)

print('frequency: %0.03f Hz'%f)
print('frequency: %0.03f GHz'%(f/1e9))

#zeros = jn_zeros(0, 3) # (for TM modes)
#zeros_der = jnp_zeros(0, 3) #derivative, (for TE modes)
#print(zeros)
#print(zeros_der)
