import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from matplotlib.pylab import *

#a = 22e-3 # Radius, r
a = np.r_[6e-3:20e-3:1000j] # Radius, r
d = 16.5e-3 # height

m = 0
n = 1
p = 2


c = 299792458 # speed of light, m/s

print('TE%i%i%i'%(m,n,p))






def TE_cylindrical(m,n,p,a,d):
    bessel_root = jnp_zeros(m, n)[-1]
    print('Bessel root:', bessel_root)
    term1 = ((c * bessel_root)/ np.pi)**2.0
    term2 = ((c * p / 2.0)**2.0) * ((2.0*a/d)**2.0)

    rhs = term1 + term2

    f = np.sqrt(rhs) / (2.0*a)
    return f

f = TE_cylindrical(m,n,p,a,d)

f_test = TE_cylindrical(0,1,2,6.32e-3, 16.5e-3)
print(f_test/1e9)

figure()
title('TE%i%i%i Cylindrical Mode, height = %0.03f mm'%(m,n,p,d *1e3))
plot(a*1e3,f/1e9)
xlabel('Radius (mm)')
grid(linestyle = ':')
show()
#print('frequency: %0.03f Hz'%f)
#print('frequency: %0.03f GHz'%(f/1e9))

#zeros = jn_zeros(0, 3) # (for TM modes)
#zeros_der = jnp_zeros(0, 3) #derivative, (for TE modes)
#print(zeros)
#print(zeros_der)
