import numpy as np
from scipy.special import jv, jn_zeros, jnp_zeros

from matplotlib.pylab import *

m = 0
n = 1
p = 2

x = np.r_[-10:10:1000j].reshape(-1,1)
y = np.r_[-10:10:1000j].reshape(1,-1)

z = jv(m,np.sqrt(x**2 + y**2))

zeros = jn_zeros(0, 3)
zeros_der = jnp_zeros(0, 3)
print(zeros)
print(zeros_der)

print(z.shape)


figure()
imshow(z)
show()
