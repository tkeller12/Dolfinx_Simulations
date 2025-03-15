import numpy as np
from matplotlib.pylab import *

t = np.r_[0:np.pi*10:1000j]

v = np.exp(1j*5*t)
#v = np.real(v)

figure()
for ix in range(10):
    figure('complex')
    plot(t, 2*ix + np.real(v * np.exp(1j*ix*2*np.pi/10)))
    figure('mag')
    plot(t, 2*ix + np.abs(v * np.exp(1j*ix*2*np.pi/10)))
show()


