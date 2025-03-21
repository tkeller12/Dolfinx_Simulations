import numpy as np
from matplotlib.pylab import *

sample_height = 16.5
sample_radius = 1.1/2.0

resonator_height = 16.5
resonator_radius = 13.8/2.0

Vs = np.pi * sample_radius**2.0 * sample_height
Vc = np.pi * resonator_radius**2.0 * resonator_height

eta = 12.33 * (Vs/Vc) * (2./3.)

print('filling factor: %0.03f'%eta)
