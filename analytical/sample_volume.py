import numpy as np

sample_radius = 1.1 / 2.0
sample_height = 10.0

max_radius = 2.0 / 2.0
resonator_height = 16.5


V_sample = (np.pi * sample_radius**2.0) * sample_height

V_max_sample = (np.pi*max_radius**2.0) * resonator_height

print(V_max_sample / V_sample)
