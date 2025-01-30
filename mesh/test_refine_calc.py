import numpy as np

pts = 100
x = np.random.randn(pts)

out = np.percentile(x, 90)

s = x[x > out]

print(x)
print(out)
print(s)


