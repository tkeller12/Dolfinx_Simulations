import numpy as np

pts = 100
x = np.random.randn(pts)

out = np.percentile(x, 99.9999999)
sorted_out = np.argsort(x)

s = x[x > out]

print(x)
print(out)
print(sorted_out)
print(s)

s2 = np.arange(len(x))[x > out]
print(s2)


