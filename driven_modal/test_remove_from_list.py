import numpy as np

a = np.r_[1:10]
b = np.r_[3,5,7]


def remove(x, remove_values):
    new_x = list(x)
    remove_values = list(remove_values)
    for each in remove_values:
        new_x.remove(each)

    return np.array(new_x)

print('a', a)
print('a', b)

c = remove(a,b)

print('a', a)
print('a', b)
print('c', c)
