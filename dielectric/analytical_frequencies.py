import math
c = 299792458.0

resonatorOD = 40e-3
resonatorHeight = 60e-3


a = resonatorOD/2 # radius
h = resonatorHeight # height

def f_from_chi_ell(chi, ell):
    k = math.sqrt((chi/a)**2 + (ell*math.pi/h)**2)
    return c/(2*math.pi)*k

# example zeros
j01 = 2.404825557695773
j11 = 3.831705970207512
j1p1 = 1.841183781340659

print('-'*50)
print("TE011:", f_from_chi_ell(j11,1)/1e9, "GHz")  # j11 == J0' first zero

print('-'*50)
print("TM010:", f_from_chi_ell(j01,0)/1e9, "GHz")
print("TM011:", f_from_chi_ell(j01,1)/1e9, "GHz")
print("TM110:", f_from_chi_ell(j11,0)/1e9, "GHz")
print("TE111:", f_from_chi_ell(j1p1,1)/1e9, "GHz")
