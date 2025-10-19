#import modules
from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities

from slepc4py import SLEPc


filename = 'cylindrical_resonator_test001.msh'

print('Creating Mesh...')
mesh, cell, facet_tags = gmshio.read_from_msh(filename, MPI.COMM_WORLD, 0, gdim=3)
print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)

c = 299792458 # speed of light, m/s

target_freq = 9e9

def mpi_print(s, rank = 0):
    if rank is None:
        print(f"Rank {comm.rank}: {s}")
    elif comm.rank == rank:
        print(f"Rank {comm.rank}: {s}")
    sys.stdout.flush()

def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target


target_eigenvalue = convert_freq_to_target(target_freq)


nev = 10

degree = 2
V = fem.functionspace(mesh, ('N2curl', degree))

lmbd0 = 1.0
k0 = 2 * np.pi / lmbd0

eps_r = 1.

print('Defining problem...')
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx
b = ufl.inner(u, v) * ufl.dx

a = fem.form(a)
b = fem.form(b)
print('Done.')

print('Applying Boundary Conditions...')
bc_facets = exterior_facet_indices(mesh.topology)
bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
print('Done.')



print('Assembling Matrix...')
A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()
print('Done.')


print('Setting up Problem...')
eps = SLEPc.EPS().create(mesh.comm)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-9
max_it = 10000
eps.setTolerances(tol=tol, max_it=max_it)
print('tol and max it:', eps.getTolerances())

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
#eps.setType(SLEPc.EPS.Type.ARNOLDI) # No Improvement, 5 eigenavlues, 10 requested
#eps.setType(SLEPc.EPS.Type.LAPACK) # All Eigenvalues


# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()
#st.setType(SLEPc.ST.Type.SHIFT) # Two eigenvalue converged
#st.setType(SLEPc.ST.Type.CAYLEY)

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
#eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)

#eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY) # not supported

#st.setType(SLEPc.ST.Type.SHIFT)
#eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

#st.setType(SLEPc.ST.Type.CAYLEY)
#eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

#eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)

eps.setTarget(target_eigenvalue)

eps.setDimensions(nev=nev)
print('Done.')
ksp =  st.getKSP()
ksp.setType('preonly')

pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('superlu_dist') # takes up additional memory


print('Solving...')
eps.solve()
eps.view()
eps.errorView()

print('Done.')

# Save the kz

ix = 0
for ix in range(eps.getConverged()):
    ix += 1
print('Total Eigenvalue:', ix)

print('Junk Eigenvalues:')
junk_values = []
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
#    if np.real(eigen_value) < -0.001:
    if np.isclose(np.real(eigen_value), 1):
        print(i, eigen_value)
        junk_values.append(i)
print('Done.')

print('Real, Non-trivial Eigenvalues:')
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
    if np.real(np.abs(eigen_value)) > 0.001:
        print(i, eigen_value)
print('Done.')

vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
vals.sort(key=lambda x: x[1].real)

eh = fem.Function(V)
print(eh)

kz_list = []

print('Summary:')
for i, kz in vals:
#    print('-'*50)
#    print('i:',i)
#    print('kz:',kz)
#    print(i, kz)
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)
    this_frequency = convert_eigenvalue_to_f(abs(kz**2.)) / 1e9
    print(i, this_frequency, ' GHz')
    freq_string = '%0.03fGHz'%this_frequency

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
    print('Error:',error)
    if error > tol:
        print('***DID NOT CONVERGE!!!***')

    # Verify, save and visualize solution
#    if error < tol and np.isclose(kz.imag, 0, atol=tol):
    if True:
        kz_list.append(kz)


#        print(f"eigenvalue: {-kz**2}")
#        print(f"kz: {kz}")
#        print(f"kz/k0: {kz / k0}")

        eh.x.scatter_forward()

#        eth, ezh = eh.split()
        eth = eh
#        eth = eh.sub(0).collapse()
#        ez = eh.sub(1).collapse()

        # Transform eth, ezh into Et and Ez
        eth.x.array[:] = eth.x.array[:]
#        ezh.x.array[:] = ezh.x.array[:] * 1j

#        print(eth.x.array)
#        print(ezh.x.array)


        gdim = mesh.geometry.dim
        V_dg = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        B = fem.Function(V_dg)
        B_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())
        B.interpolate(B_expr)
        B.x.scatter_forward()


        # Save solutions
#        with io.VTXWriter(mesh.comm, "sols_test/Et_%04i_%s.bp"%(i,freq_string), Et_dg) as f:
        with io.VTXWriter(mesh.comm, "sols_test/E_%04i.bp"%i, Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(mesh.comm, "sols_test/H_%04i.bp"%i, B) as f:
            f.write(0.0)

#        with io.VTXWriter(mesh.comm, "sols_test/Ez_%04i.bp"%i, ezh) as f:
#            f.write(0.0)


print('Script Done.')

