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

from dolfinx.mesh import CellType, create_rectangle, exterior_facet_indices, locate_entities

from slepc4py import SLEPc

#waveguide parameters
a = 0.9
b = 0.4

nx = 100
ny = 100

print('Creating Mesh...')
mesh = create_rectangle(MPI.COMM_WORLD, np.array([[0,0],[a,b]]), np.array([nx, ny]), CellType.quadrilateral)
#mesh = create_rectangle(MPI.COMM_WORLD, np.array([[0,0],[a,b]]), np.array([nx, ny]), CellType.triangle) # doesn't work with RTCE elements
print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)

# Creating dielectric function space
#D = fem.functionspace(mesh, ("DQ", 0))
#eps = fem.Function(D)

vector_degree = 2
nodal_degree = 3
degree = 2
RTCE = element("RTCE", mesh.basix_cell(), vector_degree, dtype=real_type)
Q = element("Lagrange", mesh.basix_cell(), nodal_degree, dtype=real_type)
V = fem.functionspace(mesh, mixed_element([RTCE, Q]))

#lmbd0 = 1.0
lmbd0 = 0.4
k0 = 2 * np.pi / lmbd0

eps_r = 1.

et, ez = ufl.TrialFunctions(V)
vt, vz = ufl.TestFunctions(V)

a_tt = (ufl.inner(ufl.curl(et), ufl.curl(vt)) - (k0**2) * eps_r * ufl.inner(et, vt)) * ufl.dx
b_tt = ufl.inner(et, vt) * ufl.dx
b_tz = ufl.inner(et, ufl.grad(vz)) * ufl.dx
b_zt = ufl.inner(ufl.grad(ez), vt) * ufl.dx
b_zz = (ufl.inner(ufl.grad(ez), ufl.grad(vz)) - (k0**2) * eps_r * ufl.inner(ez, vz)) * ufl.dx

a = fem.form(a_tt)
b = fem.form(b_tt + b_tz + b_zt + b_zz)



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
max_it = 1111111
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

#eps.setTarget(-((0.5 * k0) ** 2))
#eps.setTarget(-29.5273)
#eps.setTarget(-29.5273*2)
#eps.setTarget(-29.52/2)
eps.setTarget(-100)

eps.setDimensions(nev=20)
print('Done.')


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

print('Negative, Non-trivial Eigenvalues:')
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
    if np.real(eigen_value) < -0.001:
        print(i, eigen_value)
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

kz_list = []

print('Summary:')
for i, kz in vals:
#    print('-'*50)
#    print('i:',i)
#    print('kz:',kz)
#    print(i, kz)
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
#    print('Error:',error)
#    if error > tol:
#        print('***DID NOT CONVERGE!!!***')

    # Verify, save and visualize solution
#    if error < tol and np.isclose(kz.imag, 0, atol=tol):
    if True:
        kz_list.append(kz)


#        print(f"eigenvalue: {-kz**2}")
#        print(f"kz: {kz}")
#        print(f"kz/k0: {kz / k0}")

        eh.x.scatter_forward()

        eth, ezh = eh.split()
        eth = eh.sub(0).collapse()
        ez = eh.sub(1).collapse()

        # Transform eth, ezh into Et and Ez
        eth.x.array[:] = eth.x.array[:] / kz
        ezh.x.array[:] = ezh.x.array[:] * 1j

#        print(eth.x.array)
#        print(ezh.x.array)


        gdim = mesh.geometry.dim
        V_dg = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        # Save solutions
        with io.VTXWriter(mesh.comm, "sols_test/Et_%04i.bp"%i, Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(mesh.comm, "sols_test/Ez_%04i.bp"%i, ezh) as f:
            f.write(0.0)


print('Script Done.')
