#import modules
from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix, LinearProblem
from dolfinx.io import gmshio

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, locate_entities_boundary
import dolfinx.mesh
from slepc4py import SLEPc

#waveguide parameters
a = 5.0
b = 0.9
c = 0.4

nx = 8
ny = 8
nz = 12

print('Creating Mesh...')
mesh = create_box(MPI.COMM_WORLD, np.array([[0,0,0],[a,b,c]]), np.array([nx, ny, nz]), CellType.hexahedron)
print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
mesh.topology.create_connectivity(mesh.topology.dim-2,mesh.topology.dim)

degree = 1
V = fem.functionspace(mesh, ('N1curl', degree))

# Identify PEC boundary, x[0] = 0 is waveguide port
pec_facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    dim=(mesh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], a) | np.isclose(x[1], 0.0) | np.isclose(x[1], b) | np.isclose(x[2], 0.0) | np.isclose(x[2], c))

pec_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=pec_facets)

u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, pec_bc_dofs)


lmbd0 = 0.1
k0 = 2 * np.pi / lmbd0

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(mesh)


def is_port(x):
    return np.isclose(x[0], 0.0)
tdim = mesh.topology.dim
port_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, is_port)
port_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))
ds = ufl.Measure("ds", domain=mesh, subdomain_data=port_marker)


a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx - k0**2. * ufl.inner(u, v) * ufl.dx
Y = 1000.0

n = ufl.as_vector([0, 0, -1])

L_port = Y * ufl.inner(u,v) * ds # impedance boundary at waveguide port

L_inc = Y * ufl.inner(ufl.as_vector([0,0,ufl.sin(ufl.pi * x[1]/b)]),v) * ds

#L = L_port + L_inc
weak_form = a + L_port + L_inc

a = ufl.lhs(weak_form)
L = ufl.rhs(weak_form)

#a = fem.form(a)
#b = fem.form(b)



print('Applying Boundary Conditions...')
#bc_facets = exterior_facet_indices(mesh.topology)
#bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
#u_bc = fem.Function(V)
#with u_bc.x.petsc_vec.localForm() as loc:
#    loc.set(0)
#bc = fem.dirichletbc(u_bc, bc_dofs)



print('Done.')

problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
E = problem.solve()


print('Assembling Matrix...')
#A = assemble_matrix(a, bcs=[bc])
#A.assemble()
#B = assemble_matrix(b, bcs=[bc])
#B.assemble()
print('Done.')

gdim = mesh.geometry.dim
V_dg = fem.functionspace(mesh, ("DG", degree, (gdim,)))
E_dg = fem.Function(V_dg)
E_dg.interpolate(E)

# Save solutions
with io.VTXWriter(mesh.comm, "sols_test/E.bp", E_dg) as f:
    f.write(0.0)



#print('Setting up Problem...')
#eps = SLEPc.EPS().create(mesh.comm)
#eps.setOperators(A, B)
#eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
#
#tol = 1e-9
#max_it = 10000
#eps.setTolerances(tol=tol, max_it=max_it)
#print('tol and max it:', eps.getTolerances())
#
#eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)
##eps.setType(SLEPc.EPS.Type.ARNOLDI) # No Improvement, 5 eigenavlues, 10 requested
##eps.setType(SLEPc.EPS.Type.LAPACK) # All Eigenvalues
#
#
## Get ST context from eps
#st = eps.getST()
#
## Set shift-and-invert transformation
#st.setType(SLEPc.ST.Type.SINVERT)
#st.setShift(0.1)
#st.setFromOptions()
##st.setType(SLEPc.ST.Type.SHIFT) # Two eigenvalue converged
##st.setType(SLEPc.ST.Type.CAYLEY)
#
#eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
##eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_MAGNITUDE)
#
##eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_IMAGINARY) # not supported
#
##st.setType(SLEPc.ST.Type.SHIFT)
##eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
#
##st.setType(SLEPc.ST.Type.CAYLEY)
##eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)
#
##eps.setWhichEigenpairs(SLEPc.EPS.Which.LARGEST_REAL)
#
#eps.setTarget(50)
#
#eps.setDimensions(nev=4)
#print('Done.')
#
#
#print('Solving...')
#eps.solve()
#eps.view()
#eps.errorView()
#
#print('Done.')
#
## Save the kz
#
#ix = 0
#for ix in range(eps.getConverged()):
#    ix += 1
#print('Total Eigenvalue:', ix)
#
#print('Negative, Non-trivial Eigenvalues:')
#for i in range(eps.getConverged()):
#    eigen_value = eps.getEigenvalue(i)
#    if np.real(eigen_value) < -0.001:
#        print(i, eigen_value)
#print('Done.')
#
#print('Real, Non-trivial Eigenvalues:')
#for i in range(eps.getConverged()):
#    eigen_value = eps.getEigenvalue(i)
#    if np.real(np.abs(eigen_value)) > 0.001:
#        print(i, eigen_value)
#print('Done.')
#
#vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]
#
## Sort kz by real part
#vals.sort(key=lambda x: x[1].real)
#
#eh = fem.Function(V)
#print(eh)
#
#kz_list = []
#
#print('Summary:')
#for i, kz in vals:
##    print('-'*50)
##    print('i:',i)
##    print('kz:',kz)
##    print(i, kz)
#    # Save eigenvector in eh
#    eps.getEigenpair(i, eh.x.petsc_vec)
#
#    # Compute error for i-th eigenvalue
#    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
##    print('Error:',error)
##    if error > tol:
##        print('***DID NOT CONVERGE!!!***')
#
#    # Verify, save and visualize solution
##    if error < tol and np.isclose(kz.imag, 0, atol=tol):
#    if True:
#        kz_list.append(kz)
#
#
##        print(f"eigenvalue: {-kz**2}")
##        print(f"kz: {kz}")
##        print(f"kz/k0: {kz / k0}")
#
#        eh.x.scatter_forward()
#
##        eth, ezh = eh.split()
#        eth = eh
##        eth = eh.sub(0).collapse()
##        ez = eh.sub(1).collapse()
#
#        # Transform eth, ezh into Et and Ez
#        eth.x.array[:] = eth.x.array[:]
##        ezh.x.array[:] = ezh.x.array[:] * 1j
#
##        print(eth.x.array)
##        print(ezh.x.array)
#
#
#        gdim = mesh.geometry.dim
#        V_dg = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
#        Et_dg = fem.Function(V_dg)
#        Et_dg.interpolate(eth)
#
#        # Save solutions
#        with io.VTXWriter(mesh.comm, "sols_test/Et_%04i.bp"%i, Et_dg) as f:
#            f.write(0.0)
#
##        with io.VTXWriter(mesh.comm, "sols_test/Ez_%04i.bp"%i, ezh) as f:
##            f.write(0.0)
#
#
#print('Script Done.')
