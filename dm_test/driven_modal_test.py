#import modules
from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
#real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element
#from basix.ufl import element, mixed_element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix, LinearProblem
from dolfinx.io import gmshio

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, locate_entities_boundary
import dolfinx.mesh
from slepc4py import SLEPc

import sys

comm = MPI.COMM_WORLD

def mpi_print(s, rank = 0):
    if rank is None:
        print(f"Rank {comm.rank}: {s}")
    elif comm.rank == rank:
        print(f"Rank {comm.rank}: {s}")
    sys.stdout.flush()

#waveguide parameters
a = 1.5
b = 0.9
c = 0.4

fc = 1.0 / (2.0 * a)

nx = 10
ny = 10
nz = 20

mpi_print('Creating Mesh...')
mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[a,b,c]]), np.array([nx, ny, nz]), CellType.hexahedron)
#mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[a,b,c]]), np.array([nx, ny, nz]), CellType.tetrahedron)
mpi_print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)

degree = 1
V = fem.functionspace(mesh, ('N1curl', degree))



def is_pec(x):
    return np.isclose(x[0], a) | np.isclose(x[1], 0.0) | np.isclose(x[1], b) | np.isclose(x[2], 0.0) | np.isclose(x[2], c)
# Identify PEC boundary, x[0] = 0 is waveguide port
pec_facets = dolfinx.mesh.locate_entities_boundary(
    mesh,
    dim=(mesh.topology.dim - 1),
    marker=is_pec)

pec_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=mesh.topology.dim-1, entities=pec_facets)

#u_bc = fem.Function(V)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, pec_bc_dofs)


lmbd0 = 1.5*0.5
k0 = 2 * np.pi / lmbd0

#u = ufl.TrialFunction(V)
u = ufl.TrialFunction(V)
#v = ufl.TestFunction(V)
v = ufl.TestFunction(V)


#def is_port(x):
#    return np.isclose(x[0], 0.0)

def is_port(x):
    return np.isclose(x[0], 0.0)
tdim = mesh.topology.dim
#mpi_print('tdim',tdim)
#port_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim = (tdim - 1), marker = is_port)
port_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim = (tdim - 1), marker = is_port)
#mpi_print('port facets',port_facets)
port_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))
#mpi_print('port markers', port_marker)
ds = ufl.Measure("ds", domain=mesh, subdomain_data=port_marker)

V_G0 = fem.functionspace(mesh, ("DG", 0, (1,)))
port_locations = fem.Function(V_G0) ### allocate for where mesh will be refined
port_locations.x.array[:] = 0
mpi_print('port facets')
for r in range(comm.Get_size()):
    mpi_print(port_facets, r)

mpi_print('pec facets')
for r in range(comm.Get_size()):
    mpi_print(pec_facets, r)

#port_locations.x.array[port_marker] = np.full_like(port_marker, 1.0, dtype=scalar_type)



x = ufl.SpatialCoordinate(mesh)
a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx - k0**2. * ufl.inner(u, v) * ufl.dx
#Y = 1.0
Y = 377.0
#Y = 10000.0
#Y = 0.0

n = ufl.as_vector([1, 0, 0])

L_port = -0.5 * Y * ufl.inner(u,v) * ds # impedance boundary at waveguide port
#L_port = -0.5 * Y * ufl.inner(ufl.cross(n,u),v) * ds # impedance boundary at waveguide port

#L_inc = 1.0 * ufl.inner(ufl.as_vector([0,0,ufl.sin(ufl.pi * x[1]/b)]),v) * ds # incident wave
L_inc = 1.0 * ufl.inner(ufl.as_vector([0,0,ufl.sin(ufl.pi * x[1] / (b))]),v) * ds # incident wave

#L_inc = -1.0 * ufl.inner(ufl.as_vector([0,0,1]),v) * ds # incident wave
#L_inc = -1.0 * ufl.inner(ufl.as_vector([0, ufl.sin(ufl.pi * x[1]/b), 0]),v) * ds # incident wave

weak_form = a + L_port + L_inc
#weak_form = a + L_inc

a = ufl.lhs(weak_form)
L = ufl.rhs(weak_form)

#a = assemble_matrix(fem.form(a), bcs = [bc])
#a.assemble()
#L = assemble_matrix(l, bcs = [bc])
#L = assemble_matrix(fem.form(l), bcs = [bc])
#L.assemble()

#a = fem.form(a)
#b = fem.form(b)



mpi_print('Solving...')
#problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
problem = LinearProblem(
    a,
    L,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)

#problem = dolfinx.fem.petsc.LinearProblem(A, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#problem = dolfinx.fem.petsc.LinearProblem(A, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
#problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": 'mumps'})
E = problem.solve()
mpi_print('Done.')
mpi_print(problem)

gdim = mesh.geometry.dim
V_dg = fem.functionspace(mesh, ("DG", degree, (gdim,)))
E_dg = fem.Function(V_dg)
E_dg.interpolate(E)
E_dg.x.scatter_forward()

#Port_E_inc_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())


#V_tag = fem.functionspace(mesh, ("DG", 0, (gdim,)))

# Save solutions
with io.VTXWriter(mesh.comm, "sols_test/E.bp", E_dg) as f:
    f.write(0.0)
with io.VTXWriter(mesh.comm, "sols_test/port.bp", port_locations) as f:
    f.write(0.0)

with dolfinx.io.XDMFFile(mesh.comm, "sols_test/ft.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
#    xdmf.write_mesh(port_marker)
# xdmf.write_meshtags(facet_tags)

mpi_print('Script Done.')

#mpi_print('Setting up Problem...')
#eps = SLEPc.EPS().create(mesh.comm)
#eps.setOperators(A, B)
#eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)
#
#tol = 1e-9
#max_it = 10000
#eps.setTolerances(tol=tol, max_it=max_it)
#mpi_print('tol and max it:', eps.getTolerances())
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
#mpi_print('Done.')
#
#
#mpi_print('Solving...')
#eps.solve()
#eps.view()
#eps.errorView()
#
#mpi_print('Done.')
#
## Save the kz
#
#ix = 0
#for ix in range(eps.getConverged()):
#    ix += 1
#mpi_print('Total Eigenvalue:', ix)
#
#mpi_print('Negative, Non-trivial Eigenvalues:')
#for i in range(eps.getConverged()):
#    eigen_value = eps.getEigenvalue(i)
#    if np.real(eigen_value) < -0.001:
#        mpi_print(i, eigen_value)
#mpi_print('Done.')
#
#mpi_print('Real, Non-trivial Eigenvalues:')
#for i in range(eps.getConverged()):
#    eigen_value = eps.getEigenvalue(i)
#    if np.real(np.abs(eigen_value)) > 0.001:
#        mpi_print(i, eigen_value)
#mpi_print('Done.')
#
#vals = [(i, np.sqrt(-eps.getEigenvalue(i))) for i in range(eps.getConverged())]
#
## Sort kz by real part
#vals.sort(key=lambda x: x[1].real)
#
#eh = fem.Function(V)
#mpi_print(eh)
#
#kz_list = []
#
#mpi_print('Summary:')
#for i, kz in vals:
##    mpi_print('-'*50)
##    mpi_print('i:',i)
##    mpi_print('kz:',kz)
##    mpi_print(i, kz)
#    # Save eigenvector in eh
#    eps.getEigenpair(i, eh.x.petsc_vec)
#
#    # Compute error for i-th eigenvalue
#    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
##    mpi_print('Error:',error)
##    if error > tol:
##        mpi_print('***DID NOT CONVERGE!!!***')
#
#    # Verify, save and visualize solution
##    if error < tol and np.isclose(kz.imag, 0, atol=tol):
#    if True:
#        kz_list.append(kz)
#
#
##        mpi_print(f"eigenvalue: {-kz**2}")
##        mpi_print(f"kz: {kz}")
##        mpi_print(f"kz/k0: {kz / k0}")
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
##        mpi_print(eth.x.array)
##        mpi_print(ezh.x.array)
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
