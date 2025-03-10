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

#TE102 cavity parameters
a = 1.0 # waveguide a, z
b = 0.5 # waveguide b, x
d = 1.5 # length of cavity, y 

lmbd0 = 1.5*0.82
k0 = 2 * np.pi / lmbd0

fc = 1.0 / (2.0 * a)

nx = 50
ny = 20
nz = 10

filename = 'TE102_test001.msh'
mpi_print('Creating Mesh...')
mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename, comm, 0, gdim=3)
#mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[a,b,c]]), np.array([nx, ny, nz]), CellType.hexahedron)
#mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[a,b,c]]), np.array([nx, ny, nz]), CellType.tetrahedron)
mpi_print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

degree = 2
V = fem.functionspace(mesh, ('N1curl', degree))


### Boundary Conditions ###
# Identify PEC boundary, x[0] = 0 is waveguide port
#def is_pec(x):
#    return np.isclose(x[0], a) | np.isclose(x[1], 0.0) | np.isclose(x[1], b) | np.isclose(x[2], 0.0) | np.isclose(x[2], c)
#
# Identify Waveguide Port Boundary location
#def is_port(x):
#    return np.isclose(x[0], 0.0)

def np_remove(x, remove_values):
    new_x = list(x)
    remove_values = list(remove_values)
    for each in remove_values:
        new_x.remove(each)

    return np.array(new_x)

port_facets = facet_tags.find(2)

exterior_facets = exterior_facet_indices(mesh.topology)

#pec_facets = dolfinx.mesh.locate_entities_boundary(mesh,dim=(tdim - 1), marker=is_pec)
pec_facets = np_remove(exterior_facets, port_facets)

pec_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=(tdim-1), entities=pec_facets)

u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, pec_bc_dofs)

#port_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim = (tdim - 1), marker = is_port)
port_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))
ds_port = ufl.Measure("ds", domain=mesh, subdomain_data=port_marker)



u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


V_G0 = fem.functionspace(mesh, ("DG", 0, (1,)))
port_locations = fem.Function(V_G0) ### allocate for where mesh will be refined
port_locations.x.array[:] = 0


x = ufl.SpatialCoordinate(mesh)
a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx - k0**2. * ufl.inner(u, v) * ufl.dx
Y = 1.0
#Y = 377.0
#Y = 10000.0
#Y = 0.0

#n = ufl.as_vector([1, 0, 0])

TE10 = ufl.as_vector([ufl.cos(ufl.pi * x[2] / (d)),0,0])

L_port = -0.5 *  Y * ufl.inner(u,v) * ds_port(1) # impedance boundary at waveguide port
#L_inc = 1.0 * ufl.inner(ufl.as_vector([0,0,ufl.sin(ufl.pi * x[1] / (b))]),v) * ds_port(1) # incident wave
L_inc = 1.0 * ufl.inner(TE10,v) * ds_port(1) # incident wave

weak_form = a + L_port + L_inc

a = ufl.lhs(weak_form)
L = ufl.rhs(weak_form)

V_port = fem.functionspace(mesh, ("CG", degree, (gdim,)))
port = fem.Function(V_port)
#L_inc_expr = fem.Expression(L_inc, V_port.element.interpolation_points(), comm)
L_inc_expr = fem.Expression(ufl.as_vector([ufl.cos(ufl.pi * x[2] / (d)),0,0]), V_port.element.interpolation_points(), comm)
port.interpolate(L_inc_expr)
port.x.scatter_forward()


mpi_print('Solving...')
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

E = problem.solve()
mpi_print('Done.')
mpi_print(problem)

gdim = mesh.geometry.dim
V_dg = fem.functionspace(mesh, ("DG", degree, (gdim,)))
E_dg = fem.Function(V_dg)
E_dg.interpolate(E)
E_dg.x.scatter_forward()

# Calculate S-parameters
V_ref_local = fem.assemble_scalar(fem.form(ufl.inner(E,TE10) * ds_port(1)))
V_inc_local = fem.assemble_scalar(fem.form(ufl.inner(TE10,TE10) * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(ufl.inner(ufl.as_vector([1,1,1]),ufl.as_vector([1,1,1])) * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(1.0 * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(ufl.inner(E, E) * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(ufl.inner(E, E) * ufl.ds))
#V_test = fem.assemble_scalar(fem.form(ufl.inner(E, E) * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(ufl.dot(E, E) * ds_port(1)))
#V_test = fem.assemble_scalar(fem.form(ufl.dot(E_dg, E_dg) * ds_port(1)))
V_test = fem.assemble_scalar(fem.form(ufl.inner(E_dg,E_dg) * ds_port(1)))

global_value = mesh.comm.allreduce(V_test, op=MPI.SUM)
V_ref = mesh.comm.allreduce(V_ref_local, op=MPI.SUM)
V_inc = mesh.comm.allreduce(V_inc_local, op=MPI.SUM)
#V_test = fem.assemble_scalar(fem.form(ufl.inner(E,E) * ufl.dx))

mpi_print('S-Parameter Calculation')
mpi_print(V_ref)
mpi_print(V_inc)
mpi_print(V_ref/V_inc)
#mpi_print(V_test)
#mpi_print(global_value)
#mpi_print(V_ref/V_inc)
#Port_E_inc_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())


#V_tag = fem.functionspace(mesh, ("DG", 0, (gdim,)))

# Save solutions
with io.VTXWriter(mesh.comm, "sols_test/E.bp", E_dg) as f:
    f.write(0.0)
with io.VTXWriter(mesh.comm, "sols_test/port.bp", port) as f:
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
