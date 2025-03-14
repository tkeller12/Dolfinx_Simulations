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
a = 1.0 # waveguide a, x
b = 0.5 # waveguide b, z
d = 5.0 # length of cavity, y 

#lmbd0 = 1.5*0.82
#lmbd0 = 1.23
#lmbd0 = 1.21 # on resonance
lmbd0 = 1.2
k0 = 2 * np.pi / lmbd0

fc = 1.0 / (2.0 * a)

nx = 10
ny = 50
nz = 10

#beta = np.sqrt(omega**2 * mu*epsilon - (np.pi/a)**2)

filename = 'TE102_test001.msh'
mpi_print('Creating Mesh...')
#mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename, comm, 0, gdim=3)
mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[a,d,b]]), np.array([nx, ny, nz]), CellType.hexahedron)
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
def is_port(x):
    return np.isclose(x[1], 0.0)

def is_port2(x):
    return np.isclose(x[1], d)

def np_remove(x, remove_values):
    new_x = list(x)
    remove_values = list(remove_values)
    for each in remove_values:
        new_x.remove(each)

    return np.array(new_x)

port_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim = (tdim - 1), marker = is_port)
port2_facets = dolfinx.mesh.locate_entities_boundary(mesh, dim = (tdim - 1), marker = is_port2)
facets = exterior_facet_indices(mesh.topology)
port_markers = np.full(len(facets), 0, dtype=np.int32)  # default unmarked
for i, facet in enumerate(facets):
    if facet in port_facets:
        port_markers[i] = 1
    elif facet in port2_facets:
        port_markers[i] = 2

#print(port_markers)
exterior_facets = exterior_facet_indices(mesh.topology)

#pec_facets = dolfinx.mesh.locate_entities_boundary(mesh,dim=(tdim - 1), marker=is_pec)
pec_facets = np_remove(exterior_facets, port_facets)
pec_facets = np_remove(pec_facets, port2_facets)
pec_bc_dofs = fem.locate_dofs_topological(V=V, entity_dim=(tdim-1), entities=pec_facets)

u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, pec_bc_dofs)

#port_markers = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))
port_markers = dolfinx.mesh.meshtags(mesh, tdim - 1, facets, port_markers)
#port_markers = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))

print(port_markers.values)

ds_port = ufl.Measure("ds", domain=mesh, subdomain_data=port_markers)



u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)


V_G0 = fem.functionspace(mesh, ("DG", 0, (1,)))
port_locations = fem.Function(V_G0) ### allocate for where mesh will be refined
port_locations.x.array[:] = 0


x = ufl.SpatialCoordinate(mesh)
A = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx - k0**2. * ufl.inner(u, v) * ufl.dx
Y = 1.0
#Y = 377.0
#Y = 10000.0

#def calc_gamma(m, n, a, b, k0):
#    a_term = (m * np.pi / a)**2.0
#    b_term = (n * np.pi / b)**2.0
#    if (a_term + b_term) <= k0**2.0:
#        return 1j* np.sqrt(k0**2.0 - a_term - b_term)
#    else:
#        return np.sqrt(a_term + b_term - k**2.0)
#
#gamma = calc_gamma(1, 0, a, b, k0)
mpi_print('GAMMA:')
#mpi_print(gamma)
gamma = 1.0


TE10 = ufl.as_vector([0,0,ufl.sin(ufl.pi * x[0] / (a))])

L_port = 0.5 *  Y * ufl.inner(u,v) * ds_port(1) # impedance boundary at waveguide port
L_port2 = 0.5 *  Y * ufl.inner(u,v) * ds_port(2) # impedance boundary at waveguide port
#L_port_H = -1.0 * ufl.inner(u,ufl.cross(ufl.FacetNormal(mesh),ufl.curl(v))) * ds_port(1) # incident wave

L_inc = -2.0 * gamma * ufl.inner(TE10,v) * ds_port(1) # incident wave
##L_inc_H = 1.0 * ufl.inner(u,ufl.cross(ufl.FacetNormal(mesh),ufl.curl(TE10))) * ds_port(1) # incident wave
#L_inc_H = 1.0 * ufl.inner(u,ufl.cross(ufl.FacetNormal(mesh),ufl.curl(TE10))) * ds_port(1) # incident wave

#L_inc_H = -1.0 * ufl.inner(u,ufl.cross(ufl.FacetNormal(mesh),ufl.curl(v))) * ds_port(1) # incident wave

weak_form = A + L_port + L_inc + L_port2

A = ufl.lhs(weak_form)
L = ufl.rhs(weak_form)

V_port = fem.functionspace(mesh, ("CG", degree, (gdim,)))
port = fem.Function(V_port)
L_inc_expr = fem.Expression(TE10, V_port.element.interpolation_points(), comm)
port.interpolate(L_inc_expr)
port.x.scatter_forward()


mpi_print('Solving...')
problem = LinearProblem(
    A,
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
#V_ref_local = fem.assemble_scalar(fem.form(ufl.inner(E,TE10) * ds_port(1)))
#V_inc_local = fem.assemble_scalar(fem.form(ufl.inner(TE10,TE10) * ds_port(1)))

# Normalize E-field
#N_local = fem.assemble_scalar(fem.form(ufl.inner(TE10, ufl.conj(TE10)) * ds_port(1)))
#N_global = mesh.comm.allreduce(N_local, op=MPI.SUM)
#normalization_factor = np.sqrt(N_global)
#E_norm = E / normalization_factor

#N_local = fem.assemble_scalar(fem.form(ufl.inner(TE10, ufl.conj(TE10)) * ds_port(1)))
#N_global = mesh.comm.allreduce(N_local, op=MPI.SUM)
#normalization_factor = np.sqrt(N_global)
#E_norm = E / normalization_factor


#V_ref_local = abs(fem.assemble_scalar(fem.form(ufl.dot(E,TE10) * ds_port(1))))
#V_ref_local = fem.assemble_scalar(fem.form(ufl.dot(E,TE10) * ds_port(1)))
#V_inc_local = fem.assemble_scalar(fem.form(ufl.dot(TE10,TE10) * ds_port(1)))

V_ref_local = fem.assemble_scalar(fem.form(ufl.inner(E,TE10) * ds_port(1)))
V_inc_local = fem.assemble_scalar(fem.form(ufl.inner(TE10,TE10) * ds_port(1)))
V_ref = mesh.comm.allreduce(V_ref_local, op=MPI.SUM)
V_inc = mesh.comm.allreduce(V_inc_local, op=MPI.SUM)

mpi_print('S-Parameter Calculation')
mpi_print(V_ref)
mpi_print(V_inc)
mpi_print(V_ref/V_inc)
mpi_print((V_ref/V_inc) - 1)

# Save solutions
with io.VTXWriter(mesh.comm, "sols_test/E.bp", E_dg) as f:
    f.write(0.0)

with io.VTXWriter(mesh.comm, "sols_test/port.bp", port) as f:
    f.write(0.0)
#    xdmf.write_mesh(port_marker)
# xdmf.write_meshtags(facet_tags)

mpi_print('Script Done.')

