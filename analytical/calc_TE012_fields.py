
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

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, locate_entities_boundary, meshtags
import dolfinx.mesh

#from slepc4py import SLEPc


import sys

comm = MPI.COMM_WORLD

def mpi_print(s, rank = 0):
    if rank is None:
        print(f"Rank {comm.rank}: {s}")
    elif comm.rank == rank:
        print(f"Rank {comm.rank}: {s}")
    sys.stdout.flush()

#mpi_print(PETSc.ScalarType)

#TE102 cavity parameters
a = 1.0 # waveguide a, z
b = 0.5 # waveguide b, x
h = 1.7 # length of cavity, z 

nx = 50
ny = 20
nz = 10

mpi_print('Creating Mesh...')
#mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename, comm, 0, gdim=3)
mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[b,d,a]]), np.array([nx, ny, nz]), CellType.hexahedron)
mpi_print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

degree = 4
V = fem.functionspace(mesh, ("CG", degree, (gdim,)))

def TE102_mode(x):
    # x: (3, n) array of coordinates on the port boundary
    E0 = 1.0  # amplitude of excitation (can be adjusted)
    val_x = E0 * np.sin(ufl.pi * x[2] / a) * np.sin(ufl.pi * 2 * x[1]/d)

    values =  np.vstack((val_x, np.zeros_like(x[0]), np.zeros_like(x[0])))

    return values

radius = .4
def Capillary(x):
    val = np.zeros_like(x[0])
#    val[np.where(np.sqrt((x[0] - b/2)**2 + (x[1]-d/2)**2) <= radius)] = 1
#    val[np.where(np.abs((x[1]-d/2)) <= (d/4))] = 1
    val[np.where(x[1] < (d/2))] = 1
    values = np.vstack((val,val,val)) 
    values /= np.sqrt(3)
    return values

def sample(x):
    return x[1] < (d/2)
#    return np.sqrt((x[0] - b/2)**2 + (x[1]-d/2)**2) <= radius

#sample_markers = dolfinx.mesh.meshtags(mesh, gdim, facets, sample)
num_local_cells = mesh.topology.index_map(tdim).size_local
cell_indices = np.arange(num_local_cells, dtype=np.int32)

sample_cells = locate_entities(mesh, gdim, sample)

sample_markers = np.full(len(cell_indices), 0, dtype=np.int32)  # default unmarked

for i, cell in enumerate(cell_indices):
    if cell in sample_cells:
        sample_markers[i] = 1

# Get number of local cells


#mpi_print(cell_markers)
#cells = exterior_facet_indices(mesh.topology)

#cell_markers_sample = np.array([1 if sample(c) else 0 for c in cell_centers], dtype=np.int32)
ct = meshtags(mesh, gdim, cell_indices, sample_markers)
dx_sample = ufl.Measure("dx", domain=mesh, subdomain_data=ct)

E_102 = fem.Function(V)
E_102.interpolate(TE102_mode)
E_102.x.scatter_forward()

cap = fem.Function(V)
cap.interpolate(Capillary)
cap.x.scatter_forward()

B_102 = fem.Function(V)
B_form = ufl.curl(E_102)
B_expr = fem.Expression(B_form, V.element.interpolation_points())
B_102.interpolate(B_expr)
B_102.x.scatter_forward()

Bz_102 = fem.Function(V)
Bz_form = ufl.as_vector((0,0,B_102[2]))
Bz_expr = fem.Expression(Bz_form, V.element.interpolation_points())
Bz_102.interpolate(Bz_expr)
Bz_102.x.scatter_forward()

#B_102_mag = fem.Function(V)
#B_form_mag = ufl.form(B_102)
#B_form_mag = ufl.sqrt(ufl.inner(B_102, B_102))
#B_form_mag = ufl.as_vector(((B_102[0]*B_102[0])**0.5, (B_102[1]*B_102[1])**0.5, (B_102[2]*B_102[2])**0.5))
#B_expr_mag = fem.Expression(B_form_mag, V.element.interpolation_points())
#B_102_mag.interpolate(B_expr_mag)
#B_102_mag.x.scatter_forward()
#B_102_mag = np.sqrt(B_102**2.)

B_sample_local = fem.assemble_scalar(fem.form(ufl.inner(B_102,B_102) * dx_sample(1)))
B_sample = mesh.comm.allreduce(B_sample_local, op = MPI.SUM)

B_resonator_local = fem.assemble_scalar(fem.form(ufl.inner(B_102,B_102) * ufl.dx))
B_resonator = mesh.comm.allreduce(B_resonator_local, op = MPI.SUM)

volume_sample_local = fem.assemble_scalar(fem.form(1.0*dx_sample(1)))
volume_sample = mesh.comm.allreduce(volume_sample_local, op = MPI.SUM)

volume_local = fem.assemble_scalar(fem.form(1.0*ufl.Measure('dx', domain = mesh)))
volume = mesh.comm.allreduce(volume_local, op = MPI.SUM)


filling_factor = B_sample / B_resonator

mpi_print('B Sample:')
mpi_print(B_sample)
mpi_print('B Resonator:')
mpi_print(B_resonator)
mpi_print('Filling Factor:')
mpi_print(filling_factor)

#mpi_print('Volume Sample:')
#mpi_print(volume_sample)

#mpi_print('Volume:')
#mpi_print(volume)

with io.VTXWriter(mesh.comm, "sols_test/E_102.bp", E_102) as f:
    f.write(0.0)

with io.VTXWriter(mesh.comm, "sols_test/B_102.bp", B_102) as f:
    f.write(0.0)

with io.VTXWriter(mesh.comm, "sols_test/cap.bp", cap) as f:
    f.write(0.0)
