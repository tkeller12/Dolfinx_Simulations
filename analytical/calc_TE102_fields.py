
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
d = 1.7 # length of cavity, y 

nx = 10
ny = 10
nz = 10

mpi_print('Creating Mesh...')
#mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename, comm, 0, gdim=3)
mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[b,d,a]]), np.array([nx, ny, nz]), CellType.hexahedron)
mpi_print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

degree = 2
V = fem.functionspace(mesh, ("CG", degree, (gdim,)))

def TE102_mode(x):
    # x: (3, n) array of coordinates on the port boundary
    E0 = 1.0  # amplitude of excitation (can be adjusted)
    val_x = E0 * np.sin(ufl.pi * x[2] / a) * np.sin(ufl.pi * 2 * x[1]/d)

    values =  np.vstack((val_x, np.zeros_like(x[0]), np.zeros_like(x[0])))

    return values

def Capillary(x):
    pass


E_102 = fem.Function(V)
E_102.interpolate(TE102_mode)
E_102.x.scatter_forward()

B_102 = fem.Function(V)
B_form = ufl.curl(E_102)
B_expr = fem.Expression(B_form, V.element.interpolation_points())
B_102.interpolate(B_expr)
B_102.x.scatter_forward()


with io.VTXWriter(mesh.comm, "sols_test/E_102.bp", E_102) as f:
    f.write(0.0)

with io.VTXWriter(mesh.comm, "sols_test/B_102.bp", B_102) as f:
    f.write(0.0)
