import numpy as np
import ufl
from mpi4py import MPI
from dolfinx import mesh, fem, io, plot
from petsc4py import PETSc

# Define mesh parameters
length = 1.0  # Length of the waveguide
width = 0.1   # Width of the waveguide
num_elements_length = 50
num_elements_width = 10

# Create a rectangular mesh
domain = mesh.create_rectangle(
    MPI.COMM_WORLD,
    [np.array([0, 0]), np.array([length, width])],
    [num_elements_length, num_elements_width],
    cell_type=mesh.CellType.triangle,
)

# Define function space for the electric field (complex-valued)
element = ufl.FiniteElement("N1curl", domain.ufl_cell(), degree=1, form_degree=1)
V = fem.FunctionSpace(domain, element)

# Define material parameters
mu = 1.0  # Permeability (in vacuum)
epsilon = 1.0  # Permittivity (in vacuum)
omega = 2 * np.pi * 10e9  # Angular frequency (e.g., 10 GHz)

# Define the source term (e.g., a Gaussian pulse)
x = ufl.SpatialCoordinate(domain)
source_expr = ufl.exp(-((x[0] - length / 2) ** 2 + (x[1] - width / 2) ** 2) / 0.01)
source = fem.Function(V)
source.interpolate(source_expr)

# Define the variational problem
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a = (
    ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx
    - (omega ** 2) * mu * epsilon * ufl.inner(u, v) * ufl.dx
)
L = ufl.inner(source, v) * ufl.dx

# Apply boundary conditions (e.g., PEC on all boundaries)
boundary_facets = mesh.locate_entities_boundary(domain, dim=1, marker=lambda x: np.full(x.shape[1], True))
bc = fem.dirichletbc(value=fem.Constant(domain, PETSc.ScalarType(0)), dofs=fem.locate_dofs_topological(V, 1, boundary_facets))

# Assemble the system
A = fem.petsc.assemble_matrix(fem.form(a), bcs=[bc])
A.assemble()
b = fem.petsc.assemble_vector(fem.form(L))
fem.apply_lifting(b, [fem.form(a)], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fem.set_bc(b, [bc])

# Solve the linear system
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType("gmres")
solver.getPC().setType("ilu")
solver.setFromOptions()
u_solution = fem.Function(V)
solver.solve(b, u_solution.vector)
u_solution.x.scatter_forward()

# Output the solution to XDMF file for visualization
with io.XDMFFile(domain.comm, "solution.xdmf", "w") as xdmf_file:
    xdmf_file.write_mesh(domain)
    xdmf_file.write_function(u_solution)
