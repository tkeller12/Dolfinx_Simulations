#!/usr/bin/env python3
"""
Driven modal simulation of a rectangular waveguide with a waveport
excitation using dolfinx. The waveguide (dimensions a x b x L) is
driven at the inlet (z = 0) with a TE₁₀ mode:
   g(x,y,z=0) = (0, E0*sin(pi*x/a), 0).
The governing frequency-domain Maxwell’s equation (curl–curl form)
is solved in the domain:
   curl(curl(E)) - k0^2 E = 0,
with PEC (E×n=0) on all boundaries.
A lifting function g (extended into the domain) is defined so that the
total field is E = u + g, where u satisfies homogeneous Dirichlet BC.
"""

from mpi4py import MPI
import numpy as np
import ufl
from dolfinx import mesh, fem, io, plot
from petsc4py import PETSc

# Parameters (use SI units)
a = 0.02286      # waveguide width (m) – e.g. WR90: 22.86 mm
b = 0.01016      # waveguide height (m) – 10.16 mm
L = 0.5          # waveguide length (m)
f = 10e9         # operating frequency: 10 GHz
c0 = 3e8         # speed of light (m/s)
k0 = 2 * np.pi * f / c0  # vacuum wavenumber
E0 = 1.0         # amplitude of excitation

# Create a 3D box mesh for the waveguide domain: [0,a] x [0,b] x [0,L]
nx, ny, nz = 20, 10, 30
msh = mesh.create_box(MPI.COMM_WORLD,
                      [np.array([0, 0, 0]), np.array([a, b, L])],
                      [nx, ny, nz],
                      cell_type=mesh.CellType.tetrahedron)

# Define function space for the electric field E using Nedelec (H(curl)) elements.
degree = 1
#V = fem.FunctionSpace(msh, ("Nedelec 1st kind H(curl)", degree))
V = fem.functionspace(msh, ('N1curl', degree))


# --- Define the TE₁₀ lifting function ---
# We wish g(x,y,z) to equal (0, E0*sin(pi*x/a), 0) at z=0.
# For a smooth extension, we let it decay with z, e.g.,
#      g(x,y,z) = (0, E0*sin(pi*x/a)*exp(-z/L), 0)
def g_expr(x):
    # x is an array with shape (3, n); x[0] is x, x[1] is y, x[2] is z.
    values = np.zeros((3, x.shape[1]), dtype=PETSc.ScalarType)
    values[1, :] = E0 * np.sin(np.pi * x[0, :] / a) * np.exp(-x[2, :] / L)
    return values

g = fem.Function(V)
g.interpolate(g_expr)
g.x.scatter_forward()  # make sure values are available

# --- Define the variational problem ---
# We now write the PDE for the correction field u, where the total field E = u + g satisfies
#   curl(curl(E)) - k0^2 E = 0.
# Substituting E = u + g, the weak form becomes:
#   a(u,v) = - a(g,v)  for all v in V₀ (functions vanishing on the essential boundary).
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
a_form = ufl.inner(ufl.curl(u), ufl.curl(v)) * ufl.dx - k0**2 * ufl.inner(u, v) * ufl.dx
L_form = - (ufl.inner(ufl.curl(g), ufl.curl(v)) * ufl.dx - k0**2 * ufl.inner(g, v) * ufl.dx)

# --- Define essential boundary conditions ---
# We impose homogeneous Dirichlet BC (u = 0) on all boundaries.
# This will force the total field E = u + g to equal g on the inlet (where g is nonzero)
# and to be g on other boundaries (so the overall BC for E become: E = g on the inlet, and E = g on PEC walls).
# In our simulation we desire PEC walls (E = 0) on the side walls. Thus we choose our lifting g so that
# g already satisfies the desired condition on the inlet and decays (ideally g = 0) on the side walls.
# For simplicity here, we impose u = 0 on all boundary facets.
def boundary_marker(x):
    # Mark all points on the boundary.
    return np.full(x.shape[1], True, dtype=bool)
facets = mesh.locate_entities_boundary(msh, msh.topology.dim - 1, boundary_marker)
dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, facets)
u_hom = fem.Function(V)
with u_hom.x.petsc_vec.localForm() as loc:
    loc.set(0.0)
bc = fem.dirichletbc(u_hom, dofs)

# --- Assemble system ---
A = fem.petsc.assemble_matrix(fem.form(a_form), bcs=[bc])
A.assemble()
b_vec = fem.petsc.assemble_vector(fem.form(L_form))
fem.petsc.apply_lifting(b_vec, [fem.form(a_form)], bcs=[[bc]])
b_vec.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
bc.apply(b_vec)

# --- Solve the linear system ---
u_sol = fem.Function(V)
solver = PETSc.KSP().create(msh.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.CG)  # conjugate gradient; adjust if needed
solver.getPC().setType(PETSc.PC.Type.LU)
solver.setTolerances(rtol=1e-8)
solver.solve(b_vec, u_sol.x.petsc_vec)
u_sol.x.scatter_forward()

# --- Compute the total field E = u + g ---
E_sol = fem.Function(V)
E_sol.x.array[:] = u_sol.x.array + g.x.array

# --- Save the solution to file ---
with io.XDMFFile(msh.comm, "driven_waveguide.xdmf", "w") as xdmf:
    xdmf.write_mesh(msh)
    xdmf.write_function(E_sol)

# --- Optionally visualize with PyVista ---
try:
    import pyvista as pv
    from dolfinx.plot import create_vtk_mesh
    topology, cells, geometry = create_vtk_mesh(msh)
    grid = pv.UnstructuredGrid(topology, cells, geometry)
    # Compute the pointwise norm of the field E
    E_values = E_sol.x.array.reshape((-1, 3))
    grid.point_data["|E|"] = np.linalg.norm(E_values, axis=1)
    pv.plot(grid, scalars="|E|", cmap="viridis", show_edges=True, title="E Field Magnitude")
except ImportError:
    print("pyvista not available; skipping visualization.")

print("Driven modal simulation completed. Results saved in driven_waveguide.xdmf")
