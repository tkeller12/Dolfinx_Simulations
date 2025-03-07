import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import dolfinx
import dolfinx.fem
import dolfinx.io
import dolfinx.mesh
from dolfinx.mesh import CellType
import ufl

# -------------------------------
# 1. Create the mesh: a rectangular cavity.
# -------------------------------
# Domain dimensions: [0,Lx] x [0,Ly] x [0,Lz]
Lx, Ly, Lz = 1.0, 0.5, 0.3  # adjust as needed
#mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([Lx, Ly, Lz])], [10,10,10], cell_type="hexahedron")
mesh = dolfinx.mesh.create_box(MPI.COMM_WORLD, [np.array([0, 0, 0]), np.array([Lx, Ly, Lz])], [10,10,10], CellType.hexahedron)
mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)


# -------------------------------
# 2. Define the function space.
# -------------------------------
# We use an Nedelec (edge) element space for electromagnetic fields.
#element = ufl.FiniteElement("N1curl", mesh.ufl_cell(), degree=1)
#V = dolfinx.fem.FunctionSpace(mesh, element)
degree = 1
V = dolfinx.fem.functionspace(mesh, ('N1curl', degree))

# -------------------------------
# 3. Define trial and test functions.
# -------------------------------
E = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

# -------------------------------
# 4. Define physical parameters.
# -------------------------------
# Physical constants (using SI units for vacuum)
mu0 = 4*np.pi*1e-7
epsilon0 = 8.854187817e-12
eta0 = np.sqrt(mu0/epsilon0)

# For simplicity, assume free-space material (or modify with relative values)
mu_val = mu0
epsilon_val = epsilon0

# Frequency (Hz) and angular frequency
f = 10e9            # 10 GHz, for example
omega = 2*np.pi*f

# -------------------------------
# 5. Define the weak form.
# -------------------------------
# Starting from Maxwell's curl-curl equation in the frequency domain:
#   ∇×(1/μ ∇×E) - ω² ε E = 0   in Ω
# After multiplying by a test function and integrating by parts,
# we have a volume integral and a boundary (port) term.
a_vol = ufl.inner((1/mu_val) * ufl.curl(E), ufl.curl(v)) * ufl.dx - \
        omega**2 * ufl.inner(epsilon_val * E, v) * ufl.dx

# -------------------------------
# 6. Define the port boundary.
# -------------------------------
# We assume that the port is located at z=0.
# Create a facet marker for all facets where z is approximately zero.
def is_port(x):
    return np.isclose(x[2], 0.0)

tdim = mesh.topology.dim
port_facets = dolfinx.mesh.locate_entities_boundary(mesh, tdim - 1, is_port)
# Tag these facets with an integer (say, 1)
port_marker = dolfinx.mesh.meshtags(mesh, tdim - 1, port_facets, np.full(len(port_facets), 1, dtype=np.int32))
#port_marker = dolfinx.mesh.meshTags(mesh, port_facets)

# Define a measure ds over the boundary using the markers.
ds = ufl.Measure("ds", domain=mesh, subdomain_data=port_marker)

# -------------------------------
# 7. Incorporate the port impedance/excitation.
# -------------------------------
# For the TE₁₀ mode in a rectangular waveguide of width a_port (along x),
# the electric field is:
#    E_inc(x,z) = (0, E0 sin(pi*x/a_port) e^{-jβz}, 0)
# and the corresponding magnetic field is (after using Maxwell’s equations)
#    H_inc(x,z) = (E0/Z_TE10 cos(pi*x/a_port) e^{-jβz}, 0, 0)
# where the TE₁₀ modal impedance is
#    Z_TE10 = eta0 / sqrt(1 - (f_c/f)^2), with f_c = c/(2*a_port), c = 1/sqrt(mu0*epsilon0).
a_port = Lx  # assume port extends along the full x direction of the cavity face

# Calculate cutoff frequency for TE10 mode (for a rectangular waveguide, f_c = c/(2a))
c = 1/np.sqrt(mu0*epsilon0)
f_c = c/(2*a_port)
# Ensure that the operating frequency is above cutoff:
if f <= f_c:
    raise ValueError("Operating frequency must be above the TE10 cutoff.")

Z_TE10 = eta0 / np.sqrt(1 - (f_c/f)**2)
Y_port = 1/Z_TE10  # admittance at the port

# Define the incident electric field on the port (z=0)
# For TE10, the E field is polarized in y.
E0 = 1.0  # amplitude (can be scaled)
x = ufl.SpatialCoordinate(mesh)
E_inc = ufl.as_vector([0, E0 * ufl.sin(ufl.pi * x[0]/a_port), 0])
# Note: At z=0, the phase factor e^{-jβz}=1.

# In the weak form, the natural (Neumann) term on the port is replaced by an impedance condition.
# This leads to an additional bilinear term and a linear forcing term.
# The additional bilinear term is:
a_port = - Y_port * ufl.inner(E, v) * ds(1)
# And the forcing term comes from the known excitation:
# First, compute the incident magnetic field. For TE10, the dominant (x) component is:
H_inc = ufl.as_vector([E0/Z_TE10 * ufl.cos(ufl.pi * x[0]/a_port), 0, 0])
# On the port, the outward normal is n = (0, 0, -1) (assuming z increases into the domain).
n = ufl.as_vector([0, 0, -1])
# Then n x H_inc = cross(n, H_inc)
n_cross_H_inc = ufl.cross(n, H_inc)
L_port = ufl.inner(n_cross_H_inc, v) * ds(1)

# Total weak form becomes:
a = a_vol + a_port
L = L_port

# -------------------------------
# 8. Solve the driven problem.
# -------------------------------
problem = dolfinx.fem.petsc.LinearProblem(a, L, bcs=[], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
E_sol = problem.solve()

# -------------------------------
# 9. Save solution for visualization.
# -------------------------------
with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "driven_modal.xdmf", "w") as xdmf:
    xdmf.write_mesh(mesh)
    xdmf.write_function(E_sol)

if MPI.COMM_WORLD.rank == 0:
    print("Driven modal simulation complete. Solution saved to 'driven_modal.xdmf'.")

