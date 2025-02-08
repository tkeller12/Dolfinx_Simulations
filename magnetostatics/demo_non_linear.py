# Basic Libraries
import numpy as np  # For vector and matrix manipulation
import gmsh  # Interface with the software GMSH
from scipy.interpolate import interp1d, splev, splrep

# Visualization
#import pyvista
from mpi4py import MPI

# FEniCS
from ufl import (
    SpatialCoordinate,
    TestFunction,
    TrialFunction,
    dx,
    dot,
    grad,
    as_vector,
)
from dolfinx import default_scalar_type, io
from dolfinx.io import gmshio
from dolfinx.fem import (
    Constant,
    Expression,
    Function,
    functionspace,
    dirichletbc,
    locate_dofs_topological,
)
from dolfinx.mesh import locate_entities_boundary
from dolfinx.fem.petsc import LinearProblem
from dolfinx.plot import vtk_mesh

rank = MPI.COMM_WORLD.rank
gmsh.initialize()

mi0 = 4 * np.pi * 1e-7
Hc = 1000
inch = 25.4e-3

# Domínio com ar
r_air = 80e-3
z_air = 80e-3

# Ímã
r_mag = 30e-3
z_mag = 30e-3

# Dimensões do ímã e model rank
gdim = 2  # Geometric dimension of the mesh
model_rank = 0
mesh_comm = MPI.COMM_WORLD

if mesh_comm.rank == model_rank:
    # Tags
    inner_tag = 1  # interior do ímã
    outer_tag = 2  # domínio com ar

    # Cria o retângulo de ar
    air_rectangle = gmsh.model.occ.addRectangle(
        0, -z_air, 0, r_air, 2 * z_air, tag=outer_tag
    )
    gmsh.model.occ.synchronize()

    # Cria o retângulo do ímã
    magnet_rectangle = gmsh.model.occ.addRectangle(
        0, -z_mag, 0, r_mag, 2 * z_mag, tag=inner_tag
    )
    gmsh.model.occ.synchronize()

    whole_domain = gmsh.model.occ.fragment(
        [(2, air_rectangle)], [(2, magnet_rectangle)]
    )
    gmsh.model.occ.synchronize()

    gmsh.model.addPhysicalGroup(
        gdim, [inner_tag], tag=1, name="inner"
    )  # Elementos internos
    gmsh.model.addPhysicalGroup(
        gdim, [outer_tag], tag=2, name="outer"
    )  # Elementos externos
    gmsh.model.occ.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", inch / 8)
    gmsh.model.mesh.generate(gdim)
    gmsh.write("ima.msh")

# Converte a malha criada pelo gmsh para um formato que o FEniCS compreende
mesh, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, mesh_comm, model_rank, gdim=gdim
)
# Finaliza o GMSH
gmsh.finalize()

# Steel
Bi_steel = np.array(
    [
        0,
        0.221950,
        0.245515,
        0.344303,
        0.375573,
        0.454417,
        0.627981,
        0.670134,
        0.861453,
        1.075180,
        1.241074,
        1.423388,
        1.656238,
        1.686626,
        1.813505,
        1.964422,
        1.979083,
        2.012433,
        2.021337,
        2.033503,
        2.050973,
        2.052071,
        2.191983,
        2.197328,
        2.240825,
        2.309729,
        2.327795,
        2.435784,
    ]
)
Hi_steel = np.array(
    [
        0,
        225.366667,
        237.316667,
        291.793333,
        310.450000,
        358.730000,
        483.890000,
        520.136667,
        723.673333,
        1071.333333,
        1570.566667,
        2775.500000,
        6290.533333,
        7049.866667,
        12338.666667,
        26304.666667,
        28581.000000,
        36287.000000,
        39022.333333,
        43292.333333,
        50590.000000,
        51118.333333,
        134313.333333,
        138566.666667,
        168803.333333,
        223476.666667,
        237853.333333,
        321480.000000,
    ]
)

mi_steel = np.divide(Bi_steel, Hi_steel)
mi_steel[0] = mi_steel[1] / 2
# mi_steel[0] = mi0


f = interp1d(Bi_steel, mi_steel)
f2 = interp1d(Bi_steel, mi_steel, kind="quadratic")
spl = splrep(Bi_steel, mi_steel)

xnew = np.linspace(0, Bi_steel[-1], num=101)
sp2 = splev(xnew, spl)

V = functionspace(mesh, ("Lagrange", 2))
tdim = mesh.topology.dim
u = TrialFunction(V)
v = TestFunction(V)
# Define the radial coordinate r
# Define the radial coordinate r from x[0]
x = SpatialCoordinate(mesh)
r = x[0]

# Update the weak form to include the magnet
facets = locate_entities_boundary(mesh, tdim - 1, lambda x: np.full(x.shape[1], True))
dofs = locate_dofs_topological(V, tdim - 1, facets)
bc = dirichletbc(default_scalar_type(0), dofs, V)

# Update the weak shape to include the magnet
mur_air = 1  # Relative permeability in air
mu0 = 4 * np.pi * 1e-7  # Permeabilidade do vácuo
Hc = float(1000)  # Vacuum permeability


# Function to define the relative permeability in the domainio


def mu_r(x):
    return np.interp(x[0], Bi_steel, mi_steel / mu0)


MU_space = functionspace(mesh, ("DG", 2))
mu_r_function = Function(MU_space)
mu_r_function.x.array[:] = mu0
mu_r_function.interpolate(mu_r, cells0=cell_tags.find(inner_tag))
mu_r_function.x.scatter_forward()

# Convert mu_r to a spatial function in Dolfinx

from dolfinx.io import VTXWriter

with VTXWriter(MPI.COMM_WORLD, "mu_r.bp", [mu_r_function], engine="BP5") as xdmf:
    xdmf.write(0.0)

# Variational equation for the magnet
a = (1 / mu_r_function) * (1 / r) * dot(grad(u), grad(v)) * dx

# Modify the source term to include Hc
M = Constant(mesh, Hc)
L = mu0 * M * v * dx

# Solve the linear problem
A_ = Function(V)
problem = LinearProblem(
    a,
    L,
    u=A_,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)
problem.solve()

# Calculate the magnetic field
W = functionspace(mesh, ("DG", 0, (mesh.geometry.dim,)))
B = Function(W)
B_expr = Expression(
    as_vector((-(1 / r) * A_.dx(1), (1 / r) * A_.dx(0))),
    W.element.interpolation_points(),
)
B.interpolate(B_expr)

with io.VTXWriter(mesh.comm, "sols/poisson_A_.bp", A_) as f:
    f.write(0.0)



##plotter = pyvista.Plotter()
#
## Converts the grid to an UnstructuredGrid
#A_grid = pyvista.UnstructuredGrid(*vtk_mesh(V))
#
## Add the data from the A_z field
#A_grid.point_data["A_"] = A_.x.array
#A_grid.set_active_scalars("A_")
#
## Applies the deformation based on the A_z field
#warp = A_grid.warp_by_scalar("A_")
## Add the deformed mesh to the plotter
#actor = plotter.add_mesh(warp, show_edges=False)
#plotter.view_xy()
#plotter.show()
#
## Iterpolate B again to mach vtk_mesh DoF.
#Wl = functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
#Bl = Function(Wl)
#Bl.interpolate(B)
#
#topology, cell_types, geo = vtk_mesh(V)
#values = np.zeros((geo.shape[0], 3), dtype=np.float64)
#values[:, : len(Bl)] = Bl.x.array.real.reshape((geo.shape[0], len(Bl)))
#
## Create a point cloud of glyphs
#function_grid = pyvista.UnstructuredGrid(topology, cell_types, geo)
#function_grid["Bl"] = values
#glyphs = function_grid.glyph(orient="Bl", factor=0.5)
#
## Create a pyvista-grid for the mesh
#mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim)
#grid = pyvista.UnstructuredGrid(*vtk_mesh(mesh, mesh.topology.dim))
#
## Create plotter
#plotter = pyvista.Plotter()
## plotter.add_mesh(grid, style="wireframe", color="k")
#plotter.add_mesh(glyphs)
#plotter.view_xy()
## plotter.window_size = [1000, 250];
## plotter.camera.zoom(3)
#plotter.show()
