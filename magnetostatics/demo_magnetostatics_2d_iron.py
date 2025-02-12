import importlib.util

if importlib.util.find_spec("petsc4py") is not None:
    import dolfinx

    if not dolfinx.has_petsc:
        print("This demo requires DOLFINx to be compiled with PETSc enabled.")
        exit(0)
    from petsc4py.PETSc import ScalarType  # type: ignore
else:
    print("This demo requires petsc4py.")
    exit(0)

from mpi4py import MPI

import numpy as np

import ufl
from dolfinx import fem, io, mesh, plot, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner, dot
from dolfinx.mesh import exterior_facet_indices, locate_entities, locate_entities_boundary
from dolfinx.io import gmshio

comm = MPI.COMM_WORLD

box_size = 1.0

#msh = mesh.create_rectangle(
#    comm=MPI.COMM_WORLD,
#    points=((0.0, 0.0), (box_size, box_size)),
#    n=(500, 500),
#    cell_type=mesh.CellType.triangle,
#)

mesh, cell, facet_tags = gmshio.read_from_msh('iron_yoke_2d_001.msh', comm, 0, gdim=3)


msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
gdim = msh.geometry.dim
tdim = msh.topology.dim
print(gdim)

degree = 2
V = fem.functionspace(msh, ("Lagrange", degree))


#facets = mesh.locate_entities_boundary(
#    msh,
#    dim=(msh.topology.dim - 1),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
#)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`:


# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` class that
# represents the boundary condition:

#facets = mesh.locate_entities_boundary(
#    msh,
#    dim=(msh.topology.dim - 1),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
#)

#dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
#bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)


#bc_facets = exterior_facet_indices(msh.topology)
#bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
#u_bc = fem.Function(V)
#with u_bc.x.petsc_vec.localForm() as loc:
#    loc.set(0)
#bc = fem.dirichletbc(u_bc, bc_dofs)



# Next, the variational problem is defined:

J_space = fem.functionspace(msh, ("DQ", 0))
J = fem.Function(J_space)
def J_location(x):
#    a = np.logical_and(x[0] > 0.45, x[0] < 0.55)

#    a = np.logical_and(x[0] > 0.35, x[0] < 0.65)
    a = np.logical_and(x[0] > 0.85, x[0] < 0.95)
#    a2 = np.logical_and(x[0] > 0.55, x[0] < 0.65)
#    a = np.logical_or(a,a2)

    b = np.logical_and(x[1] > 0.15, x[1] < 0.25)
    c = np.logical_and(a,b)
    return c

cells_J = locate_entities(msh, msh.topology.dim, J_location)
print(cells_J)
J.x.array[cells_J] = np.full_like(cells_J, 1.0, dtype=ScalarType)


### Dirichlet Boundary conditions on all boundaries ###
#facets = locate_entities_boundary(msh, gdim - 1, lambda x: np.full(x.shape[1], True))
#dofs = fem.locate_dofs_topological(V, gdim - 1, facets)
#bc = fem.dirichletbc(default_scalar_type(0), dofs, V)

# Update the weak form to include the magnet
#x = ufl.SpatialCoordinate(msh)
#facets = locate_entities_boundary(msh, tdim - 1, lambda x: np.full(x.shape[1], True))
#dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
#bc = fem.dirichletbc(default_scalar_type(0), dofs, V)

### Dirichlet Boundary conditions on chosen boundaries ###
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
)

dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)


# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
#J = 10 * ufl.exp(-1*((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
#g = ufl.sin(5 * x[0])
#a = inner(grad(u), grad(v)) * dx
#L = inner(f, v) * dx
#L = J * v * dx
#L = dot(f, v) * dx

r = x[1]
#L =  J * v * (1./ r) * dx
L =  J * v * dx

#a = dot(grad(u), grad(v)) * dx
#a = (dot(grad(u), grad(v)) * x[1]) * dx
a = dot(grad(u), grad(v)) * (1./r) * dx

#L = J * v * dx

A_z = fem.Function(V)
#problem = LinearProblem(a, L, u=A_z, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})

problem = LinearProblem(
    a,
    L,
    u=A_z,
    bcs=[bc],
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "ksp_error_if_not_converged": True,
    },
)

uh = problem.solve()

print(len(uh.x.array))

### Calculate Curl


W = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))
#W = fem.functionspace(msh, ("DG", degree, (msh.geometry.dim, )))
#W = fem.functionspace(msh, ("CG", degree, (msh.geometry.dim, )))
B = fem.Function(W)
#B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B_expr = fem.Expression(ufl.as_vector(((1.0/r)*A_z.dx(1), (-1.0/r)*A_z.dx(0))), W.element.interpolation_points())
#B_expr = fem.Expression(ufl.as_vector(((x[1]*A_z).dx(1), -1*(x[1]*A_z).dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)

W2 = fem.functionspace(msh, ("CG", 2, (msh.geometry.dim, )))
B2 = fem.Function(W2)


with io.VTXWriter(msh.comm, "sols/poisson_J.bp", J) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "sols/poisson_A.bp", uh) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "sols/poisson_B.bp", B) as f:
    f.write(0.0)

print('Done.')

#with io.XDMFFile(msh.comm, "out_poisson/poisson.xdmf", "w") as file:
#    file.write_mesh(msh)
#    file.write_function(uh)
# -

# and displayed using [pyvista](https://docs.pyvista.org/).

# +
#try:
#    import pyvista
#
#    cells, types, x = plot.vtk_mesh(V)
#    grid = pyvista.UnstructuredGrid(cells, types, x)
#    grid.point_data["u"] = uh.x.array.real
#    grid.set_active_scalars("u")
#    plotter = pyvista.Plotter()
#    plotter.add_mesh(grid, show_edges=True)
#    warped = grid.warp_by_scalar()
#    plotter.add_mesh(warped)
#    if pyvista.OFF_SCREEN:
#        pyvista.start_xvfb(wait=0.1)
#        plotter.screenshot("uh_poisson.png")
#    else:
#        plotter.show()
#except ModuleNotFoundError:
#    print("'pyvista' is required to visualise the solution")
#    print("Install 'pyvista' with pip: 'python3 -m pip install pyvista'")
## -
