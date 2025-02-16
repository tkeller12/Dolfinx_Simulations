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
from ufl import ds, dx, grad, inner, curl, dot
from dolfinx.mesh import exterior_facet_indices, locate_entities, locate_entities_boundary

box_size = 1.0

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (box_size, box_size)),
    n=(200, 200),
    cell_type=mesh.CellType.triangle,
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
gdim = msh.geometry.dim
tdim = msh.topology.dim
print(gdim)

degree = 2
V = fem.functionspace(msh, ("Lagrange", degree))


# Next, the variational problem is defined:

#J_space = fem.functionspace(msh, ("DQ", 0))
J_space = fem.functionspace(msh, ("DG", 0))
#J = fem.Function(V)
#f.interpolate()
J = fem.Function(J_space)
print(J)
def J_location(x):
    a = np.logical_and(x[0] > 0.45, x[0] < 0.55)
    b = np.logical_and(x[1] > 0.45, x[1] < 0.55)
    c = np.logical_and(a,b)
    return c

cells_J = locate_entities(msh, msh.topology.dim, J_location)
print(cells_J)
#cells_J2 = locate_entities(msh, msh.topology.dim, J_location2)
#J_expr = fem.Expression(J_expression, V.element.interpolation_points())
#J = fem.Function(V)
#J.interpolate(J_expr)

#J.x.array[:] = 1.0
J.x.array[cells_J] = np.full_like(cells_J, 1.0, dtype=ScalarType)
#J.x.array[cells_J2] = np.full_like(cells_J2, -1.0, dtype=ScalarType)
#J.x.array[cells_J] = np.full_like(cells_J, 1000.0)#, dtype=ScalarType)

### Dirichlet Boundary conditions on chosen boundaries ###
facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size),
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

#L =  J * v * (1./ r) * dx
#L =  J * v * dx
M = fem.Function(V)
#M.interpolate(J)
M = ufl.as_vector((0.0, J))
#M = ufl.as_vector((J, 0.0))
#M.interpolate(J)

#L =  M * v * dx
#L =  M.dx(1) * v * dx
#L = (x[0]*J) * v * dx
L = dot(M, curl(v)) * dx

#a = dot(grad(u), grad(v)) * dx
#a = (dot(grad(u), grad(v)) * x[1]) * dx
a = dot(curl(u), curl(v)) * dx

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


#W = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))
#W = fem.functionspace(msh, ("DG", degree, (msh.geometry.dim, )))
W = fem.functionspace(msh, ("CG", degree, (msh.geometry.dim, )))
B = fem.Function(W)
#B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -1.0*A_z.dx(0))), W.element.interpolation_points())
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
