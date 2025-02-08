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
from dolfinx import fem, io, mesh, plot
from dolfinx.fem.petsc import LinearProblem
from ufl import ds, dx, grad, inner, dot
from dolfinx.mesh import exterior_facet_indices, locate_entities

msh = mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=((0.0, 0.0), (2.0, 1.0)),
    n=(32, 16),
    cell_type=mesh.CellType.triangle,
)
msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
gdim = msh.geometry.dim
print(gdim)

#V = fem.functionspace(msh, ("Lagrange", 3, (3,)))
V = fem.functionspace(msh, ("Lagrange", 3))


#facets = mesh.locate_entities_boundary(
#    msh,
#    dim=(msh.topology.dim - 1),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0),
#    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], 2.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], 1.0),
#)

# We now find the degrees-of-freedom that are associated with the
# boundary facets using {py:func}`locate_dofs_topological
# <dolfinx.fem.locate_dofs_topological>`:

#dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)

# and use {py:func}`dirichletbc <dolfinx.fem.dirichletbc>` to create a
# {py:class}`DirichletBC <dolfinx.fem.DirichletBC>` class that
# represents the boundary condition:

#bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)
#bc = fem.dirichletbc(value=VectorType(0), dofs=dofs, V=V)


bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

# Next, the variational problem is defined:

# +
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
x = ufl.SpatialCoordinate(msh)
f = 10 * ufl.exp(-1*((x[0] - 0.5) ** 2 + (x[1] - 0.5) ** 2) / 0.02)
#g = ufl.sin(5 * x[0])
#a = inner(grad(u), grad(v)) * dx
a = dot(grad(u), grad(v)) * dx
#L = inner(f, v) * dx
L = dot(f, v) * dx


problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()


with io.VTXWriter(msh.comm, "sols/poisson.bp", uh) as f:
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
