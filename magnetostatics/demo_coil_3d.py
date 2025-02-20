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
from basix.ufl import element
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

msh, cell_tags, facet_tags = gmshio.read_from_msh('coil_3d_test001.msh', comm, 0, gdim=3)


msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
msh.topology.create_connectivity(msh.topology.dim - 2, msh.topology.dim)
gdim = msh.geometry.dim
tdim = msh.topology.dim
print(gdim)

degree = 2
#V = fem.functionspace(msh, ("Lagrange", degree))
#el = element("Lagrange", msh.topology.cell_name(), 1, shape=(2,))
el = element("Lagrange", msh.topology.cell_name(), 2, shape=(3,))
#el = element("N2curl", msh.topology.cell_name(), 2, shape=(3,))
V = fem.functionspace(msh, el)

mu0 = 4. * np.pi * 1e-7

MU_space = fem.functionspace(msh, ("DG", 0))
mu_r_function = fem.Function(MU_space)
mu_r_function.x.array[:] = 1.0

coil_cell_tags = cell_tags.find(2)

def J_coil(x):
#    vals = np.zeros((msh.geometry.dim, x.shape[1]))

    x0 = 0 # x
    x1 = -x[2]/np.sqrt(x[1]**2 + x[2]**2)
    x2 = x[1]/np.sqrt(x[1]**2 + x[2]**2)

    return np.array((0*x[0],x1,x2))
#    return (0*x[0],-x[2]/np.sqrt(x[1]**2 + x[2]**2), )


#J = fem.Function(V2)
J = fem.Function(V)
#J = fem.Function(MU_space, dtype = np.float64)
J.interpolate(J_coil, cells0 = coil_cell_tags)
J.x.scatter_forward()




### Dirichlet Boundary conditions on all boundaries ###
#facets = locate_entities_boundary(msh, gdim - 1, lambda x: np.full(x.shape[1], True))
#dofs = fem.locate_dofs_topological(V, gdim - 1, facets)
#bc = fem.dirichletbc(default_scalar_type(0), dofs, V)

bc_facets = exterior_facet_indices(msh.topology)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)

# Update the weak form to include the magnet
#x = ufl.SpatialCoordinate(msh)
#facets = locate_entities_boundary(msh, tdim - 1, lambda x: np.full(x.shape[1], True))
#dofs = fem.locate_dofs_topological(V, tdim - 1, facets)
#bc = fem.dirichletbc(default_scalar_type(0), dofs, V)

#### Dirichlet Boundary conditions on chosen boundaries ###
#facets = mesh.locate_entities_boundary(
#    msh,
#    dim=(msh.topology.dim - 1),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], box_size),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[0], box_size) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
##    marker=lambda x: np.isclose(x[0], 0.0) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
#    marker=lambda x: np.isclose(x[0], box_size) | np.isclose(x[1], 0.0) | np.isclose(x[1], box_size),
#)

#dofs = fem.locate_dofs_topological(V=V, entity_dim=1, entities=facets)
#bc = fem.dirichletbc(value=ScalarType(0), dofs=dofs, V=V)


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

#r = x[1]
#L =  J * v * (1./ r) * dx
#L =  J * v * dx
#L =  inner(J,v) * dx
#L =  dot(J,v) * dx
L =  inner(J,v) * dx

#a = dot(grad(u), grad(v)) * dx
#a = (dot(grad(u), grad(v)) * x[1]) * dx
#a = (1.0 / mu_r_function) * inner(grad(u), grad(v)) * (1./r) * dx
a = (1.0 / mu_r_function) * inner(grad(u), grad(v)) * dx

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

print('Solving...')
uh = problem.solve()
print('Done.')

#print(len(uh.x.array))

### Calculate Curl


W = fem.functionspace(msh, ("CG", 2, (msh.geometry.dim, )))
#W = fem.functionspace(msh, ("DG", degree, (msh.geometry.dim, )))
#W = fem.functionspace(msh, ("CG", degree, (msh.geometry.dim, )))
B = fem.Function(W)
#B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
#B_expr = fem.Expression(ufl.as_vector(((1.0/r)*A_z.dx(1), (-1.0/r)*A_z.dx(0))), W.element.interpolation_points())
#B_expr = fem.Expression(ufl.as_vector(((x[1]*A_z).dx(1), -1*(x[1]*A_z).dx(0))), W.element.interpolation_points())
#B.interpolate(B_expr)

V_dg = fem.functionspace(msh, ("CG", 2, (gdim,)))
B = fem.Function(V_dg)
B_expr = fem.Expression(ufl.curl(A_z), V_dg.element.interpolation_points())
B.interpolate(B_expr)
B.x.scatter_forward()


#W2 = fem.functionspace(msh, ("CG", 2, (msh.geometry.dim, )))
#B2 = fem.Function(W2)


#with io.VTXWriter(msh.comm, "sols/poisson_J.bp", J) as f:
#    f.write(0.0)

#with io.VTXWriter(msh.comm, "sols/poisson_A.bp", uh) as f:
#    f.write(0.0)

with io.VTXWriter(msh.comm, "sols/coil_3d_B.bp", B) as f:
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
