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
from dolfinx.io import gmshio
import gmsh


box_size = 1.0

filename = 'halbach_2d_sym_001'

msh, cell_tags, facet_tags = gmshio.read_from_msh(filename + '.msh', MPI.COMM_WORLD, 0, gdim=2)


msh.topology.create_connectivity(msh.topology.dim - 1, msh.topology.dim)
gdim = msh.geometry.dim
tdim = msh.topology.dim
print(gdim)


degree = 2
V = fem.functionspace(msh, ("Lagrange", degree))

Mxy_space = fem.functionspace(msh, ("DG", 0))
marker_data = np.loadtxt(filename + '.csv', delimiter = ',')

markers = marker_data[:,0]
Mu_data = marker_data[:,1]
Mx_data = marker_data[:,2]
My_data = marker_data[:,3]

Mx = fem.Function(Mxy_space)
My = fem.Function(Mxy_space)

for ix, marker in enumerate(markers):
    tags = cell_tags.find(int(marker))
    Mx_value = Mx_data[ix]
    My_value = My_data[ix]
    Mx.x.array[tags] = np.full_like(tags, Mx_value, dtype=ScalarType)
    My.x.array[tags] = np.full_like(tags, My_value, dtype=ScalarType)

natural_bc_facets = mesh.locate_entities_boundary(
    msh,
    dim=(msh.topology.dim - 1),
    marker=lambda x: np.isclose(x[1], 0.0))

bc_facets = exterior_facet_indices(msh.topology)
bc_facets_list = list(bc_facets)
print(bc_facets)
for each in natural_bc_facets:
    if each in bc_facets_list:
        bc_facets_list.remove(each)
bc_facets = np.array(bc_facets_list)
bc_dofs = fem.locate_dofs_topological(V, msh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)


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
M = ufl.as_vector((Mx, My)) # why minus 1?
#M = ufl.as_vector((0.0, My))
#M = ufl.as_vector((J, 0.0))
#M.interpolate(J)
#J_eff = ufl.as_vector((M.dx(1), -1.0*M.dx(0)))
#L =  M * v * dx
#L =  M.dx(1) * v * dx
#L = (x[0]*J) * v * dx
L = dot(M, curl(v)) * dx
#L = inner(J_eff, v) * dx

#J_eff = curl(M)*dx
#L = dot(J_eff, v) * dx

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
M2 = fem.functionspace(msh, ("DG", 0, (msh.geometry.dim, )))
B = fem.Function(W)
#B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -A_z.dx(0))), W.element.interpolation_points())
B_expr = fem.Expression(ufl.as_vector((A_z.dx(1), -1.0*A_z.dx(0))), W.element.interpolation_points())
#B_expr = fem.Expression(ufl.as_vector(((x[1]*A_z).dx(1), -1*(x[1]*A_z).dx(0))), W.element.interpolation_points())
B.interpolate(B_expr)

W2 = fem.functionspace(msh, ("CG", 2, (msh.geometry.dim, )))
B2 = fem.Function(W2)

M_save = fem.Function(M2)
M_expr = fem.Expression(M, M2.element.interpolation_points())
M_save.interpolate(M_expr)

with io.VTXWriter(msh.comm, "sols/%s_M.bp"%filename, M_save) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "sols/%s_A.bp"%filename, uh) as f:
    f.write(0.0)

with io.VTXWriter(msh.comm, "sols/%s_B.bp"%filename, B) as f:
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
