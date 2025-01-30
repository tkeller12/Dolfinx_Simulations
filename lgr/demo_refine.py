# -*- coding:utf-8 -*-
import ufl 
import dolfinx
from mpi4py import MPI
import numpy as np 
#import pyvista
from dolfinx import plot

domain = dolfinx.mesh.create_rectangle(
    MPI.COMM_WORLD, 
    points=((-3.0, -1.0), (3.0, 1.0)), 
    n=(30, 10), 
    cell_type=dolfinx.mesh.CellType.triangle)

#we use this function as an example
def func(x):
    return np.sin(2*(x[0]+x[1])) * np.exp(-x[0]**2)

#V = dolfinx.fem.FunctionSpace(domain, ("Lagrange", 1))
V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
u = dolfinx.fem.Function(V)
u.interpolate(func)

#calculate the norm of gradient. Note that, it is a function instead of form
#When the 1st order Lagrange finite element is used, u is a piecewise linear function, gradient 
#will be discontinuous at the vertex. Here, we use zero order Discontinuous Galerkin (DG) finite element
#There is a DOF at the center of the cell, it has a one-to-one correspondence with the cell.
Vg = dolfinx.fem.functionspace(domain, ("DG", 0))
F = dolfinx.fem.Expression(
    ufl.sqrt(ufl.dot(ufl.grad(u), ufl.grad(u))), 
    Vg.element.interpolation_points())
grad_norm = dolfinx.fem.Function(Vg)
grad_norm.interpolate(F)

#sort the grad_norm and get the cell index
order = np.argsort(grad_norm.x.array)
cell_index = order[-int(0.3*order.size):-1]

#cell_index now contains the cell which we should refine
#now we get the edges of these cells
domain.topology.create_connectivity(1, 2)
edge_index = dolfinx.mesh.compute_incident_entities(domain.topology, cell_index.astype(np.int32), 2, 1)

#call refine
new_domain = dolfinx.mesh.refine(domain, edge_index, True)
new_V = dolfinx.fem.FunctionSpace(new_domain, ("Lagrange", 1))
new_u = dolfinx.fem.Function(new_V)

#Finally, we interpolate the old function onto the new mesh
interp_data= dolfinx.fem.create_nonmatching_meshes_interpolation_data(
    new_V.mesh._cpp_object, new_V.element, V.mesh._cpp_object, 1.0e-8)

new_u.interpolate(u, nmm_interpolation_data=interp_data)

#you can plot the function. As you can see, the cells on which gradient norms are large are refined. 
#cells, types, x = plot.vtk_mesh(new_V)
#grid = pyvista.UnstructuredGrid(cells, types, x)
#grid.point_data["u"] = new_u.x.array.real
#grid.set_active_scalars("u")
#plotter = pyvista.Plotter()
#plotter.add_mesh(grid, show_edges=True)
#warped = grid.warp_by_scalar()
#plotter.add_mesh(warped)
#plotter.show()
