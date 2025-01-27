#import modules
from mpi4py import MPI
import numpy as np

from petsc4py import PETSc
real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element, mixed_element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix
from dolfinx.io import gmshio

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, refine, compute_midpoints

from slepc4py import SLEPc


c = 299792458 # speed of light, m/s

target_freq = 9e9

def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target


target_eigenvalue = convert_freq_to_target(target_freq)


print('Importing Mesh...')
mesh, cell, facet_tags = gmshio.read_from_msh('mesh/lgr_3d_test3.msh', MPI.COMM_WORLD, 0, gdim=3)
print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)


degree = 2
element_type = "N2curl"
V = fem.functionspace(mesh, (element_type, degree))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx
b = ufl.inner(u, v) * ufl.dx

a = fem.form(a)
b = fem.form(b)

print('Applying Boundary Conditions...')
bc_facets = exterior_facet_indices(mesh.topology)
bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
print('Done.')



print('Assembling Matrix...')
A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()
print('Done.')


print('Setting up Problem...')
eps = SLEPc.EPS().create(mesh.comm)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-9
max_it = 10000
eps.setTolerances(tol=tol, max_it=max_it)
print('tol and max it:', eps.getTolerances())

eps.setType(SLEPc.EPS.Type.KRYLOVSCHUR)

# Get ST context from eps
st = eps.getST()

# Set shift-and-invert transformation
st.setType(SLEPc.ST.Type.SINVERT)
st.setShift(0.1)
st.setFromOptions()

eps.setWhichEigenpairs(SLEPc.EPS.Which.TARGET_REAL)

eps.setTarget(target_eigenvalue)

eps.setDimensions(nev=4)
print('Done.')


print('Solving...')
eps.solve()
eps.view()
eps.errorView()

print('Done.')


#print('freq TE_102: %0.03f'%(freq_te102/1e9))

print('Eigenvalues:')
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
    mode_freq = convert_eigenvalue_to_f(np.real(eigen_value))
#    print(i, eigen_value, np.sqrt(eigen_value), mode_freq)
    print(i, '%0.05f GHz'%(mode_freq/1e9))
#    print(i, eigen_value)

#    if np.real(np.abs(eigen_value)) > 0.001:
#        print(i, eigen_value)
print('Done.')

vals = [(i, np.sqrt(eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
#vals.sort(key=lambda x: x[1].real) # this doesn't make sense if you "target" a specific eigenvalue

eh = fem.Function(V)

kz_list = []

#print('Summary:')
for i, kz in vals:
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)

    # Verify, save and visualize solution
    kz_list.append(kz)

    eh.x.scatter_forward()

    eth = eh

    # Transform eth, ezh into Et and Ez
    eth.x.array[:] = eth.x.array[:]

    gdim = mesh.geometry.dim
    V_dg = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
    Et_dg = fem.Function(V_dg)
    Et_dg.interpolate(eth)

    B = fem.Function(V_dg)
    B_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())
    B.interpolate(B_expr)


    V_grad = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
    Grad = fem.Function(V_grad)
    grad_form = ufl.grad(eth[0]) + ufl.grad(eth[1]) + ufl.grad(eth[2])
    grad_expr = fem.Expression(grad_form, V_grad.element.interpolation_points())
    Grad.interpolate(grad_expr)

    # Save solutions
    with io.VTXWriter(mesh.comm, "sols_lgr/Et_%04i.bp"%i, Et_dg) as f:
        f.write(0.0)

    with io.VTXWriter(mesh.comm, "sols_lgr/B_%04i.bp"%i, B) as f:
        f.write(0.0)

    with io.VTXWriter(mesh.comm, "sols_lgr/Grad_%04i.bp"%i, Grad) as f:
        f.write(0.0)


def refine_criteria(x, threshold = 0.):
    return Grad.eval(mesh,x) > threshold
#    values = Grad.Vector.getArray()
values = np.abs(Grad.x.array)
threshold = np.percentile(values, 85)
#    refine_criteria(x, threshold = )
print(values)
print(mesh)
#    print(mesh.cells[1])
#    locate_entities(mesh, 1, refine_criteria)
#    refine_cells = []
#    for i norm in enumerate(threshold):
#
refined_mesh = refine(mesh, np.array([0, 1], dtype = np.int32))
print(refined_mesh)
refined_mesh = refined_mesh[0]
num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
print('Num cells',num_cells)
num_cells = refined_mesh.topology.index_map(refined_mesh.topology.dim).size_local
print('Num cells 2',num_cells)
print(refined_mesh.topology.dim)
print(refined_mesh.name)
print(refined_mesh.topology)
print(refined_mesh.geometry.x)

midpoints = compute_midpoints(mesh, 3, np.array([1], np.int32))
print('-'*50)
print(midpoints)
print(midpoints.shape)
#    refine = mesh.locate_entities(msh, dim-1, refine_criteria)
#for ix in range(num_cells):
#    x0 = (mesh.geometry.x[0] + mesh.geometry.x[1] + mesh.geometry.x[2]) / 2.0
#    tree = bb_tree(mesh, mesh.geometry.dim)
#    cell_candidates = compute_collisions_points(tree, x0)
#    cell = compute_colliding_cells(mesh, cell_candidates, x0)
#    cell = cell[ix]
#    Grad.eval(x0, first_cell)[:3]
    



#    grad_V = fem.Function(V)
#    grad_form = ufl.grad(eth[0]) + ufl.grad(eth[1]) + ufl.grad(eth[2])
#    grad_expr = fem.Expression(grad_form, V.element.interpolation_points())
#    grad_V.interpolate(grad_expr)


print('Script Done.')
