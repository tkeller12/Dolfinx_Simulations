#import ipyparallel as ipp
#cluster = ipp.Cluster(engines = "mpi", n = 4)
#rc = cluster.start_and_connect_sync()
#
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

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, refine, compute_midpoints, compute_incident_entities


from slepc4py import SLEPc

comm = MPI.COMM_WORLD


c = 299792458 # speed of light, m/s

target_freq = 9e9

def mpi_print(s):
    print(f"Rank {comm.rank}: {s}")

def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target


target_eigenvalue = convert_freq_to_target(target_freq)

mpi_print('Test')

print('Importing Mesh...')
mesh, cell, facet_tags = gmshio.read_from_msh('mesh/lgr_3d_test3.msh', comm, 0, gdim=3)
print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
gdim = mesh.geometry.dim


nev = 4

degree = 2
element_type = "N2curl"
#V = fem.functionspace(mesh, (element_type, degree)) # works
V = fem.functionspace(mesh, (element_type, degree, (gdim,)))
#V_grad = fem.functionspace(mesh, ("CG", degree, (gdim,)))


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

eps.setDimensions(nev=nev)
print('Done.')

ksp =  st.getKSP()
ksp.setType('preonly')

pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('superlu_dist')

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
#    V_dg = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
    V_dg = fem.functionspace(mesh, ("CG", degree, (gdim,)))
    Et_dg = fem.Function(V_dg)
    Et_dg.interpolate(eth)

    B = fem.Function(V_dg)
    B_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())
    B.interpolate(B_expr)

#    V_G = fem.functionspace(mesh, ("DG", 0, (gdim,)))
    V_G = fem.functionspace(mesh, ("DG", 0, (1,)))
#    G_element = element('DG', mesh.ufl_cell(), degree = 1)
#    V_G = fem.FunctionSpace(mesh, G_element)
#    V_G = fem.FunctionSpace(mesh, ("DG", 0, (1,)))
    G = fem.Function(V_G)
#    G_form = ufl.grad(eth[0]) + ufl.grad(eth[1]) + ufl.grad(eth[2])
#    G_form = ufl.grad(eth[0]) # 1d
    G_form = ufl.inner(ufl.grad(eth),ufl.grad(eth))
    print('ufl_shape:', G_form.ufl_shape)
    G_expr = fem.Expression(G_form, V_G.element.interpolation_points())
    print('Interplating...')
    G.interpolate(G_expr)
    print('here')

    ### ORDER IS 3x longer than it should be...
    # do I need a scalar space?

    order = np.argsort(G.x.array)
    print('G.x.array:', G.x.array)
    print('len(G.x.array):', len(G.x.array))
    print('order:', order)
    print('len(order):', len(order))
    cell_index = order[-int(0.3*order.size):-1]
    mesh.topology.create_connectivity(3, 1)
    mesh.topology.create_connectivity(1, 3)
    print(len(cell_index))
#    mesh.topology.create_connectivity(1, 3)
#    mesh.topology.create_connectivity(2, 3)
#    edge_index = compute_incident_entities(mesh.topology, cell_index.astype(np.int32), 2, 1)
#    edge_index = compute_incident_entities(mesh.topology, cell_index.astype(np.int32), 3, 1)
    print('cell_index:', cell_index)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print('mesh.topology.dim:',mesh.topology.dim)
    print('sizes:')
    print(mesh.topology.index_map(0).size_local)
    print(mesh.topology.index_map(1).size_local)
    print(mesh.topology.index_map(2).size_local)
    print(mesh.topology.index_map(3).size_local)
#    cells_array = np.array(range(num_cells)).astype(np.int32) # fake data for test

    edge_index = compute_incident_entities(mesh.topology, cell_index, 3, 1)
#    edge_index = compute_incident_entities(mesh.topology, cells_array, 1, 3)
#    edge_index = compute_incident_entities(mesh.topology, np.array([0]).astype(np.int32), 3, 1)
    print('edge_index:', edge_index)
    print('type(edge_index):', type(edge_index))

    new_mesh = refine(mesh, edge_index.astype(np.int32)) # works
#    new_mesh = refine(mesh, [1,2,3], True)
#    new_mesh = refine(mesh, [1,2,3])
#    new_mesh = refine(mesh, np.array([0,1,2,3])) # this works
#    new_mesh = refine(mesh) # this works
    print(new_mesh)
    new_mesh = new_mesh[0]
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    print('Old mesh cells:', num_cells)
    num_cells = new_mesh.topology.index_map(new_mesh.topology.dim).size_local
    print('New mesh cells:', num_cells)
    print(num_cells)



#    V_grad = fem.functionspace(mesh, ("DQ", degree, (gdim,)))
#    V_grad = fem.functionspace(mesh, ("CG", degree, (gdim,)))
#    Grad = fem.Function(V_grad)
#    grad_form = ufl.grad(eth[0]) + ufl.grad(eth[1]) + ufl.grad(eth[2])
#    grad_form = ufl.inner(ufl.grad(eth),ufl.grad(eth))#*ufl.dx#ufl.grad(eth[0]) + ufl.grad(eth[1]) + ufl.grad(eth[2])
#    print(type(grad_form))
#    print(grad_form.ufl_shape)
#    print(grad_form.ufl_index_dimensions)
#    print(grad_form.ufl_free_indices)
#    print(type(eth))
#    grad_expr = fem.Expression(grad_form, V_grad.element.interpolation_points())
#    Grad.interpolate(grad_expr)

    # Save solutions
    if i < nev:
        with io.VTXWriter(mesh.comm, "sols_lgr/Et_%04i.bp"%i, Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(mesh.comm, "sols_lgr/B_%04i.bp"%i, B) as f:
            f.write(0.0)

#    with io.VTXWriter(mesh.comm, "sols_lgr/Grad_%04i.bp"%i, Grad) as f:
#        f.write(0.0)


#values = np.abs(G.x.array)
#threshold = np.percentile(values, 85)
#num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
#
#x_array = np.arange(num_cells)
#
#def refine_criteria(x, threshold = threshold):
#    return Grad.eval(x.T, x_array, x_array) > threshold
##    values = Grad.Vector.getArray()
##    refine_criteria(x, threshold = )
#print(values)
#print(mesh)
#    print(mesh.cells[1])
#locate_entities(mesh, 3, refine_criteria)
#    refine_cells = []
#    for i norm in enumerate(threshold):
#
#refined_mesh = refine(mesh, np.array([0, 1], dtype = np.int32))
#print(refined_mesh)
#refined_mesh = refined_mesh[0]
#num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
#print('Num cells',num_cells)
#num_cells = refined_mesh.topology.index_map(refined_mesh.topology.dim).size_local
#print('Num cells 2',num_cells)
#print(refined_mesh.topology.dim)
#print(refined_mesh.name)
#print(refined_mesh.topology)
#print(refined_mesh.geometry.x)

#midpoints = compute_midpoints(mesh, 3, np.array([1], np.int32))
#print('-'*50)
#print(midpoints)
#print(midpoints.shape)
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
