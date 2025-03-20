
#import modules
from mpi4py import MPI
import numpy as np

import gmsh
 
from petsc4py import PETSc
#real_type = PETSc.RealType
scalar_type = PETSc.ScalarType

import ufl
from basix.ufl import element
#from basix.ufl import element, mixed_element
from dolfinx import fem, io, plot
from dolfinx.fem.petsc import assemble_matrix, LinearProblem
from dolfinx.io import gmshio

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities, locate_entities_boundary, meshtags
import dolfinx.mesh

from slepc4py import SLEPc


import sys

comm = MPI.COMM_WORLD

def mpi_print(s, rank = 0):
    if rank is None:
        print(f"Rank {comm.rank}: {s}")
    elif comm.rank == rank:
        print(f"Rank {comm.rank}: {s}")
    sys.stdout.flush()

#mpi_print(PETSc.ScalarType)

#TE102 cavity parameters
#a = 1.0 # waveguide a, z
#b = 0.5 # waveguide b, x
#d = 1.7 # length of cavity, z 

r = 22.e-3 # radius, r
h = 86.e-3 # height of cavity, z 

#sample_radius = 5e-3
epsilon = 8.854187817e-12 # F/m
mu = 4 * np.pi * 1e-7 # H/m

eta = np.sqrt(mu/epsilon)

nev = 10

mpi_print('Creating Mesh...')

filename = 'mesh/TE011_3d_mesh.msh'
if comm.rank == 1:
    gmsh.initialize()

    gmsh.model.add("2D_Rectangle")

    factory = gmsh.model.occ

#    box = factory.addBox(0,0,0,b,d,a, tag = 1)
    cylinder = factory.addCylinder(0,0,-h/2,0,0,h,r, tag = 1)

#    resonator = factory.cut([(3,1)], [(3,2)], removeTool = False)
#    resonator = factory.cut([(3,1)], [(3,2)], removeTool = True)

    factory.synchronize()
    gmsh.model.addPhysicalGroup(3, [1], tag = 1, name = 'Resonator') # need to add physcial groups
#    gmsh.model.addPhysicalGroup(3, [2], tag = 2, name = 'Sample') # need to add physcial groups
    #gmsh.option.setNumber("Mesh.Algorithm3D", 9) #R-tree, mesh looks good, good option
    gmsh.option.setNumber("Mesh.Algorithm3D", 4) #Frontal, mesh looks good, good option
    gmsh.model.mesh.generate(3)
    gmsh.model.mesh.refine()
#    gmsh.model.mesh.refine()

    gmsh.write(filename)
    gmsh.finalize()

comm.Barrier() # barrier until file save completed
mesh, cell_tags, facet_tags = gmshio.read_from_msh(filename, comm, 0, gdim=3)
#mesh = create_box(MPI.COMM_WORLD, np.array([[0.0,0.0,0.0],[b,d,a]]), np.array([nx, ny, nz]), CellType.hexahedron)
mpi_print('Done.')

mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
tdim = mesh.topology.dim
gdim = mesh.geometry.dim

c = 299792458 # speed of light, m/s

target_freq = 9e9

def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target


target_eigenvalue = convert_freq_to_target(target_freq)

degree = 2
V = fem.functionspace(mesh, ("N1curl", degree, (gdim,)))
V_CG = fem.functionspace(mesh, ("CG", degree, (gdim,)))

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)

a = (ufl.inner(ufl.curl(u), ufl.curl(v))) * ufl.dx
b = ufl.inner(u, v) * ufl.dx

a = fem.form(a)
b = fem.form(b)

mpi_print('Applying Boundary Conditions...')
bc_facets = exterior_facet_indices(mesh.topology)
bc_dofs = fem.locate_dofs_topological(V, mesh.topology.dim - 1, bc_facets)
u_bc = fem.Function(V)
with u_bc.x.petsc_vec.localForm() as loc:
    loc.set(0)
bc = fem.dirichletbc(u_bc, bc_dofs)
mpi_print('Done.')


mpi_print('Assembling Matrix...')
A = assemble_matrix(a, bcs=[bc])
A.assemble()
B = assemble_matrix(b, bcs=[bc])
B.assemble()
mpi_print('Done.')

mpi_print('Setting up Problem...')
eps = SLEPc.EPS().create(mesh.comm)
eps.setOperators(A, B)
eps.setProblemType(SLEPc.EPS.ProblemType.GNHEP)

tol = 1e-9
max_it = 10000
eps.setTolerances(tol=tol, max_it=max_it)
mpi_print('tol and max it: %s'%str(eps.getTolerances()))

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
mpi_print('Done.')

ksp =  st.getKSP()
ksp.setType('preonly')

pc = ksp.getPC()
pc.setType('lu')
pc.setFactorSolverType('superlu_dist') # takes up additional memory

mpi_print('Solving...')
eps.solve()
eps.view()
eps.errorView()

mpi_print('Done.')

eigen_values = []
mpi_print('Eigenvalues:')
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
    eigen_values.append(eigen_value)
    mode_freq = convert_eigenvalue_to_f(np.real(eigen_value))
#    if i == 0:
#        freq_list.append(float(mode_freq)/1e9)
    mpi_print('%i, %0.05f GHz'%(i,mode_freq/1e9))
mpi_print('Done.')

vals = [(i, np.sqrt(eps.getEigenvalue(i))) for i in range(nev)]
eh = fem.Function(V)

kz_list = []

for i, kz in vals:
    eigen_value = eigen_values[i]
    mode_freq = convert_eigenvalue_to_f(np.real(eigen_value))
    mpi_print('Mode frequency: %0.03f GHz'%(mode_freq/1e9))
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

    norm_local = fem.assemble_scalar(fem.form(epsilon * ufl.inner(eth,eth) * ufl.dx))
    norm = mesh.comm.allreduce(norm_local, op=MPI.SUM)
#    mpi_print('Norm: %0.03e'%norm)
    eth.x.array[:] = eth.x.array[:] / np.sqrt(norm)

    mode_power_local = fem.assemble_scalar(fem.form(epsilon * ufl.inner(eth,eth) * ufl.dx))
    mode_power = mesh.comm.allreduce(mode_power_local, op=MPI.SUM)
    mpi_print('Mode Power E: %0.03f W'%mode_power)

    gdim = mesh.geometry.dim
#        V_dg = fem.functionspace(mesh, ("CG", degree, (gdim,)))
    V_dg = fem.functionspace(mesh, ("CG", degree, (gdim,)))
#        V_dg = fem.functionspace(mesh, ("DG", interpolation_degree, (gdim,)))
    Et_dg = fem.Function(V_dg)
    Et_dg.interpolate(eth)
    Et_dg.x.scatter_forward()

    H = fem.Function(V_dg)
    const = (1./(2*np.pi*mode_freq * mu))
#    mpi_print(const)
    H_form = ufl.curl(eth)
    H_expr = fem.Expression(H_form, V_dg.element.interpolation_points())
    H.interpolate(H_expr)
    H.x.scatter_forward()
    H.x.array[:] = H.x.array[:] * const


    mode_power_H_local = fem.assemble_scalar(fem.form((1.0/mu) * ufl.inner(H,H) * ufl.dx))
    mode_power_H = mesh.comm.allreduce(mode_power_H_local, op=MPI.SUM)
    mpi_print('Mode Power H: %0.03f W'%mode_power)

    B = fem.Function(V_dg)
    B.interpolate(H)
    B.x.scatter_forward()
    B.x.array[:] = B.x.array[:] * mu


    if i < nev:
        mpi_print('Saving solution, eigenvalue %i'%i)
        with io.VTXWriter(mesh.comm, "sols_test/TE011_E_test_%02i.bp"%i, Et_dg) as f:
            f.write(0.0)
        with io.VTXWriter(mesh.comm, "sols_test/TE011_H_test_%02i.bp"%i, H) as f:
            f.write(0.0)
        with io.VTXWriter(mesh.comm, "sols_test/TE011_B_test_%02i.bp"%i, B) as f:
            f.write(0.0)


#def TE102_mode(x):
#    # x: (3, n) array of coordinates on the port boundary
#    E0 = 1.0  # amplitude of excitation (can be adjusted)
#    val_x = E0 * np.sin(ufl.pi * x[2] / a) * np.sin(ufl.pi * 2 * x[1]/d)
#
#    values =  np.vstack((val_x, np.zeros_like(x[0]), np.zeros_like(x[0])))
#
#    return values

#radius = .4
#def Capillary(x):
#    val = np.zeros_like(x[0])
##    val[np.where(np.sqrt((x[0] - b/2)**2 + (x[1]-d/2)**2) <= radius)] = 1
##    val[np.where(np.abs((x[1]-d/2)) <= (d/4))] = 1
#    val[np.where(x[1] < (d/2))] = 1
#    values = np.vstack((val,val,val)) 
#    values /= np.sqrt(3)
#    return values
#
#def sample(x):
#    return x[1] < (d/2)
##    return np.sqrt((x[0] - b/2)**2 + (x[1]-d/2)**2) <= radius
#
##sample_markers = dolfinx.mesh.meshtags(mesh, gdim, facets, sample)
#num_local_cells = mesh.topology.index_map(tdim).size_local
#cell_indices = np.arange(num_local_cells, dtype=np.int32)
#
#sample_cells = locate_entities(mesh, gdim, sample)
#
#sample_markers = np.full(len(cell_indices), 0, dtype=np.int32)  # default unmarked
#
#for i, cell in enumerate(cell_indices):
#    if cell in sample_cells:
#        sample_markers[i] = 1
#
## Get number of local cells
#
#
##mpi_print(cell_markers)
##cells = exterior_facet_indices(mesh.topology)
#
##cell_markers_sample = np.array([1 if sample(c) else 0 for c in cell_centers], dtype=np.int32)
#ct = meshtags(mesh, gdim, cell_indices, sample_markers)
#dx_sample = ufl.Measure("dx", domain=mesh, subdomain_data=ct)
#
#E_102 = fem.Function(V_CG)
#E_102.interpolate(eh)
#E_102.x.scatter_forward()
#
#cap = fem.Function(V_CG)
#cap.interpolate(Capillary)
#cap.x.scatter_forward()
#
#B_102 = fem.Function(V_CG)
#B_form = ufl.curl(E_102)
#B_expr = fem.Expression(B_form, V.element.interpolation_points())
#B_102.interpolate(B_expr)
#B_102.x.scatter_forward()
#
#Bz_102 = fem.Function(V_CG)
#Bz_form = ufl.as_vector((0,0,B_102[2]))
#Bz_expr = fem.Expression(Bz_form, V.element.interpolation_points())
#Bz_102.interpolate(Bz_expr)
#Bz_102.x.scatter_forward()
#
##B_102_mag = fem.Function(V)
##B_form_mag = ufl.form(B_102)
##B_form_mag = ufl.sqrt(ufl.inner(B_102, B_102))
##B_form_mag = ufl.as_vector(((B_102[0]*B_102[0])**0.5, (B_102[1]*B_102[1])**0.5, (B_102[2]*B_102[2])**0.5))
##B_expr_mag = fem.Expression(B_form_mag, V.element.interpolation_points())
##B_102_mag.interpolate(B_expr_mag)
##B_102_mag.x.scatter_forward()
##B_102_mag = np.sqrt(B_102**2.)
#
#B_sample_local = fem.assemble_scalar(fem.form(ufl.inner(B_102,B_102) * dx_sample(1)))
#B_sample = mesh.comm.allreduce(B_sample_local, op = MPI.SUM)
#
#B_resonator_local = fem.assemble_scalar(fem.form(ufl.inner(B_102,B_102) * ufl.dx))
#B_resonator = mesh.comm.allreduce(B_resonator_local, op = MPI.SUM)
#
#volume_sample_local = fem.assemble_scalar(fem.form(1.0*dx_sample(1)))
#volume_sample = mesh.comm.allreduce(volume_sample_local, op = MPI.SUM)
#
#volume_local = fem.assemble_scalar(fem.form(1.0*ufl.Measure('dx', domain = mesh)))
#volume = mesh.comm.allreduce(volume_local, op = MPI.SUM)
#
#
#filling_factor = B_sample / B_resonator
#
#mpi_print('B Sample:')
#mpi_print(B_sample)
#mpi_print('B Resonator:')
#mpi_print(B_resonator)
#mpi_print('Filling Factor:')
#mpi_print(filling_factor)
#
##mpi_print('Volume Sample:')
##mpi_print(volume_sample)
#
##mpi_print('Volume:')
##mpi_print(volume)
#
#with io.VTXWriter(mesh.comm, "sols_test/E_102.bp", E_102) as f:
#    f.write(0.0)
#
#with io.VTXWriter(mesh.comm, "sols_test/B_102.bp", B_102) as f:
#    f.write(0.0)
#
#with io.VTXWriter(mesh.comm, "sols_test/cap.bp", cap) as f:
#    f.write(0.0)
mpi_print('Script Done.')
