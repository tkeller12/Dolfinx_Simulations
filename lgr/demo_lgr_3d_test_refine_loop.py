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

def mpi_print(s, rank = 0):
    if rank is None:
        print(f"Rank {comm.rank}: {s}")
    elif comm.rank == rank:
        print(f"Rank {comm.rank}: {s}")

def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target


target_eigenvalue = convert_freq_to_target(target_freq)


mpi_print('Importing Mesh...')
mesh, cell, facet_tags = gmshio.read_from_msh('mesh/lgr_3d_test3.msh', comm, 0, gdim=3)
mpi_print('Done.')

#mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
#gdim = mesh.geometry.dim


nev = 4

degree = 2
element_type = "N2curl"
max_run_ix = 7
#run_ix = 3
freq_list = []

for run_ix in range(max_run_ix):
    mpi_print('RUN IX %i'%run_ix)
    num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
    mpi_print('Cells: %i'%num_cells)
    mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)
    gdim = mesh.geometry.dim
    V = fem.functionspace(mesh, (element_type, degree, (gdim,)))

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
    pc.setFactorSolverType('superlu_dist')

    mpi_print('Solving...')
    eps.solve()
    eps.view()
    eps.errorView()

    mpi_print('Done.')


    mpi_print('Eigenvalues:')
    for i in range(eps.getConverged()):
        eigen_value = eps.getEigenvalue(i)
        mode_freq = convert_eigenvalue_to_f(np.real(eigen_value))
        if i == 0:
            freq_list.append(float(mode_freq)/1e9)
        mpi_print('%i, %0.05f GHz'%(i,mode_freq/1e9))
    mpi_print('Done.')

    vals = [(i, np.sqrt(eps.getEigenvalue(i))) for i in range(eps.getConverged())]

    eh = fem.Function(V)

    kz_list = []

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
        V_dg = fem.functionspace(mesh, ("CG", degree, (gdim,)))
        Et_dg = fem.Function(V_dg)
        Et_dg.interpolate(eth)

        B = fem.Function(V_dg)
        B_expr = fem.Expression(ufl.curl(eth), V_dg.element.interpolation_points())
        B.interpolate(B_expr)

        V_G = fem.functionspace(mesh, ("DG", 0, (1,)))
        G = fem.Function(V_G)
        G_form = ufl.inner(ufl.grad(eth),ufl.grad(eth))
        G_expr = fem.Expression(G_form, V_G.element.interpolation_points())
        G.interpolate(G_expr)


        order = np.argsort(G.x.array)
#        cell_index = order[-int(0.3*order.size):-1] # 30% refinement
        cell_index = order[-int(0.05*order.size):-1] # 5% refinement
#        cell_index = order[-int(0.02*order.size):-1] # 5% refinement
        mesh.topology.create_connectivity(3, 1)
        mesh.topology.create_connectivity(1, 3)
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local

        edge_index = compute_incident_entities(mesh.topology, cell_index, 3, 1)

        new_mesh = refine(mesh, edge_index.astype(np.int32)) # works
        new_mesh = new_mesh[0]
        num_cells = mesh.topology.index_map(mesh.topology.dim).size_local
        num_cells = new_mesh.topology.index_map(new_mesh.topology.dim).size_local


        # Save solutions
        mpi_print('Saving solution %i'%run_ix)
        if i < nev:
            with io.VTXWriter(mesh.comm, "sols_lgr_%i/Et_%04i.bp"%(run_ix,i), Et_dg) as f:
                f.write(0.0)

            with io.VTXWriter(mesh.comm, "sols_lgr_%i/B_%04i.bp"%(run_ix,i), B) as f:
                f.write(0.0)

        # Update mesh for eigenvector closest to target eigenvalue
        if i == 0:
            mpi_print('Setting new mesh')
            mpi_print(num_cells)
            temp_mesh = new_mesh # set mesh equal to new mesh   

    mesh = temp_mesh

mpi_print('Script Done.')
for freq_ix, freq_value in enumerate(freq_list):
    mpi_print('%i, %0.05f'%(freq_ix,freq_value))
