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

from dolfinx.mesh import CellType, create_box, exterior_facet_indices, locate_entities

from slepc4py import SLEPc


c = 299792458 # speed of light, m/s
target_freq = 10e9


def convert_eigenvalue_to_f(k_squared):
    return c * np.sqrt(k_squared) / (2 * np.pi)

def convert_freq_to_target(freq):
    target = (freq * 2 * np.pi / c)**2.
    return target

def import_mesh(filename):
    print('Importing Mesh...')
    mesh, cell, facet_tags = gmshio.read_from_msh(filename, MPI.COMM_WORLD, 0, gdim=3)
    print('Done.')

    mesh.topology.create_connectivity(mesh.topology.dim-1,mesh.topology.dim)

    return mesh


def solve(mesh, target_eigenvalue, num_eigenvalues, element_type = "N2curl", degree = 2, tol = 1e-9, maxit = 10000):

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

    #    tol = 1e-9
    #    max_it = 10000
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

    return eps



print('Eigenvalues:')
for i in range(eps.getConverged()):
    eigen_value = eps.getEigenvalue(i)
    mode_freq = convert_eigenvalue_to_f(np.real(eigen_value))
    print(i, '%0.05f GHz'%(mode_freq/1e9))

print('Done.')

vals = [(i, np.sqrt(eps.getEigenvalue(i))) for i in range(eps.getConverged())]

# Sort kz by real part
#vals.sort(key=lambda x: x[1].real) # this doesn't make sense if you "target" a specific eigenvalue

eh = fem.Function(V)

kz_list = []

#print('Summary:')
for i, kz in vals:
#    print('-'*50)
#    print('i:',i)
#    print('kz:',kz)
#    print(i, kz)
    # Save eigenvector in eh
    eps.getEigenpair(i, eh.x.petsc_vec)

    # Compute error for i-th eigenvalue
    error = eps.computeError(i, SLEPc.EPS.ErrorType.RELATIVE)
#    print('Error:',error)
#    if error > tol:
#        print('***DID NOT CONVERGE!!!***')

    # Verify, save and visualize solution
#    if error < tol and np.isclose(kz.imag, 0, atol=tol):
    if True:
        kz_list.append(kz)
#        print('freq:', kz)


#        print(f"eigenvalue: {-kz**2}")
#        print(f"kz: {kz}")
#        print(f"kz/k0: {kz / k0}")

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

        # Save solutions
        with io.VTXWriter(mesh.comm, "sols_lgr/Et_%04i.bp"%i, Et_dg) as f:
            f.write(0.0)

        with io.VTXWriter(mesh.comm, "sols_lgr/B_%04i.bp"%i, B) as f:
            f.write(0.0)




if __name__ == "__main__":
    print('Script Start')
    target_eigenvalue = convert_freq_to_target(target_freq)

    print('Script Done.')
