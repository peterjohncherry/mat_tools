
import eps_solvers
import matrix_utils as mu

def test_jacobi_davidson():
    threshold = 0.0000001
    max_iter = 5
    jd_test = eps_solvers.eps_solver("Jacobi Davidson", threshold, max_iter)

    ndim = 10
    sparsity = 0.0001
    A = mu.make_diagonally_dominant( mu.generate_random_symmetric_matrix(ndim, ndim), sparsity)
    print( "--- A ---\n ", A)

