
import eps_solvers
import matrix_utils as mu

def test_jacobi_davidson():
    threshold = 0.0000001
    max_iter = 20
    jd_test = eps_solvers.eps_solver("Jacobi Davidson", threshold, max_iter)

    ndim = 20
    nev = 4
    sparsity = 0.0001
    A = mu.make_diagonally_dominant( mu.generate_random_symmetric_matrix(ndim, ndim), sparsity)

    jd_test.set_variables(ndim, nev, A)
    jd_test.first_iteration_init()
    jd_test.main_loop()



