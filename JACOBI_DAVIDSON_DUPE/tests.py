
import eps_solvers as es
import matrix_utils as mu
import mat_reader
import numpy as np
import davidson
import jacobi_davidson as jd

def numpy_check( matrix_orig, eig, print_eigvals = True, print_eigvecs = False):
    E, Vec = np.linalg.eig(matrix_orig)
    E = np.sort(E)

    if print_eigvals :
        print("numpy results = ", E[:eig])
    if print_eigvecs :
        print("numpy results = ", Vec[:eig])

def test_davidson():
    threshold = 0.0000001
    max_iter = 48
    ndim = 50
    nevals = 4
    sparsity = 0.1
    A = mu.make_diagonally_dominant(mu.generate_random_symmetric_matrix(ndim), sparsity)

    dave_test = davidson.Davidson(A, "Davidson", nevals, threshold, max_iter)
    dave_test.solve()
    numpy_check(A, nevals)

def test_jacobi_davidson():
    threshold = 0.0000001
    max_iter = 100


    ndim = 1000
    nevals = 4
    sparsity = 0.01
    A = mu.make_diagonally_dominant( mu.generate_random_symmetric_matrix(ndim), sparsity)
    jd_test = jd.JacobiDavidson(A, "Jacobi Davidson", nevals, threshold, max_iter)
    jd_test.set_variables(preconditioning_type="Full")
    jd_test.first_iteration_init()
    jd_test.main_loop()

    # Calculate and print eigvals from numpy for checking
    numpy_check(A,nev)

def test_fortran_file_read():
    nrows, ncols = mat_reader.read_mat_info_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.info')
    print ("nrows = ", nrows, "  ncols = ", ncols)
    mat_reader.read_binary_fortran_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.bin', nrows, ncols, datatype="real")





