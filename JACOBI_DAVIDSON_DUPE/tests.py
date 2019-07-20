import matrix_utils as mu
import numpy as np
import davidson
import jacobi_davidson_real as jdr
import jacobi_davidson_4c as jdr4c
import mat_reader as mr

def numpy_check( matrix_orig, eig, print_eigvals = True, print_eigvecs = False):
    E, Vec = np.linalg.eig(matrix_orig)
    E = np.sort(E)

    if print_eigvals :
        print("numpy results = ", E[:eig])
    if print_eigvecs :
        print("numpy results = ", Vec[:eig])

def test_davidson():
    threshold = 1e-8
    max_iter = 500
    ndim = 1000
    nevals = 4
    sparsity = 0.01
    A = mu.make_diagonally_dominant(mu.generate_random_symmetric_matrix(ndim), sparsity)
    B = A
    dave_test = davidson.Davidson(A, "Davidson", nevals, threshold, max_iter)
    dave_test.solve()
    numpy_check(B, nevals)

def test_jacobi_davidson():
    threshold = 1e-8
    max_iter = 50
    ndim = 100
    nevals = 4
    sparsity = 0.001
    A = mu.make_diagonally_dominant( mu.generate_random_symmetric_matrix(ndim), sparsity)
    # Calculate and print eigvals from numpy for checking
    numpy_check(A, nevals)
    jd_test = jdr.JacobiDavidsonReal("Jacobi Davidson", nevals, threshold, max_iter)
    jd_test.set_variables(preconditioning_type="Full")
    jd_test.read_full_matrix(file_seedname="/home/RS_FILES/4C/full_mat")
    jd_test.first_iteration_init()
    jd_test.solve()

    # Calculate and print eigvals from numpy for checking
    numpy_check(A,nevals)

def test_fortran_file_read():
    nrows, ncols = mat_reader.read_mat_info_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.info')
    print ("nrows = ", nrows, "  ncols = ", ncols)
    mat_reader.read_binary_fortran_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.bin', nrows, ncols, datatype="real")

def test_jacobi_davidson_4c():
    threshold = 1e-8
    max_iter = 50
    nevals = 3
    jd_test = jdr4c.JacobiDavidson4C("Jacobi Davidson", nevals, threshold, max_iter)
    jd_test.read_full_matrix(file_seedname = "/home/peter/RS_FILES/4C/full_mat")
    evals = mr.read_fortran_array("/home/peter/RS_FILES/4C/lapack_eigvals")
    evals = np.float64(evals)
    np.savetxt("/home/peter/RS_FILES/4C/lapack_eigvals", evals, fmt='%10.5f')
    jd_test.initialize_tda()





