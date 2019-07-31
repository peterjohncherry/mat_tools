import matrix_utils as mu
import numpy as np
import jacobi_davidson_4c as jdr4c
import mat_reader as mr


def numpy_check(matrix_orig, num_eig, print_eigvals=True, print_eigvecs=False):
    eigvals, eigvecs = np.linalg.eig(matrix_orig)
    eigvals = np.sort(eigvals)

    if print_eigvals :
        print("numpy results = ", eigvals[:num_eig])
    if print_eigvecs :
        print("numpy results = ", eigvecs[:num_eig])


def test_fortran_file_read():
    nrows, ncols = mat_reader.read_mat_info_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.info')
    print ("nrows = ", nrows, "  ncols = ", ncols)
    mat_reader.read_binary_fortran_file('/home/peter/SMALL_PROGS/FORTRAN_MAT_OUTPUT/mat1_test.bin',
                                        nrows, ncols, datatype="real")


def test_jacobi_davidson_4c():
    nevals = 3
    jd_test = jdr4c.JacobiDavidson4C(num_eigenvalues=nevals,
                                     rs_filename="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/TDA/4c-HF.out_scf")
    jd_test.initialize()
    jd_test.read_full_matrix(file_seedname="/home/peter/RS_FILES/4C/full_mat")  # should really do in initialization
    jd_test.solve()


def test_array_reading():
    evals = mr.read_fortran_array("/home/peter/RS_FILES/4C/lapack_eigvals")
    evals = np.float64(evals)
    np.savetxt("/home/peter/RS_FILES/4C/lapack_eigvals", evals, fmt='%10.5f')
