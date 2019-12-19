import numpy as np
import jacobi_davidson_tda_4c as jdr4c
import jacobi_davidson_full_4c as jd_f_4c
import mat_reader as mr
import matrix_utils as utils


# This just diagonalizes a matrix directly, used to check if the approximate Jacobi Davidson routines are working
def numpy_check(matrix_orig, num_eig, print_eigvals=True, print_eigvecs=False):
    eigvals, eigvecs = np.linalg.eig(matrix_orig)
    eigvals = np.sort(eigvals)

    if print_eigvals:
        print("numpy results = ", eigvals[:num_eig])
    if print_eigvecs:
        print("numpy results = ", eigvecs[:num_eig])


# Tests the file reading
def test_fortran_file_read():
    nrows, ncols, datatype = mr.read_array_info_file('REF/INPUT/MAT_READER/mat1_test.info')
    print("nrows = ", nrows, "  ncols = ", ncols)
    my_array = mr.read_binary_fortran_file('REF/INPUT/MAT_READER/mat1_test.bin', datatype, nrows, ncols)
    np.savetxt("OUTPUT/MAT_READER/my_matrix.txt", my_array)


# TDA version
def test_jacobi_davidson_tda_4c():
    nevals = 3
    jd_test = jdr4c.JacobiDavidsonTDA4C(num_eigenvalues=nevals,
                                        rs_filename="REF/INPUT/4c-HF.out_scf")
    jd_test.read_full_matrix(file_seedname="REF/INPUT/TDA/full_mat")
    jd_test.solve()


# FULL matrix version
def test_jacobi_davidson_full_4c():
    nevals = 5
    jd_test = jd_f_4c.JacobiDavidsonFull4C(num_eigenvalues=nevals,
                                           rs_filename="REF/INPUT/4c-HF.out_scf")
    jd_test.read_full_matrix(file_seedname="REF/INPUT/FULL/full_mat")
    jd_test.solve()


# tests that the routines to orthogonalize two vectors are working OK
def test_v1_v2_orthogonalization():
    # new_v1 should be [ 0 0 0 1 1 1 ] with angle of 45 degrees
    v1 = np.array([1, 1, 1, 1, 1, 1])
    v2 = np.array([1, 1, 1, 0, 0, 0])
    print("v1 = ", v1)
    print("v2 = ", v2)
    new_v1, angle = utils.orthogonalize_v1_against_v2(v1, v2)
    print("new_v1 = ", new_v1)
    print("angle = ", np.degrees(np.arcsin(angle)))
