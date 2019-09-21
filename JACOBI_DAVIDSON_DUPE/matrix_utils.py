import numpy as np
import sys


# symmetrizes matrix
def symmetrize_mat(mat):
    # type : (numpy.matrix) -> numpy.matrix
    new_mat = mat / 2 + mat.transpose() / 2
    return new_mat


# Generates a real random matrix with nrows and ncols
# actually not random; diagonal elements increase with indexes
def generate_random_symmetric_matrix(dim):
    mat = np.random.rand(dim, dim)
    return symmetrize_mat(mat)


# Scales down off diagonal elements to make matrix amenable to solution by Davidson
def make_diagonally_dominant(mat_in, sparsity):
    mat_out = mat_in
    for ii in range(np.size(mat_in, 0)):
        for jj in range(np.size(mat_in, 1)):
            if ii != jj:
                scale = np.power(sparsity, abs(ii-jj))
                if scale > 1e-6:
                    mat_out[ii, jj] = mat_in[ii, jj]*scale
                else:
                    mat_out[ii, jj] = 0.0
    return mat_out


# Orthonormalizes vec w.r.t. mat using modified Gramm Schmidt
# matrix is a set of vectors stored as _columns_
def orthonormalize(vec, mat):
    for ii in range(np.size(mat, 1)):
        vec = vec - np.dot(vec, mat[:, ii])*mat[:, ii]
    return vec/np.linalg.norm(vec)


def test_orthogonality(a, name="mat"):
    ata = np.dot(a.T, a)
    for ii in range(ata.shape[0]):
        for jj in range(ata.shape[1]):
            if (ii != jj) and (np.abs(ata[ii, jj]) > 1e-3):
                print(name+"^{T}"+name+"[", ii, ",", jj, "] = ", ata[ii, jj])
                sys.exit("Orthogonalization failed!")


def modified_gram_schmidt(a):
    ncols = a.shape[0]
    qmat = np.zeros_like(a)
    for j in range(ncols):
        q = a[:, j]
        for i in range(j):
            rij = np.vdot(qmat[:, i], q)
            q = q - rij*qmat[:, i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj, 0.0):
            raise ValueError("invalid input matrix")
        else:
            qmat[:, j] = q/rjj
    return qmat


# Normalize v, and return pair, where second argument is the "relative shrinkage" of v
# If ||(1-AA*)v|| <<  ||v||,  then component of v orthogonal to A, and the resulting orthogonalized
# v is more likely to be problematically influenced by noise
def orthonormalize_v_against_mat_check(v, mat):
    ncols = mat.shape[1]
    orig_mod_v = np.linalg.norm(v)
    for ii in range(ncols):
        v = v - np.vdot(mat[:, ii], v) * mat[:, ii]
    new_mod_v = np.linalg.norm(v)
    if new_mod_v > 1e-10:
        return v/new_mod_v, new_mod_v/orig_mod_v
    else:
        return v, 1e-10


# Returns normed vector, second return value is false if norm of vector was too small for accurate
def normalize(vec, threshold=1e-10):
    buff = vec
    norm = np.linalg.norm(buff)
    if norm > threshold:
        return buff/norm, True
    else:
        return buff, False


def orthogonalize_v1_against_v2(v1, v2, arctan_norm_angle_thresh=1e-8, norm_thresh=1e-12):
    v1_norm_orig = np.linalg.norm(v1)
    print("v1_norm_orig = ", v1_norm_orig)
    print("check1 : np.linalg.norm(v1)= ", np.linalg.norm(v1))
    v1new, check1 = normalize(v1, norm_thresh)
    if check1:
        print("check2 : np.linalg.norm(v2)= ", np.linalg.norm(v2))
        v2new, check2 = normalize(v2, norm_thresh)
        if check2:
            print("check3 : np.linalg.norm(v1-v2) =", np.linalg.norm(v1new-v2new))
            vnew, successful_norm = normalize((v1new - v2new), norm_thresh)
            if not successful_norm:
                print("Normalization in matrix_utils.orthogonalize_v1_against_v2(v1, v2) failed : "
                      "||v1-v2|| < " + str(norm_thresh))
                return vnew, successful_norm
            else:
                return vnew, (np.linalg.norm(vnew) / v1_norm_orig > arctan_norm_angle_thresh)
        else:
            exit("normalization of v2 failed")
            return v1, False
    else:
        print("normalization of v1 failed")
        return v1, False


# Normalize v against vectors stored as columns in A
def orthonormalize_v_against_mat(v, mat):
    ncols = mat.shape[1]
    for ii in range(ncols):
        v = v - np.vdot(mat[:, ii], v) * mat[:, ii]
    return normalize(v)


def print_only_large_imag(vec, name=" "):
    if name != " ":
        print(name, end=' = ')
    for elem in vec:
        if abs(elem.imag) < 0.000001:
            print(elem.real, end=' ')
        else:
            print(elem, end=' ')
    print("\n")


def sort_eigvecs_and_vals(eigvals, eigvecs):
    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs


# checks whether columns of matrix vspace are normalized
def check_normalization(vspace, thresh=1e-10, name="???"):
    for ii in range(len(vspace)):
        vnorms = np.linalg.norm(vspace[:, ii])
        if (vnorms - 1) > thresh:
            sys.exit("normalization of " + name + " failed... Aborting!")


def remove_imag_part(complex_array):
    for ii in range(complex_array.shape[0]):
        for jj in range(complex_array.shape[1]):
            if abs(np.imag(complex_array[ii, jj])) < 1e-12:
                complex_array[ii, jj] = np.real(complex_array[ii, jj]) + 0.0j


# crap, should be done with forvec
def zero_small_parts(complex_array, thresh=1e-10):
    if len(complex_array.shape) == 2:
        for ii in range(complex_array.shape[0]):
            for jj in range(complex_array.shape[1]):
                if abs(np.imag(complex_array[ii, jj])) < thresh:
                    complex_array[ii, jj] = complex_array[ii, jj].real
                if abs(np.real(complex_array[ii, jj])) < thresh:
                    complex_array[ii, jj] = complex_array[ii, jj].imag

    elif len(complex_array.shape) == 1:
        for ii in range(complex_array.shape[0]):
            if abs(np.imag(complex_array[ii])) < 1e-10:
                complex_array[ii] = complex_array[ii].real
            if abs(np.real(complex_array[ii])) < 1e-10:
                complex_array[ii] = complex_array[ii].imag


def print_largest_component_of_vector(vec, name):
    print(name + "[" + str(vec.argmax()) + "] = ", vec.max())


def print_largest_component_of_vector_bundle(vspace, space_name):
    print("------------------------- " + space_name + "----------------------------")
    for iv in range(vspace.shape[1]):
        print_largest_component_of_vector(vspace[:, iv], space_name + "_{" + str(iv) + "}")


def find_nonzero_elems(seedname, input_array, threshold=1e-10):
    non_zero_ids = np.argwhere(np.abs(input_array) > threshold)
    outfile = open(seedname + "_nonzero.txt", "w+")
    for idx in non_zero_ids:
        outfile.write(str(idx) + " =  " + str(input_array[idx]) + "\n")
    outfile.close()


def print_nonzero_numpy_elems(my_arr, arr_name="??", thresh=1e-10):
    if len(my_arr.shape) == 2:
        for ii in range(my_arr.shape[0]):
            for jj in range(my_arr.shape[1]):
                if abs(my_arr[ii, jj]) > thresh:
                    print(arr_name + "[" + str(ii) + "," + str(jj) + "] = ", my_arr[ii, jj])

    elif len(my_arr.shape) == 1:
        for ii in range(my_arr.shape[0]):
            if abs(my_arr[ii]) > thresh:
                print(arr_name + "[" + str(ii) + "] = ", my_arr[ii])


def check_for_nans(arr_list, name_list, exit_on_nan=True):
    for ii in range(len(arr_list)):
        if np.isnan(arr_list[ii]).any():
            print(name_list[ii], "has a NaN")
            if exit_on_nan:
                exit()


def save_arrs_to_file(arr_list, arr_names, parent_folder="/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/"):
    for ii in range(len(arr_list)):
        np.savetxt(parent_folder+arr_names[ii] + ".txt", arr_list[ii])
