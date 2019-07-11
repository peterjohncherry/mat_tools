import numpy as np
import sys

# symmetrizes matrix
def symmetrize_mat(mat):
    # type : (numpy.matrix) -> numpy.matrix
    new_mat = mat / 2 + mat.transpose() / 2
    return new_mat

# generates a real random matrix with nrows and ncols
def generate_random_symmetric_matrix(nrows, ncols):
    # type: (int, int) -> matrix
    return symmetrize_mat(np.random.rand(nrows, ncols))

#Scales down off diagonal elements to make matrix amenable to solution by Davidson
def make_diagonally_dominant(mat_in, sparsity) :
    mat_out = mat_in
    for ii in range(np.size(mat_in,0)):
        for jj in range(np.size(mat_in, 1)):
            if ii != jj:
                mat_out[ii,jj] = mat_in[ii,jj]*(sparsity**abs(ii-jj))
    return mat_out