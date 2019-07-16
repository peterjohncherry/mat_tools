import numpy as np
import sys

# symmetrizes matrix
def symmetrize_mat(mat):
    # type : (numpy.matrix) -> numpy.matrix
    new_mat = mat / 2 + mat.transpose() / 2
    return new_mat

# generates a real random matrix with nrows and ncols
#actually not random; diagonal elements increase with indexes
def generate_random_symmetric_matrix(dim):
    # type: (int, int) -> matrix
    mat = np.random.rand(dim,dim)
    return symmetrize_mat(mat)

#Scales down off diagonal elements to make matrix amenable to solution by Davidson
def make_diagonally_dominant(mat_in, sparsity) :
    mat_out = mat_in
    for ii in range(np.size(mat_in, 0)):
        for jj in range(np.size(mat_in, 1)):
            if ii != jj:
                scale = np.power(sparsity, abs(ii-jj))
                if scale > 1e-6 :
                    mat_out[ii,jj] = mat_in[ii,jj]*scale
                else :
                    mat_out[ii, jj] = 0.0
    return mat_out

# orthonormalizes vec w.r.t. mat using modified Gramm Schmidt
# matrix is a set of vectors stored as _columns_
def orthonormalize(vec, mat):
    for ii in range(np.size(mat,1)):
        vec = vec - np.dot(vec, mat[:,ii])*mat[:,ii]
    return vec/np.linalg.norm(vec)

def test_orthogonality(A, name ="A"):
    AtA = np.dot(A.T, A)
    for ii in range(AtA.shape[0]):
        for jj in range(AtA.shape[1]):
            if (ii != jj and np.abs(AtA[ii,jj]) > 1e-3):
                print(name+"^{T}"+name+"[",ii,",",jj,"] = ", AtA[ii,jj])
                sys.exit("Orthogonalization failed!")

def modifiedGramSchmidt(A):
    ncols = A.shape[0]
    Q = np.zeros(A.shape, dtype=A.dtype)
    for j in range(ncols):
        q = A[:,j]
        for i in range(j):
            rij = np.vdot(Q[:,i],q)
            q = q - rij*Q[:,i]
        rjj = np.linalg.norm(q, ord=2)
        if np.isclose(rjj,0.0):
            raise ValueError("invalid input matrix")
        else:
            Q[:,j] = q/rjj
    return Q


# Normalize v, and return pair, where second argument is the "relative shrinkage" of v
# If ||(1-AA*)v|| <<  ||v||,  then component of v orthogonal to A, and the resulting orthogonalized
# v is more likely to be problematically influenced by noise
def orthonormalize_v_against_A_check(v,A):
    ncols = A.shape[1]
    orig_mod_v = np.linalg.norm(v)
    for ii in range(ncols):
        v = v - np.vdot(A[:, ii], v) * A[:, ii]
    new_mod_v = np.linalg.norm(v)
    #print("new_mod_v = ", new_mod_v)
    #print("orig_mod_v = ", orig_mod_v)
    if new_mod_v > 1e-10 :
        return v/new_mod_v, new_mod_v/orig_mod_v
    else :
        return v, 1e-10

# Normalize v against vectors stored as columns in A
def orthonormalize_v_against_A(v,A):
    ncols = A.shape[1]
    for ii in range(ncols):
        v = v - np.vdot(A[:, ii], v) * A[:, ii]
    return  v / np.linalg.norm(v)

def print_only_large_imag( vec,name = " " ):
    if name != " ":
        print(name, end=' = ')
    for elem in vec :
        if ( abs(elem.imag) < 0.000001):
            print (elem.real, end=' ')
        else:
            print(elem, end=' ')
    print ("\n")

def sort_eigvecs_and_vals(eigvals, eigvecs):

    idx = eigvals.argsort()
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    return eigvals, eigvecs

#checks whether columns of matrix vspace are normalized
def check_normalization(vspace, thresh = 1e-10, name = "???"):
    bad_norms = []

    for ii in range(iter):
        vnorm =np.linalg.norm(self.vspace[:, ii])
        if ( vnorm -1 > thresh ):
            vnorms.append((str(ii)+ "norm = ", vnorm ))
    if len(vnorm) > 0:
        for a,b in vnorms :
            print(a,b)
        sys.exit("normalization of " + name + " failed... Aborting!")