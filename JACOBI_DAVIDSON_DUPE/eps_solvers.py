import numpy as np
import mat_reader as mr
import sys


class Solver:

    def __init__(self, rs_filename, num_eigenvalues, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="TDA", symmetry="general", pe_rot=False):

        # Solver parameters
        self.nev = num_eigenvalues  # number of eigenvalues to solve for
        self.pe_rot = pe_rot  # Allow rotations between +ive and -ive energy virtual orbitals
        self.symmetry = symmetry  # Symmetry to be used when constructing guess
        self.threshold = threshold  # convergence threshold for solver
        self.solver = solver  # Solver type
        self.method = method  # approximation for preconditioning
        self.complex_precision = np.complex128  # precision for internal arrays
        self.real_precision = np.float64        # precision for internal arrays

        # Information to be read in from ReSpect SCF output
        self.rs_filename = rs_filename  # seedname for ReSpect SCF files
        self.evals_1e = None  # 1-electron eigenvalues
        self.mat_orig = None  # Matrix whose eigenvectors we seek
        self.esorted = None   # Sorted evals_1e
        self.eindex = None    # indexes used to sort evals_1e; needed for guess construction
        self.ndims = -1  # number of spherical GTOs
        self.ndimc = -1  # number of Cartesian GTOs
        self.nocc = -1   # number of occupied orbitals
        self.nvirt = -1  # number of virtual orbitals
        self.ndim = -1   # length of ritz_vector
        self.nov = None  # nov = nocc*nvirt = ndim(in TDA) = ndim/2 (in FULL)

        # Internal arrays to be used in solver
        self.u_vecs = None
        self.r_vecs = None
        self.submat = None
        self.teta = None

        self.get_basis_info(rs_filename)

        # maximum dimension of subspace to solve for
        if maxdim_subspace != -1:
            self.maxs = maxdim_subspace
        else:
            for ii in range(10):
                if self.nev * (10 - ii) < self.ndimc:
                    self.maxs = self.nev * (10 - ii)
                    break

    def numpy_test(self, print_eigvals=True, print_eigvecs=False):
        eigvals, eigvecs = np.linalg.eig(self.mat_orig)
        eigvals = np.sort(eigvals)

        if print_eigvals:
            print("numpy results = ", np.real(eigvals[:self.nev]))

        if print_eigvecs:
            print("numpy results = ", eigvecs[:self.nev])

    def get_basis_info(self, rs_filename):
        infofile = open(rs_filename, "r")
        got_ndimc = False
        got_ndims = False
        got_nocc = False
        for line in infofile.readlines():
            if line.find("Total number of spherical GTO functions") != -1:
                print(int(line.split()[-1]))
                self.ndims = int(line.split()[-1]) * 4
                got_ndims = True
            elif line.find("Total number of cartesian GTO functions") != -1:
                self.ndimc = int(line.split()[-1]) * 4
                got_ndimc = True
            elif line.find("number electrons") != -1:
                self.nocc = int(line.split()[-1])
                got_nocc = True

        if not got_ndimc:
            print("could not extract number of Cartesian GTO functions (ndimc)")
        if not got_ndims:
            print("could not extract number of spherical GTO functions (ndims)")
        if not got_nocc:
            print("could not extract number of occupied orbitals (nocc)")

        if not (got_nocc and got_ndimc and got_ndims):
            sys.exit("Unable to extract all variables from respect output. Aborting :(")

        self.nvirt = self.ndimc * 2 - self.nocc

    def read_full_matrix(self, file_seedname="REF/TDA/full_mat"):
        print("reading in full matrix", file_seedname)
        self.mat_orig = mr.read_fortran_array(file_seedname)
        self.ndim = np.size(self.mat_orig, 0)
        print("ndim = ", self.ndim)
        print("self.mat_orig.shape = ", self.mat_orig.shape, "\n")
        np.savetxt(file_seedname+"_py", self.mat_orig, fmt='%10.5f')

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=np.float64)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def read_1e_eigvals_and_eigvecs(self, seedname):
        evals_1e_all = mr.read_fortran_array(seedname)
        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_orig.txt", evals_1e_all)

        num_pos_evals = self.nvirt + self.nocc
        print("num_pos_evals = ", num_pos_evals)
        if self.pe_rot:
            self.evals_1e = np.zeros(2*num_pos_evals, dtype=np.float64)
            self.evals_1e[:num_pos_evals] = evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = np.zeros(num_pos_evals, dtype=np.float64)
            self.evals_1e = evals_1e_all[num_pos_evals:]

        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_post.txt", self.evals_1e)
        self.eindex = np.argsort(self.evals_1e)
