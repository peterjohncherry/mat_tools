import numpy as np
import mat_reader as mr
import sys


class Solver:

    def __init__(self, rs_filename, num_eigenvalues, restart=False, threshold=1e-4, maxdim_subspace= 30,
                 solver="Jacobi_Davidson", method="TDA", symmetry="general", pe_rot=False):

        self.rs_filename = rs_filename
        self.nev = num_eigenvalues  # number of eigenvalues to solve for
        self.threshold = threshold  # convergence threshold for solver
        self.solver = solver  # Solver type
        self.solver_type = self.solver
        self.method = method  # approximation for preconditioning

        self.pe_rot = pe_rot  # Rotations between positive and negative energy virtual orbitals
        self.restart = restart  # True if restarting from previous cycle
        self.symmetry = symmetry
        self.P0_tsymm = self.symmetry

        # Solver specific variables, not set here
        self.submat = None
        self.teta = None
        self.t_vec = None

        self.dnorm = None
        self.skip = None
        self.nov = None

        # Guess space arrays
        self.u_vecs = None
        self.vspace = None
        self.wspace = None
        self.r_vecs = None
        self.u_hats = None

        # Ritz value arrays
        self.evals_1e = None
        self.eindex = None

        self.ndims = -1
        self.ndimc = -1
        self.nocc = -1
        self.nvirt = -1

        self.mat_orig = None
        self.ndim = None
        self.esorted = None

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

    def read_full_matrix(self, file_seedname="/home/peter/RS_FILES/4C/full_mat"):
        print("reading in full matrix", file_seedname)
        self.mat_orig = mr.read_fortran_array(file_seedname)
        self.ndim = np.size(self.mat_orig, 0)
        print("ndim = ", self.ndim)
        print("self.mat_orig.shape = ", self.mat_orig.shape, "\n")
        np.savetxt(file_seedname+"_py", self.mat_orig, fmt='%10.5f')
