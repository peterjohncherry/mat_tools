import numpy as np
import mat_reader as mr

class Solver:

    def __init__(self, rs_filename, num_eigenvalues, restart=False, threshold=1e-8, maxdim_subspace=-1,
                 solver="Jacobi_Davidson", method="TDA", symmetry="general", pe_rot=False):

        self.rs_filename = rs_filename
        self.num_eigenvalues = num_eigenvalues  # number of eigenvalues to solve for
        self.nev = self.num_eigenvalues
        self.threshold = threshold
        self.get_basis_info(rs_filename)

        self.threshold = threshold  # convergence threshold for solver
        self.solver = solver  # Solver type
        self.solver_type = self.solver
        self.method = method  # approximation for preconditioning

        self.pe_rot = pe_rot  # Rotations between positive and negative energy virtual orbitals
        self.restart = restart  # True if restarting from previous cycle
        self.symmetry = symmetry
        self.P0_tsymm = self.symmetry

        self.edegen = -99999999  # I DON'T KNOW
        self.t4skip = -99999999  # I DON'T KNOW

        # maximum dimension of subspace to solve for
        if maxdim_subspace != -1:
            self.maxdim_subspace = maxdim_subspace
        else:
            for ii in range(10):
                if self.num_eigenvalues * (10 - ii) < self.ndimc:
                    self.maxdim_subspace = (self.num_eigenvalues) * (10 - ii)
                    break

        self.maxs = self.maxdim_subspace

    def get_basis_info(self, rs_filename):
        infofile = open(rs_filename, "r")

        for line in infofile.readlines():
            # print(line.find("Total number of cartesian GTO functions"))
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

    def read_full_matrix(self, file_seedname = "/home/peter/RS_FILES/4C/full_mat"):
        print("reading in full matrix", file_seedname)
        self.mat_orig = mr.read_fortran_array(file_seedname)
        self.ndim = np.size(self.mat_orig,0)
        print("ndim = ", self.ndim )
        np.savetxt(file_seedname+"_py", self.mat_orig, fmt='%10.5f')





