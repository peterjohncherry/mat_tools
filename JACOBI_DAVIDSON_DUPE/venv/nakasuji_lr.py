import numpy as np
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils
import decimal


class JacobiDavidsonFull4C(eps_solvers.Solver):

    def __init__(self, rs_filename, num_eigenvalues, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="Full", symmetry="general", pe_rot=False):
        super().__init__(rs_filename, num_eigenvalues, threshold, maxdim_subspace, solver, method, symmetry,
                         pe_rot)

        decimal.getcontext().prec = 32
        # Guess space arrays - original right-handed guesses
        self.vspace_r = None
        self.wspace_r = None

        # Guess space arrays - pairs of right-handed guesses
        self.vspace_rp = None
        self.wspace_rp = None

        # Guess space arrays - original of left-handed guesses#
        self.vspace_l = None
        self.wspace_l = None

        # Guess space arrays - pairs of left-handed guesses
        self.vspace_lp = None
        self.wspace_lp = None

        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2 * self.nov
        self.cycle = 1

        self.read_1e_eigvals_and_eigvecs(seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        self.get_esorted_general()

    def read_1e_eigvals_and_eigvecs(self, seedname):
        evals_1e_all = mr.read_fortran_array(seedname)
        num_pos_evals = self.nvirt + self.nocc
        if self.pe_rot:
            self.evals_1e = np.zeros(2*num_pos_evals, dtype=self.real_precision)
            self.evals_1e[:num_pos_evals] = evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = np.zeros(num_pos_evals, dtype=self.real_precision)
            self.evals_1e = evals_1e_all[num_pos_evals:]

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=self.real_precision)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self, symmetry: object = 'general'):
        # Ritz values and vectors
        self.teta = np.zeros(self.nev, dtype=self.complex_precision)
        self.u_vecs = np.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.construct_guess(iev, symmetry)

        # Residual values for convergence checking
        self.r_vecs = np.zeros((self.ndim, 2*self.nev), dtype=self.complex_precision)

    def main_loop(self):
        np.set_printoptions(precision=16)
        it = 0
        while it <= 1000:
            print("\n\n=====================================================")
            print("cycle = ", self.cycle, "it = ", it)
            print("=====================================================")

            # if it > self.maxs:
            if it > 70:
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            max_cycle = 3
            if self.cycle > max_cycle:
                sys.exit("got to cycle " + str(max_cycle) + ", exit")

            for iev in range(self.nev):
                self.iev = iev



            self.cycle = self.cycle + 1

    def build_subspace_matrix(self):
        sb_dim = 0
        for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
            if vs is not None:
                sb_dim = sb_dim + vs.shape[1]
        submat = np.zeros((sb_dim, sb_dim), self.complex_precision)

        if self.vspace_r is not None:
            print("self.vspace_r.shape = ", self.vspace_r.shape)
        if self.vspace_l is not None:
            print("self.vspace_l.shape = ", self.vspace_l.shape)
        if self.vspace_rp is not None:
            print("self.vspace_rp.shape = ", self.vspace_rp.shape)
        if self.vspace_lp is not None:
            print("self.vspace_lp.shape = ", self.vspace_lp.shape)

        print("sb_dim = ", sb_dim)

        vi = 0
        for vs in [self.vspace_l, self.vspace_lp]:
            if vs is not None:
                for ii in range(vs.shape[1]):
                    wj = 0
                    for ws in [self.wspace_r, self.wspace_rp ]:
                        if ws is not None:
                            for jj in range(ws.shape[1]):
                                submat[vi+ii, wj+jj] = np.vdot(vs[:, ii], ws[:, jj])
                            wj = wj + ws.shape[1]
                vi = vi + vs.shape[1]
        return submat


    def get_residual_vectors(self, submat):
        return dnorm

    # Construct initial guess
    def construct_guess(self, iguess):
        guess = np.zeros(self.nov, dtype=self.complex_precision)
        guess[self.eindex[iguess]] = self.complex_precision(1.0 + 0.0j)

    # Construct pair according to symmetry relations
    # x :  [X,Y] --> [ Y*, X* ]
    # Ax : [X,Y] --> [ -Y*, -X* ]
    def get_pair(self, pair_type, vec_in):

    # This restructures the input vector so that it's "pair" as constructed using get_pair
    # will also be orthogonal to the vector space
    def orthonormalize_pair(self, vec):

    def get_new_tvec(self, iev):

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc + virt_orb] - self.evals_1e[occ_orb]

    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig, vec)

    @staticmethod
    def save_array_as_vectors(my_arr, name):
        for ii in range(my_arr.shape[1]):
            np.savetxt(name+"_i"+str(ii), my_arr[:, ii])
