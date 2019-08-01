import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils

class JacobiDavidsonFull4C(eps_solvers.Solver):

    def initialize(self):

        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2*self.nov
        print("self.ndim = ", self.ndim )

        self.eindex = np.empty(self.nov)
        for ii in range(self.nov):
            self.eindex[ii] = ii

        self.read_1e_eigvals_and_eigvecs(seedname = "/home/peter/RS_FILES/4C/1el_eigvals")

        self.u_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.construct_guess()


    def construct_guess(self):
        for iev in range(self.nev):
            self.u_vecs[:, iev] = self.tddft4_driver_guess(iev)

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

    # iguess : index of the eigenvector being built
    def tddft4_driver_guess(self, iguess):
        if self.P0_tsymm == "general":
            return self.build_general_guess_vec(iguess)
        elif self.P0_tsymm == "symmetric":
            print( "Not implemented symmetric guess")
            #return self.build_symmetric_guess_vec(iguess)

    # guess used for open-shell systems
    def build_general_guess_vec(self, iguess):
        guess = np.zeros(self.ndim, dtype=np.complex64)
        guess[self.eindex[iguess]] = 1.0+0.0j
        return guess

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self):
        #for convergence checking
        self.skip = np.full(self.nev, False)
        self.dnorm = np.zeros(self.nev, dtype=np.float32)

        #temporary variables, not really needed here
        self.t_vec = np.zeros(self.ndim, dtype=np.complex64)
        self.w_tmp = np.zeros_like(self.t_vec)
        self.v_tmp = np.zeros_like(self.t_vec)

        # Guess vectors, residuals, etc.,
        self.vspace = np.zeros((self.ndim, self.maxs), dtype=np.complex64)
        self.wspace = np.zeros_like(self.vspace)

        self.r_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.u_hats = np.zeros_like(self.r_vecs)

        # Subspace Hamiltonian
        self.submat = np.empty((2*self.maxs, 2*self.maxs), dtype=np.complex64)
        self.teta = np.zeros(self.nev, dtype=np.complex64)

    def main_loop(self):
        print("main loop NOT IMPLEMENTED")