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
        self.ndim = 2*(self.nocc*self.nvirt)
        print("self.ndim = ", self.ndim )

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
        self.submat= np.empty((2*self.maxs, 2*self.maxs), dtype=np.complex64)
        self.teta = np.zeros(self.nev, dtype=np.complex64)

    def main_loop(self):
        print("nothing doing....")