import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils

class JacobiDavidsonFull4C(eps_solvers.Solver):

    def initialize(self):

        print ("self.maxs = ", self.maxs)
        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2*self.nov
        print("self.ndim = ", self.ndim )

        self.read_1e_eigvals_and_eigvecs(seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        self.get_esorted_general()

        self.u_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        #self.construct_guess()

        for iev in range(self.nev):
            self.u_vecs[self.eindex[iev], iev] = 1.0+ 0.0j


    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=np.float64)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    def construct_guess(self):
        if self.nov < self.ndim :
            for iev in range(self.nev):
                self.u_vecs[self.nov:, iev] = 0.0 + 0.0j

        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.tddft4_driver_guess(iev)

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

    # iguess : index of the eigenvector being built
    def tddft4_driver_guess(self, iguess):
        if self.P0_tsymm == "general":
            return self.build_general_guess_vec(iguess)
        elif self.P0_tsymm == "symmetric":
            print( "Not implemented symmetric guess")
            #return self.build_symmetric_guess_vec(iguess)

    # guess used for open-shell systems
    def build_general_guess_vec(self, iguess):
        guess = np.zeros(self.nov, dtype=np.complex64)
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

        it = 0
        while it < self.maxs:

            if (it + 2*self.nev) > self.maxs :
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            for iev in range(self.nev):

                if self.skip[iev]:
                    continue

                if it < 2*self.nev :
                    self.t_vec = self.u_vecs[:,iev]
                else :
                    self.get_t_vec()

                for ii in range(it-1):
                    v_tmp = self.get_pair('x', ii)
                    self.t_vec, vtangle1 = utils.orthonormalize_v_against_mat_check(self.t_vec, self.vspace)
                    self.t_vec, vtangle2 = utils.orthonormalize_v1_against_v2(self.t_vec, v_tmp)
                    print("vtangle1 = ", vtangle1)
                    print("vtangle2 = ", vtangle2)
                    if max(vtangle1, vtangle2) > 1e-10:
                        print("angle between new guess vector and current guess space is small!",
                              max(vtangle1, vtangle2))
                        continue

            it += 1

    def get_t_vec(self):
        print("get_t_vec NOT IMPLEMENTED!!")

    def get_pair(self, pair_type, iev):

        vec_out = np.empty(self.ndim, dtype=np.complex64)
        n2 = int(self.ndim/2)
        if pair_type == 'x':
            vec_out[n2:] = np.conj(self.vspace[:n2, iev])
            vec_out[:n2] = np.conj(self.vspace[n2:, iev])

        elif pair_type == 'Ax':
            vec_out[n2:] = -np.conj(self.vspace[:n2, iev])
            vec_out[:n2] = -np.conj(self.vspace[n2:, iev])

        else:
            sys.exit("ABORTING!! Unknown pair_type specified in eigenvector construction")

        return vec_out