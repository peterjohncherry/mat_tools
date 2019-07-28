import numpy as np
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils
import JacobiDavidson4c_TDA as jd

class JacobiDavidson4C(eps_solvers.Solver):

    def initialize(self):
        ns2 = int(self.ndims / 2)
        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.eindex = np.empty(self.nov)
        for ii in range(self.nov):
            self.eindex[ii] = ii
        self.read_1e_eigvals_and_eigvecs()

        if self.P0_tsymm == "symmetric":
            self.get_esorted_symmetric()
        elif self.P0_tsymm == "general":
            self.get_esorted_general()
        else:
            sys.exit("have not implemented guess for symmetry " + self.P0_tsymm)

    def read_1e_eigvals_and_eigvecs(self):
        self.evals_1e_all = mr.read_fortran_array("/home/peter/RS_FILES/4C/1el_eigvals")
        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_orig.txt", self.evals_1e_all)

        num_pos_evals = self.nvirt + self.nocc
        print ("num_pos_evals = ", num_pos_evals)
        if self.pe_rot:
            self.evals_1e = np.zeros(2*num_pos_evals, dtype=np.float64)
            self.evals_1e[:num_pos_evals] = self.evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = self.evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = np.zeros(num_pos_evals, dtype=np.float64)
            self.evals_1e = self.evals_1e_all[num_pos_evals:]

        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_post.txt",self.evals_1e)
        self.eindex = np.argsort(self.evals_1e)

    def get_esorted_general(self):
    # build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc,self.nvirt), dtype=np.float64)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii,jj] = self.evalai(ii,jj)


        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    # gets energy differences
    # if symmetry is involved, then will get sorted eigvals in sets of 4
    def evalai(self, occ_orb, virt_orb):
        return  self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    def evalai(self, occ_orb, virt_orb):
        return  self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self):
        # This initialization is not really needed in python, but keep for consistency with F95 code until it is working

        # Eval and norm arrays, and vector to remember convergence
        self.teta = np.zeros(self.nev, dtype=np.float64)
        self.dnorm = np.zeros_like(self.teta)
        self.skip = np.full(self.nev, False, dtype=np.bool)

        self.nov = self.nocc * self.nvirt  # Dimension of guess vecs (E_a-E_i)^{-1}

        # Guess space arrays
        self.u_vecs = np.zeros((self.ndim, self.nev), dtype = np.complex64) # Ritz vectors
        self.vspace = np.zeros_like(self.u_vecs)                            # trial vectors
        self.wspace = np.zeros_like(self.u_vecs)                            # Hv
        self.r_vecs = np.zeros_like(self.u_vecs)                            # residual vectors
        self.u_hat = np.zeros_like(self.u_vecs)                             # Can't remember name....

        self.evecs = self.construct_guess()

        self.submat = np.zeros((self.maxs, self.maxs), dtype=np.complex64) # H represented in the space of trial vectors

    def construct_guess(self):
        for ii in range(self.nev):
            self.u_vecs[:,ii] = self.tddft4_driver_guess(ii)

    # iguess : index of the eigenvector being built
    def tddft4_driver_guess(self, iguess ):
        if self.P0_tsymm == "general":
            return self.build_general_guess_vec(iguess)
        elif self.P0_tsymm == "symmetric":
            return self.build_symmetric_guess_vec(iguess)

    # guess used for open-shell systems
    def build_general_guess_vec(self, iguess):
        guess = np.zeros(self.nov, dtype=np.complex64)
        guess[self.eindex[iguess]] = 1.0+0.0j
        return guess

    # guess used for closed-shell systems
    # NOT FINISHED, BUT ONLY TESTING OPEN-SHELL (GENERAL) ROUTINES FOR NOW
    def build_symmetric_guess_vec(self, iguess):
        znorm = (1.0+0.0j) / np.sqrt(2.0+0.0j)
        ii = iguess%4
        jj = (iguess+3)/4
        kk = 4*(jj-1) +1

        i0 = self.eindex[kk]
        i1 = self.eindex[kk+1]
        i2 = self.eindex[kk+2]
        i3 = self.eindex[kk+3]

        guess = np.zeros(self.nov, dtype=np.complex64)
        if ii == 1:
            guess[i0] = 1.0 * znorm  # ai(00)
            guess[i3] = 1.0 * znorm  # ai(11)
        elif ii == 2:
            guess[i0] =  1.0j * znorm  # ai(00)
            guess[i3] = -1.0j * znorm  # ai(11)
        elif ii == 3:
            guess[i1] = 1.0* znorm  # ai(01)
            guess[i2] = 1.0 * znorm  # ai(10)
        elif ii == 0:
            guess[i1] = 1.0j * znorm  # ai(01)
            guess[i2] = 1.0j * znorm  # ai(10)

        return guess



    def main_loop(self):

        skip = np.full(self.nev, False)
        not_converged = True
        first = True
        iter = 0
        t_vec = np.ndarray(self.nov, np.complex64)
        while not_converged and (iter-self.maxs)<0:
            iold = iter

            if (iter + self.nev > self.maxs) :
                sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

            for iev in range(self.nev) :
                iter = iter + 1
                if skip[iev]:
                    continue
                print ("iter = ", iter)
                if first :
                    self.t_vec = self.u_vecs[:,iev]
                else :
                    old_t_vec = self.t_vec
                    self.get_new_tvec(iev)
                    print("np.vdot(self.t_vec, old_t_vec) = ",np.vdot(self.t_vec, old_t_vec) )

                for ii in range(iter-1):
                    utils.orthonormalize_v_against_A_check(self.t_vec, self.vspace)


            #q =  self.new_orth(self.vspace)
            #print( "np.linalg.norm(q) = ", np.linalg.norm(q))
            for ii in range(self.vspace.shape[1]):
                print("np.vdot(self.t_vec, self.vspace[:,"+str(ii)+"]) = ", np.vdot(self.t_vec, self.vspace[:,ii]))
                #print("np.vdot(q, self.vspace[:," + str(ii) + "]) = ", np.vdot(q, self.vspace[:, ii]))

            if first:
                self.vspace[:,iev] = t_vec
            else :
                self.vspace = np.c_[self.vspace, self.t_vec]
        #    print("pre self.vspace.shape = ", self.vspace.shape)
            utils.test_orthogonality(self.vspace, name="vspace+t")

            if first:
                self.wspace[:,iev] = self.t_vec
                first = False
            else :
                self.wspace = np.c_[self.wspace, self.t_vec]


    #1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    def get_new_tvec(self, iev):

        print("iev = ", iev)
        v1 = np.ndarray(self.nov, np.complex64)  # v1 = M^{-1}*r
        v2 = np.ndarray(self.nov, np.complex64)  # v2 = M^{-1}*u
        teta_iev = self.teta[iev]

        if self.method == "TDA":
            print("into t_vec construction")
            for ii in range(self.nocc):
                for jj in range(self.nvirt):
                    ediff = (self.evalai(ii, jj) - teta_iev)
                    idx = jj+self.nvirt*ii
                    if abs(ediff) > 1e-8:
                        ediff = 1 / ediff
                        v1[idx] = self.r_vecs[idx, iev] * ediff
                        v2[idx] = self.u_vecs[idx, iev] * ediff
                    else :
                        v1[idx] = 0.0+0.0j
                        v2[idx] = 0.0+0.0j

            uMinvu = np.vdot(self.u_vecs[:, iev], v2)
            if abs(uMinvu)> 1e-8:
                factor = np.vdot(self.u_vecs[:, iev], v1) / uMinvu
                self.t_vec = factor*v2-v1
            else :
                self.t_vec = -v1
        else:
            sys.exit("Have not written FULL preconditioner yet")













    def get_esorted_symmetric():
    # Build sorted list of eigval differences with symmetry constraints imposed.
    # Blocks of 4 identical energy differences due to time reversal symmetry i.e.,
    # E_{0i}-E_{0a}) = (E_{1i}-E_{0a}) = (E_{1i}-E_{1a}) = (E_{0i}-E_{1a})
    # where 0i and 1a index an occupied Kramers pair, and 0a and 1a index a virtual Kramers pair
    # if obeys symmetry relations, can use smaller arrays, as eigvals/vecs are coupled
        self.esorted4 = np.ndarray(self.nov / 4)
        self.eindex4 = np.ndarray(self.nov / 4)

        for ii in range(self.nov / 4):
            self.eindex[ii] = ii

        ij = 0
        for ii in range(0, self.nocc, 2):
            for jj in range(0, self.nvir, 2):
                ij = ij + 1
                self.esorted4[ij] = self.esorted[(ii - 1) * self.nvir + jj]

        for ii in range(0, (self.nov / 4 - 1)):
            imin = np.argmin[self.esorted[ii:self.nov / 4]]
            if imin != 0:
                self.esorted4[[ii, imin]] = self.esorted4[[imin, ii]]
                self.eindex4[[ii, imin]] = self.eindex4[[imin, ii]]

        for ii in range(nov / 4):
            jj = (self.eindex4[ii] - 1) / (self.nvir / 2)
            imin = 2 * nvir * jj + (self.eindex[ii] - 1) % ((self.nvir / 2) + 1)
            self.eindex[4 * ii - 3] = imin
            self.eindex[4 * ii - 2] = imin + self.nvir
            self.eindex[4 * ii - 1] = imin + 1
            self.eindex[4 * ii] = imin + self.nvir + 1

