import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils

class JacobiDavidson4C(eps_solvers.Solver):
    #np.set_printoptions(precision=3)

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
        self.u_hats = np.zeros_like(self.u_vecs)                             # Can't remember name....

        self.construct_guess()

        self.submat = np.zeros((self.maxs, self.maxs), dtype=np.complex64) # H represented in the space of trial vectors

    def construct_guess(self):
        for iev in range(self.nev):
            self.u_vecs[:,iev] = self.tddft4_driver_guess(iev)

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

    def main_loop(self):

        iter = 0

        #while not_converged and (iter-self.maxs)<0:
        while iter < self.maxs:

            if (iter + self.nev > self.maxs) :
                sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

            for iev in range(self.nev):

                if iter <self.nev:
                    self.t_vec = self.u_vecs[:,iev]
                else :
                    self.get_new_tvec(iev)


                self.t_vec[abs(self.t_vec)< 1e-10] = 0.0 +0.0j

                self.t_vec, vt_angle = utils.orthonormalize_v_against_A_check(self.t_vec, self.vspace)
                if vt_angle < 1e-8 :
                    print ("Warning! Angle of t_vec with respect to vspace is small : ", vt_angle)

                self.t_vec[abs(self.t_vec)< 1e-10] = 0.0 +0.0j

                if iter < self.nev:
                    self.vspace[:,iev] = self.t_vec
                else :
                    self.vspace = np.c_[self.vspace, self.t_vec]

                utils.print_largest_component_of_vector(self.t_vec, "t")
                new_w = self.sigma_constructor()
                new_w[abs(new_w) < 1e-10] = 0.0 + 0.0j
                if iter < self.nev:
                    #self.wspace[(abs(self.wspace[:,iev]) < 1e-10), iev] = 0.0 + 0.0j # should use this, but zeros everything??
                    self.wspace[:,iev] = new_w
                else :
                    #self.wspace = np.c_[self.wspace, self.sigma_constructor()]
                    self.wspace = np.c_[self.wspace, new_w]

                iter =iter+1

            utils.zero_small_parts(self.wspace)
            utils.zero_small_parts(self.vspace)

            for iev in range(self.vspace.shape[1]):
                np.savetxt("v_"+str(iev)+".txt", self.vspace[:, iev])
                np.savetxt("w_"+str(iev)+".txt", self.wspace[:, iev])
                utils.find_nonzero_elems("v_" + str(iev), self.vspace[:, iev], 1e-10)
                utils.find_nonzero_elems("w_" + str(iev), self.wspace[:, iev], 1e-10)

            self.submat = np.matmul(self.wspace.T, self.vspace)
            utils.zero_small_parts(self.submat)
            np.savetxt("submat_" + str(iter), self.submat)
            self.teta, hdiag = la.eig(self.submat)
            #utils.zero_small_parts(self.teta)

            np.savetxt("teta_" + str(iter), self.teta)
            np.savetxt("hdiag_" + str(iter), hdiag)
            # u_{i} = h_{ij}*v_{i},            --> eigenvectors of submat represented in vspace
            # \hat{u}_{i} = hvec_{i}*w_{i},    --> eigenvectors of submat represented in wspace
            # r_{i} = \hat{u}_{i} - teta_{i}*v_{i}
            for iteta in range(self.nev):
                self.u_vecs[:, iteta] = np.matmul(self.vspace, hdiag[:, iteta])
                self.u_hats[:, iteta] = np.matmul(self.wspace, hdiag[:, iteta])
                self.r_vecs[:, iteta] = self.u_hats[:, iteta] - self.teta[iteta]*self.u_vecs[:, iteta]
                self.dnorm[iteta] = la.norm(self.r_vecs[:,iteta])

            print ("dnorm = ", self.dnorm)

        print ("self.teta = ", self.teta)
      #  print ("self.submat = ", self.submat)


    #1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    # t = x.M^{-1}.u_{k} - u_{k}
    # x = ( u*_{k}.M^{-1}r_{k} ] / ( u*_{k}.M^{-1}.u_{k} ) = e/c
    def get_new_tvec(self, iev):
        v1 = np.ndarray(self.nov, np.complex64)  # v1 = M^{-1}*r
        v2 = np.ndarray(self.nov, np.complex64)  # v2 = M^{-1}*u
        teta_iev = self.teta[iev]
        print("teta["+str(iev)+"] = ", teta_iev)

        if self.method == 'TDA':
            for ii in range(self.nocc):
                for jj in range(self.nvirt):
                    ediff = (self.evalai(ii, jj) - teta_iev)
                    idx = jj+self.nvirt*ii
                    if abs(ediff) > 1e-8:
                        ediff = 1 / ediff
                        v1[idx] = self.r_vecs[idx, iev] * ediff
                        v2[idx] = self.u_vecs[idx, iev] * ediff
                    else :
                        print("Warning, (E_{a}-E_{i})<1e-8")
                        v1[idx] = 0.0+0.0j
                        v2[idx] = 0.0+0.0j

            print("||v1|| = ", la.norm(v1))
            print("||v2|| = ", la.norm(v2))
            print("||u["+str(iev)+"] = ", la.norm(self.u_vecs[:,iev]))
            uMinvu = np.vdot(self.u_vecs[:, iev], v2)
            if abs(uMinvu)> 1e-8:
                uMinvr = np.vdot(self.u_vecs[:, iev], v1)
                factor = uMinvr / uMinvu
                print("uMinvr = ", uMinvr)
                print("factor = ", factor)
                self.t_vec = factor*v2-v1
            else :
                self.t_vec = -v1
        else:
            sys.exit("Have not written FULL preconditioner yet")

    #Extend w space by doing H*t
    def sigma_constructor(self):
        return np.matmul(self.mat_orig, self.t_vec)


        #for elem in complex_array:
        #    if abs(np.imag(elem)) < 1e-12 :
        #        elem = np.real(elem) +0.0j
        #    else :
        #        print("WARNING! imaginary component of array is :", np.imag(complex_array))

    def tv_orth_check(self):
        for ii in range(self.vspace.shape[1]):
            vtoverlap = np.vdot(self.t_vec, self.vspace[:, ii])
            if abs(vtoverlap) > 1e-10:
                print("np.vdot(self.t_vec, self.vspace[:," + str(ii) + "]) = ",
                      np.vdot(self.t_vec, self.vspace[:, ii]), end=' ')
                print("   ||t_vec[" + str(iter) + "]|| =", la.norm(self.t_vec))

    ############################ SYMMETRIZED ROUTINES FOR LATER INCORPORATION ######################################
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

