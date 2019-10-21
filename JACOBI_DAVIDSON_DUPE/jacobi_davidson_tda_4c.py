import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils


class JacobiDavidsonTDA4C(eps_solvers.Solver):
    np.set_printoptions(precision=4)

    def __init__(self, rs_filename, num_eigenvalues, restart=False, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="TDA", symmetry="general", pe_rot=False):
        super().__init__(rs_filename, num_eigenvalues, restart, threshold, maxdim_subspace, solver, method, symmetry,
                         pe_rot)
        # Guess space arrays
        self.vspace = None
        self.wspace = None

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()


    # gets energy differences
    # if symmetry is involved, then will get sorted eigvals in sets of 4
    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    def initialize_first_iteration(self):
        # This initialization is not really needed in python, but keep for consistency with F95 code until it is working
        # Eval and norm arrays, and vector to remember convergence

        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.eindex = np.arange(self.nov)

        self.read_1e_eigvals_and_eigvecs("/home/peter/RS_FILES/4C/KEEPERS_TDA/1el_eigvals")
        self.get_esorted_general()

        self.set_arrays()
        self.construct_guess()

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=np.float64)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def set_arrays(self):
        self.teta = np.zeros(self.nev, dtype=np.float64)
        self.dnorm = np.zeros_like(self.teta)
        self.skip = np.full(self.nev, False, dtype=np.bool)
        self.nov = self.nocc * self.nvirt  # Dimension of guess vectors, one for each possible (E_a-E_i)^{-1}

        # Guess space arrays
        self.u_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)   # Ritz vectors
        self.vspace = np.zeros_like(self.u_vecs)                            # trial vectors
        self.wspace = np.zeros_like(self.u_vecs)                            # H*v
        self.r_vecs = np.zeros_like(self.u_vecs)                            # residual vectors
        self.u_hats = np.zeros_like(self.u_vecs)                            # H*u
        self.submat = np.zeros((self.maxs, self.maxs), dtype=np.complex64)  # H represented in trial vector space

    def construct_guess(self):
        for iev in range(self.nev):
            self.u_vecs[:, iev] = self.tddft4_driver_guess(iev)

    # iguess : index of the eigenvector being built
    def tddft4_driver_guess(self, iev):
        guess = np.zeros(self.ndim, dtype=np.complex64)
        guess[self.eindex[iev]] = 1.0+0.0j
        return guess

    def main_loop(self):

        it = 0
        skip = np.full(self.maxs, False)
        # While not_converged and (it-self.maxs)<0:
        self.maxs = 50
        while it < self.maxs:

            if (it + self.nev) >= self.maxs:
                sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

            for iev in range(self.nev):

                if it < self.nev:
                    self.t_vec = self.u_vecs[:, iev]
                else:
                    self.get_new_tvec(iev)

                self.t_vec, vt_angle = utils.orthonormalize_v_against_mat_check(self.t_vec, self.vspace)
                if vt_angle < 1e-8:
                    print("Warning! Angle of t_vec with respect to vspace is small : ", vt_angle)

                if it < self.nev:
                    self.vspace[:, iev] = self.t_vec
                else:
                    self.vspace = np.c_[self.vspace, self.t_vec]

                new_w = self.sigma_constructor()
                if it < self.nev:
                    self.wspace[:, iev] = new_w
                else:
                    self.wspace = np.c_[self.wspace, new_w]

                it = it+1

            self.submat = np.matmul(np.conjugate(self.wspace.T), self.vspace)
            ritz_vals, hdiag = la.eig(self.submat)

            # sort eigenvalues and eigenvectors
            ev_idxs = ritz_vals.argsort()
            ritz_vals = ritz_vals[ev_idxs]
            hdiag = hdiag[:, ev_idxs]

            # u_{i} = h_{ij}*v_{i},            --> eigenvectors of submat represented in vspace
            # \hat{u}_{i} = h_{ij}*w_{i},    --> eigenvectors of submat represented in wspace
            # r_{i} = \hat{u}_{i} - teta_{i}*v_{i}
            for iteta in range(self.u_vecs.shape[1]):
                self.u_vecs[:, iteta] = np.matmul(self.vspace, hdiag[:, iteta])
                self.u_hats[:, iteta] = np.matmul(self.wspace, hdiag[:, iteta])
                self.r_vecs[:, iteta] = self.u_hats[:, iteta] - ritz_vals[iteta] * self.u_vecs[:, iteta]
                self.dnorm[iteta] = la.norm(self.r_vecs[:, iteta])

            self.teta = ritz_vals
            for ii in range(self.nev):
                if self.dnorm[ii] <= self.threshold:
                    skip[ii] = True
                else:
                    skip[ii] = False

            print("self.teta = ", self.teta)
            if False in skip[:self.nev]:
                print("Not converged on iteration ", it)
                print("self.dnorm = ", self.dnorm, "skip = ", skip)
            else:
                print("Final eigenvalues = ", np.real(self.teta[:self.nev]))
                sys.exit("Converged!!")

    # 1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    # t = x.M^{-1}.u_{k} - u_{k}
    # x = ( u*_{k}.M^{-1}r_{k} ] / ( u*_{k}.M^{-1}.u_{k} ) = e/c
    def get_new_tvec(self, iev):
        v1 = np.ndarray(self.nov, np.complex64)  # v1 = M^{-1}*r
        v2 = np.ndarray(self.nov, np.complex64)  # v2 = M^{-1}*u
        teta_iev = self.teta[iev]

        idx = 0
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = (self.evalai(ii, jj) - teta_iev)
                if abs(ediff) > 1e-8:
                    ediff = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * ediff
                    v2[idx] = self.u_vecs[idx, iev] * ediff
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = 0.0+0.0j
                    v2[idx] = 0.0+0.0j
                idx += 1

        u_m1_u = np.vdot(self.u_vecs[:, iev], v2)
        if abs(u_m1_u) > 1e-8:
            u_m1_r = np.vdot(self.u_vecs[:, iev], v1)
            factor = u_m1_r / u_m1_u
            # print("uMinvr = ", u_m1_r, "uMinvu = ", u_m1_u, "factor = ", factor)
            self.t_vec = factor*v2-v1
        else:
            self.t_vec = -v1

    # Extend w space by doing H*t
    def sigma_constructor(self):
        return np.matmul(self.mat_orig, self.t_vec)

    def print_guess_arrays(self):
        utils.zero_small_parts(self.u_hats)
        utils.zero_small_parts(self.u_vecs)
        utils.zero_small_parts(self.r_vecs)
        for iev in range(self.u_vecs.shape[1]):
            np.savetxt("u_" + str(iev) + ".txt", self.u_vecs[:, iev])
            np.savetxt("U_" + str(iev) + ".txt", self.u_hats[:, iev])
            np.savetxt("r_" + str(iev) + ".txt", self.r_vecs[:, iev])