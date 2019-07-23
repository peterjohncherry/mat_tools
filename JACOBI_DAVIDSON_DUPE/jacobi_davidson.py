import numpy as np
import matrix_utils as mu
import sys
import eps_solvers

class JacobiDavidson(eps_solvers.Solver):

    ####################################################################################################################
    ####################################################################################################################
    def set_variables(self, preconditioning_type = "Full"):
        if self.ndim % 2:
            sys.exit("ABORTING! Array must have even number of dimensions, " + str(ndim) + " is not even. \n")
        self.preconditioning_type = preconditioning_type

    ####################################################################################################################
    # This is the call back routine; just using matrix multiplication as we can store full matrix
    ####################################################################################################################
    def sigma_constructor(self, vec):
        tmp = np.matmul(self.mat_orig, vec)
        return tmp / np.linalg.norm(tmp)

    ####################################################################################################################
    # Prepare first iteration
    ####################################################################################################################
    def first_iteration_init(self):
        # initialize v1, w1, h1, theta, and residuals
        self.vspace = np.eye(self.ndim, self.nev, dtype=complex)
        print ("vspace = ", self.vspace)

        self.wspace = np.empty_like(self.vspace, dtype=complex)
        for ii in range(self.nev):
            self.wspace[:, ii] = self.sigma_constructor(self.vspace[:, ii])

        self.h11 = np.ndarray((self.nev, self.nev), dtype=complex)
        for ii in range(self.nev):
            for jj in range(self.nev):
                self.h11[ii, jj] = np.matmul(self.vspace[:, ii], self.wspace[:, jj])

        self.teta, self.v_vec = np.linalg.eigh(self.h11)
        self.teta, self.v_vec = mu.sort_eigvecs_and_vals(self.teta, self.v_vec)

        self.u_vec = np.empty_like(self.vspace, dtype=complex)
        for ii in range(self.nev):
            self.u_vec[:, ii] = self.vspace[:, ii]

        self.r_vec = np.ndarray((self.ndim, self.nev), dtype=complex)
        for ii in range(self.nev):
            self.r_vec[:, ii] = self.wspace[:, ii] - self.teta[ii] * self.u_vec[:, ii]

        self.dnorm = np.ndarray(self.nev)
        self.skip = np.full(self.nev, False, dtype=bool)
        for ii in range(self.nev):
            self.dnorm[ii] = np.sqrt(np.linalg.norm(self.r_vec))
            if self.dnorm[ii] < self.threshold:
                self.skip[ii] = True

    ####################################################################################################################
    # gets list of energy differences
    # This probably isn't applicable here, as I don't see how I can use it
    ####################################################################################################################
    def eval_ai(self, eval):
        self.esorted = np.ndarray(nocc * nvir)
        for ii in range(nocc):
            for jj in range(nvir):
                self.esorted = eval[jj] - eval[ii]
        esorted.sort()

    ####################################################################################################################
    # Use preconditioned matrix to get guess vector
    # In "Full", eval_ai is replaced with diagonal elements of matrix... don't know if this is sensible
    # Necessary for now as matrix lacks the appropriate pair structure
    ####################################################################################################################
    def preconditioning(self, vecinp, iev):

        vecout = np.ndarray(self.ndim, dtype=complex)

        if self.preconditioning_type == "Full":
            # for ii in range(int(self.ndim/2)):
            #    vecout[2*ii-1] = vecinp[2*ii-1]/(self.mat_orig[ii,ii] - self.teta[iev])
            #    vecout[2*ii] = vecinp[2*ii]/(self.mat_orig[ii,ii] - self.teta[iev])
            for ii in range(self.ndim):
                vecout[ii] = vecinp[ii] / (self.teta[iev]- self.mat_orig[ii, ii])

        elif self.preconditioning_type == "Diag":
            for ii in range(self.ndim):
                vecout[ii] = vecinp[ii] / self.teta[iev]

        else:
            sys.exit("ABORTING! Preconditioning type " + self.preconditioning_type + " is unknown \n")

        return vecout

    ####################################################################################################################
    # Calculates t vector for expansion of guess space
    # if approx = "Exact" --> e = (u M^{-1}r) /(u M^{-1} u)
    # if approx = "basic" --> e =
    ####################################################################################################################
    def get_epsilon(self, approx="basic"):
        if approx == "basic":
            e = self__ddot(n, u, v1)
            e = e / self__ddot(n, u, v2)
        else:
            sys.exit("I haven't programmed any other approximations to epsilon other than \"approx\".... ")

    ####################################################################################################################
    # Calculates t vector for expansion of guess space
    # t = e*M^{-1}u -M^{-1}r
    # e = (u M ^ {-1}r) / (u M ^ {-1} u)
    ####################################################################################################################
    def get_t(self, iev):
        e = 0.0
        v1 = self.preconditioning(self.r_vec[:, iev], iev)  # v1 = M^{-1}r
        v2 = self.preconditioning(self.u_vec[:, iev], iev)  # v2 = M^{-1}u

        # e = (u M^{-1}r) /(u M^{-1} u)
        e = np.dot(self.u_vec[:, iev], v1) / np.dot(self.u_vec[:, iev], v2)

        return e * v2 - v1

    ####################################################################################################################
    # Main loop of eigensolver
    ####################################################################################################################
    def solve(self):
        iter = self.nev

        # build up initial guess vectors
        for jj in range(0, self.nev):
            self.v_vec[:, jj] = self.v_vec[:, jj] / np.linalg.norm(self.v_vec[:, jj])

        # Entering inner loop
        while iter < self.maxs:
            print("iter = ", iter)
            if ((iter + self.nev) > self.maxs):
                convergence_failure(iter)

            for iev in range(self.nev):
                if self.skip[iev]:
                    continue

                # Calculate t, extend v_space and w_space
                t_vec = self.get_t(iev)
                t_vec, relative_shrinkage = mu.orthonormalize_v_against_A_check(t_vec, self.vspace)
                print ("relative_shrinkage = ", relative_shrinkage)

                if relative_shrinkage < 1e-2 :
                    print("Skipping v due to shrinkage")
                    continue

                #adding new t_vec to vspace
                self.vspace = np.c_[self.vspace, t_vec]
                mu.test_orthogonality(self.vspace, name="vspace+t")
                self.wspace = np.c_[self.wspace, self.sigma_constructor(t_vec)]

                iter = iter + 1

            tmp = np.ndarray((iter, iter), dtype=complex)
            for ii in range(iter):
                for jj in range(iter):
                    tmp[ii, jj] = np.dot(self.vspace[:, ii], self.wspace[:, jj])

            # get Ritz Values
            self.teta, hdiag = np.linalg.eig(tmp)

            # get new ritz vecs (u_vec)
            # get u_hat = A*u_vec , get residual r_vec = u_hat- teta*u_vec , calculate ||r||
            u_hat = np.zeros_like(self.u_vec, dtype=complex)
            self.u_vec.fill(0.0)
            for iev in range(self.nev):
                for jj in range(iter):
                    self.u_vec[:, iev] = self.u_vec[:, iev] + hdiag[jj, iev] * self.vspace[:, jj]
                    u_hat[:, iev] = u_hat[:, iev] + hdiag[jj, iev] * self.wspace[:, jj]

                self.r_vec[:, iev] = u_hat[:, iev] - self.teta[iev] * self.u_vec[:, iev]

                self.dnorm[iev] = np.linalg.norm(self.r_vec[:, iev])
                self.skip[iev] = (self.dnorm[iev] < self.threshold)

            # mu.print_only_large_imag(self.teta[:self.nev], " teta")
            self.teta.sort()
            mu.print_only_large_imag(self.teta, " teta")
            for ii in range(self.nev):
                if self.skip[ii]:
                    print("Converged self.teta[", ii, "] = ", self.teta[ii])
            if self.skip[:self.nev].min():
                print("Converged!!")
                return

    def check_mat_norms(self):
        if np.linalg.norm(self.vspace) < 1:
            sys.exit("vspace norm is tiny !! = " + str(np.linalg.norm(self.vspace)) + "ABORTING!")

        if np.linalg.norm(self.wspace) < 1:
            sys.exit("wspace norm is tiny !! = " + str(np.linalg.norm(self.wspace)) + "ABORTING!")

        if np.linalg.norm(self.h11) < 1:
            sys.exit("h11 norm is tiny !! = " + str(np.linalg.norm(self.h11)) + "ABORTING!")

    def convergence_failure(self, iter, print_evals, print_evecs):
        print("WARNING: Maximum number of iterations was reached in eigenproblem solver ")
        if print_evals :
            mu.print_only_large_imag(self.teta, "teta")
            print("npevals = ", self.npevals)
        if print_evecs :
            print("npevecs = ", self.npevecs)
        sys.exit("(iter + nev )= (" + str(iter) + "+" + str(self.nev) + ") > " + str(self.maxs))