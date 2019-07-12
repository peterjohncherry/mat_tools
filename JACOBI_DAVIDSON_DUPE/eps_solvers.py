import numpy as np
import matrix_utils as mu
import sys

class eps_solver:

    ####################################################################################################################
    ####################################################################################################################
    def __init__(self, solver_type, threshold, maxs, preconditioning_type = "Full" ) :
        self.solver_type = solver_type
        self.threshold = threshold
        self.maxs = maxs
        self.preconditioning_type = preconditioning_type
        print ("initializing " + solver_type + " solver")

    ####################################################################################################################
    ####################################################################################################################
    def set_variables(self, ndim, nev, mat_orig):
        if ndim % 2 :
            sys.exit("ABORTING! Array must have even number of dimensions, "+ str(ndim)+ " is not even. \n")
        self.ndim = ndim
        self.nev  = nev
        self.mat_orig = mat_orig
        self.npevals, self.npevecs = np.linalg.eig(mat_orig)
        mu.sort_eigvecs_and_vals(self.npevals, self.npevecs)


    ####################################################################################################################
    # This is the call back routine; just using matrix multiplication as we can store full matrix
    ####################################################################################################################
    def sigma_constructor(self, vec):
        tmp = np.matmul(self.mat_orig, vec)
        return tmp/np.linalg.norm(tmp)

    ####################################################################################################################
    # Prepare first iteration
    ####################################################################################################################
    def first_iteration_init(self):

        #initialize v1, w1, h1, theta, and residuals
        self.vspace = np.eye(self.ndim, self.nev, dtype=complex)

        self.wspace =np.empty_like(self.vspace, dtype=complex)
        for ii in range(self.nev):
            self.wspace[:, ii] = self.sigma_constructor(self.vspace[:,ii])


        self.h11 = np.ndarray((self.nev, self.nev), dtype=complex)
        for ii in range(self.nev):
            for jj in range(self.nev):
                self.h11[ii,jj] = np.matmul(self.vspace[:,ii], self.wspace[:,jj])


        self.u_vec = np.empty_like(self.vspace, dtype=complex)
        for ii in range(self.nev):
            self.u_vec[:,ii] = self.vspace[:,ii]

        self.teta = np.linalg.eigvals(self.h11)
        print("self.teta = ", self.teta)
        self.teta.sort()
        print("self.teta sorted = ", self.teta)

        self.r_vec = np.ndarray((self.ndim, self.nev), dtype=complex)
        for ii in range(self.nev):
            self.r_vec[:,ii] = self.wspace[:, ii] - self.teta[ii]*self.u_vec[:,ii]

        self.dnorm = np.ndarray(self.nev)
        self.skip  = np.full(self.nev, False, dtype=bool)
        for ii in range(self.nev):
            self.dnorm[ii] = np.sqrt(np.linalg.norm(self.r_vec))
            if self.dnorm[ii] < self.threshold :
                self.skip[ii] = True

    ####################################################################################################################
    # gets list of energy differences
    # This probably isn't applicable here, as I don't see how I can use it
    ####################################################################################################################
    def eval_ai(self, eval):
        self.esorted = np.ndarray(nocc*nvir)
        for ii in range(nocc):
            for jj in range(nvir):
                self.esorted = eval[jj]- eval[ii]
        esorted.sort()

    ####################################################################################################################
    # Use preconditioned matrix to get guess vector
    # In "Full", eval_ai is replaced with diagonal elements of matrix... don't know if this is sensible
    # Necessary for now as matrix lacks the appropriate pair structure
    ####################################################################################################################
    def preconditioning(self, vecinp, iev ):

        vecout = np.ndarray(self.ndim, dtype=complex)

        if self.preconditioning_type == "Full":
            #for ii in range(int(self.ndim/2)):
            #    vecout[2*ii-1] = vecinp[2*ii-1]/(self.mat_orig[ii,ii] - self.teta[iev])
            #    vecout[2*ii] = vecinp[2*ii]/(self.mat_orig[ii,ii] - self.teta[iev])
            for ii in range(self.ndim):
                vecout[ii] = vecinp[ii]/(self.mat_orig[ii,ii] - self.teta[iev])

        elif self.preconditioning_type == "Diag":
            for ii in range(self.ndim):
                vecout[ii] = vecinp[ii]/self.teta[iev]

        else :
            sys.exit("ABORTING! Preconditioning type " + self.preconditioning_type + " is unknown \n")

        return vecout

    ####################################################################################################################
    # Calculates t vector for expansion of guess space
    # if approx = "Exact" --> e = (u M^{-1}r) /(u M^{-1} u)
    # if approx = "basic" --> e =

    ####################################################################################################################
    def get_epsilon(self, approx = "basic"):
        if approx == "basic" :
            e = self__ddot(n, u, v1)
            e = e / self__ddot(n, u, v2)
        else :
            sys.exit("I haven't programmed any other approximations to epsilon other than \"approx\".... ")

    ####################################################################################################################
    # Calculates t vector for expansion of guess space
    # t = e*M^{-1}u -M^{-1}r
    # e = (u M ^ {-1}r) / (u M ^ {-1} u)
    ####################################################################################################################
    def get_t(self, iev):
        e = 0.0
        v1 = self.preconditioning(self.r_vec[:, iev], iev) # v1 = M^{-1}r
        v2 = self.preconditioning(self.u_vec[:, iev], iev) # v2 = M^{-1}u

        # e = (u M^{-1}r) /(u M^{-1} u)
        e = np.dot(self.u_vec[:,iev], v1)/np.dot(self.u_vec[:,iev], v2)

        return e*v2 - v1

    ####################################################################################################################
    # Main loop of eigensolver
    ####################################################################################################################
    def main_loop(self):
        iter = self.nev

        while iter < self.maxs :
            print("iter = ", iter)
            if ((iter + self.nev) > self.maxs ) :
               print ("WARNING: Maximum number of iterations was reached in eigenproblem solver ")
               mu.print_only_large_imag(self.teta, "teta")
               print("npevals = ", self.npevals)
               sys.exit("(iter + nev )= (" +str(iter)+ "+" +str(self.nev) +") > " + str(self.maxs) )

            for iev in range(self.nev):
                #Skip calculation of eigenvalue if already converged
                if self.skip[iev]:
                   # print("eigval "+ str(iev)+ " already converged, skipping")
                    continue

                #Calculate t, extend v_space and w_space
                t_vec = self.get_t(iev)


                t_vec = mu.orthonormalize_v_against_A(t_vec, self.vspace)
                self.vspace = np.c_[self.vspace, t_vec]
                mu.test_orthogonality(self.vspace, name ="vspace+t")


                self.wspace = np.c_[self.wspace, self.sigma_constructor(t_vec)]
                self.check_mat_norms()

                iter = iter +1

            tmp = np.ndarray((iter,iter), dtype=complex)
            for ii in range(iter):
                for jj in range(iter):
                    tmp[ii,jj] = np.dot(self.vspace[:, ii], self.wspace[:, jj])

            # get Ritz Values
            self.teta, hdiag = np.linalg.eig(tmp)
          #  mu.sort_eigvecs_and_vals(self.teta, hdiag)
            # get new ritz vecs (u_vec)
            # get u_hat = A*u_vec , get residual r_vec = u_hat- teta*u_vec , calculate ||r||
            u_hat = np.zeros_like(self.u_vec, dtype =complex)
            self.u_vec.fill(0.0)
            for iev in range(self.nev):
                for jj in range(iter):
                    self.u_vec[:,iev] = self.u_vec[:,iev] + hdiag[jj, iev]*self.vspace[:,jj]
                    u_hat[:,iev] = u_hat[:,iev] + hdiag[jj, iev]*self.wspace[:, jj]

                self.r_vec[:,iev]= u_hat[:,iev] - self.teta[iev]*self.u_vec[:,iev]

                self.dnorm[iev] = np.linalg.norm(self.r_vec[:, iev])
                self.skip[iev]= (self.dnorm[iev] < self.threshold)

           # mu.print_only_large_imag(self.teta[:self.nev], " teta")
            self.teta.sort()
            mu.print_only_large_imag(self.teta, " teta")
            for ii in range(self.nev):
                if self.skip[ii] :
                    print ("Converged self.teta[",ii,"] = ", self.teta[ii])
            if self.skip[:self.nev].min() :

                print ("Converged!!")
                return

            #u_hat = np.zeros_like(self.u_vec)

    def check_mat_norms(self):
        if np.linalg.norm(self.vspace) < 1:
            sys.exit("vspace norm is tiny !! = " + str(np.linalg.norm(self.vspace)) + "ABORTING!")

        if np.linalg.norm(self.wspace) < 1:
            sys.exit("wspace norm is tiny !! = " + str(np.linalg.norm(self.wspace)) + "ABORTING!")

        if np.linalg.norm(self.h11) < 1:
            sys.exit("h11 norm is tiny !! = " + str(np.linalg.norm(self.h11)) + "ABORTING!")
