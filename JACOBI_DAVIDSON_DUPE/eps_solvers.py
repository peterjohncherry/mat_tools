import numpy as np
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

    ####################################################################################################################
    # This is the call back routine; just using matrix multiplication as we can store full matrix
    ####################################################################################################################
    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig, vec)

    ####################################################################################################################
    # Prepare first iteration
    ####################################################################################################################
    def first_iteration_init(self):

        #initialize v1, w1, h1, theta, and residuals
        self.vspace = np.eye(self.ndim, self.nev)

        self.wspace =np.empty_like(self.vspace)
        for ii in range(self.nev):
            self.wspace[:, ii] = self.sigma_constructor(self.vspace[:,ii])

        self.h11 = np.ndarray((self.nev, self.nev))
        for ii in range(self.nev):
            for jj in range(self.nev):
                np.matmul(self.vspace[:,ii], self.wspace[:,jj])

        self.u_vec = np.empty_like(self.vspace)
        for ii in range(self.nev):
            self.u_vec[:,ii] = self.vspace[:,ii]

        self.teta = np.linalg.eigvals(self.h11)

        self.r_vec = np.ndarray((self.ndim, self.nev))
        for ii in range(self.nev):
            self.r_vec[:,ii] = self.wspace[:, ii] - self.teta[ii]*self.u_vec[:,ii]

        self.dnorm = np.ndarray(self.nev)
        self.skip  = np.ndarray(self.nev)
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
    ####################################################################################################################
    def preconditioning(self, vecinp ):

        vecout = np.ndarray(self.ndim)

        if self.preconditioning_type == "Full":
            for ii in range(int(self.ndim/2)):
                vecout[2*ii-1] = vecinp[2*ii-1]#/(self.mat_orig[ii,ii] - self.teta[ii])
                vecout[2*ii] = vecinp[2*ii]#/(self.mat_orig[ii,ii] - self.teta[ii])

        elif self.preconditioning_type == "TDA":
            for ii in range(ndim/2):
                vecout[ii] = vecinp[ii]/self.teta[ii]

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
        v1 = self.preconditioning(self.r_vec[:, iev]) # v1 = M^{-1}r
        v2 = self.preconditioning(self.u_vec[:, iev]) # v2 = M^{-1}u

        # e = (u M^{-1}r) /(u M^{-1} u)
        e = np.dot(self.u_vec[:,iev], v1)/np.dot(self.u_vec[:,iev], v2)

        return e*v2 - v1

    ####################################################################################################################
    # Main loop of eigensolver
    ####################################################################################################################
    def main_loop(self):
        iter = self.nev

        while iter < self.maxs :

            if ((iter + self.nev) > self.maxs ) :
               print ("WARNING: Maximum number of iterations was reached in eigenproblem solver ")
               sys.exit()

            for iev in range(self.nev):

                if self.skip[iev] :
                    continue

                t_vec = self.get_t(iev)

            iter = iter +1