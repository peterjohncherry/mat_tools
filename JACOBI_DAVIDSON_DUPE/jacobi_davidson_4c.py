import numpy as np
import matrix_utils as mu
import sys
import eps_solvers

class JacobiDavidson4C(eps_solvers.Solver):

    ####################################################################################################################
    ####################################################################################################################
    def set_variables(self, preconditioning_type = "Full", ):

        if self.ndim % 2:
            sys.exit("ABORTING! Array must have even number of dimensions, " + str(ndim) + " is not even. \n")

        self.preconditioning_type = preconditioning_type

    ####################################################################################################################
    ####################################################################################################################

    def initialize_tda(self):
        self.skip = np.full(self.nev, False, dtype=np.bool)
        self.dnorm  =np.zeros(self.nev,dtype= np.float64)
        self.teta = np.zeros(self.nev, dtype=np.float64)
        self.r_vec = np.zeros((self.ndim, self.nev), dtype= np.complex64)
        self.u_hat = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.t_vec = np.zeros(self.ndim, dtype=np.complex64)
        self.vspace = np.zeros((self.ndim, self.maxs),dtype= np.complex64)
        self.wspace = np.zeros((self.ndim, self.maxs),dtype= np.complex64)
        self.submat = np.zeros((self.maxs, self.maxs),dtype= np.complex64)

        self.iter = 0
        self.first = True # For the first iteration?

    def main_loop(self):
        iold = self.iter

        if (self.iter + self.nev > self.maxs) :
            print("WARNING in EPS solver: Maximum number of iteration reached")
            sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

        iev =1
        while iev >self.nev :
            iev = iev + 1
            if self.skip[iev] :
                continue

            self.get_new_t_vec()
            if self.first :
                print("set t_vec to u_vec guess")
                #t_vec = self.u_vec[:,iev]



    #1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    def get_new_tvec(self):
        print ("not_done")







