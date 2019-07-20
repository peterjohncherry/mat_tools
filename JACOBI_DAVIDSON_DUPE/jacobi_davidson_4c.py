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


