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

