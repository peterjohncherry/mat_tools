import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils

class JacobiDavidsonFull4C(eps_solvers.Solver):

    def initialize(self):

        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndims = 2*(self.nocc*self.nvirt)
        print("self.ndims = ", self.ndims )
