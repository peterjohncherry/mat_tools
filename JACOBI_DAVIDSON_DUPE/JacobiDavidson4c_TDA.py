import numpy as np
import sys
import eps_solvers
import mat_reader as mr

class JacobiDavidson4C(eps_solvers.Solver):

    def tda_solver(self):
        print("into of JacobiDavidson4C::tda_solver")
        iter = 0
        skip = np.full(False)
        first = True
        print("out of JacobiDavidson4C::tda_solver")