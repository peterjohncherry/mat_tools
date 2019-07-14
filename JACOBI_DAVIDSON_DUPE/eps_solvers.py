import numpy as np
import matrix_utils as mu
import jacobi_davidson as jd
import sys

class eps_solver:

    def __init__(self, solver_type, threshold, maxs, solver_name = "jd"):
        self.solver_type = solver_type
        self.threshold = threshold
        self.maxs = maxs

        if (solver_name == "jd"):
            return jd.jacobi_davidson( self )




