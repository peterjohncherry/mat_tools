import numpy as np
import matrix_utils as mu
import sys

class eps_solver:

    def __init__(self, solver_type, threshold, maxs, preconditioning_type="Full"):
        self.solver_type = solver_type
        self.threshold = threshold
        self.maxs = maxs
        self.preconditioning_type = preconditioning_type
        print("initializing " + solver_type + " solver")

