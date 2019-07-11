import numpy as np

class eps_solver:

    def __init__(self, solver_type, threshold, maxs ) :
        solver_type = solver_type
        threshold = threshold
        maxs = maxs
        print ("initializing " + solver_type + " solver")

    def set_variables(self, ndim, nev, teta, u_vec, mat_orig):
        ndim= ndim
        teta = teta
        u_vec = u_vec
        nev = nev
        mat_orig = mat_orig

    # This is the call back routine; just using matrix multiplication as we can store full matrix here
    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig_, vec)

    def first_iteration_init():
        vspace = np.eye(nev_, nev_)
        print ("vspace = \n", vspace)
        wspace =np.empty_like(vspace)

        for ii in range(nev):
            wspace[:, ii] = self.sigma_constructor(vspace[:,ii])
            print ("wspace = \n", wspace)
