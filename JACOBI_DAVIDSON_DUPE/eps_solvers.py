import numpy as np

class eps_solver:


    def __init__(self, solver_type, threshold, maxs ) :
        self.solver_type = solver_type
        self.threshold = threshold
        self.maxs = maxs
        print ("initializing " + solver_type + " solver")

    def set_variables(self, ndim, nev, mat_orig):
        self.ndim = ndim
        self.nev  = nev
        self.mat_orig = mat_orig

    # This is the call back routine; just using matrix multiplication as we can store full matrix here
    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig, vec)

    # Prepare first iteration
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
            self.r_vec = self.wspace[:, ii] - self.teta[ii]*self.u_vec[:,ii]

        self.dnorm = np.ndarray(self.nev)
        self.skip  = np.ndarray(self.nev)
        for ii in range(self.nev):
            self.dnorm[ii] = np.sqrt(np.linalg.norm(self.r_vec))
            if self.dnorm[ii] < self.threshold :
                self.skip[ii] = True

