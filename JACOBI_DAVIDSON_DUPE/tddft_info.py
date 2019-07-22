
class TddftInfo :

    def __init__( self, restart, num_eigenvalues, maxdim_subspace, threshold, solver = "Jacobi_Davidson",
                  method = "TDA", symmetry = "general", pe_rot= True):

        self.solver = solver # Solver type
        self.method = method  # approximation for preconditioning
        self.pe_rot = pe_rot  # I DON'T KNOW
        self.restart = restart # True if restarting from previous cycle
        self.num_eigenvalues = num_eigenvalues # number of eigenvalues to solve for
        self.maxdim_subspace = maxdim_subspace # maximum dimension of subspace to solve for
        self.threshold = threshold # convergence threshold for solver
        self.symmetry = symmetry
        self.edgen = -99999999 # I DON'T KNOW
        self.t4skip = -99999999 # I DON'T KNOW
