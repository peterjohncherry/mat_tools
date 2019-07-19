import numpy as np
import mat_reader as mr

class Solver:

    def __init__(self, solver_type, nev, threshold, maxs):
        self.solver_type = solver_type
        self.nev = nev
        self.threshold = threshold
        self.maxs = maxs

    def read_full_matrix(self, file_seedname = "/home/peter/RS_FILES/4C/full_mat"):
        print("reading in full matrix", file_seedname)
        self.mat_orig = mr.read_fortran_array(file_seedname)
        self.ndim = np.size(self.mat_orig,0)
        print("ndim = ", self.ndim )
        np.savetxt(file_seedname+"_py", self.mat_orig, fmt='%10.5f')





