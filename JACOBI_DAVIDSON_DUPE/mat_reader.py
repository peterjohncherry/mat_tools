import sys
import numpy as np
from scipy.io import FortranFile

def read_binary_fortran_file (name, nrows = 10 , ncols = 10, datatype = "real"):
    input_array=np.ndarray((nrows*ncols))
    if (datatype == "real"):
        fmat = FortranFile(name, 'r')
        input_array = fmat.read_reals(dtype=np.float64)
        input_array = input_array.reshape((nrows,ncols)).transpose()
        print (input_array)
        fmat.close()
    else:
        sys.exit("reading of datatype \"" +  datatype + "\" is not implemented ")
    return input_array


