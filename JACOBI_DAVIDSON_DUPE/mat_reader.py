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

#reads the matrix info file which is generated to aid reading of fortran binary file
def read_mat_info_file(name):

    infofile = open(name,"r")
    for line in infofile.readlines():
        line_array = line.split()
        if line_array[0] == "nrows":
            nrows = int(line_array[2])
        elif line_array[0] == "ncols":
            ncols = int(line_array[2])

    return nrows, ncols


