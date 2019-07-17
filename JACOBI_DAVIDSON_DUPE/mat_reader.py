import sys
import numpy as np
from scipy.io import FortranFile

def read_binary_fortran_file (name, nrows, ncols, datatype):
    input_array=np.ndarray((nrows*ncols))
    if (datatype == "real"):
        fmat = FortranFile(name, 'r')
        input_array = fmat.read_reals(dtype=np.float64)
        input_array = input_array.reshape((nrows,ncols)).transpose()
        print (input_array)
        fmat.close()

    elif (datatype == "int"):
        fmat = FortranFile(name, 'r')
        input_array = fmat.read_ints(dtype=np.int32)
        input_array = input_array.reshape((nrows,ncols)).transpose()
        print (input_array)
        fmat.close()

    elif ( datatype == "complex" ):
        sys.exit("reading of complex matrices is not directly possible, must write out real and imag parts as two seperate real matrices\n")

    else :
        sys.exit("reading of datatype \"" +  datatype + "\" is not implemented\n ")

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
        elif line_array[0] == "datatype":
            datatype = str(line_array[2])

    return nrows, ncols, datatype

def read_fortran_matrix(seedname):
    nrows, ncols, datatype = read_mat_info_file("/home/peter/RS_FILES/"+seedname+".info")

    if datatype == "complex":
        name ="/home/peter/RS_FILES/"+seedname+"_real.bin"
        mat_real =  read_binary_fortran_file(name, nrows, ncols, datatype="real")
        name = "/home/peter/RS_FILES/" + seedname+"_imag.bin"
        mat_imag = read_binary_fortran_file(name, nrows, ncols, datatype="real")
        return mat_real + 1j*mat_imag

    elif datatype == "real":
        name ="/home/peter/RS_FILES/"+seedname+".bin"
        mat_real =  read_binary_fortran_file(name, nrows, ncols, datatype="real")
        return mat_real

    elif datatype == "int":
        name ="/home/peter/RS_FILES/"+seedname+".bin"
        mat_int =  read_binary_fortran_file(name, nrows, ncols, datatype="int")
        return mat_int