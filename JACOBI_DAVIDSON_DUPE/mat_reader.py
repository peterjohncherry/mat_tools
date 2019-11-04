# python libraries
import glob

# external libraries
import sys
import numpy as np
from scipy.io import FortranFile


def read_binary_fortran_file(name, datatype, dim0, dim1=1, real_precision=np.float64):

    if dim1 == 1:
        fmat = FortranFile(name, 'r')
        input_array = fmat.read_reals(dtype=real_precision)
        fmat.close()

    else:
        if datatype == "real":
            fmat = FortranFile(name, 'r')
            input_array = fmat.read_reals(dtype=real_precision)
            input_array = input_array.reshape((dim0, dim1)).transpose()
            fmat.close()

        elif datatype == "int":
            fmat = FortranFile(name, 'r')
            input_array = fmat.read_ints(dtype=np.int32)
            input_array = input_array.reshape((dim0, dim1)).transpose()
            # fmat.close()

        elif datatype == "complex":
            sys.exit("reading of complex matrices is not directly possible, must write out real and imag parts "
                     "as two separate real matrices\n")

        else:
            sys.exit("reading of datatype \"" + datatype + "\" is not implemented\n ")

    return input_array


# Reads the matrix info file which is generated to aid reading of fortran binary file
def read_array_info_file(name):

    infofile = open(name, "r")
    dim0 = -1
    dim1 = -1
    datatype = "unread"
    for line in infofile.readlines():
        line_array = line.split()
        if line_array[0] == "dim0":
            dim0 = int(line_array[2])
        elif line_array[0] == "dim1":
            dim1 = int(line_array[2])
        elif line_array[0] == "datatype":
            datatype = str(line_array[2])

    if dim1 != -1 and dim0 != -1 and datatype != "unread":
        return dim0, dim1, datatype
    else:
        sys.exit("error reading .info file")


def read_fortran_array(seedname):
    nrows, ncols, datatype = read_array_info_file(seedname+".info")

    if datatype == "complex":
        name = seedname + "_real.bin"
        array_real = read_binary_fortran_file(name, "real", nrows, ncols)
        name = seedname + "_imag.bin"
        array_imag = read_binary_fortran_file(name, "real", nrows, ncols)
        return array_real + 1j*array_imag

    elif datatype == "real":
        name = seedname + ".bin"
        array_real = read_binary_fortran_file(name, datatype, nrows, ncols)
        return array_real

    elif datatype == "int":
        name = seedname + ".bin"
        array_ints = read_binary_fortran_file(name, datatype, nrows, ncols)
        return array_ints

    else:
        sys.exit("Not implemented " + datatype + " ABORTING")


def read_numpy_array(name):
    return np.load(name)


def get_seedname_list(base_name):
    seedname_list = []
    for infile in sorted(glob.glob(base_name + '*')):
        if infile.endswith('.info'):
            seedname = infile[:-5]
            seedname_list.append(seedname)
    return seedname_list


def read_array_sequence(basename, save_as_numpy_file=True, save_as_text_file=False, get_ndarray_list=False):
    # dir_path = "/home/peter/RS_FILES/"
    seedname_list = get_seedname_list(basename)

    if save_as_numpy_file:
        for seedname in seedname_list:
            np.save(seedname, read_fortran_array(seedname))

    if save_as_text_file:
        for seedname in seedname_list:
            np.savetxt(seedname + ".txt", read_fortran_array(seedname))

    if get_ndarray_list:
        array_list = []
        for seedname in seedname_list:
            array_list.append(read_fortran_array(seedname))
        return array_list
