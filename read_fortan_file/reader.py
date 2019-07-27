import sys
import os
import numpy as np
from scipy.io import FortranFile

def read_binary_fortran_file(name, datatype, dim0, dim1=1 ):


    input_array=np.ndarray((dim0*dim1))

    if dim1 == 1 :
        if (datatype == "real"):
            fmat = FortranFile(name, 'r')
            input_array = fmat.read_reals(dtype=np.float64)
            fmat.close()
        elif (datatype == "int"):
            fmat = FortranFile(name, 'r')
            input_array = fmat.read_ints(dtype=np.int32)
            fmat.close()
    else :
      if (datatype == "real"):
          fmat = FortranFile(name, 'r')
          input_array = fmat.read_reals(dtype=np.float64)
          input_array = input_array.reshape((dim0,dim1)).transpose()
          print (input_array)
          fmat.close()

      elif (datatype == "int"):
          fmat = FortranFile(name, 'r')
          input_array = fmat.read_ints(dtype=np.int32)
          input_array = input_array.reshape((dim0,dim1)).transpose()
          print (input_array)
          fmat.close()

      elif ( datatype == "complex" ):
          sys.exit("reading of complex matrices is not directly possible, must write out real and imag parts as two seperate real matrices\n")

      else :
          sys.exit("reading of datatype \"" +  datatype + "\" is not implemented\n ")

    return input_array

#reads the matrix info file which is generated to aid reading of fortran binary file
def read_array_info_file(name):

    infofile = open(name,"r")
    for line in infofile.readlines():
        line_array = line.split()
        if line_array[0] == "dim0":
            dim0 = int(line_array[2])
        elif line_array[0] == "dim1":
            dim1 = int(line_array[2])
        elif line_array[0] == "datatype":
            datatype = str(line_array[2])

    return dim0, dim1, datatype

def read_fortran_array(seedname):
    nrows, ncols, datatype = read_array_info_file(seedname+".info")

    if datatype == "complex" :
        datatype = "real"
        name =seedname+"_real.bin"
        array_real =  read_binary_fortran_file(name, datatype, nrows, ncols)
        name = seedname+"_imag.bin"
        array_imag = read_binary_fortran_file( name, datatype, nrows, ncols)
        return array_real + 1j*array_imag

    elif datatype == "real" :
        datatype = "real"
        name =seedname+".bin"
        array_real =  read_binary_fortran_file(name, datatype, nrows, ncols)
        return array_real

    elif datatype == "int" :
        datatype ="int"
        name =seedname+".bin"
        array_ints =  read_binary_fortran_file(name, datatype, nrows, ncols)
        return array_ints

    else :
        sys.exit("Not implemented " + datatype + " ABORTING" )


if __name__=='__main__':
        import sys
        if len(sys.argv)==2:
            dir_path = os.path.dirname(os.path.realpath(__file__))
            seedname = dir_path + "/" +sys.argv[1]
            print ("seedname = ",  seedname )
            np.save(dir_path+"/"+seedname, read_fortran_array(seedname))

        elif len(sys.argv)==3:
            seedname = sys.argv[2] + "/"+ sys.argv[1]
            print ("seedname = ",  seedname )
            read_fortran_array(seedname)
        else :
            print(" input should be of form  ./read_fortran_matrix seedname working_directory, where " \
                    "the relevant files can be found in {working_directory}+/+{seedname}\n" )
