import numpy as np
import matrix_utils as mu
import sys
import eps_solvers
import tddft_info

class JacobiDavidson4C(eps_solvers.Solver):

    ####################################################################################################################
    # preconditioning type : "TDA" or "Full"
    # P0symm (time reversal symmetry) : "general" or "symmetric"
    ####################################################################################################################
    def set_variables(self, nocc, nvirt, nbasis, preconditioning_type = "Full", P0_tsymm = "general" ):

        #if self.ndim % 2:
        #    sys.exit("ABORTING! Array must have even number of dimensions, " + str(self.ndim) + " is not even. \n")

        self.preconditioning_type = preconditioning_type
        self.P0_tsymm = P0_tsymm
        self.nocc = nocc
        self.nvirt = nvirt
        self.nbasis = nbasis

        ####################################################################################################################
    ####################################################################################################################

    def initialize_tda(self):
        #This initialization is not really needed in python, but keep for consistencies sake
        self.skip = np.full(self.nev, False, dtype=np.bool)
        self.dnorm  =np.zeros(self.nev,dtype= np.float64)
        self.teta = np.zeros(self.nev, dtype=np.float64)
        self.r_vec = np.zeros((self.ndim, self.nev), dtype= np.complex64)
        self.u_hat = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.t_vec = np.zeros(self.ndim, dtype=np.complex64)
        self.vspace = np.zeros((self.ndim, self.maxs),dtype= np.complex64)
        self.wspace = np.zeros((self.ndim, self.maxs),dtype= np.complex64)
        self.submat = np.zeros((self.maxs, self.maxs),dtype= np.complex64)

        self.nov = self.nocc*self.nvirt
        self.iter = 0
        self.first = True # For the first iteration

        self.read_1e_eigvals_and_eigvecs()

        self.construct_guess()

    def read_1e_eigvals_and_eigvecs(self):
        self.evals_1e = np.zeros(self.nocc+self.nvirt, dtype =np.float64)
        self.evecs_1e = np.zeros((self.ndim, self.nocc + self.nvirt), dtype=np.complex64)
        self.eindex_1e = np.argsort(self.evals_1e)


    def construct_guess(self):
        #dummy setting whilst writing
        self.guess = np.ndarray(self.nov, np.complex64 )
        iguess = 1

        for ii in range(self.nev):
            self.tddft4_driver_guess(iguess, self.eindex_1e, self.nov)

            for jj in range(self.nov):
                self.evecs[jj,ii] = self.guess[jj]

    # gets energy differences
    # if symmetry is involved, then will get sorted eigvals in sets of 4
    # ai :
    def evalai(self, ai):
        iocc = ((ai-1)/self.nvir) + 1
        ivir = self.nocc+mod((ai - 1), self.nvir) + 1


    def main_loop(self):
        iold = self.iter
        self.tddft4_driver_guess
        if (self.iter + self.nev > self.maxs) :
            print("WARNING in EPS solver: Maximum number of iteration reached")
            sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

        iev =1
        while iev >self.nev :
            iev = iev + 1
            if self.skip[iev] :
                continue

            self.get_new_t_vec()
            if self.first :
                print("set t_vec to u_vec guess")
                #t_vec = self.u_vec[:,iev]


    # iguess : index of eigenvalue used for constructing guess
    # P0_tsymm : string specifying kind of time reversal symmetry
    # eindex : array of indices of eigenvalues in sorted order e.g., eindex = [1,0,2] if evals = [3,-1, 10]
    # n : the index of eigenvector being built
    def tddft4_driver_guess(self, iguess, eindex, n):

        z1 = np.complex64(1.0+0.0j)
        zi = np.complex64(0.0+1.0j)
        znorm = np.complex64(1.0) / np.sqrt(2.0)

        #dummy whilst writing
        guess = np.empty(self.nov, dtype = np.complex64)

        if self.P0_tsymm == "general":
            i0 = 1#eindex[iguess]
            guess[i0] = np.complex64(1.0)

        elif self.P0_tsymm == "symmetric":
            ii = iguess%4
            jj = (iguess+3)/4
            kk = 4*(jj-1) +1

            i0 = eindex[kk]
            i1 = eindex[kk+1]
            i2 = eindex[kk+2]
            i3 = eindex[kk+3]

            if ii == 1:
                guess[i0] = z1 * znorm  # ai(00)
                guess[i3] = z1 * znorm  # ai(11)
            elif ii == 2:
                guess[i0] = zi * znorm  # ai(00)
                guess[i3] = -zi * znorm  # ai(11)
            elif ii == 3:
                guess[i1] = z1 * znorm  # ai(01)
                guess[i2] = -z1 * znorm  # ai(10)
            elif ii == 0:
                guess[i1] = zi * znorm  # ai(01)
                guess[i2] = zi * znorm  # ai(10)






        #1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    def get_new_tvec(self):
        print ("not_done")


    def tddft4_initialize(self, ndimc, ndims, nocc, tddft_info, symmetry):

        ns2 = ndims / 2

        # time reversal symmetry of the determinant
        self.P0_tsymm == symmetry

        #find out what this variable means; I think it refers to whether the thing is restricted
        self.pe_rot = True
        if self.pe_rot :
            nvir = ndims -nocc
        else :
            nvir = ndims/2 - nocc

        self.pe_rot = tddft_info.pe_rot
        self.restart = tddft_info.restart
        self.edegen = tddft_info.edegen
        self.t4skip = tddft_info.t4skip

        self.ndimc = ndimc
        self.ndims = ndims
        self.nocc = nocc
        self.nvir = nvir

        self.max_dim_subspace = tddft_info.max_dim_subspace

        #read 1 electron eigvals from charlie file, just generate random one for now
        els = np.ndarray.rand(self.ndims)

        # resort the eigenvalues so that the evals are
        for ii in range(ns2) :
            self.eval[ii] = self.els[ii + ns2]

        if self.pe_rot :
            for ii in range(ns2):
                self.eval[ii+ns2] = els

        self.nov = self.nvir * self.nocc

        for ii in range(nov):
            self.esorted = self.evalai(ii, self.nocc, self.nvir, self.eval)

        for ii in range(nov):
            self.eindex[ii] = ii

        # if obeys symmetry relations, can use smaller arrays, as eigvals/vecs are coupled
        if self.P0_tsymm == "symmetric" :
            self.esorted4 = np.ndarray(self.nov/4)
            self.eindex4 = np.ndarray(self.nov/4)

            for ii in range(nov/4):
                self.eindex[ii] = ii

            ij = 0
            for ii in range(0,self.nocc,2):
                for jj in range(0, self.nvir, 2):
                    ij = ij+1
                    self.esorted4[ij] = self.esorted[ (ii-1)*self.nvir + jj]

            for ii in range(0, (self.nov/4-1)):
                imin = np.amin[self.esorted[ii:self.nov/4]]
                if imin != 0:
                    self.esorted4[[ii, imin]] = self.esorted4[[imin, ii]]
                    self.eindex4[[ii, imin]] = self.eindex4[[imin, ii]]

            for ii in range(nov/4):
                jj = (self.eindex4[ii]-1)/(self.nvir/2)
                imin = 2*nvir*jj + (self.eindex[ii]-1)%((self.nvir/2)+1)
                self.eindex[4*ii-3] = imin
                self.eindex[4 * ii - 2] = imin + self.nvir
                self.eindex[4 * ii - 1] = imin + 1
                self.eindex[4 * ii ] = imin + self.nvir +1

        if self.P0_tsymm == "general":
            for ii in range(self.nov-1):
                imin = np.amin[self.esorted[ii:self.nov]]
                if imin != 0:
                    self.esorted[[ii, imin]] = self.esorted[[imin, ii]]
                    self.eindex[[ii, imin]] = self.eindex[[imin, ii]]




    #def swapi(self, a, n, i1, i2):









        # method == TDA











