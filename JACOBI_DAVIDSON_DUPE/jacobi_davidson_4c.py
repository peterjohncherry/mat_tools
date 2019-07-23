import numpy as np
import sys
import eps_solvers

class JacobiDavidson4C(eps_solvers.Solver):

    def initialize(self):
        ns2 = int(self.ndims / 2)
        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc
        # read 1 electron eigvals from charlie file, just generate random one for now
        els = np.empty(self.ndims)

        # resort the eigenvalues so that the evals are
        self.evals = np.empty(self.ndims)
        for ii in range(ns2):
            self.evals[ii] = els[ii + ns2]

        # If pe_rot is true we allow mixing between virtual orbitals with positive and negative energies
        if self.pe_rot:
            for ii in range(ns2):
                self.evals[ii + ns2] = els[ii]

        self.nov = self.nvirt * self.nocc

        self.eindex = np.empty(self.nov)
        for ii in range(self.nov):
            self.eindex[ii] = ii

        self.esorted = np.empty(self.nov)
        if self.P0_tsymm == "symmetric":
            self.get_esorted_symmetric()
        elif self.P0_tsymm == "general":
            self.get_esorted_general()
        else:
            sys.exit("have not implemented guess for symmetry " + self.P0_tsymm)

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self):
        #This initialization is not really needed in python, but keep for consistencies sake
        self.skip = np.full(self.nev, False, dtype=np.bool)
        self.dnorm = np.zeros(self.nev,dtype=np.float64)
        self.teta = np.zeros(self.nev, dtype=np.float64)
        self.r_vec = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.u_hat = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.t_vec = np.zeros(self.ndim, dtype=np.complex64)
        self.vspace = np.zeros((self.ndim, self.maxs), dtype=np.complex64)
        self.wspace = np.zeros((self.ndim, self.maxs), dtype=np.complex64)
        self.submat = np.zeros((self.maxs, self.maxs), dtype=np.complex64)

        self.nov = self.nocc*self.nvirt
        self.iter = 0
        self.first = True # For the first iteration
        self.eindex = np.empty(self.nov)
        self.evecs = np.empty((self.nov, self.nev), dtype = np.complex64)

        self.read_1e_eigvals_and_eigvecs()

        if self.method == "TDA":
            self.construct_guess_tda()
        elif self.method == "FULL":
            self.construct_guess_full()
        else:
            sys.exit("Preconditioner \""+self.preconditioner+"\" is not implemented \n")


    def get_esorted_general(self):
    # build sorted list of eigval differences without imposing any symmetry constraints
        for ii in range(self.nov-1):
            imin = np.argmin(self.esorted[ii:self.nov])
            if imin != 0:
                self.esorted[[ii, imin]] = self.esorted[[imin, ii]]
                self.eindex[[ii, imin]] = self.eindex[[imin, ii]]


    def get_esorted_symmetric():
    # Build sorted list of eigval differences with symmetry constraints imposed.
    # Blocks of 4 identical energy differences due to time reversal symmetry i.e.,
    # E_{0i}-E_{0a}) = (E_{1i}-E_{0a}) = (E_{1i}-E_{1a}) = (E_{0i}-E_{1a})
    # where 0i and 1a index an occupied Kramers pair, and 0a and 1a index a virtual Kramers pair
    # if obeys symmetry relations, can use smaller arrays, as eigvals/vecs are coupled
            self.esorted4 = np.ndarray(self.nov / 4)
            self.eindex4 = np.ndarray(self.nov / 4)

            for ii in range(self.nov / 4):
                self.eindex[ii] = ii

            ij = 0
            for ii in range(0, self.nocc, 2):
                for jj in range(0, self.nvir, 2):
                    ij = ij + 1
                    self.esorted4[ij] = self.esorted[(ii - 1) * self.nvir + jj]

            for ii in range(0, (self.nov / 4 - 1)):
                imin = np.argmin[self.esorted[ii:self.nov / 4]]
                if imin != 0:
                    self.esorted4[[ii, imin]] = self.esorted4[[imin, ii]]
                    self.eindex4[[ii, imin]] = self.eindex4[[imin, ii]]

            for ii in range(nov / 4):
                jj = (self.eindex4[ii] - 1) / (self.nvir / 2)
                imin = 2 * nvir * jj + (self.eindex[ii] - 1) % ((self.nvir / 2) + 1)
                self.eindex[4 * ii - 3] = imin
                self.eindex[4 * ii - 2] = imin + self.nvir
                self.eindex[4 * ii - 1] = imin + 1
                self.eindex[4 * ii] = imin + self.nvir + 1

    def read_1e_eigvals_and_eigvecs(self):
        self.evals_1e = np.zeros(self.ndims, dtype =np.float64)
        self.evecs_1e = np.zeros((self.ndim, self.nocc + self.nvirt), dtype=np.complex64)
        self.eindex_1e = np.argsort(self.evals_1e)
        self.eindex = np.argsort(self.evals_1e)

    def construct_guess_tda(self):
        self.guess = np.zeros(self.nov, np.complex64)
        for ii in range(self.nev):
            self.tddft4_driver_guess(ii)
            for jj in range(self.nov):
                self.evecs[jj,ii] = self.guess[jj]

    def construct_guess_full(self):
        sys.exit("Have not implemented guess for FULL method")

    # gets energy differences
    # if symmetry is involved, then will get sorted eigvals in sets of 4
    # ai :
    def evalai(self, ai):
        iocc = (int)((ai-1)/self.nvirt) + 1
        ivir = self.nocc+((ai - 1)%self.nvirt) + 1
        return self.evals[ivir] - self.evals[iocc]

    def main_loop(self):
        iold = self.iter
        #self.tddft4_driver_guess
        if (self.iter + self.nev > self.maxs) :
            print("WARNING in EPS solver: Maximum number of iteration reached")
            sys.exit("WARNING in EPS solver: Maximum number of iteration reached")

        iev = 1
        while iev >self.nev :
            iev = iev + 1
            if self.skip[iev] :
                continue

            self.get_new_t_vec()
            if self.first :
                print("set t_vec to u_vec guess")
                #t_vec = self.u_vec[:,iev]

    # iguess : index of the eigenvector being built
    def tddft4_driver_guess(self, iguess ):
        if self.P0_tsymm == "general":
            self.build_general_guess_vec(iguess)
        elif self.P0_tsymm == "symmetric":
            self.build_symmetric_guess_vec(iguess)

    # guess used for open-shell systems
    def build_general_guess_vec(self, iguess):
            self.guess[self.eindex[iguess]] = 1.0+0.0j

    # guess used for closed-shell systems
    #NOT FINISHED, BUT ONLY TESTING OPEN-SHELL (GENERAL) ROUTINES FOR NOW
    def build_symmetric_guess_vec(self, iguess):

        znorm = (1.0+0.0j) / np.sqrt(2.0+0.0j)
        ii = iguess%4
        jj = (iguess+3)/4
        kk = 4*(jj-1) +1

        i0 = self.eindex[kk]
        i1 = self.eindex[kk+1]
        i2 = self.eindex[kk+2]
        i3 = self.eindex[kk+3]

        if ii == 1:
            self.guess[i0] = 1.0 * znorm  # ai(00)
            self.guess[i3] = 1.0 * znorm  # ai(11)
        elif ii == 2:
            self.guess[i0] =  1.0j * znorm  # ai(00)
            self.guess[i3] = -1.0j * znorm  # ai(11)
        elif ii == 3:
            self.guess[i1] = 1.0* znorm  # ai(01)
            self.guess[i2] = 1.0 * znorm  # ai(10)
        elif ii == 0:
            self.guess[i1] = 1.0j * znorm  # ai(01)
            self.guess[i2] = 1.0j * znorm  # ai(10)

    #1. Find orthogonal complement t_vec of the u_vec using preconditioned matrix ( A - teta*I )
    def get_new_tvec(self):
        print ("not_done")












