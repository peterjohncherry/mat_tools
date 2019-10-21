import numpy as np


def get_esorted_symmetric(self):
    # Build sorted list of eigval differences with symmetry constraints imposed.
    # Blocks of 4 identical energy differences due to time reversal symmetry i.e.,
    # E_{0i}-E_{0a}) = (E_{1i}-E_{0a}) = (E_{1i}-E_{1a}) = (E_{0i}-E_{1a})
    # where 0i and 1a index an occupied Kramers pair, and 0a and 1a index a virtual Kramers pair
    # if obeys symmetry relations, can use smaller arrays, as eigvals/vecs are coupled
    esorted4 = np.ndarray(self.nov / 4)
    eindex4 = np.ndarray(self.nov / 4)

    for ii in range(self.nov / 4):
        self.eindex[ii] = ii

    ij = 0
    for ii in range(0, self.nocc, 2):
        for jj in range(0, self.nvirt, 2):
            ij = ij + 1
            esorted4[ij] = self.esorted[(ii - 1) * self.nvirt + jj]

    for ii in range(0, (self.nov / 4 - 1)):
        imin = np.argmin[self.esorted[ii:self.nov / 4]]
        if imin != 0:
            esorted4[[ii, imin]] = esorted4[[imin, ii]]
            eindex4[[ii, imin]] = eindex4[[imin, ii]]

    for ii in range(self.nov / 4):
        jj = (eindex4[ii] - 1) / (self.nvirt / 2)
        imin = 2 * self.nvirt * jj + (self.eindex[ii] - 1) % ((self.nvirt / 2) + 1)
        self.eindex[4 * ii - 3] = imin
        self.eindex[4 * ii - 2] = imin + self.nvirt
        self.eindex[4 * ii - 1] = imin + 1
        self.eindex[4 * ii] = imin + self.nvirt + 1


# guess used for closed-shell systems
# NOT FINISHED, BUT ONLY TESTING OPEN-SHELL (GENERAL) ROUTINES FOR NOW
def build_symmetric_guess_vec(self, iguess):
    znorm = (1.0 + 0.0j) / np.sqrt(2.0 + 0.0j)
    ii = iguess % 4
    jj = (iguess + 3) / 4
    kk = 4 * (jj - 1) + 1

    i0 = self.eindex[kk]
    i1 = self.eindex[kk + 1]
    i2 = self.eindex[kk + 2]
    i3 = self.eindex[kk + 3]

    guess = np.zeros(self.nov, dtype=np.complex64)
    if ii == 1:
        guess[i0] = 1.0 * znorm  # ai(00)
        guess[i3] = 1.0 * znorm  # ai(11)
    elif ii == 2:
        guess[i0] = 1.0j * znorm  # ai(00)
        guess[i3] = -1.0j * znorm  # ai(11)
    elif ii == 3:
        guess[i1] = 1.0 * znorm  # ai(01)
        guess[i2] = 1.0 * znorm  # ai(10)
    elif ii == 0:
        guess[i1] = 1.0j * znorm  # ai(01)
        guess[i2] = 1.0j * znorm  # ai(10)

    return guess
