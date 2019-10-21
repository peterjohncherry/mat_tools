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

# Sorting by size of residual
    def sort_by_rnorm(self):
        for it in range(len(self.wspace)):
            self.submat = np.matmul(np.conjugate(self.wspace.T), self.vspace)
            np.savetxt("submat_" + str(it), self.submat, fmt='%.4f')  # TESTING
            ritz_vals, hdiag = la.eig(self.submat)

            np.savetxt("teta_" + str(it), ritz_vals, fmt='%.4f')  # TESTING
            np.savetxt("hdiag_" + str(it), hdiag, fmt='%.4f')  # TESTING

            # Calculate Ritz vectors with the lowest residual norms and put them into u
            tmp_dnorm = self.dnorm
            if it > self.nev:
                for iteta in range(it):
                    tmp_u_vec = np.matmul(self.vspace, hdiag[:, iteta])
                    tmp_u_hat = np.matmul(self.wspace, hdiag[:, iteta])
                    tmp_r_vec = tmp_u_hat - ritz_vals[iteta] * tmp_u_vec
                    tmp_r_norm = la.norm(tmp_r_vec)

                    if iteta < self.nev:
                        self.u_vecs[:, iteta] = tmp_u_vec
                        self.u_hats[:, iteta] = tmp_u_hat
                        self.r_vecs[:, iteta] = tmp_r_vec
                        self.teta[iteta] = ritz_vals[iteta]
                    else:
                        # If new Ritz vector has low residual norm, replace existing Ritz vector with max residual norm
                        max_norm_loc = np.argmax(tmp_dnorm)
                        self.u_vecs[:, max_norm_loc] = tmp_u_vec
                        self.u_hats[:, max_norm_loc] = tmp_u_hat
                        self.r_vecs[:, max_norm_loc] = tmp_r_vec
                        self.dnorm[max_norm_loc] = tmp_r_norm
                        tmp_dnorm[max_norm_loc] = tmp_r_norm

                rsorted_idxs = self.dnorm.argsort()[::-1]
                self.u_vecs = self.u_vecs[:, rsorted_idxs]
                self.u_hats = self.u_hats[:, rsorted_idxs]
                self.r_vecs = self.r_vecs[:, rsorted_idxs]
                self.dnorm = self.dnorm[rsorted_idxs]


    def tv_orth_check(self):
        for ii in range(self.vspace.shape[1]):
            vtoverlap = np.vdot(self.t_vec, self.vspace[:, ii])
            if abs(vtoverlap) > 1e-10:
                print("np.vdot(self.t_vec, self.vspace[:," + str(ii) + "]) = ",
                      np.vdot(self.t_vec, self.vspace[:, ii]), end=' ')
                print("   ||t_vec[" + str(iter) + "]|| =", la.norm(self.t_vec))