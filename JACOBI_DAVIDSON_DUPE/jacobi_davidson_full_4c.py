import numpy as np
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils


class JacobiDavidsonFull4C(eps_solvers.Solver):

    def __init__(self, rs_filename, num_eigenvalues, restart=False, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="Full", symmetry="general", pe_rot=False):
        super().__init__(rs_filename, num_eigenvalues, restart, threshold, maxdim_subspace, solver, method, symmetry,
                         pe_rot)
        # Guess space arrays - original right-handed guesses
        self.vspace_r = None
        self.wspace_r = None

        # Guess space arrays - pairs of right-handed guesses
        self.vspace_rp = None
        self.wspace_rp = None

        # Guess space arrays - original of left-handed guesses#
        self.vspace_l = None
        self.wspace_l = None

        # Guess space arrays - pairs of left-handed guesses
        self.vspace_lp = None
        self.wspace_lp = None

        self.ndim = 2 * self.nov
        self.cycle = 0

        super().read_1e_eigvals_and_eigvecs(
            seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        super().get_esorted_general()

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self, symmetry: object = 'general'):
        self.u_vecs = np.zeros((self.ndim, self.nev), np.complex64)
        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.construct_guess(iev, symmetry)

        # for convergence checking
        self.skip = np.full(self.nev, False)
        self.dnorm = np.zeros(self.nev, dtype=np.float32)
        self.r_vecs = np.zeros((self.ndim, 2*self.nev), dtype=np.complex64)

        # Subspace Hamiltonian
        self.teta = np.zeros(self.nev, dtype=np.complex64)

    def main_loop(self):
        # v_space[:,0:maxs2] are original vectors, v_space[:,maxs2:maxs] are symmetric pairs
        # v_space[:, ii ] are right eigvecs if i is odd, and left eigenvectors if ii is even

        it = 0

        while it <= 10:
            print("\n\n=====================================================")
            print("cycle = ", self.cycle, "it = ", it)
            print("=====================================================")
            if it > self.maxs:
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            if self.cycle > 2:
                exit("got to cycle 3, exit")
            for iev in range(self.nev):
                # if self.skip[iev]:
                #     continue
                if it < self.nev:
                    t_vec = self.u_vecs[:, iev]
                else:
                    t_vec = self.get_new_tvec(iev)

                np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/t_vec" + str(it) + ".txt", t_vec)

                self.extend_right_handed_spaces(t_vec, it)
                self.extend_left_handed_spaces(it)
                it = it + 1

            # Build subspace matrix : v*Av = v*w
            submat = self.build_subspace_matrix()
#            utils.zero_small_parts(submat)
            utils.print_nonzero_numpy_elems(submat, arr_name="submat", thresh=1e-6)

            np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/submat" + str(it) + ".txt", submat)

            self.get_residual_vectors(submat)

            self.cycle = self.cycle + 1

    def extend_right_handed_spaces(self, t_vec, it):
        print("extending right_handed_spaces")
        # from t_vec = [Y, X]  get t_vec_pair = [ Y*, X* ]
        t_vec_pair = self.get_pair('x', t_vec)

        # Get coefficients for symmetrization
        d1, d2 = self.orthonormalize_pair(t_vec)

        # Build symmetrized t_vec using coeffs, and extend vspace and wspace
        if self.vspace_r is None:

            self.vspace_r = np.ndarray((self.ndim, 1), np.complex64)
            self.vspace_r[:, 0] = d1 * t_vec + d2 * t_vec_pair

            self.wspace_r = np.ndarray((self.ndim, 1), np.complex64)
            self.wspace_r[:, 0] = self.sigma_constructor(self.vspace_r[:, 0])

            self.vspace_rp = np.ndarray((self.ndim, 1), np.complex64)
            self.vspace_rp[:, 0] = self.get_pair('x', self.vspace_r[:, 0])

            self.wspace_rp = np.ndarray((self.ndim, 1), np.complex64)
            self.wspace_rp[:, 0] = self.get_pair('Ax', self.wspace_r[:, 0])
        else:
            self.vspace_r = np.c_[self.vspace_r, (d1 * t_vec + d2 * t_vec_pair)]
            self.vspace_rp = np.c_[self.vspace_rp, self.get_pair('x', self.vspace_r[:, it])]
            self.wspace_r = np.c_[self.wspace_r, self.sigma_constructor(self.vspace_r[:, it])]
            self.wspace_rp = np.c_[self.wspace_rp, self.get_pair('Ax', self.wspace_r[:, it])]

        # just to test, remove later
        self.zero_check_and_save_rh(it)

    def extend_left_handed_spaces(self, it):
        # good_t_vec is only true if the left eigenvector just generated has as sufficiently large component
        # which is orthogonal to the space spanned by the right eigenvectors
        t_vec_l = self.get_left_evec(self.vspace_r[:, it])
        t_vec_l, good_t_vec = utils.orthogonalize_v1_against_v2(t_vec_l, self.vspace_r[:, it])
        print("good_t_vec = ", good_t_vec)
        if good_t_vec:
            if self.vspace_l is None:
                self.vspace_l = np.ndarray((self.ndim, 1), np.complex64)
                self.vspace_l[:, 0] = t_vec_l

                self.wspace_l = np.ndarray((self.ndim, 1), np.complex64)
                self.wspace_l[:, 0] = self.sigma_constructor(self.vspace_r[:, 0])

                self.vspace_lp = np.ndarray((self.ndim, 1), np.complex64)
                self.vspace_lp[:, 0] = self.get_pair('x', self.vspace_r[:, 0])

                self.wspace_lp = np.ndarray((self.ndim, 1), np.complex64)
                self.wspace_lp[:, 0] = self.get_pair('Ax', self.wspace_r[:, 0])
            else:
                self.vspace_l = np.c_[self.vspace_l, t_vec_l]
                self.vspace_lp = np.c_[self.vspace_lp, self.get_pair('x', self.vspace_l[:, -1])]
                self.wspace_l = np.c_[self.wspace_l, self.sigma_constructor(self.vspace_l[:, -1])]
                self.wspace_lp = np.c_[self.wspace_lp, self.get_pair('Ax', self.wspace_l[:, -1])]

            # just to test, remove later
            self.zero_check_and_save_lh(it)

    # Ugly, but will slowly swap out parts for more sensible approach
    def build_subspace_matrix(self):

        sb_dim = 0
        for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
            if vs is not None:
                sb_dim = sb_dim + vs.shape[1]
        submat = np.zeros((sb_dim, sb_dim), np.complex64)

        vi = 0
        for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
            if vs is not None:
                for ii in range(vs.shape[1]):
                    wj = 0
                    for ws in [self.wspace_r, self.wspace_l, self.wspace_rp,  self.wspace_lp]:
                        if ws is not None:
                            for jj in range(ws.shape[1]):
                                submat[vi+ii, wj+jj] = np.vdot(vs[:, ii], ws[:, jj])
                            wj = wj + ws.shape[1]
                vi = vi + vs.shape[1]
        return submat

    @staticmethod
    # [X,Y] --> [X, -Y]
    def get_left_evec(vec):
        buff = np.zeros_like(vec)
        buff[:int(len(vec)/2)] = vec[:int(len(vec)/2)]
        buff[int(len(vec)/2):] = -vec[int(len(vec)/2):]
        return buff

    def get_residual_vectors(self, submat):
        # Ritz values and Ritz vectors defined in trial vspace
        ritz_vals, hevecs = np.linalg.eig(submat)
        utils.zero_small_parts(hevecs)

        # ordering eigenvalues and eigenvectors so first set of evals are positive and run in ascending order
        # (eigenvalues in the spectrum are always in positive and negative pairs)
        print("theta orig = ", ritz_vals)
        idx = ritz_vals.argsort()
        t2 = int(len(ritz_vals)/2)
        for ii in range(t2):
            idx[ii], idx[ii+t2] = idx[ii+t2], idx[ii]

        print("idx = ", idx)
        ritz_vals = ritz_vals[idx]
        hevecs = hevecs[:, idx]
        print("theta sorted = ", ritz_vals)

        for ii in range(hevecs.shape[1]):
            np.savetxt("hevecs_"+str(ii)+"_"+str(self.cycle)+".txt", hevecs[:, ii])

        # Construction of Ritz vectors from eigenvectors
        self.u_vecs = np.zeros((self.ndim, self.nev), np.complex64)
        for iev in range(self.nev):
            vi = 0
            for vs in [self.vspace_r, self.vspace_l,self.vspace_rp, self.vspace_lp]:
                if vs is not None:
                    for ii in range(vs.shape[1]):
                        self.u_vecs[:, iev] = self.u_vecs[:, iev] + hevecs[ii+vi, iev] * vs[:, ii]
                    vi = vi + vs.shape[1]

        for ii in range(self.u_vecs.shape[1]):
            np.savetxt("u_vecs_"+str(ii)+"_"+str(self.cycle)+".txt", self.u_vecs[:, ii])

        # Construction of u_hat from Ritz_vectors and w_spaces,
        dnorm = np.zeros(self.nev, np.complex64)
        self.r_vecs = np.zeros((self.ndim, self.nev), np.complex64)
        u_hats = np.zeros((self.ndim, self.nev), np.complex64)
        for iev in range(self.nev):
            wj = 0
            for ws in [self.wspace_r, self.wspace_l, self.wspace_rp, self.wspace_lp]:
                if ws is not None:
                    for jj in range(ws.shape[1]):
                        u_hats[:, iev] = u_hats[:, iev] + hevecs[jj+wj, iev] * ws[:, jj]
                    wj = wj + ws.shape[1]
            # Calculation of residual vectors, and residual norms
            self.r_vecs[:, iev] = u_hats[:, iev] - self.u_vecs[:, iev] * ritz_vals[iev]

        utils.zero_small_parts(self.r_vecs)
        utils.zero_small_parts(self.u_vecs)
        for ii in range(u_hats.shape[1]):
            np.savetxt("u_hats_" + str(ii) + "_" + str(self.cycle) + ".txt", u_hats[:, ii])
            np.savetxt("r_vecs_" + str(ii) + "_" + str(self.cycle) + ".txt", self.r_vecs[:, ii])
            dnorm[ii] = np.linalg.norm(self.r_vecs[:, ii])

        print("dnorm = ", dnorm)
        self.teta = ritz_vals[:self.nev]

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    # Construct initial guess
    def construct_guess(self, iguess, symmetry_type):
        guess = np.zeros(self.nov, dtype=np.complex64)

        if symmetry_type == 'general':
            guess[self.eindex[iguess]] = 1.0 + 0.0j

        elif symmetry_type == '4c':
            component = iguess % 4
            mo_number = iguess+3/4
            start_elem = int(4*(mo_number-1)+1)

            i0 = int(self.eindex[start_elem])
            i1 = int(self.eindex[start_elem+1])
            i2 = int(self.eindex[start_elem+2])
            i3 = int(self.eindex[start_elem+3])

            if component == 1:
                guess[i0] = 1.0/np.sqrt(2.0) + 0.0j
                guess[i3] = 1.0/np.sqrt(2.0) + 0.0j
            elif component == 2:
                guess[i0] = 0.0 + 1.0j/np.sqrt(2.0)
                guess[i3] = 0.0 - 1.0j/np.sqrt(2.0)
            elif component == 3:
                guess[i1] = 1.0/np.sqrt(2.0) + 0.0
                guess[i2] = -1.0/np.sqrt(2.0) + 0.0
            else:
                guess[i1] = 0.0 + 1.0j/np.sqrt(2.0)
                guess[i2] = 0.0 - 1.0j/np.sqrt(2.0)
        else:
            print("Symmetry type ", symmetry_type, " not implemented")

        return guess

    # Construct pair according to symmetry relations
    # x :  [X,Y] --> [ Y*, X* ]
    # Ax : [X,Y] --> [ -Y*, -X* ]
    def get_pair(self, pair_type, vec_in):
        vec_out = np.empty(self.ndim, dtype=np.complex64)
        n2 = int(self.ndim/2)
        if pair_type == 'x':
            vec_out[n2:] = np.conj(vec_in[:n2])
            vec_out[:n2] = np.conj(vec_in[n2:])
        elif pair_type == 'Ax':
            vec_out[n2:] = -vec_in[:n2].conj()
            vec_out[:n2] = -vec_in[n2:].conj()
        else:
            sys.exit("ABORTING!! Unknown pair_type specified in eigenvector construction")

        return vec_out

    # This restructures the input vector so that it's "pair" as constructed using get_pair
    # will also be orthogonal to the vector space
    def orthonormalize_pair(self, vec):
        d1 = 0.0
        d2 = 0.0
        z = np.dot(vec[:int(self.ndim/2)], vec[int(self.ndim/2):])

        d = np.real(z * np.conj(z))
        r = 1.0 - 4.0 * d
        if r >= 1e-12:
            if np.abs(d) > 1e-30:
                t = 0.0 + np.sqrt(r)
                r1 = np.real(z)/np.sqrt(d)
                r2 = np.imag(z)/np.sqrt(d)
                d1 = -r1 * np.sqrt(0.5*t/r) + r2*np.sqrt(0.5*t/r)*1.0j
                d2 = np.sqrt(2.0*d/(t*r)) + 0.0j
            else:
                print("WARNING!, small d in orthonormalize pair routine")
                d1 = 1.0 + 0.0j
                d2 = 0.0 + 0.0j
        else:
            print("WARNING!, small r in orthonormalize pair routine")

        return d1, d2

    def get_new_tvec(self, iev):
        v1 = np.ndarray(self.ndim, np.complex64)  # v1 = M^{-1}*r

        v2 = np.ndarray(self.ndim, np.complex64)  # v2 = M^{-1}*u
        idx = 0
        # First half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = (self.evalai(ii, jj) - np.real(self.teta[iev]))
                if abs(ediff) > 1e-8:
                    dtmp = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = 0.0 + 0.0j
                    v2[idx] = 0.0 + 0.0j
                idx += 1

        # Second half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = -self.evalai(ii, jj) - np.real(self.teta[iev])
                if abs(ediff) > 1e-8:
                    dtmp = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = 0.0 + 0.0j
                    v2[idx] = 0.0 + 0.0j
                idx += 1
        np.savetxt("v1_" + str(iev) + ".txt", v1)
        np.savetxt("v2_" + str(iev) + ".txt", v2)

        u_m1_u = np.vdot(self.u_vecs[:, iev], v2)
        u_m1_r = np.vdot(self.u_vecs[:, iev], v1)
        print("uMinvr = ", u_m1_r, "uMinvu = ", u_m1_u)
        if abs(u_m1_u) > 1e-8:
            u_m1_r = np.vdot(self.u_vecs[:, iev], v1)
            factor = u_m1_r / np.real(u_m1_u)
            return factor*v2-v1
        else:
            return -v1

    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig, vec)

    def zero_check_and_save_rh(self, it):
        utils.check_for_nans([self.vspace_r, self.vspace_rp, self.wspace_r, self.wspace_rp],
                             ["vspace_r", "vspace_rp", "wspace_r", "wspace_rp"])
        utils.zero_small_parts(self.vspace_r)
        utils.zero_small_parts(self.wspace_r)
        utils.zero_small_parts(self.vspace_rp)
        utils.zero_small_parts(self.wspace_rp)
        for ii in range(self.vspace_r.shape[1]):
            utils.save_arrs_to_file([self.vspace_r[:, it], self.vspace_rp[:, it], self.wspace_r[:, it],
                                     self.wspace_rp[:, it]],
                                    ["vspace_r" + str(it), "vspace_rp" + str(it), "wspace_r" + str(it),
                                     "wspace_rp" + str(it)])

    def zero_check_and_save_lh(self, it):
        utils.check_for_nans([self.vspace_l, self.vspace_lp, self.wspace_l, self.wspace_lp],
                             ["vspace_l", "vspace_lp", "wspace_l", "wspace_lp"])
        utils.zero_small_parts(self.vspace_l)
        utils.zero_small_parts(self.wspace_l)
        utils.zero_small_parts(self.vspace_lp)
        utils.zero_small_parts(self.wspace_lp)
        for ii in range(self.vspace_l.shape[1]):
            utils.save_arrs_to_file([self.vspace_l[:, ii], self.vspace_lp[:, ii], self.wspace_l[:, ii],
                                     self.wspace_lp[:, ii]],
                                    ["vspace_l" + str(ii), "vspace_lp" + str(ii), "wspace_l" + str(ii),
                                     "wspace_lp" + str(ii)])

    def get_numpy_evals(self):
        evals, evecs = np.linalg.eig(self.mat_orig)
        np.set_printoptions(threshold=sys.maxsize)
        print("numpy evals =\n ", sorted(abs(np.real(evals)), reverse=False))
