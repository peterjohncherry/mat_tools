import numpy as np
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils
import decimal


class JacobiDavidsonFull4C(eps_solvers.Solver):

    def __init__(self, rs_filename, num_eigenvalues, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="Full", symmetry="general", pe_rot=False):
        super().__init__(rs_filename, num_eigenvalues, threshold, maxdim_subspace, solver, method, symmetry,
                         pe_rot)

        self.save_dir = "/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/TXTS/"
        decimal.getcontext().prec = 32
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

        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2 * self.nov
        self.cycle = 1

        self.read_1e_eigvals_and_eigvecs(seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        self.get_esorted_general()

    def read_1e_eigvals_and_eigvecs(self, seedname):
        evals_1e_all = mr.read_fortran_array(seedname)
        num_pos_evals = self.nvirt + self.nocc
        if self.pe_rot:
            self.evals_1e = np.zeros(2*num_pos_evals, dtype=self.real_precision)
            self.evals_1e[:num_pos_evals] = evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = np.zeros(num_pos_evals, dtype=self.real_precision)
            self.evals_1e = evals_1e_all[num_pos_evals:]

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=self.real_precision)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self, symmetry: object = 'general'):
        # Ritz values and vectors
        self.teta = np.zeros(self.nev, dtype=self.complex_precision)
        self.u_vecs = np.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.construct_guess(iev, symmetry)

        # Residual values for convergence checking
        self.r_vecs = np.zeros((self.ndim, 2*self.nev), dtype=self.complex_precision)

    def main_loop(self):
        np.set_printoptions(precision=16)
        it = 0
        while it <= 1000:
            print("\n\n=====================================================")
            print("cycle = ", self.cycle, "it = ", it)
            print("=====================================================")

            # if it > self.maxs:
            if it > 70:
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            max_cycle = 3
            if self.cycle > max_cycle:
                sys.exit("got to cycle " + str(max_cycle) + ", exit")

            for iev in range(self.nev):
                self.iev = iev
                self.extend_right_handed_spaces(iev)
                if self.cycle > 1:
                    self.extend_left_handed_spaces()
                it = it + 1

            self.save_spaces()

            # Build subspace matrix : v*Av = v*w
            submat = self.build_subspace_matrix()
            dnorm = self.get_residual_vectors(submat)

            skip = np.ndarray(self.nev, np.bool)
            for ii in range(self.nev):
                if dnorm[ii] <= self.threshold:
                    skip[ii] = True
                else:
                    skip[ii] = False

            print("self.teta = ", self.teta)
            if False in skip[:self.nev]:
                print("Not converged on iteration ", it)
                print("dnorm = ", dnorm, "skip = ", skip)
            else:
                print("Final eigenvalues = ", np.real(self.teta[:self.nev]))
                sys.exit("Converged!!")

            self.cycle = self.cycle + 1

    def extend_right_handed_spaces(self, iev):

        if self.cycle == 1:
            t_vec = self.u_vecs[:, iev]
        else:
            t_vec = self.get_new_tvec(iev)

            for ii in range(self.vspace_r.shape[1]):
                t_vec = utils.rs_self__orthogonalize(t_vec, self.vspace_r[:, ii])

        t_vec, good_t_vec = utils.rs_self_normalize(t_vec)
        if good_t_vec:
            # Get coefficients for symmetrization
            d1, d2 = self.orthonormalize_pair(t_vec)

            # from t_vec = [Y, X]  get t_vec_pair = [ Y*, X* ]
            t_vec_pair = self.get_pair('x', t_vec)
            t_vec = d1 * t_vec + d2 * t_vec_pair
            t_vec_pair = self.get_pair('x', t_vec)
            np.savetxt(self.save_dir + "t_vec_r_c" + str(self.cycle) + "_i" + str(iev) + ".txt", t_vec)
            np.savetxt(self.save_dir + "t_vec_rp_c" + str(self.cycle) + "_i" + str(iev) + ".txt", t_vec_pair)

            if self.vspace_r is None:
                self.vspace_r = np.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_r[:, 0] = t_vec
                self.wspace_r = np.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_r[:, 0] = self.sigma_constructor(t_vec)
            else:
                self.vspace_r = np.c_[self.vspace_r, t_vec]
                self.wspace_r = np.c_[self.wspace_r, self.sigma_constructor(t_vec)]

            # Build symmetrized t_vec using coeffs, and extend vspace and wspace
            if self.vspace_rp is None:
                self.vspace_rp = np.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_rp[:, 0] = t_vec_pair
                self.wspace_rp = np.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_rp[:, 0] = self.sigma_constructor(t_vec_pair)
            else:
                self.vspace_rp = np.c_[self.vspace_rp, t_vec_pair]
                self.wspace_rp = np.c_[self.wspace_rp, self.sigma_constructor(t_vec_pair)]

    def extend_left_handed_spaces(self):
        # good_t_vec is only true if the left eigenvector just generated has as sufficiently large component
        # which is orthogonal to the space spanned by the right eigenvectors
        t_vec_l = self.get_left_evec(self.vspace_r[:, -1])
        for ii in range(self.vspace_r.shape[1]):
            t_vec_l = utils.rs_self__orthogonalize(t_vec_l, self.vspace_r[:, ii])

        t_vec_l, good_t_vec = utils.rs_self_normalize(t_vec_l)
        t_vec_l_pair = self.get_pair('x', t_vec_l)
        np.savetxt(self.save_dir + "t_vec_l_c" + str(self.cycle) + "_i" + str(self.iev) + ".txt", t_vec_l)
        np.savetxt(self.save_dir + "t_vec_lp_c" + str(self.cycle) + "_i" + str(self.iev) + ".txt", t_vec_l_pair)

        if good_t_vec:
            if self.vspace_l is None:
                self.vspace_l = np.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_l[:, 0] = t_vec_l

                self.wspace_l = np.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_l[:, 0] = self.sigma_constructor(t_vec_l)

                self.vspace_lp = np.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_lp[:, 0] = t_vec_l_pair

                self.wspace_lp = np.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_lp[:, 0] = self.sigma_constructor(t_vec_l_pair)
            else:
                self.vspace_l = np.c_[self.vspace_l, t_vec_l]
                self.vspace_lp = np.c_[self.vspace_lp, t_vec_l_pair]
                self.wspace_l = np.c_[self.wspace_l, self.sigma_constructor(t_vec_l)]
                self.wspace_lp = np.c_[self.wspace_lp, self.sigma_constructor(t_vec_l_pair)]

    # Ugly, but will slowly swap out parts for more sensible approach
    def build_subspace_matrix(self):
        sb_dim = 0
        for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
            if vs is not None:
                sb_dim = sb_dim + vs.shape[1]
        submat = np.zeros((sb_dim, sb_dim), self.complex_precision)

        if self.vspace_r is not None:
            print("self.vspace_r.shape = ", self.vspace_r.shape)
        if self.vspace_l is not None:
            print("self.vspace_l.shape = ", self.vspace_l.shape)
        if self.vspace_rp is not None:
            print("self.vspace_rp.shape = ", self.vspace_rp.shape)
        if self.vspace_lp is not None:
            print("self.vspace_lp.shape = ", self.vspace_lp.shape)

        print("sb_dim = ", sb_dim)

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
        buff = np.empty_like(vec)
        buff[:int(len(vec)/2)] = vec[:int(len(vec)/2)]
        buff[int(len(vec)/2):] = -vec[int(len(vec)/2):]
        return buff

    def get_residual_vectors(self, submat):
        # Ritz values and Ritz vectors defined in trial vspace
        np.savetxt(self.save_dir+"submat" + str(submat.shape[0]) + ".txt", np.real(submat))
        ritz_vals, ritz_vecs = np.linalg.eig(submat)

        # ordering eigenvalues and eigenvectors so first set of evals are positive and run in ascending order
        # (eigenvalues in the spectrum are always in positive and negative pairs)
        idx = ritz_vals.argsort()
        tmp_ritz_vals = ritz_vals[idx]
        np.savetxt(self.save_dir+"ritz_vals_c"+str(self.cycle)+".txt", tmp_ritz_vals)
        t2 = int(len(ritz_vals)/2)
        for ii in range(t2):
            idx[ii], idx[ii+t2] = idx[ii+t2], idx[ii]

        ritz_vals = ritz_vals[idx]
        ritz_vecs = ritz_vecs[:, idx]
        print("====== ritz_vals sorted ====== \n", ritz_vals)

        # Construction of Ritz vectors from eigenvectors
        self.u_vecs = np.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            vi = 0
            for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
                if vs is not None:
                    for ii in range(vs.shape[1]):
                        self.u_vecs[:, iev] = self.u_vecs[:, iev] + ritz_vecs[ii+vi, iev] * vs[:, ii]
                    vi = vi + vs.shape[1]

        # Construction of u_hat from Ritz_vectors and w_spaces,
        dnorm = np.zeros(self.nev, self.complex_precision)
        self.r_vecs = np.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            wj = 0
            u_hat = np.zeros(self.ndim, self.complex_precision)
            for ws in [self.wspace_r, self.wspace_l, self.wspace_rp, self.wspace_lp]:
                if ws is not None:
                    for jj in range(ws.shape[1]):
                        u_hat = u_hat + ritz_vecs[jj+wj, iev] * ws[:, jj]
                    wj = wj + ws.shape[1]

            # Calculation of residual vectors, and residual norms
            self.r_vecs[:, iev] = u_hat - self.u_vecs[:, iev] * ritz_vals[iev]
            dnorm[iev] = np.linalg.norm(self.r_vecs[:, iev])

        print("dnorm = ", dnorm)
        self.teta = ritz_vals[:self.nev]
        return dnorm

    # Construct initial guess
    def construct_guess(self, iguess, symmetry_type):
        guess = np.zeros(self.nov, dtype=self.complex_precision)

        if symmetry_type == 'general':
            guess[self.eindex[iguess]] = self.complex_precision(1.0 + 0.0j)

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
        vec_out = np.empty(self.ndim, dtype=self.complex_precision)
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
        d1 = self.complex_precision(0.0 + 0.0j)
        d2 = self.complex_precision(0.0 + 0.0j)
        z = self.complex_precision(0.0 + 0.0j)
        nh = int(self.ndim/2)
        for ii in range(nh):
            z = z + vec[ii] * vec[ii+nh]

        # z = np.dot(vec[:int(self.ndim/2)], vec[int(self.ndim/2):])
        one = self.complex_precision(1.0 + 0.0j)
        four = self.complex_precision(4.0 + 0.0j)
        half = self.complex_precision(0.5 + 0.0j)
        two = self.complex_precision(2.0 + 0.0j)
        d = self.complex_precision(z * np.conj(z) + 0.0j)
        r = self.complex_precision(one - four * d + 0.0j)
        sqrt1 = self.complex_precision(0.0 + 1.0j)

        print("determining t_vec coefficients")
        print("z = ", z)
        print("d = ", d)
        print("r = ", r)

        if r >= 1e-12:
            if np.abs(d) > 1e-12:
                t = self.complex_precision(one + np.sqrt(r) + 0.0j)
                print("t = ", t)
                print("np.sqrt(d) = ", np.sqrt(d))
                print("np.real(z) = ", np.real(z))
                print("np.imag(z) = ", np.imag(z))
                r1 = np.real(z)/np.sqrt(d)
                r2 = np.imag(z)/np.sqrt(d)

                print("r1 = ", r1)
                print("r2 = ", r2)

                d1 = self.complex_precision(-r1 * np.sqrt(half*t/r) + r2*np.sqrt(half*t/r)*sqrt1)
                d2 = self.complex_precision(np.sqrt(two*d/(t*r)) + 0.0j)
            else:
                print("WARNING!, small d in orthonormalize pair routine")
                d1 = self.complex_precision(1.0 + 0.0j)
                d2 = self.complex_precision(0.0 + 0.0j)
        else:
            print("WARNING!, small r in orthonormalize pair routine")

        print("d1, d2 = ", d1, d2)
        return d1, d2

    def get_new_tvec(self, iev):
        v1 = np.ndarray(self.ndim, self.complex_precision)  # v1 = M^{-1}*r
        v2 = np.ndarray(self.ndim, self.complex_precision)  # v2 = M^{-1}*u
        idx = 0
        one = self.complex_precision(1.0 + 0.0j)
        # First half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = (self.evalai(ii, jj) - np.real(self.teta[iev]))
                if abs(ediff) > 1e-8:
                    dtmp = one / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = self.complex_precision(0.0 + 0.0j)
                    v2[idx] = self.complex_precision(0.0 + 0.0j)
                idx += 1

        # Second half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = -self.evalai(ii, jj) - np.real(self.teta[iev])
                if abs(ediff) > 1e-8:
                    dtmp = one / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = self.complex_precision(0.0 + 0.0j)
                    v2[idx] = self.complex_precision(0.0 + 0.0j)
                idx += 1

        u_m1_u = np.vdot(self.u_vecs[:, iev], v2)
        print("u_m1_u"+str(iev) + " = ", u_m1_u)
        if abs(u_m1_u) > 1e-8:
            u_m1_r = np.vdot(self.u_vecs[:, iev], v1)
            print("u_m1_r"+str(iev) + " = ", u_m1_r)
            factor = u_m1_r / np.real(u_m1_u)
            return factor*v2-v1
        else:
            return -v1

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc + virt_orb] - self.evals_1e[occ_orb]

    def sigma_constructor(self, vec):
        return np.matmul(self.mat_orig, vec)

    def save_spaces(self):
        if self.vspace_r is not None:
            self.save_array_as_vectors(self.vspace_r, self.save_dir + "v_vec_r_c"+str(self.cycle))
        if self.vspace_rp is not None:
            self.save_array_as_vectors(self.vspace_rp, self.save_dir + "v_vec_rp_c"+str(self.cycle))
        if self.vspace_l is not None:
            self.save_array_as_vectors(self.vspace_l, self.save_dir + "v_vec_l_c"+str(self.cycle))
        if self.vspace_lp is not None:
            self.save_array_as_vectors(self.vspace_lp, self.save_dir + "v_vec_lp_c"+str(self.cycle))
        if self.wspace_r is not None:
            self.save_array_as_vectors(self.wspace_r, self.save_dir + "w_vec_r_c"+str(self.cycle))
        if self.wspace_rp is not None:
            self.save_array_as_vectors(self.wspace_rp, self.save_dir + "w_vec_rp_c"+str(self.cycle))
        if self.wspace_l is not None:
            self.save_array_as_vectors(self.wspace_l, self.save_dir + "w_vec_l_c"+str(self.cycle))
        if self.wspace_lp is not None:
            self.save_array_as_vectors(self.wspace_lp, self.save_dir + "w_vec_lp_c"+str(self.cycle))

    @staticmethod
    def save_array_as_vectors(my_arr, name):
        for ii in range(my_arr.shape[1]):
            np.savetxt(name+"_i"+str(ii)+".txt", my_arr[:, ii])

    def get_numpy_evals(self):
        evals, evecs = np.linalg.eig(self.mat_orig)
        np.set_printoptions(threshold=sys.maxsize)
        print("numpy evals =\n ", sorted(abs(np.real(evals)), reverse=False))
