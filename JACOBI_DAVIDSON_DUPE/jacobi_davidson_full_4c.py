import numpy as np
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils
import decimal
import scipy as sp
import scipy.linalg as sp_la
D = decimal.Decimal


class JacobiDavidsonFull4C(eps_solvers.Solver):

    def __init__(self, rs_filename, num_eigenvalues, threshold=1e-4, maxdim_subspace=6,
                 solver="Jacobi_Davidson", method="Full", symmetry="general", pe_rot=False):
        super().__init__(rs_filename, num_eigenvalues, threshold, maxdim_subspace, solver, method, symmetry,
                         pe_rot)

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
            self.evals_1e = sp.zeros(2*num_pos_evals, dtype=self.real_precision)
            self.evals_1e[:num_pos_evals] = evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = sp.zeros(num_pos_evals, dtype=self.real_precision)
            self.evals_1e = evals_1e_all[num_pos_evals:]

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = sp.ndarray((self.nocc, self.nvirt), dtype=self.real_precision)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = sp.reshape(self.esorted, self.nov)
        self.eindex = sp.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self, symmetry: object = 'general'):
        # Ritz values and vectors
        self.teta = sp.zeros(self.nev, dtype=self.complex_precision)
        self.u_vecs = sp.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.construct_guess(iev, symmetry)

        # Residual values for convergence checking
        self.r_vecs = sp.zeros((self.ndim, 2*self.nev), dtype=self.complex_precision)

    def main_loop(self):
        sp.set_printoptions(precision=16)
        it = 0
        while it <= 1000:
            print("\n\n=====================================================")
            print("cycle = ", self.cycle, "it = ", it)
            print("=====================================================")

            # if it > self.maxs:
            if it > 40:
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            if self.cycle > 2:
                sys.exit("got to cycle 3, exit")

            for iev in range(self.nev):
                self.iev = iev
                self.extend_right_handed_spaces(iev)
                if self.cycle > 1:
                    self.extend_left_handed_spaces(iev)
                it = it + 1

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
                print("Final eigenvalues = ", self.real_precision(self.teta[:self.nev]))
                sys.exit("Converged!!")

            self.cycle = self.cycle + 1

    def extend_right_handed_spaces(self, iev):

        if self.cycle == 1:
            t_vec = self.u_vecs[:, iev]
        else:
            t_vec = self.get_new_tvec(iev)
            sp.savetxt("t_vec_preorth_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", t_vec)
            for ii in range(self.vspace_r.shape[1]):
                t_vec = utils.rs_self__orthogonalize(t_vec, self.vspace_r[:, ii])

                #   t_vec, good_t_vec = utils.orthogonalize_v1_against_v2(t_vec, self.vspace_r[:, ii])
                good_t_vec = True
            #   if not good_t_vec:
            #       sys.exit("Cannot orthonormalize new t_vec with vspace_r! Aborting! \n"
            #                "you must at least able to extend the right hand space by 1 vector")

            #if self.vspace_rp is not None:
                #for ii in range(self.vspace_rp.shape[1]):
                #    t_vec, good_t_vec = utils.orthogonalize_v1_against_v2(t_vec, self.vspace_rp[:, ii])
                #    if not good_t_vec:
                #        sys.exit("Cannot orthonormalize new t_vec with vspace_rp! Aborting! \n"
                #                 "you must at least able to extend the right hand space by 1 vector")

        t_vec = t_vec/sp_la.norm(t_vec)
        sp.savetxt("t_vec_postorth_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", t_vec)

        # Get coefficients for symmetrization
        d1, d2 = self.orthonormalize_pair(t_vec)
        # from t_vec = [Y, X]  get t_vec_pair = [ Y*, X* ]
        t_vec_pair = self.get_pair('x', t_vec)
        sp.savetxt("t_vec_pair_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", t_vec_pair)
        t_vec_final = d1 * t_vec + d2 * t_vec_pair
        sp.savetxt("t_vec_final_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", t_vec_final)
        print("\n")

        if self.vspace_r is None:
            self.vspace_r = sp.ndarray((self.ndim, 1), self.complex_precision)
            self.vspace_r[:, 0] = d1 * t_vec + d2 * t_vec_pair
            self.wspace_r = sp.ndarray((self.ndim, 1), self.complex_precision)
            self.wspace_r[:, 0] = self.sigma_constructor(self.vspace_r[:, 0])
        else:
            self.vspace_r = sp.c_[self.vspace_r, (d1 * t_vec + d2 * t_vec_pair)]
            self.wspace_r = sp.c_[self.wspace_r, self.sigma_constructor(self.vspace_r[:, -1])]

        # Build symmetrized t_vec using coeffs, and extend vspace and wspace
        if self.vspace_rp is None:
            self.vspace_rp = sp.ndarray((self.ndim, 1), self.complex_precision)
            self.vspace_rp[:, 0] = self.get_pair('x', self.vspace_r[:, 0])
            self.wspace_rp = sp.ndarray((self.ndim, 1), self.complex_precision)
            self.wspace_rp[:, 0] = self.get_pair('Ax', self.wspace_r[:, 0])
        else:
            self.vspace_rp = sp.c_[self.vspace_rp, self.get_pair('x', self.vspace_r[:, -1])]
            self.wspace_rp = sp.c_[self.wspace_rp, self.get_pair('Ax', self.wspace_r[:, -1])]

        self.zero_check_and_save_rh()

    def extend_left_handed_spaces(self,it):
        # good_t_vec is only true if the left eigenvector just generated has as sufficiently large component
        # which is orthogonal to the space spanned by the right eigenvectors
        t_vec_l = self.get_left_evec(self.vspace_r[:, -1])
    #    t_vec_l, good_t_vec = utils.orthogonalize_v1_against_v2(t_vec_l, self.vspace_r[:, -1])
        for ii in range(self.vspace_r.shape[1]):
            t_vec_l = utils.rs_self__orthogonalize(t_vec_l, self.vspace_r[:, ii])

        t_vec_l_norm = sp_la.norm(t_vec_l)
        if t_vec_l_norm < 1e-8:
            good_t_vec = False
        else :
            t_vec_l = t_vec_l / t_vec_l_norm
            good_t_vec = True

        sp.savetxt("t_vec_lp_c" + str(self.cycle) + "_i", t_vec_l)

        if good_t_vec:
            if self.vspace_l is None:
                self.vspace_l = sp.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_l[:, 0] = t_vec_l

                self.wspace_l = sp.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_l[:, 0] = self.sigma_constructor(self.vspace_r[:, 0])

                self.vspace_lp = sp.ndarray((self.ndim, 1), self.complex_precision)
                self.vspace_lp[:, 0] = self.get_pair('x', self.vspace_r[:, 0])

                self.wspace_lp = sp.ndarray((self.ndim, 1), self.complex_precision)
                self.wspace_lp[:, 0] = self.get_pair('Ax', self.wspace_r[:, 0])
            else:
                self.vspace_l = sp.c_[self.vspace_l, t_vec_l]
                self.vspace_lp = sp.c_[self.vspace_lp, self.get_pair('x', self.vspace_l[:, -1])]
                self.wspace_l = sp.c_[self.wspace_l, self.sigma_constructor(self.vspace_l[:, -1])]
                self.wspace_lp = sp.c_[self.wspace_lp, self.get_pair('Ax', self.wspace_l[:, -1])]

            self.zero_check_and_save_lh()

    # Ugly, but will slowly swap out parts for more sensible approach
    def build_subspace_matrix(self):
        sb_dim = 0
        for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
            if vs is not None:
                sb_dim = sb_dim + vs.shape[1]
        submat = sp.zeros((sb_dim, sb_dim), self.complex_precision)

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
                                submat[vi+ii, wj+jj] = sp.vdot(vs[:, ii], ws[:, jj])
                            wj = wj + ws.shape[1]
                vi = vi + vs.shape[1]
        return submat

    @staticmethod
    # [X,Y] --> [X, -Y]
    def get_left_evec(vec):
        buff = sp.empty_like(vec)
        buff[:int(len(vec)/2)] = vec[:int(len(vec)/2)]
        buff[int(len(vec)/2):] = -vec[int(len(vec)/2):]
        return buff

    def get_residual_vectors(self, submat):
        # Ritz values and Ritz vectors defined in trial vspace
        sp.savetxt("submat" + str(submat.shape[0]) + ".txt", self.real_precision(submat))
        ritz_vals, ritz_vecs_l, ritz_vecs = sp_la.eig(submat, left=True)

        # ordering eigenvalues and eigenvectors so first set of evals are positive and run in ascending order
        # (eigenvalues in the spectrum are always in positive and negative pairs)
        idx = ritz_vals.argsort()
        t2 = int(len(ritz_vals)/2)
        for ii in range(t2):
            idx[ii], idx[ii+t2] = idx[ii+t2], idx[ii]

        print("ritz_vals orig = ", ritz_vals)
        ritz_vals = ritz_vals[idx]
        ritz_vecs = ritz_vecs[:, idx]
        print("ritz_vals sorted = ", ritz_vals)

        # Construction of Ritz vectors from eigenvectors
        self.u_vecs = sp.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            vi = 0
            for vs in [self.vspace_r, self.vspace_l, self.vspace_rp, self.vspace_lp]:
                if vs is not None:
                    for ii in range(vs.shape[1]):
                        self.u_vecs[:, iev] = self.u_vecs[:, iev] + ritz_vecs[ii+vi, iev] * vs[:, ii]
                    vi = vi + vs.shape[1]

        # Construction of u_hat from Ritz_vectors and w_spaces,
        dnorm = sp.zeros(self.nev, self.complex_precision)
        self.r_vecs = sp.zeros((self.ndim, self.nev), self.complex_precision)
        for iev in range(self.nev):
            wj = 0
            u_hat = sp.zeros(self.ndim, self.complex_precision)
            for ws in [self.wspace_r, self.wspace_l, self.wspace_rp, self.wspace_lp]:
                if ws is not None:
                    for jj in range(ws.shape[1]):
                        u_hat = u_hat + ritz_vecs[jj+wj, iev] * ws[:, jj]
                    wj = wj + ws.shape[1]

            # Calculation of residual vectors, and residual norms
            self.r_vecs[:, iev] = u_hat - self.u_vecs[:, iev] * ritz_vals[iev]
            dnorm[iev] = sp_la.norm(self.r_vecs[:, iev])

        print("dnorm = ", dnorm)
        self.teta = ritz_vals[:self.nev]
        return dnorm

    # Construct initial guess
    def construct_guess(self, iguess, symmetry_type):
        guess = sp.zeros(self.nov, dtype=self.complex_precision)

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

            sqm1 = self.complex_precision(0.0 + 1.0j)
            sq2r = self.complex_precision(self.real_precision(1.0)/sp.sqrt(self.real_precision(2.0)) + 0.0j)

            if component == 1:
                guess[i0] = sq2r
                guess[i3] = sq2r
            elif component == 2:
                guess[i0] = sq2r*sqm1
                guess[i3] = -sq2r*sqm1
            elif component == 3:
                guess[i1] = sq2r
                guess[i2] = -sq2r
            else:
                guess[i1] = sqm1*sq2r
                guess[i2] = - sqm1*sq2r
        else:
            print("Symmetry type ", symmetry_type, " not implemented")

        return guess

    # Construct pair according to symmetry relations
    # x :  [X,Y] --> [ Y*, X* ]
    # Ax : [X,Y] --> [ -Y*, -X* ]
    def get_pair(self, pair_type, vec_in):
        vec_out = sp.empty(self.ndim, dtype=self.complex_precision)
        n2 = int(self.ndim/2)
        if pair_type == 'x':
            vec_out[n2:] = sp.conj(vec_in[:n2])
            vec_out[:n2] = sp.conj(vec_in[n2:])
        elif pair_type == 'Ax':
            vec_out[n2:] = -vec_in[:n2].conj()
            vec_out[:n2] = -vec_in[n2:].conj()
        else:
            sys.exit("ABORTING!! Unknown pair_type specified in eigenvector construction")

        return vec_out

    # This restructures the isput vector so that it's "pair" as constructed using get_pair
    # will also be orthogonal to the vector space
    def orthonormalize_pair(self, vec):
        d1 = self.complex_precision(0.0 + 0.0j)
        d2 = self.complex_precision(0.0 + 0.0j)
        z = self.complex_precision(0.0 + 0.0j)
        nh = int(self.ndim/2)
        sp.savetxt("t_vec_op_i"+str(self.iev)+"_c"+str(self.cycle)+".txt", vec)
        for ii in range(nh):
            z = z + vec[ii] * vec[ii+nh]

        # z = sp.dot(vec[:int(self.ndim/2)], vec[int(self.ndim/2):])
        d = self.real_precision(z * np.conj(z))
        r = self.real_precision(1.0) - self.real_precision(4.0) * d
        print("determining t_vec coefficients")
        print("z = ", z)
        print("d = ", d)
        print("r = ", r)

        if r >= 1e-12:
            if np.abs(d) > 1e-12:
                t = self.real_precision(1.0) + sp.sqrt(r)
                print("t = ", t)

                print("sp.sqrt(d) = ", sp.sqrt(d))
                print("sp.real(z) = ", sp.real(z))
                print("sp.imag(z) = ", sp.imag(z))
                r1 = sp.real(z)/sp.sqrt(d)
                r2 = sp.imag(z)/sp.sqrt(d)

                print("r1 = ", r1)
                print("r2 = ", r2)

                d1 = self.complex_precision(-r1 * sp.sqrt(self.real_precision(0.5)*t/r) + r2*sp.sqrt(self.real_precision(0.5)*t/r)*1.0j)
                d2 = self.complex_precision(sp.sqrt(self.real_precision(2.0)*d/(t*r)) + 0.0j)
            else:
                print("WARNING!, small d in orthonormalize pair routine")
                d1 = self.complex_precision(1.0 + 0.0j)
                d2 = self.complex_precision(0.0 + 0.0j)
        else:
            print("WARNING!, small r in orthonormalize pair routine")

        print("d1, d2 = ", d1, d2)
        return d1, d2

    def get_new_tvec(self, iev):
        v1 = sp.ndarray(self.ndim, self.complex_precision)  # v1 = M^{-1}*r
        v2 = sp.ndarray(self.ndim, self.complex_precision)  # v2 = M^{-1}*u
        idx = 0

        # First half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = (self.evalai(ii, jj) - sp.real(self.teta[iev]))
                if np.abs(ediff) > 1e-8:
                    dtmp = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = self.complex_precision(self.real_precision(0.0) + self.real_precision(0.0j))
                    v2[idx] = self.complex_precision(self.real_precision(0.0) + self.real_precision(0.0j))
                idx += 1

        # Second half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = -self.evalai(ii, jj) - sp.real(self.teta[iev])
                if abs(ediff) > 1e-8:
                    dtmp = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * dtmp
                    v2[idx] = self.u_vecs[idx, iev] * dtmp
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = self.complex_precision(self.real_precision(0.0) + self.real_precision(0.0j))
                    v2[idx] = self.complex_precision(self.real_precision(0.0) + self.real_precision(0.0j))
                idx += 1

        sp.savetxt("v1_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", v1)
        sp.savetxt("v2_c" + str(self.cycle) + "_i" + str(iev + 1) + ".txt", v2)

        u_m1_u = sp.vdot(self.u_vecs[:, iev], v2)
        print("u_m1_u = ", u_m1_u)
        if abs(u_m1_u) > 1e-8:
            u_m1_r = sp.vdot(self.u_vecs[:, iev], v1)
            print("u_m1_r = ", u_m1_r)
            factor = u_m1_r / sp.real(u_m1_u)
            return factor*v2-v1
        else:
            return -v1

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc + virt_orb] - self.evals_1e[occ_orb]

    def sigma_constructor(self, vec):
        return sp.matmul(self.mat_orig, vec)

    def zero_check_and_save_rh(self):
        utils.check_for_nans([self.vspace_r, self.vspace_rp, self.wspace_r, self.wspace_rp],
                             ["vspace_r", "vspace_rp", "wspace_r", "wspace_rp"])
        utils.zero_small_parts(self.vspace_r)
        utils.zero_small_parts(self.wspace_r)
        utils.zero_small_parts(self.vspace_rp)
        utils.zero_small_parts(self.wspace_rp)
        for ii in range(self.vspace_r.shape[1]):
            utils.save_arrs_to_file([self.vspace_r[:, ii], self.vspace_rp[:, ii], self.wspace_r[:, ii],
                                     self.wspace_rp[:, ii]],
                                    ["vspace_r_c" + str(self.cycle) + "_i" + str(ii + 1),
                                     "vspace_rp_c" + str(self.cycle) + "_i" + str(ii + 1),
                                     "wspace_r_c" + str(self.cycle) + "_i" + str(ii + 1),
                                     "wspace_rp_c" + str(self.cycle) + "_i" + str(ii + 1)])

    def zero_check_and_save_lh(self):
        utils.check_for_nans([self.vspace_l, self.vspace_lp, self.wspace_l, self.wspace_lp],
                             ["vspace_l", "vspace_lp", "wspace_l", "wspace_lp"])
        utils.zero_small_parts(self.vspace_l)
        utils.zero_small_parts(self.wspace_l)
        utils.zero_small_parts(self.vspace_lp)
        utils.zero_small_parts(self.wspace_lp)
        for ii in range(self.vspace_l.shape[1]):
            utils.save_arrs_to_file([self.vspace_l[:, ii], self.vspace_lp[:, ii], self.wspace_l[:, ii],
                                     self.wspace_lp[:, ii]],
                                    ["vspace_l_c" + str(self.cycle) + "_i" + str(ii+1),
                                     "vspace_lp_c" + str(self.cycle) + "_i" + str(ii+1),
                                     "wspace_l_c" + str(self.cycle) + "_i" + str(ii+1),
                                     "wspace_lp_c" + str(self.cycle) + "_i" + str(ii+1)])

    def get_numpy_evals(self):

        evals, evecs_l, evecs = sp_la.eig(self.mat_orig, left=True)
        sp.set_printoptions(threshold=sys.maxsize)
        print("numpy evals =\n ", sorted(abs(sp.real(evals)), reverse=False))
