import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils


class JacobiDavidsonFull4C(eps_solvers.Solver):

    def main_loop(self):
        # v_space[:,0:maxs2] are original vectors, v_space[:,maxs2:maxs] are symmetric pairs
        # v_space[:, ii ] are right eigvecs if i is odd, and left eigenvectors if ii is even
        maxs2 = int(self.maxs/2)
        it = 0
        print("maxs2 = ", maxs2)
        print("nev = ", self.nev)
        evals, evecs = np.linalg.eig(self.mat_orig)
        np.set_printoptions(threshold=sys.maxsize)
        print("numpy evals =\n ", sorted(abs(np.real(evals)), reverse=False))


        while it < 3:
            print("it = ", it)

            if it > self.maxs :
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            for iev in range(self.nev):
               # if self.skip[iev]:
               #     continue

                utils.print_nonzero_numpy_elems(self.u_vecs, arr_name = "u_vecs")
                if it < self.nev:
                    t_vec = self.u_vecs[:, iev]
                else :
                    t_vec = self.get_new_tvec(iev)

                # from t_vec = [Y, X]  get t_vec_pair = [ Y*, X* ]
                t_vec_pair = self.get_pair('x', t_vec)

                utils.print_nonzero_numpy_elems(t_vec, arr_name="t_vec")
                utils.print_nonzero_numpy_elems(t_vec_pair, arr_name="t_vec_pair")

                # Get coefficients for symmetrization
                d1, d2 = self.orthonormalize_pair(t_vec)

                # Build symmetrized t_vec using coeffs, and extend vspace and wspace
                self.vspace[:, it] = d1*t_vec + d2*t_vec_pair
                self.wspace[:, it] = self.sigma_constructor(self.vspace[:, it])
                self.vspace[:, it + maxs2] = self.get_pair('x', self.vspace[:, it])
                self.wspace[:, it + maxs2] = self.get_pair('Ax', self.wspace[:, it])
                utils.zero_small_parts(self.vspace)
                utils.zero_small_parts(self.wspace)
                np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/wspace" + str(it) + ".txt", self.wspace[:, it])
                np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/vspace" + str(it) + ".txt", self.vspace[:, it])
                np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/wspace" + str(it) + ".txt", self.wspace[:, it + maxs2])
                np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/vspace" + str(it) + ".txt", self.vspace[:, it + maxs2])


                it += 1
                # Now build left eigenvectors
                t_vec = self.vspace[:, it-1]
                t_vec = self.get_left_evec(t_vec)
                t_vec, t_angle = utils.orthogonalize_v1_against_v2(t_vec, self.vspace[:, it-1])
                self.vspace[:, it] = t_vec
                self.wspace[:, it] = self.sigma_constructor(t_vec)
                self.vspace[:, it + maxs2] = self.get_pair('x', self.vspace[:, it])
                self.wspace[:, it + maxs2] = self.get_pair('Ax', self.wspace[:, it])

            # Build subspace matrix : v*Av = v*w
            submat = np.zeros((2*it, 2*it), np.complex64)
            for ii in range(2*it):
                for jj in range(2*it):
                    submat[ii, jj] = np.vdot(self.vspace[:, ii], self.wspace[:, jj])

            utils.zero_small_parts(submat)
            utils.zero_small_parts(self.vspace)
            utils.zero_small_parts(self.wspace)
            np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/wspace"+str(it)+".txt", self.wspace)
            np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/vspace"+str(it)+".txt", self.vspace)
            np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/submat" + str(it) + ".txt", submat)

#           print("submat = \n", np.real(submat))
            theta, hevecs = np.linalg.eig(submat)
            print("theta = ", theta)
            utils.print_nonzero_numpy_elems(hevecs, arr_name="hevecs")


            self.r_vecs = np.zeros((self.ndim, self.nev), np.complex64)
            u_hat = np.zeros(self.ndim, np.complex64)
            dnorm = np.zeros(self.nev, np.complex64)
            for iev in range(iev):
                for ii in range(it):
                    self.u_vecs[:, iev] = self.u_vecs[:, iev] + hevecs[iev, ii]*self.vspace[:, ii]
                    u_hat = u_hat + hevecs[iev, ii] * self.wspace[:, ii]
                self.r_vecs[:, iev] = u_hat - self.u_vecs[:, iev]*theta[iev]
                dnorm[iev] = np.linalg.norm(self.r_vecs[:, iev])
            utils.print_nonzero_numpy_elems(self.u_vecs, arr_name="u_vecs")
            #utils.print_nonzero_numpy_elems(u_hat, arr_name="u_hat")
            #utils.print_nonzero_numpy_elems(self.r_vecs, arr_name="r_vecs")

            print("dnorm = ", dnorm)

    def get_left_evec(self, vec):
        vec[int(len(vec)/2):] = vec[int(len(vec)/2):]
        return vec

    def initialize(self, symmetry_type = '4c'):

        print("self.maxs = ", self.maxs)
        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2*self.nov
        print("self.ndim = ", self.ndim )

        self.read_1e_eigvals_and_eigvecs(seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        self.get_esorted_general()

        self.u_vecs = np.zeros((self.ndim, self.nev), np.complex64)
        for iev in range(self.nev):
            self.u_vecs[:self.nov, iev] = self.construct_guess(iev, 'general')

    def get_esorted_general(self):
        # Build sorted list of eigval differences without imposing any symmetry constraints
        self.esorted = np.ndarray((self.nocc, self.nvirt), dtype=np.float64)
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                self.esorted[ii, jj] = self.evalai(ii, jj)

        self.esorted = np.reshape(self.esorted, self.nov)
        self.eindex = np.argsort(self.esorted)
        self.esorted = self.esorted[self.eindex]

    def evalai(self, occ_orb, virt_orb):
        return self.evals_1e[self.nocc+virt_orb] - self.evals_1e[occ_orb]

    def read_1e_eigvals_and_eigvecs(self, seedname):
        evals_1e_all = mr.read_fortran_array(seedname)
        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_orig.txt", evals_1e_all)

        num_pos_evals = self.nvirt + self.nocc
        print("num_pos_evals = ", num_pos_evals)
        if self.pe_rot:
            self.evals_1e = np.zeros(2*num_pos_evals, dtype=np.float64)
            self.evals_1e[:num_pos_evals] = evals_1e_all[num_pos_evals:]
            self.evals_1e[num_pos_evals:] = evals_1e_all[:num_pos_evals]
        else:
            self.evals_1e = np.zeros(num_pos_evals, dtype=np.float64)
            self.evals_1e = evals_1e_all[num_pos_evals:]

        np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/evals_post.txt", self.evals_1e)

    def solve(self):
        self.initialize_first_iteration()
        self.main_loop()

    def initialize_first_iteration(self):
        # for convergence checking
        self.skip = np.full(self.nev, False)
        self.dnorm = np.zeros(self.nev, dtype=np.float32)

        # Guess vectors, residuals, etc.,
        self.vspace = np.zeros((self.ndim, 4*self.maxs), dtype=np.complex64)
        self.wspace = np.zeros_like(self.vspace)
        self.r_vecs = np.zeros((self.ndim, 2*self.nev), dtype=np.complex64)

        # Subspace Hamiltonian
        self.teta = np.zeros(self.nev, dtype=np.complex64)

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
                guess[i1] = 0.0 + 1.0j/ np.sqrt(2.0)
                guess[i2] = 0.0 - 1.0j/ np.sqrt(2.0)
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
        r = 1.0 - 4.0 *d
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
        teta_iev = self.teta[iev]
        idx = 0
        # First half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = (self.evalai(ii, jj) - teta_iev)
                if abs(ediff) > 1e-8:
                    ediff = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * ediff
                    v2[idx] = self.u_vecs[idx, iev] * ediff
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = 0.0 + 0.0j
                    v2[idx] = 0.0 + 0.0j
                idx += 1
        # Second half of vector
        for ii in range(self.nocc):
            for jj in range(self.nvirt):
                ediff = -(self.evalai(ii, jj) + teta_iev)
                if abs(ediff) > 1e-8:
                    ediff = 1 / ediff
                    v1[idx] = self.r_vecs[idx, iev] * ediff
                    v2[idx] = self.u_vecs[idx, iev] * ediff
                else:
                    print("Warning, (E_{a}-E_{i})<1e-8")
                    v1[idx] = 0.0 + 0.0j
                    v2[idx] = 0.0 + 0.0j
                idx += 1
        e = np.vdot(self.u_vecs[:, iev], v1)
        c = np.vdot(self.u_vecs[:, iev], v2)
        print("uMinvr = ", e )
        print("uMinvu = ", c )
        e = e/np.real(c)

        return v2 - e*v1

    def sigma_constructor(self, vec):
        #np.savetxt("/home/peter/MAT_TOOLS/JACOBI_DAVIDSON_DUPE/mat_orig.txt", self.mat_orig)
        return np.matmul(self.mat_orig,vec)