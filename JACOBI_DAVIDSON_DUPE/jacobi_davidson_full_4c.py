import numpy as np
import numpy.linalg as la
import sys
import eps_solvers
import mat_reader as mr
import matrix_utils as utils

class JacobiDavidsonFull4C(eps_solvers.Solver):

    def initialize(self):

        print ("self.maxs = ", self.maxs)
        if self.pe_rot:
            self.nvirt = self.ndims - self.nocc
        else:
            self.nvirt = int(self.ndims / 2) - self.nocc

        self.nov = self.nvirt * self.nocc
        self.ndim = 2*self.nov
        print("self.ndim = ", self.ndim )

        self.read_1e_eigvals_and_eigvecs(seedname="/home/peter/CALCS/RS_TESTS/TDDFT-os/4C/FULL/RS_FILES/KEEPERS/1el_eigvals")
        self.get_esorted_general()

        self.u_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        for iev in range(self.nev):
            self.u_vecs[self.eindex[iev], iev] = 1.0 + 0.0j

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

        # temporary variables, not really needed here
        self.t_vec = np.zeros(self.ndim, dtype=np.complex64)
        self.w_tmp = np.zeros_like(self.t_vec)
        self.v_tmp = np.zeros_like(self.t_vec)

        # Guess vectors, residuals, etc.,
        self.vspace = np.zeros((self.ndim, self.maxs), dtype=np.complex64)
        self.wspace = np.zeros_like(self.vspace)

        self.r_vecs = np.zeros((self.ndim, self.nev), dtype=np.complex64)
        self.u_hats = np.zeros_like(self.r_vecs)

        # Subspace Hamiltonian
        self.submat = np.empty((2*self.maxs, 2*self.maxs), dtype=np.complex64)
        self.teta = np.zeros(self.nev, dtype=np.complex64)

    def main_loop(self):
        self.maxs = 13
        it = 0
        while it < self.maxs:

            if (it + 2*self.nev) > self.maxs :
                sys.exit("Exceeded maximum number of iterations. ABORTING!")

            for iev in range(self.nev):
                np.savetxt("TXTS/u_vecs_"+str(it)+"_"+str(self.nev)+".txt", self.u_vecs[:, iev] )

                if self.skip[iev]:
                    continue

                if it < 2*self.nev:
                    self.t_vec = self.u_vecs[:, iev]
                else:
                    self.t_vec = self.get_t_vec(iev)

                for ii in range(it-1):
                    v_tmp = self.get_pair('x', self.t_vec)
                    np.savetxt("TXTS/v_tmp" + "_it" + str(it) + "_ii" + str(ii)+".txt", v_tmp )
                    print("||v_tmp|| = ", np.linalg.norm(v_tmp))

                    self.t_vec, vtangle1 = utils.orthonormalize_v_against_mat_check(self.t_vec, self.vspace)
                    print("||v_tmp_o1|| = ", np.linalg.norm(self.t_vec))
                    np.savetxt("TXTS/t_vec_o1" + "_it" + str(it) + "_ii" + str(ii) + ".txt", self.t_vec)

                    self.t_vec, vtangle2 = utils.orthogonalize_v1_against_v2(self.t_vec, v_tmp)
                    print("||v_tmp_o2|| = ", np.linalg.norm(self.t_vec))
                    np.savetxt("TXTS/t_vec_o2" + "_it" + str(it) + "_ii" + str(ii) + ".txt", v_tmp)

                    if max(vtangle1, vtangle2) < 1e-10:
                        print("angle between new guess vector and current guess space is small!",
                              max(vtangle1, vtangle2))
                        continue

                v_tmp = self.get_pair('x', self.t_vec)
                #  self__add(ndim, t1, t_vec, t2, v_tmp)

                d1, d2 = self.orthonormalize_pair(self.t_vec)
                self.t_vec = d1*self.t_vec + d2*v_tmp
            it += 1

    # n =  size of t_vec, u and residual
    # teta : ritz value
    #
    def get_t_vec(self, iev):
        n = self.ndim
        t = self.u_vecs[:, iev]
        u = self.u_vecs[:, iev]
        r = np.zeros_like(u)
        teta = self.teta[iev]

        v1 = np.zeros_like(u)
        v2 = np.zeros_like(u)

        #Minv = np.zeros_like()

        #call self__preconditioning(n, teta, v1, r)   ! v1 = M ^ (-1) * r
        #call self__preconditioning(n, teta, v2, u)   ! v2 = M ^ (-1) * u

        e = np.dot(r, v1)
        c = np.dot(u, v2)
        print( "uMinvr = ", e)
        print( "uMinvu = ", c)

        if (abs(np.real(c)) < 1.0e-8):
            e = 0.0 + 0.0j
        else:
            e = e / np.real(c)

        #call self__sum(n, t, e, v2, -dcmplx(1.0,0.0), v1)

        #call memrm('self__get_t', 'v1', v1) deallocate(v1)
        #call memrm('self__get_t', 'v2', v2); deallocate(v2)

        # in case of  time - reversal symmetry trial vector should be t - symmetric
        #select case(trim(self__P0_tsymm))
        #case('symmetric');
        #self__make_mo_ts(t, n)
        #end select

        print("get_t_vec NOT IMPLEMENTED!!")
        return t

    def get_pair(self, pair_type, vec_in):

        vec_out = np.empty(self.ndim, dtype=np.complex64)
        n2 = int(self.ndim/2)
        print ("n2 = ", n2)
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
                r2 = np.real(z)/np.sqrt(d)
                d1 = -r1 * np.sqrt(0.5*t/r) + r2*np.sqrt(0.5*t/r)*1.0j
                d2 = np.sqrt(2.0*d/(t*r)) + 0.0j
            else:
                print("WARNING!, small d in orthonormalize pair routine")
                d1 = 1.0 + 0.0j
                d2 = 0.0 + 0.0j
        else:
            print("WARNING!, small r in orthonormalize pair routine")

        return d1, d2
