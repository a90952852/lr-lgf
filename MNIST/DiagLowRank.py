import numpy as np
import jax.numpy as jnp
import jax
from scipy.sparse.linalg import svds
from scipy.linalg import svd
import scipy
import matplotlib.pyplot as plt


class Diag_LowRank(object):

    def __init__(self, shape_weights,num_classes):
        self.mPi = jnp.zeros((shape_weights,1))
        self.Pi_t = [jnp.ones((1, shape_weights)), jnp.zeros((shape_weights,num_classes)), jnp.zeros((num_classes,num_classes))]

    def add_diag(self, D):
        self.Pi_t[0] += D
        # update the SVD of the whole thing

    def block_matrix(self, C, H):
        b, c, _ = C.shape

        # Create block matrices using broadcasting
        upper_block = jnp.concatenate([C, jnp.zeros_like(C)], axis=2)
        lower_block = jnp.concatenate([jnp.zeros_like(H), H], axis=2)

        return jnp.concatenate([upper_block, lower_block], axis=1)

    def add_low(self, J, H):
        U = self.Pi_t[1]
        C = self.Pi_t[2]
        left_vec = np.zeros((J.shape[2], J.shape[0]+1, C.shape[0])) # d, b, c
        C_12 = scipy.linalg.sqrtm(C)
        left_vec[:,0,:] = U @ C_12
        for b in range(J.shape[0]):
            H_12_b = scipy.linalg.sqrtm(H[b]+1e-4*jnp.eye(H.shape[1]))
            left_vec[:,b+1,:J.shape[1]] = (1/np.sqrt(J.shape[0]))*J[b].T @ H_12_b
        new_Ul, new_Sl, new_Vl = self.truncated_svd(left_vec, m=C.shape[0])


        self.Pi_t[1], self.Pi_t[2] = new_Ul, jnp.diag(new_Sl**2)


    def truncated_svd(self, M, m):
        M_reshaped = M.reshape(M.shape[0], M.shape[1] * M.shape[2])
        U, s, Vt = svds(np.array(M_reshaped) , k=m)
        #U_exact, s_exac, Vt_exact = svd(np.array(M_reshaped), full_matrices=False)
        idx = np.argsort(s)[::-1]
        U = U[:, idx]
        s = s[idx]
        Vt = Vt[idx, :]
        return U, s, Vt

    def compute_inv_diag(self, D):
        return jnp.diag(1.0 / jnp.diag(D))

    def compute_inv_sum_diag_dlr(self, Q, Pts=None, Pms=None, t=None):
        A_1 = (1/(self.Pi_t[0]))
        L = (1/(Q + A_1))
        Up = A_1.T * self.Pi_t[1]
        Cp = jnp.diag(1/jnp.diag(self.Pi_t[2])) + (self.Pi_t[1].T @ Up)
        left = L.T * Up
        mid = (Cp - Up.T @ left)
        mid_inv = jnp.linalg.pinv(mid)
        self.Pi_t = [L, left, mid_inv]
        if Pts != None:
            Pts.append([A_1, Up, jnp.linalg.pinv(Cp)])
            Pms.append([A_1+Q, Up, jnp.linalg.pinv(Cp)])
            return Pts, Pms



    def update_mPi(self, theta_star):
        self.mPi = theta_star[None,:] * self.Pi_t[0]
        temp1 = theta_star[None, :] @ self.Pi_t[1]
        temp2 = temp1 @ self.Pi_t[2]
        self.mPi += (temp2 @ self.Pi_t[1].T)
        self.mPi = self.mPi.T


