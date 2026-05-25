# ----------------------------------------------------------------------------------------------------------------
# This file contains the code supplementary to the paper:
# Robust Moment-Based Estimation via Spectral Gradient Reweighting

# Copyright (c) 2026 Liu Zhang

# SPDX-License-Identifier: GPL-3.0-only

# This program is free software: you can redistribute it and/or modify it under the terms of 
# the GNU General Public License as published by the Free Software Foundation, version 3.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; 
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. 
# See the GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along with this program. 
# If not, see <https://www.gnu.org/licenses/>. 

# Author: Liu Zhang (lz1619@princeton.edu), Program in Applied and Computational Mathematics, Princeton University
# Last update date: 2026-05-25
# ----------------------------------------------------------------------------------------------------------------

import numpy as np
import math
from functools import reduce
from utils import symmetrization

# ---------------------------------
from scipy.special import logsumexp
from scipy import linalg


# -----------------------------------------------------------------------------
# The following line is added for DGMM to set seed for reproducibility: 
import random
SEED = 4142
random.seed(SEED)                  # Python built-in RNG
np.random.seed(SEED)               # NumPy global RNG (for any legacy code)
rng = np.random.default_rng(SEED)  # New NumPy Generator
# -----------------------------------------------------------------------------

class GaussianMixtureMoments(object):
 
    def __init__(self, pi, A, V, X, L = None, order_weights = None):
        self.pi = pi
        self.A = A
        self.V = V

        self.n_dim, self.n_components = self.A.shape
        _, self.rank, _ = self.V.shape
        
        self.covariances = np.zeros((self.n_dim, self.n_dim, self.n_components))
        for j in range(self.n_components):
            self.covariances[:, :, j] = self.V[:, :, j] @ self.V[:, :, j].T

        self.X = X
        _, self.n_points = self.X.shape

        self.L = L
        if order_weights is None:
            self.order_weights = np.ones(self.L+1)
        else:
            self.order_weights = order_weights

        self.B = None
        self.B_n = None 
        self.K = None
        self.ViTVjVjTVi_p = None
        self.ViViTVjVjT_p_Ai = None
        self.ViViTVjVjT_p_ViViT_Aj = None
        self.AiT_VjVjTViViT_p = None
        self.AjT_ViViT_VjVjTViViT_p = None
        self.ATX = None
        self.XTVjVjTX = None

    def compute_logprob_Gaussian(self):
        logprob_Gaussian = np.empty((self.n_points, self.n_components))
        epsilon = 1e-7

        for j in range(self.n_components):
            try:
                cv_chol = linalg.cholesky(self.covariances[:,:,j], lower=True)
            except linalg.LinAlgError:
                cv_chol = linalg.cholesky(self.covariances[:,:,j] + epsilon * np.eye(self.n_dim), lower=True)
            
            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = linalg.solve_triangular(cv_chol, (self.X.T - self.A[:,j]).T, lower=True).T
            logprob_Gaussian[:, j] = - .5 * (np.sum(cv_sol ** 2, axis=1) + self.n_dim * np.log(2 * np.pi) + cv_log_det)

        return logprob_Gaussian
    
    def compute_logprob_GMM(self):
        logprob_Gaussian = (self.compute_logprob_Gaussian() + np.log(self.pi))
        logprob_GMM = logsumexp(logprob_Gaussian, axis=1)
        return logprob_GMM

    def compute_model_moments(self):
        M_model = [1.0]  
        for k in range(1, self.L+1):
            shape = (self.n_dim,) * k
            M_k = np.zeros(shape, dtype=float)

            for j in range(self.n_components):
                M_k += self.pi[0, j] * reduce(lambda a, b: np.multiply.outer(a, b), [self.A[:, j]] * k)

                for l in range(1, int(np.floor(k/2)+1)):
                    if k-2*l == 0:
                        tensor = reduce(lambda a, b: np.multiply.outer(a, b), [self.V[:, :, j] @ self.V[:, :, j].T] * l)
                    else: 
                        tensor_A = reduce(lambda a, b: np.multiply.outer(a, b), [self.A[:, j]] * (k-2*l))
                        tensor_V = reduce(lambda a, b: np.multiply.outer(a, b), [self.V[:, :, j] @ self.V[:, :, j].T] * l)
                        tensor = np.multiply.outer(tensor_A, tensor_V)
                    M_k += self.pi[0, j] * (math.factorial(k) / (math.factorial(k-2*l) * math.factorial(l) * (2 ** l))) * symmetrization(tensor)
            M_model.append(M_k)
        return M_model
    
    def _precomputation_for_cumulants(self):
        highest_matrix_power = int(self.L/2)

        if self.ViTVjVjTVi_p is None or self.ViViTVjVjT_p_Ai is None or self.ViViTVjVjT_p_ViViT_Aj is None:
            self.ViTVjVjTVi_p = np.ones((self.rank, self.rank, self.n_components, self.n_components, highest_matrix_power+1))
            self.ViViTVjVjT_p_Ai = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))
            self.ViViTVjVjT_p_ViViT_Aj = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))

            for j in range(self.n_components):
                for i in range(self.n_components): 
                    self.ViTVjVjTVi_p[:, :, i, j, 0] = np.identity(self.rank)
                    self.ViViTVjVjT_p_Ai[:, i, j, 0] = self.A[:,i]
                    self.ViViTVjVjT_p_ViViT_Aj[:, i, j, 0] = self.V[:,:,i] @ (self.V[:,:,i].T @ self.A[:, j])

                    for p in range(1, highest_matrix_power+1):
                        if p == 1:
                            self.ViTVjVjTVi_p[:, :, i, j, p] = (self.V[:,:,i].T @ self.V[:,:,j]) @ (self.V[:,:,j].T @ self.V[:,:,i])
                        if p >= 2: 
                            self.ViTVjVjTVi_p[:, :, i, j, p] = self.ViTVjVjTVi_p[:, :, i, j, p-1] @ self.ViTVjVjTVi_p[:, :, i, j, 1]

                        self.ViViTVjVjT_p_Ai[:, i, j, p] = self.V[:,:,i] @ (self.V[:,:,i].T @ (self.V[:,:,j] @ (self.V[:,:,j].T @ self.ViViTVjVjT_p_Ai[:, i, j, p-1])))
                        self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p] = self.V[:,:,i] @ (self.V[:,:,i].T @ (self.V[:,:,j] @ (self.V[:,:,j].T @ self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p-1])))
       

    def compute_cumulants(self):
        self.K = np.zeros((self.n_components, self.n_components, self.L+1))
        self.K[:, :, 0] = 1
        ATA = self.A.T @ self.A
        self.K[:, :, 1] = ATA   

        if self.ViViTVjVjT_p_Ai is None:
            self._precomputation_for_cumulants()
        
        for j in range(self.n_components):
            for i in range(self.n_components):
                for l in range(2, self.L+1):
                    if l % 2 == 0:
                        self.K[i, j, l] = math.factorial(l - 1) * np.trace(self.ViTVjVjTVi_p[:, :, i, j, int(l/2)]) + math.factorial(l)/2 * self.A[:,i].T @ (self.V[:,:,j] @ (self.V[:,:,j].T @ self.ViViTVjVjT_p_Ai[:, i, j, int((l-2)/2)])) + math.factorial(l)/2 * self.A[:,j].T @ self.ViViTVjVjT_p_ViViT_Aj[:, i, j, int((l-2)/2)]
                    
                    else:     
                        self.K[i, j, l] = math.factorial(l) * self.A[:,j].T @ self.ViViTVjVjT_p_Ai[:, i, j, int((l-1)/2)]


    def compute_bell_polynomials(self):
        self.B = np.empty((self.n_components, self.n_components, self.L+1), dtype=np.float64)
        if self.K is None:
            self.compute_cumulants()
        self.B[:, :, 0] = 1
        self.B[:, :, 1] = self.K[:, :, 1]
        for k in range(2, self.L+1):
            self.B[:, :, k] = np.sum([math.comb(k - 1, l) * self.B[:, :, l] * self.K[:, :, k - l] for l in range(k)], axis=0)
        return self.B
    
    def compute_inner_product_of_moment_moment(self):
        if self.B is None:
            self.compute_bell_polynomials()

        inner_product_of_moment_moment = np.zeros(self.L+1)
        W = np.outer(self.pi[0], self.pi[0])
        inner_product_of_moment_moment[1:] = np.tensordot(W, self.B[:, :, 1:], axes=([0, 1], [0, 1]))
        return inner_product_of_moment_moment


    def compute_inner_product_of_moment_vector(self, for_each_sample=False):
        if self.ATX is None or self.XTVjVjTX is None:
            self._precomputation_for_F2()

        B0 = np.ones((self.n_components, self.n_points), dtype=np.float64)
        B1 = self.ATX
        pi_vec = self.pi[0].reshape(self.n_components, 1)
        
        inner_product_of_moment_vector = np.empty(self.L+1, dtype=np.float64)
        inner_product_of_moment_vector[0] = np.sum(pi_vec * B0)  
        inner_product_of_moment_vector[1] = np.sum(pi_vec * B1)

        if not for_each_sample:
            for k in range(2, self.L+1):
                current = B1 * self.ATX + (k - 1) * B0 * self.XTVjVjTX
                inner_product_of_moment_vector[k] = np.sum(pi_vec * current)
                B0, B1 = B1, current

            return inner_product_of_moment_vector
        else:
            inner_product_of_moment_vector_n = np.empty((self.L+1, self.n_points), dtype=np.float64)
            inner_product_of_moment_vector_n[0, :] = np.sum(pi_vec * B0, axis=0)
            inner_product_of_moment_vector_n[1, :] = np.sum(pi_vec * B1, axis=0)

            for k in range(2, self.L+1):
                current = B1 * self.ATX + (k - 1) * B0 * self.XTVjVjTX
                inner_product_of_moment_vector[k] = np.sum(pi_vec * current)
                inner_product_of_moment_vector_n[k, :] = np.sum(pi_vec * current, axis=0)
                B0, B1 = B1, current

            return inner_product_of_moment_vector, inner_product_of_moment_vector_n
        
    def _precomputation_for_F1(self):
        highest_matrix_power = int(self.L /2)
        self.ViTVjVjTVi_p = np.ones((self.rank, self.rank, self.n_components, self.n_components, highest_matrix_power+1))
        self.ViViTVjVjT_p_Ai = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))  
        self.ViViTVjVjT_p_ViViT_Aj = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))
        self.AiT_VjVjTViViT_p = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))
        self.AjT_ViViT_VjVjTViViT_p = np.ones((self.n_dim, self.n_components, self.n_components, highest_matrix_power+1))


        for j in range(self.n_components):
            for i in range(self.n_components): 
                self.ViTVjVjTVi_p[:, :, i, j, 0] = np.identity(self.rank)

                self.ViViTVjVjT_p_Ai[:, i, j, 0] = self.A[:,i]
                self.AiT_VjVjTViViT_p[:, i, j, 0] = self.A[:,i]

                self.ViViTVjVjT_p_ViViT_Aj[:, i, j, 0] = self.V[:,:,i] @ (self.V[:,:,i].T @ self.A[:, j])
                self.AjT_ViViT_VjVjTViViT_p[:, i, j, 0] = (self.A[:, j].T @ self.V[:,:,i]) @ self.V[:,:,i].T 

                for p in range(1, highest_matrix_power+1):   
                    if p == 1:
                        self.ViTVjVjTVi_p[:, :, i, j, p] = (self.V[:,:,i].T @ self.V[:,:,j]) @ (self.V[:,:,j].T @ self.V[:,:,i])
                    if p >= 2: 
                        self.ViTVjVjTVi_p[:, :, i, j, p] = self.ViTVjVjTVi_p[:, :, i, j, p-1] @ self.ViTVjVjTVi_p[:, :, i, j, 1]

                    self.ViViTVjVjT_p_Ai[:, i, j, p] = self.V[:,:,i] @ (self.V[:,:,i].T @ (self.V[:,:,j] @ (self.V[:,:,j].T @ self.ViViTVjVjT_p_Ai[:, i, j, p-1])))
                    self.AiT_VjVjTViViT_p[:, i, j, p] = (((self.AiT_VjVjTViViT_p[:, i, j, p-1] @ self.V[:,:,j]) @ self.V[:,:,j].T) @ self.V[:,:,i]) @ self.V[:,:,i].T

                    self.AjT_ViViT_VjVjTViViT_p[:, i, j, p] = (((self.AjT_ViViT_VjVjTViViT_p[:, i, j, p-1] @ self.V[:,:,j]) @ self.V[:,:,j].T) @ self.V[:,:,i]) @ self.V[:,:,i].T
                    self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p] = self.V[:,:,i] @ (self.V[:,:,i].T @ (self.V[:,:,j] @ (self.V[:,:,j].T @ self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p-1])))
      

    def _precomputation_for_F2(self, noise_covariance=None):
        self.ATX = self.A.T @ self.X 
        self.XTVjVjTX = np.empty((self.n_components, self.n_points), dtype=np.float64)
        for j in range(self.n_components):
            cov_j = self.V[:, :, j] @ self.V[:, :, j].T
            if noise_covariance is not None:
                cov_j = cov_j + noise_covariance
            w = cov_j @ self.X  
            self.XTVjVjTX[j, :] = np.sum(self.X * w, axis=0)

    
    def _precompute(self, noise_covariance=None):
        self._precomputation_for_F1()
        self._precomputation_for_F2(noise_covariance=noise_covariance)
        self.compute_bell_polynomials()

    
    def _compute_F1(self):
        if self.B is None:
            self.compute_bell_polynomials()
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)

        inner_product_of_moment_moment = self.compute_inner_product_of_moment_moment()
        F1 = self.order_weights.reshape(1, -1) @ inner_product_of_moment_moment.reshape(-1, 1)

        return float(np.asarray(F1).squeeze())


    def _compute_grad_F1(self):
        grad_F1_wrt_pi = np.zeros((1, self.n_components))
        grad_F1_wrt_A = np.zeros((self.n_dim, self.n_components))
        grad_F1_wrt_V = np.zeros((self.n_dim, self.rank, self.n_components))
            
        if self.ViTVjVjTVi_p is None or self.ViViTVjVjT_p_Ai is None or self.ViViTVjVjT_p_ViViT_Aj is None:
            self._precomputation_for_F1()
        if self.B is None:
            self.compute_bell_polynomials()
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)

        for k in range(1, self.L + 1):
            grad_F1_wrt_pi += 2 * self.order_weights[k] * self.pi @ self.B[:, :, k]
        
        for j in range(self.n_components):
            for i in range(self.n_components):                
                for k in range(1, self.L+1):
                    for l in range(1, k+1):
                        if l == 1:
                            grad_F1_wrt_A[:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * (self.A[:,i].reshape((self.n_dim, 1)))[:,0]

                        elif l % 2 == 0:
                            grad_F1_wrt_A[:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * self.ViViTVjVjT_p_ViViT_Aj[:, i, j, int((l-2)/2)]
                            
                            for p in range(int((l-2)/2) +1):
                                grad_F1_wrt_V[:,:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * self.ViViTVjVjT_p_Ai[:, i, j, p].reshape(-1, 1) @ (self.AiT_VjVjTViViT_p[:, i, j, int((l-2)/2) - p] @ self.V[:,:,j]).reshape(1, -1)

                            for p in range(int((l-2)/2)):
                                grad_F1_wrt_V[:,:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p].reshape(-1, 1) @ (self.AjT_ViViT_VjVjTViViT_p[:, i, j, int((l-2)/2) - 1 - p] @ self.V[:,:,j]).reshape(1, -1)
                     
                            grad_F1_wrt_V[:,:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * self.V[:,:,i] @ (self.ViTVjVjTVi_p[:, :, i, j, int((l-2)/2)] @ (self.V[:,:,i].T @ self.V[:,:,j]))
                                
                        elif l % 2 != 0:
                            grad_F1_wrt_A[:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * self.ViViTVjVjT_p_Ai[:, i, j, int((l-1)/2)]
                            
                            for p in range(int((l-1)/2)):
                                grad_F1_wrt_V[:,:,j] += self.pi[0, j] * self.pi[0, i] * self.order_weights[k] * math.comb(k, l) * self.B[i, j, k - l] * math.factorial(l) * (self.ViViTVjVjT_p_ViViT_Aj[:, i, j, p].reshape(-1, 1) @ (self.AiT_VjVjTViViT_p[:, i, j, int((l-1)/2-1-p)] @ self.V[:,:,j]).reshape(1, -1) + self.ViViTVjVjT_p_Ai[:, i, j, p].reshape(-1, 1) @ (self.AjT_ViViT_VjVjTViViT_p[:, i, j, int((l-1)/2-1-p)] @ self.V[:,:,j]).reshape(1, -1))
            
            grad_F1_wrt_A[:,j] *= 2
            grad_F1_wrt_V[:,:,j] *= 2

        return grad_F1_wrt_pi, grad_F1_wrt_A, grad_F1_wrt_V



    def _compute_F2(self):
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)

        inner_product_of_moment_vector = self.compute_inner_product_of_moment_vector(for_each_sample = False)
        F2 = np.sum(inner_product_of_moment_vector[1:] * self.order_weights[1:]) / self.n_points

        return F2
    

    def _compute_grad_F2(self):
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)
        if self.ATX is None or self.XTVjVjTX is None:
            self._precomputation_for_F2()

        grad_F2_wrt_pi = np.zeros((1, self.n_components))
        grad_F2_wrt_A = np.zeros((self.n_dim, self.n_components))
        grad_F2_wrt_V = np.zeros((self.n_dim, self.rank, self.n_components))
        
        B0 = np.ones((self.n_components, self.n_points), dtype=np.float64)
        B1 = self.ATX

        Y = np.einsum('in,irj->nrj', self.X, self.V, optimize='optimal')  # Y shape: (n_points, rank, self.n_components)
        pi_reshaped = self.pi[0].reshape(1, -1)

        grad_F2_wrt_pi += np.sum(B1, axis = 1) * self.order_weights[1] / self.n_points
        grad_F2_wrt_A += (np.dot(self.X, B0.T) * pi_reshaped) * 1 * self.order_weights[1] / self.n_points

        for k in range(2, self.L+1):
            current = B1 * self.ATX + (k - 1) * B0 * self.XTVjVjTX
            grad_F2_wrt_pi += np.sum(current, axis = 1) * self.order_weights[k] / self.n_points
            grad_F2_wrt_A += (np.dot(self.X, B1.T) * pi_reshaped) * k * self.order_weights[k] / self.n_points
            grad_F2_wrt_V += np.einsum('j, jn, in, nrj->irj', self.pi[0,:], B0, self.X, Y, optimize='optimal') * self.order_weights[k] * k * (k-1) / self.n_points
            B0, B1 = B1, current

        return grad_F2_wrt_pi, grad_F2_wrt_A, grad_F2_wrt_V
    

    def compute_F(self):
        if self.B is None:
            self.compute_bell_polynomials()
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)

        F1 = self._compute_F1()
        F2 = self._compute_F2()
        F = F1 - 2 * F2

        return float(np.asarray(F).squeeze())

    def compute_grad_F(self):
        if self.B is None:
            self.compute_bell_polynomials()
        if self.order_weights is None:
            self.order_weights = np.ones(self.L+1)

        grad_F1_wrt_pi, grad_F1_wrt_A, grad_F1_wrt_V = self._compute_grad_F1()
        grad_F2_wrt_pi, grad_F2_wrt_A, grad_F2_wrt_V = self._compute_grad_F2()

        grad_F_wrt_pi = grad_F1_wrt_pi - 2 * grad_F2_wrt_pi
        grad_F_wrt_A = grad_F1_wrt_A - 2 * grad_F2_wrt_A
        grad_F_wrt_V = grad_F1_wrt_V - 2 * grad_F2_wrt_V

        return grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V

    def F_and_grad_F(self):
        self._precompute()    
        F = self.compute_F()
        grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V = self.compute_grad_F()
        return F, grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V


    def F_and_grad_F1(self):
        self._precompute()
        F = self.compute_F()
        grad_F1_wrt_pi, grad_F1_wrt_A, grad_F1_wrt_V = self._compute_grad_F1()
        return F, grad_F1_wrt_pi, grad_F1_wrt_A, grad_F1_wrt_V
    

    def compute_grad_F2_per_sample(self):
        if self.order_weights is None:
            self.order_weights = np.ones(self.L + 1)
        if (getattr(self, "ATX", None) is None) or (getattr(self, "XTVjVjTX", None) is None):
            self._precomputation_for_F2()

        B0 = np.ones((self.n_components, self.n_points), dtype=np.float64)      
        B1 = self.ATX                              

        Y = np.einsum('in,irj->nrj', self.X, self.V, optimize=True)

        pi_row = np.asarray(self.pi).reshape(-1)     # (K,)
        grad_F2_wrt_pi_n = np.zeros((self.n_points, self.n_components), dtype=np.float64)
        grad_F2_wrt_A_n  = np.zeros((self.n_points, self.n_dim, self.n_components), dtype=np.float64)
        grad_F2_wrt_V_n  = np.zeros((self.n_points, self.n_dim, self.rank, self.n_components), dtype=np.float64)

        if self.L >= 1:
            grad_F2_wrt_pi_n += (B1.T) * self.order_weights[1]  
            grad_F2_wrt_A_n  += np.einsum('in,j->nij', self.X, pi_row, optimize=True) * self.order_weights[1]

        for k in range(2, self.L + 1):
            current = B1 * self.ATX + (k - 1.0) * B0 * self.XTVjVjTX 
            grad_F2_wrt_pi_n += (current.T) * self.order_weights[k]  
            grad_F2_wrt_A_n  += np.einsum('in,jn,j->nij', self.X, B1, pi_row, optimize=True) * (k * self.order_weights[k])
            grad_F2_wrt_V_n  += np.einsum('j,jn,in,nrj->nirj', pi_row, B0, self.X, Y, optimize=True) * (k * (k - 1.0) * self.order_weights[k])
            B0, B1 = B1, current

        return grad_F2_wrt_pi_n, grad_F2_wrt_A_n, grad_F2_wrt_V_n


    def _compute_grad_F2_per_sample_per_order(self):
        if (getattr(self, "ATX", None) is None) or (getattr(self, "XTVjVjTX", None) is None):
            self._precomputation_for_F2()

        B0 = np.ones((self.n_components, self.n_points), dtype=np.float64)
        B1 = self.ATX
        Y = np.einsum('in,irj->nrj', self.X, self.V, optimize=True)
        pi_row = np.asarray(self.pi).reshape(-1)

        g_pi = np.zeros((self.L+1, self.n_points, self.n_components))
        g_A  = np.zeros((self.L+1, self.n_points, self.n_dim, self.n_components))
        g_V  = np.zeros((self.L+1, self.n_points, self.n_dim, self.rank, self.n_components))

        if self.L >= 1:
            g_pi[1] = B1.T
            g_A[1]  = np.einsum('in,j->nij', self.X, pi_row, optimize=True)

        for k in range(2, self.L + 1):
            current = B1 * self.ATX + (k - 1.0) * B0 * self.XTVjVjTX
            g_pi[k] = current.T
            g_A[k]  = np.einsum('in,jn,j->nij', self.X, B1, pi_row, optimize=True) * k
            g_V[k]  = np.einsum('j,jn,in,nrj->nirj', pi_row, B0, self.X, Y, optimize=True) * (k * (k - 1.0))
            B0, B1 = B1, current
            
        return g_pi, g_A, g_V
