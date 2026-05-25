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

from sklearn.cluster import kmeans_plusplus
from sklearn.utils.extmath import randomized_svd
from scipy.linalg import svd
from numpy.linalg import pinv
from scipy.linalg import solve_triangular
from sklearn.utils.extmath import randomized_svd
import math 
import numpy as np

from upstream.rpcholesky import rpcholesky

# -----------------------------------------------------------------------------
# The following line is added for DGMM to set seed for reproducibility: 
import random
SEED = 4142
random.seed(SEED)                  # Python built-in RNG
np.random.seed(SEED)               # NumPy global RNG (for any legacy code)
rng = np.random.default_rng(SEED)  # New NumPy Generator
# -----------------------------------------------------------------------------

def _adaptive_nystrom_power(X, k, n_components, rank_max,
    m_landmarks, landmark_indices, factorization_rank=None, random_state=4142, landmark_indices_prev=None, sampling_method='kmeans', factorization_method='rpchol', tol=1e-6,
):
    _, n_points = X.shape 

    # Landmark selection using k-means++ sampling (Oglic and Gartner, 2017).
    if m_landmarks is None:
        m_landmarks = min(n_components * math.comb(rank_max + k - 1, k), n_points)
    else:
        m_landmarks = min(int(m_landmarks), n_points)
    if m_landmarks <= 0:
        raise ValueError("m_landmarks must be positive.")

    if landmark_indices is None or len(landmark_indices) != m_landmarks:
        if sampling_method != 'kmeans':
            raise ValueError("Only sampling_method='kmeans' is currently implemented.")
        _, landmark_indices = kmeans_plusplus(
            X.T,
            n_clusters=m_landmarks,
            random_state=random_state,
        )

    landmarks = X[:, landmark_indices]
    inner_products = X.T @ landmarks  
    C = inner_products ** k           
    
    inner_landmarks = landmarks.T @ landmarks  
    W = inner_landmarks ** k                 
    
    delta = 1e-10 * np.trace(W) / m_landmarks
    A = W + delta * np.eye(m_landmarks)
    v = C.T @ np.ones(n_points) 

    if factorization_method == 'rpchol':
        if rpcholesky is not None:
            factorization_rank = m_landmarks if factorization_rank is None else min(factorization_rank, m_landmarks)
            rpchol_obj = rpcholesky(A, factorization_rank)
            pivots = np.asarray(rpchol_obj.get_indices(), dtype=int)
            if pivots.shape[0] == m_landmarks:
                A_perm = A[pivots, :][:, pivots]
                v_perm = v[pivots]
                L_chol_reg = np.linalg.cholesky(A_perm)
                u = solve_triangular(L_chol_reg, v_perm, lower=True, check_finite=False)
                y_perm = solve_triangular(L_chol_reg.T, u, lower=False, check_finite=False)
                y = np.empty_like(v)
                y[pivots] = y_perm
                row_sums = C @ y
                return row_sums, landmark_indices
        factorization_method = 'pchol'

    if factorization_method == 'pchol':
        L_chol_reg  = np.linalg.cholesky(A)
        u = solve_triangular(L_chol_reg,    v, lower=True,  check_finite=False)
        y = solve_triangular(L_chol_reg.T,  u, lower=False, check_finite=False)
        row_sums = C @ y                  
        return row_sums, landmark_indices
    
    else:
        raise ValueError("Invalid factorization_method. Choose 'rpchol' or 'pchol'.")


def compute_moment_sums(X, L, n_components, rank_max, m_landmarks=None, exact = False,
                        sampling_method='kmeans', factorization_method='rpchol',
                        lambda_reg=1e-6, mu_shift=1e-6, tol=1e-6, r_leverage=50, random_state=4142):
    _, n_points = X.shape
    
    sum_matrix = np.zeros((2*L+1, n_points))  # row 0 unused; k=1,...,2L

    if exact is True:
        block_size = min(256, n_points)
        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            gram_block = X[:, start:end].T @ X
            gram_power = gram_block.copy()
            for k in range(1, 2 * L + 1):
                if k > 1:
                    gram_power *= gram_block
                sum_matrix[k, start:end] = np.sum(gram_power, axis=1)
    
    else:
        landmark_indices = None
        landmark_indices_prev = None
        factorization_rank = None 

        for k in range(1, 2*L+1):
            row_sums, landmark_indices = _adaptive_nystrom_power(X, k, n_components, rank_max,
                m_landmarks, landmark_indices, factorization_rank, sampling_method=sampling_method, factorization_method=factorization_method, tol=tol, random_state=random_state,
            )
            sum_matrix[k, :] = row_sums

    diag_matrix = np.zeros((L+1, n_points))
    norms_sq = np.sum(X**2, axis=0)
    for k in range(1, L+1):
        diag_matrix[k, :] = norms_sq ** k
    
    return sum_matrix, diag_matrix

def compute_exact_moment_sums(X, L, block_size=256):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError("X must have shape (n_dim, n_points).")
        if L <= 0:
            raise ValueError("L must be positive.")
        if block_size <= 0:
            raise ValueError("block_size must be positive.")

        _, n_points = X.shape
        vec_vec_sum = np.zeros((2 * L + 1, n_points), dtype=np.float64)
        vec_vec_diag = np.zeros((L + 1, n_points), dtype=np.float64)

        norms_sq = np.sum(X * X, axis=0)
        for k in range(1, L + 1):
            vec_vec_diag[k, :] = norms_sq ** k

        for start in range(0, n_points, block_size):
            end = min(start + block_size, n_points)
            gram_block = X[:, start:end].T @ X
            gram_block_power = gram_block.copy()

            for k in range(1, 2 * L + 1):
                if k > 1:
                    gram_block_power *= gram_block
                vec_vec_sum[k, start:end] = np.sum(gram_block_power, axis=1)

        return vec_vec_sum, vec_vec_diag