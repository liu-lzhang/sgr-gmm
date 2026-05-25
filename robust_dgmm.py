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

from gm_moments import GaussianMixtureMoments
from gmm import GMM
from utils import params_to_theta_grad, params_to_theta_grad_per_sample, theta_to_params
from robust_gradient_reweighting import RobustGradientReweighting
from NystromApprox import compute_exact_moment_sums, compute_moment_sums

import numpy as np
import numpy.linalg as npl
from scipy.optimize import minimize
from functools import reduce
from sympy.utilities.iterables import multiset_permutations
from sklearn.metrics import normalized_mutual_info_score
import copy
import math
from typing import Callable, Optional

# -----------------------------------------------------------------------------
# The following line is added for DGMM to set seed for reproducibility:
import random

SEED = 4142
random.seed(SEED)                  # Python built-in RNG
np.random.seed(SEED)               # NumPy global RNG (for any legacy code)
rng = np.random.default_rng(SEED)  # New NumPy Generator
# -----------------------------------------------------------------------------

class RobustDGMM(GMM):

    def __init__(
        self,
        theta_init,
        rank,
        n_components,
        L=4,
        W_init="diagonal",
        W_step="diagonal",
        step_max=5,
        step_tol=1e-04,
        iter_max=100,
        softmax_reparam=True,
        softmax_temperature=1.5,
        vec_vec_sum=None,
        vec_vec_diag=None,
        M_sample=None,
        M_model=None,
        contamination_epsilon=0.1,
        noise_covariance=None,
        reweight_interval=5,
        robust_weight_tol=1e-4,
        auto_compute_moment_sums=True,
        moment_sum_block_size=256,
        max_points_for_exact_moment_sums=2000,
        algorithm2_kwargs=None,
        algorithm2_eta_rho=None,
        algorithm2_eta_w=None,
        algorithm2_step_max_outer=10,
        algorithm2_step_max_inner=200,
        algorithm2_threshold_const=None,
        algorithm2_target_accuracy=1e-6,
        algorithm2_use_exact_diameter="auto",
        algorithm2_exact_diameter_auto_threshold=512,
        algorithm2_min_outer_iterations_for_stabilization=5,
        algorithm2_stabilization_patience=2,
        algorithm2_warm_start_inner_weights=True,
        algorithm2_initial_center_strategy="geometric_median",
        algorithm2_step_size_strategy="max_safe",
        algorithm2_warm_start_across_reweightings=True,
        algorithm2_reuse_previous_centers=True,
        pre_reweight_before_optimization=True,
        robustify_diagonal_order_weights=True,
        recompute_order_weights_after_reweight=True,
        algorithm2_verbose=False,
        robust_order_weights_verbose=False,
        store_per_order_weight_history=True,
    ):
        super().__init__(
            theta_init,
            rank,
            n_components,
            L=L,
            W_init=W_init,
            W_step=W_step,
            step_max=step_max,
            step_tol=step_tol,
            iter_max=iter_max,
            softmax_reparam=softmax_reparam,
            softmax_temperature=softmax_temperature,
            vec_vec_sum=vec_vec_sum,
            vec_vec_diag=vec_vec_diag,
            M_sample=M_sample,
            M_model=M_model,
        )

        if reweight_interval <= 0:
            raise ValueError("reweight_interval must be positive.")
        if robust_weight_tol < 0:
            raise ValueError("robust_weight_tol must be nonnegative.")
        if moment_sum_block_size <= 0:
            raise ValueError("moment_sum_block_size must be positive.")
        if max_points_for_exact_moment_sums <= 0:
            raise ValueError("max_points_for_exact_moment_sums must be positive.")

        if isinstance(algorithm2_use_exact_diameter, str):
            if algorithm2_use_exact_diameter != "auto":
                raise ValueError(
                    "algorithm2_use_exact_diameter must be a bool or the string 'auto'."
                )
        elif not isinstance(algorithm2_use_exact_diameter, (bool, np.bool_)):
            raise ValueError(
                "algorithm2_use_exact_diameter must be a bool or the string 'auto'."
            )
        if algorithm2_exact_diameter_auto_threshold <= 0:
            raise ValueError(
                "algorithm2_exact_diameter_auto_threshold must be positive."
            )
        if algorithm2_min_outer_iterations_for_stabilization < 0:
            raise ValueError(
                "algorithm2_min_outer_iterations_for_stabilization must be nonnegative."
            )
        if algorithm2_stabilization_patience <= 0:
            raise ValueError(
                "algorithm2_stabilization_patience must be positive."
            )

        self.noise_covariance = None if noise_covariance is None else np.asarray(noise_covariance, dtype=np.float64)
        self.reweight_interval = reweight_interval
        self.robust_weight_tol = robust_weight_tol
        self.auto_compute_moment_sums = auto_compute_moment_sums
        self.moment_sum_block_size = moment_sum_block_size
        self.max_points_for_exact_moment_sums = max_points_for_exact_moment_sums

        self.contamination_epsilon = contamination_epsilon

        self.algorithm2_use_exact_diameter = algorithm2_use_exact_diameter
        self.algorithm2_exact_diameter_auto_threshold = int(
            algorithm2_exact_diameter_auto_threshold
        )
        self.algorithm2_min_outer_iterations_for_stabilization = int(
            algorithm2_min_outer_iterations_for_stabilization
        )
        self.algorithm2_stabilization_patience = int(
            algorithm2_stabilization_patience
        )

        if algorithm2_kwargs is None:
            self.algorithm2_kwargs = {}
        else:
            self.algorithm2_kwargs = copy.deepcopy(algorithm2_kwargs)

        explicit_algorithm2_kwargs = {
            "eta_rho": algorithm2_eta_rho,
            "eta_w": algorithm2_eta_w,
            "step_max_outer": algorithm2_step_max_outer,
            "step_max_inner": algorithm2_step_max_inner,
            "threshold_const": algorithm2_threshold_const,
            "target_accuracy": algorithm2_target_accuracy,
            "warm_start_inner_weights": algorithm2_warm_start_inner_weights,
            "initial_center_strategy": algorithm2_initial_center_strategy,
            "step_size_strategy": algorithm2_step_size_strategy,
            "verbose": algorithm2_verbose,
            "warn_on_theory_gap": False,
        }
        for key, value in explicit_algorithm2_kwargs.items():
            self.algorithm2_kwargs[key] = value

        self.algorithm2_warm_start_across_reweightings = bool(
            algorithm2_warm_start_across_reweightings
        )
        self.algorithm2_reuse_previous_centers = bool(
            algorithm2_reuse_previous_centers
        )

        self.robustify_diagonal_order_weights = bool(robustify_diagonal_order_weights)
        self.pre_reweight_before_optimization = bool(pre_reweight_before_optimization)
        self.recompute_order_weights_after_reweight = bool(recompute_order_weights_after_reweight)
        self.robust_order_weights_verbose = bool(robust_order_weights_verbose)
        self.store_per_order_weight_history = bool(store_per_order_weight_history)

        self.reset_robust_state()

    def reset_robust_state(self):
        self.n_iter_ = 0
        self.theta_opt_ = None
        self.robust_weights = None
        self.robust_per_order_weights = None
        self.noise_factor_ = None

        self.robust_weights_history = []
        self.robust_theta_history = []
        self.robust_objective_history = []
        self.robust_theta_change_history = []
        self.robust_weight_change_history = []
        self.robust_reweighter_history = []
        self.robust_gradient_cloud_history = []
        self.robust_order_weight_history = []
        self.robust_order_weight_strategy_history = []
        self.robust_per_order_weights_history = []
        self.robust_per_order_centers_history = []
        self.robust_reweight_stage_history = []

        self.algorithm2_previous_sample_weights = None
        self.algorithm2_previous_fixed_center = None
        self.algorithm2_previous_per_order_weights = None
        self.algorithm2_previous_per_order_centers = None

    def uniform_sample_weights(self):
        if self.n_points is None:
            raise RuntimeError("n_points must be known before the sample weights are initialized.")
        return np.ones(self.n_points, dtype=np.float64) / self.n_points

    def normalize_sample_weights(self, sample_weights):
        if self.n_points is None:
            raise RuntimeError("n_points must be known before sample weights are validated.")

        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights.shape != (self.n_points,):
            raise ValueError("sample_weights must have shape ({},), got {}.".format(self.n_points, sample_weights.shape,))
        if np.any(sample_weights < 0):
            raise ValueError("sample_weights must be nonnegative.")
        if not np.all(np.isfinite(sample_weights)):
            raise ValueError("sample_weights must contain only finite values.")

        total_mass = np.sum(sample_weights)
        if total_mass <= 0:
            raise ValueError("sample_weights must sum to a positive value.")

        return sample_weights / total_mass

    def _normalize_per_order_sample_weights(self, sample_weights):
        if sample_weights is None:
            return np.tile(self.uniform_sample_weights(), (self.L, 1))

        sample_weights = np.asarray(sample_weights, dtype=np.float64)

        if sample_weights.ndim == 1:
            normalized = self.normalize_sample_weights(sample_weights)
            return np.tile(normalized, (self.L, 1))

        if sample_weights.ndim != 2:
            raise ValueError("sample_weights must be either one- or two-dimensional.")

        if sample_weights.shape != (self.L, self.n_points):
            raise ValueError("sample_weights must have shape ({}, {}), got {}.".format(self.L, self.n_points, sample_weights.shape,))

        return np.vstack([self.normalize_sample_weights(sample_weights[k_idx, :]) for k_idx in range(self.L)])


    @staticmethod
    def _symmetrize_matrix(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        return 0.5 * (matrix + matrix.T)

    def noise_model_is_active(self, noise_covariance=None):
        if self.n_dim is None:
            raise RuntimeError("n_dim must be initialized before the noise model is queried.")

        effective_noise_covariance = self.noise_covariance if noise_covariance is None else noise_covariance
        if effective_noise_covariance is None:
            return False

        effective_noise_covariance = np.asarray(effective_noise_covariance, dtype=np.float64)
        if effective_noise_covariance.shape != (self.n_dim, self.n_dim):
            raise ValueError("noise_covariance must have shape ({}, {}), got {}.".format(self.n_dim, self.n_dim, effective_noise_covariance.shape,))

        return bool(np.any(np.abs(effective_noise_covariance) > 0.0))

    def _compute_noise_factor(self, noise_covariance=None):
        if self.n_dim is None:
            raise RuntimeError("n_dim must be initialized before the noise factor is computed.")

        if noise_covariance is None and self.noise_factor_ is not None:
            return self.noise_factor_

        effective_noise_covariance = self.noise_covariance if noise_covariance is None else noise_covariance
        if effective_noise_covariance is None:
            factor = np.zeros((self.n_dim, 0), dtype=np.float64)
            if noise_covariance is None:
                self.noise_factor_ = factor
            return factor

        Sigma_omega = np.asarray(effective_noise_covariance, dtype=np.float64)
        if Sigma_omega.shape != (self.n_dim, self.n_dim):
            raise ValueError("noise_covariance must have shape ({}, {}), got {}.".format(self.n_dim, self.n_dim, Sigma_omega.shape,))

        Sigma_omega = self._symmetrize_matrix(Sigma_omega)
        eigenvalues, eigenvectors = np.linalg.eigh(Sigma_omega)
        scale = max(1.0, float(np.max(np.abs(eigenvalues))))
        tolerance = 1e-12 * scale

        if np.min(eigenvalues) < -100.0 * tolerance:
            raise ValueError("noise_covariance must be positive semidefinite.")

        eigenvalues = np.clip(eigenvalues, a_min=0.0, a_max=None)
        positive_mask = eigenvalues > tolerance
        if np.any(positive_mask):
            factor = eigenvectors[:, positive_mask] * np.sqrt(eigenvalues[positive_mask]).reshape(1, -1)
        else:
            factor = np.zeros((self.n_dim, 0), dtype=np.float64)

        factor = np.asarray(factor, dtype=np.float64)
        if noise_covariance is None:
            self.noise_factor_ = factor
        return factor

    def _make_effective_gaussian_mixture_moments(self, theta, order_weights=None, noise_covariance=None):
        if self.X is None:
            raise RuntimeError("X must be initialized before the effective moments object is built.")

        pi, A, V = theta_to_params(theta,self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature, )

        if order_weights is None:
            order_weights = np.ones(self.L + 1, dtype=np.float64)
        else:
            order_weights = np.asarray(order_weights, dtype=np.float64)

        noise_factor = self._compute_noise_factor(noise_covariance)
        if noise_factor.shape[1] == 0:
            V_effective = np.asarray(V, dtype=np.float64).copy()
        else:
            shared_noise_factor = np.repeat(noise_factor[:, :, np.newaxis], self.n_components, axis=2)
            V_effective = np.concatenate((V, shared_noise_factor), axis=1)

        gmm_effective = GaussianMixtureMoments(pi, A, V_effective, self.X, self.L, order_weights,)
        gmm_effective._precompute()

        return pi, A, V, V_effective, gmm_effective

    def _compute_diagonal_order_weights_from_terms(self, model_terms, cross_term_sums, cross_terms_per_sample):
        if self.vec_vec_sum is None:
            raise RuntimeError(
                "vec_vec_sum must be available before diagonal DGMM weights are computed."
            )

        model_terms = np.asarray(model_terms, dtype=np.float64)
        cross_term_sums = np.asarray(cross_term_sums, dtype=np.float64)
        cross_terms_per_sample = np.asarray(cross_terms_per_sample, dtype=np.float64)

        if model_terms.shape != (self.L + 1,):
            raise ValueError("model_terms must have shape (L+1,).")
        if cross_term_sums.shape != (self.L + 1,):
            raise ValueError("cross_term_sums must have shape (L+1,).")
        if cross_terms_per_sample.shape != (self.L + 1, self.n_points):
            raise ValueError("cross_terms_per_sample must have shape ({}, {}), got {}.".format(self.L + 1, self.n_points, cross_terms_per_sample.shape,))

        if self.vec_vec_sum.shape != (2 * self.L + 1, self.n_points):
            raise ValueError("vec_vec_sum must have shape ({}, {}), got {}.".format(2 * self.L + 1, self.n_points, self.vec_vec_sum.shape,))

        if self.vec_vec_diag is not None:
            vec_vec_diag = np.asarray(self.vec_vec_diag, dtype=np.float64)
            if vec_vec_diag.shape != (self.L + 1, self.n_points):
                raise ValueError("vec_vec_diag must have shape ({}, {}), got {}.".format(self.L + 1, self.n_points, vec_vec_diag.shape,))
        else:
            norms_sq = np.sum(self.X * self.X, axis=0)
            vec_vec_diag = np.zeros((self.L + 1, self.n_points), dtype=np.float64)
            for k in range(1, self.L + 1):
                vec_vec_diag[k, :] = norms_sq ** k

        order_weights = np.zeros(self.L + 1, dtype=np.float64)

        for k in range(1, self.L + 1):
            numerator_k = np.sum(model_terms[k] - 2.0 * cross_terms_per_sample[k, :] + vec_vec_diag[k, :]) / self.n_points

            denominator_k = 0.0
            for k_prime in range(1, self.L + 1):
                denominator_k += 2.0 * cross_term_sums[k] * cross_term_sums[k_prime]
                denominator_k += np.sum(
                    self.n_points * model_terms[k] * model_terms[k_prime]
                    - 2.0 * model_terms[k] * cross_term_sums[k_prime]
                    - 2.0 * model_terms[k_prime] * cross_term_sums[k]
                    + model_terms[k] * self.vec_vec_sum[k_prime, :]
                    + model_terms[k_prime] * self.vec_vec_sum[k, :]
                    + 2.0 * self.n_points
                    * cross_terms_per_sample[k, :]
                    * cross_terms_per_sample[k_prime, :]
                    - 2.0 * cross_terms_per_sample[k, :] * self.vec_vec_sum[k_prime, :]
                    - 2.0 * cross_terms_per_sample[k_prime, :] * self.vec_vec_sum[k, :]
                    + self.vec_vec_sum[k + k_prime, :]
                )

            denominator_k = denominator_k / (self.n_points ** 2)
            if denominator_k <= 0.0 or not np.isfinite(denominator_k):
                raise FloatingPointError("Encountered a nonpositive diagonal-DGMM denominator at order k={}.".format(k))

            order_weights[k] = numerator_k / denominator_k

        return order_weights

    def _compute_diagonal_order_weights_from_terms_weighted(
        self,
        model_terms,
        cross_terms_per_sample,
        sample_weights,
    ):
        if self.X is None:
            raise RuntimeError("X must be initialized before robust order weights are computed.")

        model_terms = np.asarray(model_terms, dtype=np.float64)
        cross_terms_per_sample = np.asarray(cross_terms_per_sample, dtype=np.float64)
        per_order_sample_weights = self._normalize_per_order_sample_weights(sample_weights)

        if model_terms.shape != (self.L + 1,):
            raise ValueError("model_terms must have shape (L+1,).")
        if cross_terms_per_sample.shape != (self.L + 1, self.n_points):
            raise ValueError("cross_terms_per_sample must have shape ({}, {}), got {}.".format( self.L + 1, self.n_points, cross_terms_per_sample.shape,))

        if self.vec_vec_diag is not None:
            vec_vec_diag = np.asarray(self.vec_vec_diag, dtype=np.float64)
            if vec_vec_diag.shape != (self.L + 1, self.n_points):
                raise ValueError("vec_vec_diag must have shape ({}, {}), got {}.".format(self.L + 1, self.n_points, vec_vec_diag.shape,))
        else:
            norms_sq = np.sum(self.X * self.X, axis=0)
            vec_vec_diag = np.zeros((self.L + 1, self.n_points), dtype=np.float64)
            for k in range(1, self.L + 1):
                vec_vec_diag[k, :] = norms_sq ** k

        residual_diag = (model_terms[1:, np.newaxis] - 2.0 * cross_terms_per_sample[1:, :] + vec_vec_diag[1:, :])

        numerator = np.sum((per_order_sample_weights ** 2) * residual_diag, axis=1)
        numerator = np.where(numerator >= 0.0, numerator, np.where(np.abs(numerator) <= 128.0 * np.finfo(np.float64).eps, 0.0, numerator,),)

        denominator = np.zeros(self.L, dtype=np.float64)
        block_size = int(max(1, self.moment_sum_block_size))

        for start in range(0, self.n_points, block_size):
            end = min(start + block_size, self.n_points)
            gram_block = self.X[:, start:end].T @ self.X
            gram_power = gram_block.copy()

            residual_blocks = np.empty((self.L, end - start, self.n_points), dtype=np.float64)
            for k_idx in range(self.L):
                if k_idx > 0:
                    gram_power *= gram_block
                residual_blocks[k_idx, :, :] = (model_terms[k_idx + 1] - cross_terms_per_sample[k_idx + 1, start:end][:, np.newaxis] - cross_terms_per_sample[k_idx + 1, :][np.newaxis, :] + gram_power)

            for k_idx in range(self.L):
                residual_k = residual_blocks[k_idx, :, :]
                weights_k_left = per_order_sample_weights[k_idx, start:end]
                weights_k_right = per_order_sample_weights[k_idx, :]

                for k_prime_idx in range(self.L):
                    combined_left = (weights_k_left * per_order_sample_weights[k_prime_idx, start:end])
                    combined_right = (weights_k_right * per_order_sample_weights[k_prime_idx, :])
                    denominator[k_idx] += np.sum((combined_left[:, np.newaxis] * combined_right[np.newaxis, :]) * residual_k * residual_blocks[k_prime_idx, :, :])

        order_weights = np.zeros(self.L + 1, dtype=np.float64)
        order_weights[1:] = numerator / denominator
        return order_weights

    def compute_order_weights_diagonal_robust(self, theta, noise_covariance=None):
        if self.X is None:
            raise RuntimeError("X must be initialized before order weights are computed.")

        if self.noise_model_is_active(noise_covariance) is False:
            self.compute_order_weights_diagonal_GMM(theta)
            return np.asarray(self.order_weights, dtype=np.float64).copy()

        _, _, _, _, gmm_effective = self._make_effective_gaussian_mixture_moments(theta, order_weights=np.ones(self.L + 1, dtype=np.float64),noise_covariance=noise_covariance,)

        model_terms = gmm_effective.compute_inner_product_of_moment_moment()
        cross_term_sums, cross_terms_per_sample = gmm_effective.compute_inner_product_of_moment_vector(for_each_sample=True)

        self.order_weights = self._compute_diagonal_order_weights_from_terms(model_terms, cross_term_sums, cross_terms_per_sample,)
        return np.asarray(self.order_weights, dtype=np.float64).copy()

    def compute_robust_order_weights(self, theta, sample_weights=None, noise_covariance=None):
        if (self.robustify_diagonal_order_weights is False or sample_weights is None):
            return self.compute_order_weights_diagonal_robust(theta, noise_covariance=noise_covariance,)

        per_order_sample_weights = self._normalize_per_order_sample_weights(sample_weights)
        uniform = np.tile(self.uniform_sample_weights(), (self.L, 1))
        if np.max(np.abs(per_order_sample_weights - uniform)) <= 128.0 * np.finfo(np.float64).eps:
            return self.compute_order_weights_diagonal_robust(theta, noise_covariance=noise_covariance,)

        _, _, _, _, gmm_effective = self._make_effective_gaussian_mixture_moments(theta, order_weights=np.ones(self.L + 1, dtype=np.float64), noise_covariance=noise_covariance,)

        model_terms = gmm_effective.compute_inner_product_of_moment_moment()
        _, cross_terms_per_sample = gmm_effective.compute_inner_product_of_moment_vector(for_each_sample=True)

        self.order_weights = self._compute_diagonal_order_weights_from_terms_weighted(model_terms, cross_terms_per_sample, per_order_sample_weights,)
        return np.asarray(self.order_weights, dtype=np.float64).copy()

    def _record_order_weights(self, strategy_name):
        if self.order_weights is None:
            raise RuntimeError("order_weights must be initialized before they are recorded.")

        self.robust_order_weight_history.append(np.asarray(self.order_weights, dtype=np.float64).copy())
        self.robust_order_weight_strategy_history.append(str(strategy_name))

    def _record_per_order_weights(self, per_order_weights, stage_name, per_order_centers=None):
        if self.store_per_order_weight_history is False:
            return

        per_order_weights = np.asarray(per_order_weights, dtype=np.float64)
        if per_order_weights.shape != (self.L, self.n_points):
            raise ValueError("per_order_weights must have shape ({}, {}), got {}.".format(self.L, self.n_points, per_order_weights.shape))

        normalized_per_order_weights = np.vstack(
            [self.normalize_sample_weights(per_order_weights[k_idx, :]) for k_idx in range(self.L)]
        )
        self.robust_per_order_weights_history.append(normalized_per_order_weights.copy())
        self.robust_reweight_stage_history.append(str(stage_name))

        if per_order_centers is None:
            self.robust_per_order_centers_history.append(None)
        else:
            per_order_centers = np.asarray(per_order_centers, dtype=np.float64)
            if per_order_centers.ndim != 2 or per_order_centers.shape[0] != self.L:
                raise ValueError("per_order_centers must have shape ({}, p_k) when provided; got {}.".format(self.L, per_order_centers.shape,))
            self.robust_per_order_centers_history.append(per_order_centers.copy())

    def maybe_refresh_order_weights(self, theta, W_option_current_step, sample_weights=None, stage_name="reweight",):
        if W_option_current_step != "diagonal":
            return

        self.order_weights = self.compute_robust_order_weights(theta, sample_weights=sample_weights,)
        self._record_order_weights("{}:diagonal_algorithm1".format(stage_name))

    def compute_per_order_per_sample_gradients(self, theta, noise_covariance=None):
        pi, _, _, _, gmm_effective = self._make_effective_gaussian_mixture_moments(theta, order_weights=np.ones(self.L + 1, dtype=np.float64), noise_covariance=noise_covariance,)

        grad_pi_per_order, grad_A_per_order, grad_V_per_order_effective = (gmm_effective._compute_grad_F2_per_sample_per_order())
        grad_V_per_order = grad_V_per_order_effective[:, :, :, : self.rank, :]

        dim_theta = int(np.asarray(theta, dtype=np.float64).shape[0])
        per_order_gradient_clouds = np.zeros((self.L, self.n_points, dim_theta), dtype=np.float64)

        for k in range(1, self.L + 1):
            per_order_gradient_clouds[k - 1, :, :] = params_to_theta_grad_per_sample(pi, grad_pi_per_order[k, :, :], grad_A_per_order[k, :, :, :], grad_V_per_order[k, :, :, :, :], self.softmax_reparam, self.softmax_temperature,)

        return per_order_gradient_clouds

    def compute_order_weighted_per_sample_gradients(self, theta, noise_covariance=None):
        if self.order_weights is None:
            raise RuntimeError("order_weights must be initialized before per-sample gradients are aggregated.")

        per_order_gradient_clouds = self.compute_per_order_per_sample_gradients(theta, noise_covariance=noise_covariance,)
        return np.tensordot(
            np.asarray(self.order_weights[1:], dtype=np.float64),
            per_order_gradient_clouds,
            axes=(0, 0),
        )

    def _resolve_algorithm2_use_exact_diameter(self):
        if isinstance(self.algorithm2_use_exact_diameter, str):
            if self.algorithm2_use_exact_diameter != "auto":
                raise ValueError("algorithm2_use_exact_diameter must be a bool or the string 'auto'.")
            if self.n_points is None:
                raise RuntimeError("n_points must be initialized before the Algorithm 2 diameter mode is resolved.")
            return bool(self.n_points <= self.algorithm2_exact_diameter_auto_threshold)
        return bool(self.algorithm2_use_exact_diameter)

    def build_algorithm2_kwargs(self):
        kwargs = copy.deepcopy(self.algorithm2_kwargs)
        kwargs["use_exact_diameter"] = self._resolve_algorithm2_use_exact_diameter()

        if "step_max_outer" not in kwargs:
            raise RuntimeError("Algorithm 2 requires step_max_outer to be specified.")
        step_max_outer = int(kwargs["step_max_outer"])
        if step_max_outer <= 0:
            raise RuntimeError("Algorithm 2 requires step_max_outer to be positive.")

        min_outer_iterations_for_stabilization = int(
            self.algorithm2_min_outer_iterations_for_stabilization
        )
        stabilization_patience = int(self.algorithm2_stabilization_patience)

        if kwargs.get("threshold_const") is None:
            max_min_outer = max(1, step_max_outer - stabilization_patience + 1)
            min_outer_iterations_for_stabilization = min(min_outer_iterations_for_stabilization, max_min_outer,)
            max_patience = max(1, step_max_outer - min_outer_iterations_for_stabilization + 1,)
            stabilization_patience = min(stabilization_patience, max_patience,)

        kwargs["min_outer_iterations_for_stabilization"] = (min_outer_iterations_for_stabilization)
        kwargs["stabilization_patience"] = stabilization_patience
        return kwargs

    def update_sample_weights(self, theta):
        if self.contamination_epsilon is None or self.contamination_epsilon <= 0:
            uniform_weights = self.uniform_sample_weights()
            reweighter = RobustGradientReweighting(contamination_epsilon=0.0, n_points=self.n_points, dim_p=self.theta_init.shape[0], **self.build_algorithm2_kwargs(),)
            per_sample_gradients = self.compute_order_weighted_per_sample_gradients(theta)
            self.algorithm2_previous_sample_weights = uniform_weights.copy()
            self.algorithm2_previous_fixed_center = None
            return uniform_weights, reweighter, per_sample_gradients

        per_sample_gradients = self.compute_order_weighted_per_sample_gradients(theta)

        initial_sample_weights = None
        if (self.algorithm2_warm_start_across_reweightings and self.algorithm2_previous_sample_weights is not None):
            initial_sample_weights = self.algorithm2_previous_sample_weights

        initial_fixed_center = None
        if (self.algorithm2_reuse_previous_centers and self.algorithm2_previous_fixed_center is not None):
            initial_fixed_center = self.algorithm2_previous_fixed_center

        reweighter = RobustGradientReweighting(contamination_epsilon=self.contamination_epsilon, n_points=self.n_points, dim_p=per_sample_gradients.shape[1], **self.build_algorithm2_kwargs(),)
        reweighter.fit(per_sample_gradients, initial_sample_weights=initial_sample_weights, initial_fixed_center=initial_fixed_center,)

        robust_weights = self.normalize_sample_weights(reweighter.sample_weights)

        self.algorithm2_previous_sample_weights = robust_weights.copy()
        if reweighter.location_estimate is not None:
            self.algorithm2_previous_fixed_center = np.asarray(reweighter.location_estimate, dtype=np.float64,).copy()
        elif reweighter.fixed_center is not None:
            self.algorithm2_previous_fixed_center = np.asarray(reweighter.fixed_center, dtype=np.float64,).copy()
        else:
            self.algorithm2_previous_fixed_center = None

        self.robust_reweighter_history.append(reweighter)
        self.robust_gradient_cloud_history.append(per_sample_gradients.copy())

        return robust_weights, reweighter, per_sample_gradients

    def update_sample_weights_per_order(self, theta):
        per_order_gradient_clouds = self.compute_per_order_per_sample_gradients(theta)

        if self.contamination_epsilon is None or self.contamination_epsilon <= 0:
            uniform = self.uniform_sample_weights()
            per_order_weights = np.tile(uniform, (self.L, 1))
            self.algorithm2_previous_per_order_weights = per_order_weights.copy()
            self.algorithm2_previous_per_order_centers = None
            return per_order_weights, [None] * self.L, per_order_gradient_clouds

        alg2_kwargs = self.build_algorithm2_kwargs()
        per_order_weights = np.empty((self.L, self.n_points), dtype=np.float64)
        reweighter_list = []

        previous_per_order_weights = self.algorithm2_previous_per_order_weights
        previous_per_order_centers = self.algorithm2_previous_per_order_centers

        if (
            previous_per_order_weights is not None
            and previous_per_order_weights.shape != (self.L, self.n_points)
        ):
            previous_per_order_weights = None
        if previous_per_order_centers is not None:
            previous_per_order_centers = np.asarray(
                previous_per_order_centers,
                dtype=np.float64,
            )
            if previous_per_order_centers.ndim != 2 or previous_per_order_centers.shape[0] != self.L:
                previous_per_order_centers = None

        new_per_order_centers = [None] * self.L

        for k_idx in range(self.L):
            gradient_cloud_k = per_order_gradient_clouds[k_idx]

            initial_sample_weights = None
            if (
                self.algorithm2_warm_start_across_reweightings
                and previous_per_order_weights is not None
            ):
                initial_sample_weights = previous_per_order_weights[k_idx, :]

            initial_fixed_center = None
            if (
                self.algorithm2_reuse_previous_centers
                and previous_per_order_centers is not None
                and previous_per_order_centers.shape[1] == gradient_cloud_k.shape[1]
            ):
                initial_fixed_center = previous_per_order_centers[k_idx, :]

            reweighter_k = RobustGradientReweighting(contamination_epsilon=self.contamination_epsilon, n_points=self.n_points, dim_p=gradient_cloud_k.shape[1], **copy.deepcopy(alg2_kwargs),)
            reweighter_k.fit(gradient_cloud_k, initial_sample_weights=initial_sample_weights, initial_fixed_center=initial_fixed_center,)
            per_order_weights[k_idx, :] = self.normalize_sample_weights(reweighter_k.sample_weights)
            if reweighter_k.location_estimate is not None:
                new_per_order_centers[k_idx] = np.asarray(reweighter_k.location_estimate, dtype=np.float64,).copy()
            elif reweighter_k.fixed_center is not None:
                new_per_order_centers[k_idx] = np.asarray(reweighter_k.fixed_center, dtype=np.float64,).copy()

            reweighter_list.append(reweighter_k)

        self.algorithm2_previous_per_order_weights = per_order_weights.copy()
        if all(center is not None for center in new_per_order_centers):
            self.algorithm2_previous_per_order_centers = np.vstack([center.reshape(1, -1) for center in new_per_order_centers])
        else:
            self.algorithm2_previous_per_order_centers = None

        self.robust_reweighter_history.append(reweighter_list)
        self.robust_gradient_cloud_history.append(per_order_gradient_clouds)

        return per_order_weights, reweighter_list, per_order_gradient_clouds


    def F_and_grad_F_order_weights_robust(self, theta, sample_weights=None, noise_covariance=None, contamination_epsilon=None, return_diagnostics=False,):
        _ = contamination_epsilon

        theta = np.asarray(theta, dtype=np.float64)
        if theta.ndim != 1:
            raise ValueError("theta must be a one-dimensional parameter vector.")
        if self.order_weights is None:
            raise RuntimeError("order_weights must be initialized before the robust objective is evaluated.")
        if self.X is None:
            raise RuntimeError("X must be initialized before the robust objective is evaluated.")

        per_order_sample_weights = self._normalize_per_order_sample_weights(sample_weights)

        order_weights = np.asarray(self.order_weights, dtype=np.float64)
        if order_weights.shape != (self.L + 1,):
            raise ValueError("order_weights must have shape ({},), got {}.".format(self.L + 1, order_weights.shape,))

        pi, _, _, _, gmm_effective = self._make_effective_gaussian_mixture_moments(theta, order_weights=np.ones(self.L + 1, dtype=np.float64), noise_covariance=noise_covariance,)

        model_terms = np.asarray(gmm_effective.compute_inner_product_of_moment_moment(), dtype=np.float64,)
        model_term_gradients_theta = np.zeros((self.L, theta.shape[0]), dtype=np.float64)

        for k in range(1, self.L + 1):
            unit_order_weights = np.zeros(self.L + 1, dtype=np.float64)
            unit_order_weights[k] = 1.0
            gmm_effective.order_weights = unit_order_weights

            grad_pi_k, grad_A_k, grad_V_k_effective = gmm_effective._compute_grad_F1()
            grad_V_k = grad_V_k_effective[:, : self.rank, :]
            model_term_gradients_theta[k - 1, :] = params_to_theta_grad(pi, grad_pi_k, grad_A_k, grad_V_k, self.softmax_reparam, self.softmax_temperature,)

        _, cross_terms_per_sample = gmm_effective.compute_inner_product_of_moment_vector(for_each_sample=True)
        grad_pi_per_order, grad_A_per_order, grad_V_per_order_effective = (gmm_effective._compute_grad_F2_per_sample_per_order())
        grad_V_per_order = grad_V_per_order_effective[:, :, :, : self.rank, :]

        weighted_cross_terms = np.zeros(self.L, dtype=np.float64)
        weighted_cross_gradients_theta = np.zeros((self.L, theta.shape[0]), dtype=np.float64)

        for k in range(1, self.L + 1):
            gradient_cloud_k = params_to_theta_grad_per_sample(pi, grad_pi_per_order[k, :, :], grad_A_per_order[k, :, :, :], grad_V_per_order[k, :, :, :, :], self.softmax_reparam, self.softmax_temperature,)

            weights_k = per_order_sample_weights[k - 1, :]
            weighted_cross_terms[k - 1] = float(weights_k @ cross_terms_per_sample[k, :])
            weighted_cross_gradients_theta[k - 1, :] = np.tensordot(weights_k, gradient_cloud_k, axes=(0, 0),)

        F_robust = float(
            np.sum(order_weights[1:] * (model_terms[1:] - 2.0 * weighted_cross_terms)))
        grad_F_wrt_theta_robust = np.sum(order_weights[1:, np.newaxis] * (model_term_gradients_theta - 2.0 * weighted_cross_gradients_theta), axis=0,)

        diagnostics = {
            "model_terms": model_terms[1:].copy(),
            "weighted_cross_terms": weighted_cross_terms.copy(),
            "order_weights": order_weights[1:].copy(),
            "sample_weights": per_order_sample_weights.copy(),
            "constant_term_omitted": True,
        }

        if return_diagnostics:
            return F_robust, grad_F_wrt_theta_robust, diagnostics
        return F_robust, grad_F_wrt_theta_robust

    def contamination_model_is_active(self):
        if self.contamination_epsilon is None:
            return False
        return bool(self.contamination_epsilon > 0)


    def sample_weights_are_uniform(self, sample_weights):
        sample_weights = self.normalize_sample_weights(sample_weights)
        return bool(np.max(np.abs(sample_weights - self.uniform_sample_weights()))<= 128.0 * np.finfo(np.float64).eps)



    def initialize_order_weights_robust(self, theta_init_current_step, W_option_current_step, W_current_step):
        if W_option_current_step == "given_order_weights":
            self.order_weights = np.asarray(W_current_step, dtype=np.float64).copy()

        elif W_option_current_step == "identity":
            self.order_weights = np.ones(self.L + 1, dtype=np.float64)

        elif W_option_current_step == "diagonal":
            self.order_weights = self.compute_order_weights_diagonal_robust(theta_init_current_step)

        elif W_option_current_step == "diagonal_truncation_average":
            if self.n_moment_conditions is None:
                self.n_moment_conditions_k = np.zeros(self.L + 1, dtype=int)
                for k in range(1, self.L + 1):
                    self.n_moment_conditions_k[k] = int(self.n_dim ** k)
                self.n_moment_conditions = int(np.sum(self.n_moment_conditions_k))
            if self.M_sample is None:
                self.M_sample = self.compute_sample_moments()

            pi_current_step, A_current_step, V_current_step = theta_to_params(theta_init_current_step, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature,)
            self.M_model = GaussianMixtureMoments(pi_current_step, A_current_step, V_current_step, self.X, self.L, None,).compute_model_moments()

            S_hat = self.compute_S_hat(theta_init_current_step)
            W_diag = np.diagonal(np.linalg.pinv(S_hat))
            index_start_k = 0
            index_end_k = 0
            self.order_weights = np.ones(self.L + 1, dtype=np.float64)
            for k in range(1, self.L + 1):
                index_start_k += self.n_moment_conditions_k[k - 1]
                index_end_k += self.n_moment_conditions_k[k]
                self.order_weights[k] = (W_diag[index_start_k:index_end_k].sum() / self.n_moment_conditions_k[k])

        else:
            raise NotImplementedError(
                "RobustDGMM currently supports only the order-weight options "
                "'given_order_weights', 'identity', 'diagonal', and "
                "'diagonal_truncation_average'."
            )

        self.order_weights = np.asarray(self.order_weights, dtype=np.float64)
        self._record_order_weights("initialize:{}".format(W_option_current_step))

    def one_step_robust_dgmm(self, theta_init_current_step, W_option_current_step, W_current_step, step_i=1,):
        self.initialize_order_weights_robust(theta_init_current_step, W_option_current_step, W_current_step,)

        theta_current = np.copy(theta_init_current_step)
        uniform_weights = self.uniform_sample_weights()
        current_per_order_weights = np.tile(uniform_weights, (self.L, 1))
        current_aggregate_weights = uniform_weights.copy()

        self.robust_theta_history.append(theta_current.copy())
        self.robust_weights_history.append(current_aggregate_weights.copy())
        self._record_per_order_weights(current_per_order_weights, "initialize_uniform")

        if step_i == 1:
            ftol = 1e-8
            gtol = 1e-8
        else:
            ftol = 1e-8 * 1e-02 ** (step_i - 1)
            gtol = 1e-8 * 1e-02 ** (step_i - 1)
            if ftol < 1e-14:
                ftol = 1e-14
            elif gtol < 1e-14:
                gtol = 1e-14

        remaining_iter = int(self.iter_max)
        reweighting_active = self.contamination_model_is_active()

        if reweighting_active and self.pre_reweight_before_optimization:
            new_per_order_weights, _, _ = self.update_sample_weights_per_order(theta_current)
            new_aggregate = np.mean(new_per_order_weights, axis=0)
            weight_change = float(np.max([np.linalg.norm(new_per_order_weights[k] - current_per_order_weights[k], ord=1) for k in range(self.L)]))

            self.robust_weight_change_history.append(weight_change)
            self.robust_weights_history.append(new_aggregate.copy())

            current_per_order_weights = new_per_order_weights
            current_aggregate_weights = new_aggregate
            self.robust_weights = current_aggregate_weights.copy()
            self._record_per_order_weights(current_per_order_weights, "pre_reweight", self.algorithm2_previous_per_order_centers,)

            if self.recompute_order_weights_after_reweight:
                self.maybe_refresh_order_weights(theta_current, W_option_current_step, current_per_order_weights, stage_name="pre_reweight",)

            if weight_change < self.robust_weight_tol:
                reweighting_active = False

        while remaining_iter > 0:
            if reweighting_active:
                burst_iter = min(self.reweight_interval, remaining_iter)
            else:
                burst_iter = remaining_iter

            use_base_objective = (reweighting_active is False and self.noise_model_is_active() is False and self.sample_weights_are_uniform(current_aggregate_weights))

            if use_base_objective:
                objective_function = self.F_and_grad_F_order_weights
            else:
                objective_function = lambda theta: self.F_and_grad_F_order_weights_robust(theta, sample_weights=current_per_order_weights,)

            result = minimize(fun=objective_function, x0=theta_current, args=(), method="L-BFGS-B", jac=True,
                options={"maxcor": 50, "maxiter": burst_iter, "maxls": 20, "disp": False, "ftol": ftol, "gtol": gtol,},
            )

            theta_next = np.asarray(result.x, dtype=np.float64)
            theta_change = np.linalg.norm(theta_next - theta_current)

            self.theta_opt_ = theta_next.copy()
            self.n_iter_ += int(result.nit)
            self.robust_objective_history.append(float(result.fun))
            self.robust_theta_change_history.append(float(theta_change))
            self.robust_theta_history.append(theta_next.copy())

            theta_current = theta_next

            iterations_used = max(int(result.nit), 1)
            remaining_iter -= iterations_used

            if reweighting_active and remaining_iter > 0:
                new_per_order_weights, _, _ = self.update_sample_weights_per_order(theta_current)
                new_aggregate = np.mean(new_per_order_weights, axis=0)
                weight_change = float(np.max([np.linalg.norm(new_per_order_weights[k] - current_per_order_weights[k], ord=1) for k in range(self.L)]))

                self.robust_weight_change_history.append(weight_change)
                self.robust_weights_history.append(new_aggregate.copy())

                current_per_order_weights = new_per_order_weights
                current_aggregate_weights = new_aggregate
                self.robust_weights = current_aggregate_weights.copy()
                self._record_per_order_weights(current_per_order_weights, "burst_reweight", self.algorithm2_previous_per_order_centers,)

                if self.recompute_order_weights_after_reweight:
                    self.maybe_refresh_order_weights(theta_current, W_option_current_step, current_per_order_weights, stage_name="burst_reweight",)

                if weight_change < self.robust_weight_tol:
                    reweighting_active = False

            if theta_change < self.step_tol and reweighting_active is False:
                break

        self.theta_opt_ = theta_current.copy()
        self.robust_weights = self.normalize_sample_weights(current_aggregate_weights)
        self.robust_per_order_weights = current_per_order_weights.copy()

        if self.store_per_order_weight_history and len(self.robust_per_order_weights_history) == 0:
            self._record_per_order_weights(self.robust_per_order_weights, "final")

        if (len(self.robust_weights_history) == 0 or np.linalg.norm(self.robust_weights_history[-1] - self.robust_weights, ord=1) > 0):
            self.robust_weights_history.append(self.robust_weights.copy())

    def maybe_compute_moment_sums(self):
        if self.auto_compute_moment_sums is False:
            return

        uses_diagonal_weights = False
        for weight_option in (self.W_init, self.W_step):
            if isinstance(weight_option, str) and weight_option == "diagonal":
                uses_diagonal_weights = True
                break

        if uses_diagonal_weights is False:
            return

        if self.vec_vec_sum is not None and self.vec_vec_diag is not None:
            return

        if self.n_points <= self.max_points_for_exact_moment_sums:
            self.vec_vec_sum, self.vec_vec_diag = compute_exact_moment_sums(self.X, self.L, block_size=self.moment_sum_block_size,)
        else:
            self.vec_vec_sum, self.vec_vec_diag = compute_moment_sums(self.X, self.L, self.n_components, self.rank, m_landmarks=None,)

    def i_step_robust_dgmm(self):
        if type(self.W_init) != str:
            if np.ndim(self.W_init) == 1:
                W_option_current_step = "given_order_weights"
                W_current_step = np.copy(self.W_init)
            else:
                raise NotImplementedError("RobustDGMM does not support a full weighting matrix in W_init.")
        else:
            W_option_current_step = self.W_init
            W_current_step = None

        theta_init_current_step = np.copy(self.theta_init)

        for i in range(1, self.step_max + 1):
            self.one_step_robust_dgmm(theta_init_current_step, W_option_current_step, W_current_step, step_i=i,)

            if np.linalg.norm(self.theta_opt_ - theta_init_current_step) < self.step_tol:
                break

            if type(self.W_step) != str:
                if np.ndim(self.W_step) == 1:
                    W_option_current_step = "given_order_weights"
                    W_current_step = np.copy(self.W_step)
                else:
                    raise NotImplementedError("RobustDGMM does not support a full weighting matrix in W_step.")
            else:
                W_option_current_step = self.W_step
                W_current_step = None

            theta_init_current_step = np.copy(self.theta_opt_)

    def fit(self, X):
        self.reset_robust_state()

        self.X = np.asarray(X, dtype=np.float64)
        if self.X.ndim != 2:
            raise ValueError("X must have shape (n_dim, n_points).")
        self.n_dim, self.n_points = self.X.shape

        if self.noise_covariance is None:
            self.noise_covariance = np.zeros((self.n_dim, self.n_dim), dtype=np.float64)
        else:
            self.noise_covariance = np.asarray(self.noise_covariance, dtype=np.float64)

        if self.noise_covariance is not None:
            if self.noise_covariance.shape != (self.n_dim, self.n_dim):
                raise ValueError("noise_covariance must have shape ({}, {}), got {}.".format(self.n_dim, self.n_dim, self.noise_covariance.shape,))

        self.maybe_compute_moment_sums()
        self.robust_weights = self.uniform_sample_weights()

        if self.W_step is None or self.step_max == 1:
            if type(self.W_init) != str:
                if np.ndim(self.W_init) == 1:
                    W_option_current_step = "given_order_weights"
                    W_current_step = np.copy(self.W_init)
                else:
                    raise NotImplementedError(
                        "RobustDGMM does not support a full weighting matrix in W_init."
                    )
            else:
                W_option_current_step = self.W_init
                W_current_step = None

            self.one_step_robust_dgmm(self.theta_init, W_option_current_step, W_current_step, step_i=1,
            )
        else:
            self.i_step_robust_dgmm()

        return self.theta_opt_, self.n_iter_

    def fit_robust(self, X):
        """Compatibility alias with the unfinished robust method in ``gmm.py``."""
        return self.fit(X)


__all__ = ["RobustDGMM"]
