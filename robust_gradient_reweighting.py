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


import math
import warnings
from typing import Literal, Optional
import numpy as np


StepSizeStrategy = Literal["max_safe", "theorem"]
InitialCenterStrategy = Literal["geometric_median", "coordinatewise_median", "sample_mean",]


class RobustGradientReweighting:
    def __init__(
        self,
        contamination_epsilon: float,
        n_points: Optional[int] = None,
        dim_p: Optional[int] = None,
        eta_rho: Optional[float] = None,
        eta_w: Optional[float] = None,
        step_max_outer: int = 10,
        step_max_inner: int = 50,
        threshold_const: Optional[float] = None,
        target_accuracy: float = 1e-4,
        min_outer_iterations_for_stabilization: int = 5,
        stabilization_patience: int = 2,
        use_exact_diameter: bool = False,
        geom_median_tol: float = 1e-8,
        geom_median_max_iter: int = 200,
        initial_center_strategy: InitialCenterStrategy = "geometric_median",
        use_outer_center_safeguard: bool = False,
        center_safeguard_max_backtracking: int = 12,
        center_safeguard_reduction_factor: float = 0.5,
        center_safeguard_objective_tol: float = 1e-12,
        stop_on_rejected_center_update: bool = False,
        warm_start_inner_weights: bool = True,
        warn_on_theory_gap: bool = True,
        projection_tol: float = 1e-12,
        projection_max_iter: int = 200,
        verbose: bool = False,
        step_size_strategy: StepSizeStrategy = "max_safe",
    ) -> None:
        if not np.isfinite(contamination_epsilon) or contamination_epsilon < 0.0 or contamination_epsilon >= 0.5:
            raise ValueError("contamination_epsilon must satisfy 0 <= contamination_epsilon < 1/2.")
        if n_points is not None and n_points <= 0:
            raise ValueError("n_points must be positive when provided.")
        if dim_p is not None and dim_p <= 0:
            raise ValueError("dim_p must be positive when provided.")
        if eta_rho is not None and (not np.isfinite(eta_rho) or eta_rho < 0.0):
            raise ValueError("eta_rho must be nonnegative when provided.")
        if eta_w is not None and (not np.isfinite(eta_w) or eta_w < 0.0):
            raise ValueError("eta_w must be nonnegative when provided.")
        if step_max_outer <= 0:
            raise ValueError("step_max_outer must be a positive integer.")
        if step_max_inner <= 0:
            raise ValueError("step_max_inner must be a positive integer.")
        if threshold_const is not None and (
            not np.isfinite(threshold_const) or threshold_const < 0.0
        ):
            raise ValueError("threshold_const must be nonnegative when provided.")
        if not np.isfinite(target_accuracy) or target_accuracy < 0.0:
            raise ValueError("target_accuracy must be nonnegative.")
        if min_outer_iterations_for_stabilization < 0:
            raise ValueError(
                "min_outer_iterations_for_stabilization must be nonnegative."
            )
        if stabilization_patience <= 0:
            raise ValueError("stabilization_patience must be positive.")
        if not np.isfinite(geom_median_tol) or geom_median_tol <= 0.0:
            raise ValueError("geom_median_tol must be strictly positive.")
        if geom_median_max_iter <= 0:
            raise ValueError("geom_median_max_iter must be positive.")
        if initial_center_strategy not in {
            "geometric_median",
            "coordinatewise_median",
            "sample_mean",
        }:
            raise ValueError(
                "initial_center_strategy must be 'geometric_median', "
                "'coordinatewise_median', or 'sample_mean'."
            )
        if center_safeguard_max_backtracking < 0:
            raise ValueError("center_safeguard_max_backtracking must be nonnegative.")
        if not np.isfinite(center_safeguard_reduction_factor) or not (
            0.0 < center_safeguard_reduction_factor < 1.0
        ):
            raise ValueError(
                "center_safeguard_reduction_factor must satisfy 0 < factor < 1."
            )
        if (
            not np.isfinite(center_safeguard_objective_tol)
            or center_safeguard_objective_tol < 0.0
        ):
            raise ValueError("center_safeguard_objective_tol must be nonnegative.")
        if not np.isfinite(projection_tol) or projection_tol <= 0.0:
            raise ValueError("projection_tol must be strictly positive.")
        if projection_max_iter <= 0:
            raise ValueError("projection_max_iter must be positive.")
        if step_size_strategy not in {"max_safe", "theorem"}:
            raise ValueError("step_size_strategy must be 'max_safe' or 'theorem'.")

        if warn_on_theory_gap and contamination_epsilon >= (1.0 / 3.0):
            warnings.warn(
                "contamination_epsilon >= 1/3 lies outside the strict contraction regime proved "
                "by Lemma 6.22 / Theorem 6.25 in robustness.pdf. The algorithm is "
                "still executed, but the outer-loop guarantee is then heuristic.",
                RuntimeWarning,
                stacklevel=2,
            )

        self.contamination_epsilon = float(contamination_epsilon)
        self.n_points = n_points
        self.dim_p = dim_p
        self.eta_rho = eta_rho
        self.eta_w = eta_w
        self.step_max_outer = int(step_max_outer)
        self.step_max_inner = int(step_max_inner)
        self.threshold_const = threshold_const
        self.target_accuracy = float(target_accuracy)
        self.min_outer_iterations_for_stabilization = int(
            min_outer_iterations_for_stabilization
        )
        self.stabilization_patience = int(stabilization_patience)
        self.use_exact_diameter = bool(use_exact_diameter)
        self.geom_median_tol = float(geom_median_tol)
        self.geom_median_max_iter = int(geom_median_max_iter)
        self.initial_center_strategy = initial_center_strategy
        self.use_outer_center_safeguard = bool(use_outer_center_safeguard)
        self.center_safeguard_max_backtracking = int(center_safeguard_max_backtracking)
        self.center_safeguard_reduction_factor = float(
            center_safeguard_reduction_factor
        )
        self.center_safeguard_objective_tol = float(center_safeguard_objective_tol)
        self.stop_on_rejected_center_update = bool(stop_on_rejected_center_update)
        self.warm_start_inner_weights = bool(warm_start_inner_weights)
        self.warn_on_theory_gap = bool(warn_on_theory_gap)
        self.projection_tol = float(projection_tol)
        self.projection_max_iter = int(projection_max_iter)
        self.verbose = bool(verbose)
        self.step_size_strategy = step_size_strategy

        self.per_sample_gradients: Optional[np.ndarray] = None

        self.normalizing_scale: Optional[float] = None
        self.eta_rho_effective: Optional[float] = None
        self.eta_w_effective: Optional[float] = None
        self.weight_cap_: Optional[float] = None
        self.sample_weights: Optional[np.ndarray] = None
        self.fixed_center: Optional[np.ndarray] = None
        self.weighted_mean: Optional[np.ndarray] = None
        self.location_estimate: Optional[np.ndarray] = None
        self.initial_geometric_median: Optional[np.ndarray] = None
        self.initial_center: Optional[np.ndarray] = None
        self.weighted_second_moment: Optional[np.ndarray] = None
        self.weighted_covariance_about_mean: Optional[np.ndarray] = None

        self.weighted_covariance: Optional[np.ndarray] = None
        self.spectral_norm: Optional[float] = None

        self.sample_weights_history: list[np.ndarray] = []
        self.fixed_center_history: list[np.ndarray] = []
        self.spectral_norm_history: list[float] = []
        self.location_estimate_history: list[np.ndarray] = []
        self.center_update_step_size_history: list[float] = []
        self.center_update_accepted_history: list[bool] = []
        self.center_safeguard_score_history: list[float] = []
        self.weight_change_history: list[float] = []
        self.center_shift_history: list[float] = []
        self.relative_center_shift_history: list[float] = []
        self.averaged_inner_loss_history: list[float] = []

        self.n_iter_ = 0
        self.n_outer_iter_ = 0
        self.converged_ = False
        self.stopped_reason_: Optional[str] = None

    def _uniform_weights(self) -> np.ndarray:
        if self.n_points is None:
            raise RuntimeError("n_points must be known before uniform weights are created.")
        return np.full(self.n_points, 1.0 / self.n_points, dtype=np.float64)

    def _validate_initial_sample_weights(self,initial_sample_weights: Optional[np.ndarray],) -> Optional[np.ndarray]:
        if initial_sample_weights is None:
            return None
        if self.n_points is None:
            raise RuntimeError(
                "n_points must be initialized before initial_sample_weights are validated."
            )

        weights = np.asarray(initial_sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (self.n_points,):
            raise ValueError(
                "initial_sample_weights must have shape ({},), got {}.".format(
                    self.n_points,
                    weights.shape,
                )
            )
        if not np.all(np.isfinite(weights)):
            raise ValueError("initial_sample_weights must contain only finite values.")
        if np.any(weights < 0.0):
            raise ValueError("initial_sample_weights must be nonnegative.")

        total_mass = float(np.sum(weights))
        if total_mass <= 0.0:
            raise ValueError("initial_sample_weights must sum to a positive value.")
        weights = weights / total_mass

        if self.contamination_epsilon <= 0.0:
            return self._uniform_weights()

        cap = 1.0 / ((1.0 - self.contamination_epsilon) * self.n_points)
        if np.max(weights) > cap + 1024.0 * np.finfo(np.float64).eps:
            weights = self._project_onto_capped_simplex_relative_entropy(weights)
        return weights

    def _validate_initial_fixed_center(self,initial_fixed_center: Optional[np.ndarray],) -> Optional[np.ndarray]:
        if initial_fixed_center is None:
            return None
        if self.dim_p is None:
            raise RuntimeError("dim_p must be initialized before initial_fixed_center is validated.")

        center = np.asarray(initial_fixed_center, dtype=np.float64).reshape(-1)
        if center.shape != (self.dim_p,):
            raise ValueError("initial_fixed_center must have shape ({},), got {}.".format(self.dim_p, center.shape,))
        if not np.all(np.isfinite(center)):
            raise ValueError("initial_fixed_center must contain only finite values.")
        return center

    def _reset_fit_state(self) -> None:
        self.sample_weights = self._uniform_weights()
        self.fixed_center = None
        self.weighted_mean = None
        self.location_estimate = None
        self.initial_geometric_median = None
        self.initial_center = None
        self.weighted_second_moment = None
        self.weighted_covariance_about_mean = None
        self.weighted_covariance = None
        self.spectral_norm = None
        self.sample_weights_history = []
        self.fixed_center_history = []
        self.spectral_norm_history = []
        self.location_estimate_history = []
        self.center_update_step_size_history = []
        self.center_update_accepted_history = []
        self.center_safeguard_score_history = []
        self.weight_change_history = []
        self.center_shift_history = []
        self.relative_center_shift_history = []
        self.averaged_inner_loss_history = []
        self.n_iter_ = 0
        self.n_outer_iter_ = 0
        self.converged_ = False
        self.stopped_reason_ = None

    def _validate_input_gradients(self, per_sample_gradients: np.ndarray) -> np.ndarray:
        gradients = np.asarray(per_sample_gradients, dtype=np.float64)
        if gradients.ndim != 2:
            raise ValueError("per_sample_gradients must have shape (n_points, dim_p).")
        if gradients.shape[0] == 0 or gradients.shape[1] == 0:
            raise ValueError("per_sample_gradients must be nonempty.")
        if not np.all(np.isfinite(gradients)):
            raise ValueError("per_sample_gradients must contain only finite values.")

        n_points, dim_p = gradients.shape
        if self.n_points is None:
            self.n_points = n_points
        if self.dim_p is None:
            self.dim_p = dim_p

        if n_points != self.n_points:
            raise ValueError(f"Expected n_points = {self.n_points}, got {n_points}.")
        if dim_p != self.dim_p:
            raise ValueError(f"Expected dim_p = {self.dim_p}, got {dim_p}.")

        return gradients

    def _compute_geometric_median(self, gradients: np.ndarray) -> np.ndarray:
        center = np.mean(gradients, axis=0)

        for _ in range(self.geom_median_max_iter):
            differences = gradients - center[None, :]
            distances = np.linalg.norm(differences, axis=1)

            close_mask = distances <= self.geom_median_tol
            if np.any(close_mask):
                return gradients[np.argmin(distances)].copy()

            inverse_distances = 1.0 / distances
            next_center = (np.sum(gradients * inverse_distances[:, None], axis=0) / np.sum(inverse_distances))

            if np.linalg.norm(next_center - center) <= (self.geom_median_tol * max(1.0, np.linalg.norm(center))):
                return next_center

            center = next_center

        return center

    @staticmethod
    def _compute_coordinatewise_median(gradients: np.ndarray) -> np.ndarray:
        return np.median(gradients, axis=0).astype(np.float64, copy=False)

    def _initialize_fixed_center(self, gradients: np.ndarray) -> np.ndarray:
        geometric_median = self._compute_geometric_median(gradients)
        self.initial_geometric_median = geometric_median.copy()

        if self.initial_center_strategy == "geometric_median":
            center = geometric_median
        elif self.initial_center_strategy == "coordinatewise_median":
            center = self._compute_coordinatewise_median(gradients)
        else:
            center = np.mean(gradients, axis=0)

        self.initial_center = np.asarray(center, dtype=np.float64).copy()
        return self.initial_center.copy()

    def _compute_normalizing_scale(self, gradients: np.ndarray) -> float:
        if self.n_points is None:
            raise RuntimeError("n_points must be known before computing the normalizing scale.")
        if self.n_points <= 1:
            return 0.0

        if self.use_exact_diameter:
            n = gradients.shape[0]
            if n <= 4096:
                sq_norms = np.sum(gradients * gradients, axis=1)  # (N,)
                gram = gradients @ gradients.T                     # (N, N) via BLAS
                pairwise_sq = sq_norms[:, None] + sq_norms[None, :] - 2.0 * gram
                return float(np.max(pairwise_sq))
            else:
                normalizing_scale = 0.0
                sq_norms = np.sum(gradients * gradients, axis=1)
                block_size = min(512, n)
                for start in range(0, n, block_size):
                    end = min(start + block_size, n)
                    gram_block = gradients[start:end] @ gradients.T  # (block, N)
                    pairwise_block = sq_norms[start:end, None] + sq_norms[None, :] - 2.0 * gram_block
                    normalizing_scale = max(normalizing_scale, float(np.max(pairwise_block)))
                return normalizing_scale

        reference_center = np.mean(gradients, axis=0)
        centered = gradients - reference_center[None, :]
        squared_radii = np.sum(centered * centered, axis=1)
        return 4.0 * float(np.max(squared_radii))

    @staticmethod
    def _squared_radius_about_center(gradients: np.ndarray, center: np.ndarray) -> float:
        centered = gradients - center[None, :]
        return float(np.max(np.sum(centered * centered, axis=1)))

    def _augment_normalizing_scale_for_center(self, center: np.ndarray) -> None:
        if self.per_sample_gradients is None:
            raise RuntimeError("per_sample_gradients must be initialized before scale augmentation.")
        if self.normalizing_scale is None:
            raise RuntimeError("normalizing_scale must be initialized before scale augmentation.")
        center_radius_sq = self._squared_radius_about_center(self.per_sample_gradients, np.asarray(center, dtype=np.float64),)
        self.normalizing_scale = max(float(self.normalizing_scale), center_radius_sq)

    def _initialize_step_sizes(self) -> None:
        if self.normalizing_scale is None:
            raise RuntimeError(
                "normalizing_scale must be computed before step sizes are initialized."
            )
        if self.dim_p is None:
            raise RuntimeError("dim_p must be known before step sizes are initialized.")

        if self.normalizing_scale <= 0.0:
            self.eta_w_effective = 0.0
            self.eta_rho_effective = 0.0
            return

        eta_cap = 0.5 / self.normalizing_scale

        if self.eta_w is None:
            if self.step_size_strategy == "theorem":
                log_cap = (math.log(1.0 / (1.0 - self.contamination_epsilon)) if self.contamination_epsilon > 0.0 else 0.0)
                eta_w = math.sqrt(log_cap / self.step_max_inner) / self.normalizing_scale
            else:
                eta_w = eta_cap
        else:
            eta_w = float(self.eta_w)

        if self.eta_rho is None:
            if self.dim_p == 1:
                eta_rho = 0.0
            elif self.step_size_strategy == "theorem":
                log_dim = math.log(float(self.dim_p))
                eta_rho = math.sqrt(log_dim / self.step_max_inner) / self.normalizing_scale
            else:
                eta_rho = eta_cap
        else:
            eta_rho = float(self.eta_rho)

        self.eta_w_effective = min(eta_w, eta_cap)
        self.eta_rho_effective = min(eta_rho, eta_cap) if self.dim_p > 1 else 0.0

    @staticmethod
    def _compute_weighted_mean(gradients: np.ndarray, weights: np.ndarray) -> np.ndarray:
        return weights @ gradients

    @staticmethod
    def _symmetrize(matrix: np.ndarray) -> np.ndarray:
        return 0.5 * (matrix + matrix.T)

    def _compute_weighted_second_moment(self,centered_gradients: np.ndarray, weights: np.ndarray,) -> np.ndarray:
        weighted_centered_gradients = centered_gradients * weights[:, None]
        gain_matrix = weighted_centered_gradients.T @ centered_gradients
        return self._symmetrize(gain_matrix)

    def _compute_spectral_norm(self, symmetric_matrix: np.ndarray) -> float:
        eigenvalues = np.linalg.eigvalsh(self._symmetrize(symmetric_matrix))
        return float(np.max(eigenvalues))

    def _compute_density_matrix(self, cumulative_gain_matrix: np.ndarray) -> np.ndarray:
        if self.dim_p is None:
            raise RuntimeError("dim_p must be known before computing the density matrix.")
        if self.dim_p == 1:
            return np.ones((1, 1), dtype=np.float64)
        if self.eta_rho_effective == 0.0:
            return np.eye(self.dim_p, dtype=np.float64) / float(self.dim_p)

        scaled_matrix = self.eta_rho_effective * self._symmetrize(cumulative_gain_matrix)
        eigenvalues, eigenvectors = np.linalg.eigh(scaled_matrix)
        shifted_eigenvalues = eigenvalues - np.max(eigenvalues)
        exp_eigenvalues = np.exp(shifted_eigenvalues)
        trace_exp = float(np.sum(exp_eigenvalues))
        if not np.isfinite(trace_exp) or trace_exp <= 0.0:
            raise FloatingPointError("Failed to compute a valid Gibbs density matrix.")

        density_matrix = ((eigenvectors * exp_eigenvalues.reshape(1, -1)) @ eigenvectors.T) / trace_exp
        density_matrix = self._symmetrize(density_matrix)
        density_matrix /= float(np.trace(density_matrix))
        return density_matrix

    @staticmethod
    def _compute_loss_values(centered_gradients: np.ndarray, density_matrix: np.ndarray) -> np.ndarray:
        projected = centered_gradients @ density_matrix
        losses = np.sum(projected * centered_gradients, axis=1)
        return np.maximum(losses, 0.0)

    def _compute_trimmed_location_objective(self, gradients: np.ndarray, center: np.ndarray,) -> float:
        if self.n_points is None:
            raise RuntimeError("n_points must be known before computing safeguard objectives.")

        trim_count = max(1, int(math.floor((1.0 - self.contamination_epsilon) * self.n_points)))
        distances = np.linalg.norm(gradients - center[None, :], axis=1)
        trimmed_distances = np.partition(distances, trim_count - 1)[:trim_count]
        return float(np.sum(trimmed_distances))

    def _safeguarded_center_update(self, gradients: np.ndarray, current_center: np.ndarray, proposed_center: np.ndarray,) -> tuple[np.ndarray, bool, float, float]:
        current_center = np.asarray(current_center, dtype=np.float64)
        proposed_center = np.asarray(proposed_center, dtype=np.float64)
        update_direction = proposed_center - current_center

        current_objective = self._compute_trimmed_location_objective(gradients, current_center)
        if np.linalg.norm(update_direction) <= self.geom_median_tol:
            return current_center.copy(), True, 0.0, current_objective

        tolerance = self.center_safeguard_objective_tol * max(1.0, current_objective)

        step_size = 1.0
        for _ in range(self.center_safeguard_max_backtracking + 1):
            candidate_center = current_center + step_size * update_direction
            candidate_objective = self._compute_trimmed_location_objective(gradients, candidate_center,)
            if candidate_objective <= current_objective + tolerance:
                return candidate_center, True, step_size, candidate_objective
            step_size *= self.center_safeguard_reduction_factor

        return current_center.copy(), False, 0.0, current_objective

    def _project_onto_capped_simplex_relative_entropy(self, weights_tilde: np.ndarray,) -> np.ndarray:
        if self.n_points is None:
            raise RuntimeError("n_points must be known before projecting onto the capped simplex.")

        positive_weights = np.maximum(weights_tilde, np.finfo(np.float64).tiny)

        cap = 1.0 / ((1.0 - self.contamination_epsilon) * self.n_points)
        order = np.argsort(-positive_weights)
        sorted_weights = positive_weights[order]
        suffix_sums = np.cumsum(sorted_weights[::-1])[::-1]

        projected_sorted = None
        tol = self.projection_tol

        for n_capped in range(self.n_points + 1):
            remaining_mass = 1.0 - n_capped * cap
            if remaining_mass < -tol:
                break

            free_sum = float(suffix_sums[n_capped]) if n_capped < self.n_points else 0.0
            if free_sum <= 0.0:
                continue

            scale = remaining_mass / free_sum
            if scale < 0.0:
                continue

            left_ok = (n_capped == 0 or scale * sorted_weights[n_capped - 1] >= cap - tol)
            right_ok = (n_capped == self.n_points or scale * sorted_weights[n_capped] <= cap + tol)
            if not (left_ok and right_ok):
                continue

            projected_sorted = np.empty_like(sorted_weights)
            if n_capped > 0:
                projected_sorted[:n_capped] = cap
            if n_capped < self.n_points:
                projected_sorted[n_capped:] = scale * sorted_weights[n_capped:]
            break

        if projected_sorted is None:

            def projected_mass(scale: float) -> float:
                return float(np.sum(np.minimum(cap, scale * positive_weights)))

            scale_low = 0.0
            scale_high = 1.0
            while projected_mass(scale_high) < 1.0:
                scale_high *= 2.0

            for _ in range(self.projection_max_iter):
                scale_mid = 0.5 * (scale_low + scale_high)
                if projected_mass(scale_mid) < 1.0:
                    scale_low = scale_mid
                else:
                    scale_high = scale_mid
                if scale_high - scale_low <= tol * max(1.0, scale_high):
                    break

            scale = 0.5 * (scale_low + scale_high)
            projected = np.minimum(cap, scale * positive_weights)
        else:
            projected = np.empty_like(projected_sorted)
            projected[order] = projected_sorted

        projected = np.maximum(projected, 0.0)
        projected = np.minimum(projected, cap)
        mass = float(np.sum(projected))
        if not np.isfinite(mass) or mass <= 0.0:
            raise FloatingPointError("Projection onto the capped simplex failed.")

        projected /= mass
        return projected

    @staticmethod
    def _relative_center_shift(current_center: np.ndarray, next_center: np.ndarray, reference_scale: float,) -> float:
        numerator = float(np.linalg.norm(next_center - current_center))
        denominator = max(reference_scale, float(np.linalg.norm(current_center)), float(np.linalg.norm(next_center)), )
        return numerator / denominator

    def fit(self, per_sample_gradients: np.ndarray, initial_sample_weights: Optional[np.ndarray] = None, initial_fixed_center: Optional[np.ndarray] = None,) -> "RobustGradientReweighting":
        self.per_sample_gradients = self._validate_input_gradients(per_sample_gradients)
        self._reset_fit_state()
        self.weight_cap_ = 1.0 / ((1.0 - self.contamination_epsilon) * self.n_points)

        initial_sample_weights = self._validate_initial_sample_weights(initial_sample_weights)
        initial_fixed_center = self._validate_initial_fixed_center(initial_fixed_center)

        self.normalizing_scale = self._compute_normalizing_scale(self.per_sample_gradients)

        if initial_fixed_center is None:
            current_fixed_center = self._initialize_fixed_center(self.per_sample_gradients)
        else:
            current_fixed_center = initial_fixed_center.copy()
            self.initial_geometric_median = current_fixed_center.copy()
            self.initial_center = current_fixed_center.copy()

        self._augment_normalizing_scale_for_center(current_fixed_center)
        self._initialize_step_sizes()

        if self.normalizing_scale <= 0.0:
            shared_gradient = np.mean(self.per_sample_gradients, axis=0)
            zero_matrix = np.zeros((self.dim_p, self.dim_p), dtype=np.float64)
            self.fixed_center = shared_gradient.copy()
            self.weighted_mean = shared_gradient.copy()
            self.location_estimate = shared_gradient.copy()
            self.initial_geometric_median = shared_gradient.copy()
            self.initial_center = shared_gradient.copy()
            self.weighted_second_moment = zero_matrix.copy()
            self.weighted_covariance_about_mean = zero_matrix.copy()
            self.weighted_covariance = zero_matrix.copy()
            self.spectral_norm = 0.0
            self.converged_ = True
            self.stopped_reason_ = "degenerate_gradient_cloud"
            return self

        if initial_sample_weights is None:
            previous_outer_weights = self._uniform_weights()
        else:
            previous_outer_weights = initial_sample_weights.copy()
            self.sample_weights = previous_outer_weights.copy()

        consecutive_stable_outer_steps = 0

        for outer_step in range(self.step_max_outer):
            self.n_outer_iter_ = outer_step + 1
            self.fixed_center_history.append(current_fixed_center.copy())

            centered_gradients = self.per_sample_gradients - current_fixed_center[None, :]

            centered_gradients_T = centered_gradients.T 

            if self.warm_start_inner_weights and (outer_step > 0 or initial_sample_weights is not None):
                inner_weights = previous_outer_weights.copy()
            else:
                inner_weights = self._uniform_weights()

            cumulative_gain_matrix = np.zeros((self.dim_p, self.dim_p), dtype=np.float64)
            averaged_weights = np.zeros(self.n_points, dtype=np.float64)
            averaged_gain_matrix = np.zeros((self.dim_p, self.dim_p), dtype=np.float64)
            average_loss_value = 0.0

            eta_w_eff = self.eta_w_effective
            if eta_w_eff is None:
                raise RuntimeError("eta_w_effective was not initialized.")
            eta_rho_eff = self.eta_rho_effective
            dim_p = self.dim_p

            for _ in range(self.step_max_inner):
                self.n_iter_ += 1

                weighted_T = centered_gradients_T * inner_weights[None, :] 
                gain_matrix = weighted_T @ centered_gradients 
                gain_matrix = 0.5 * (gain_matrix + gain_matrix.T)

                averaged_weights += inner_weights
                averaged_gain_matrix += gain_matrix
                cumulative_gain_matrix += gain_matrix

                if dim_p == 1:
                    density_matrix = np.ones((1, 1), dtype=np.float64)
                elif eta_rho_eff == 0.0:
                    density_matrix = np.eye(dim_p, dtype=np.float64) / float(dim_p)
                else:
                    scaled_matrix = eta_rho_eff * cumulative_gain_matrix
                    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (scaled_matrix + scaled_matrix.T))
                    shifted = eigenvalues - eigenvalues[-1]  
                    exp_eig = np.exp(shifted)
                    trace_exp = float(np.sum(exp_eig))
                    density_matrix = (
                        (eigenvectors * exp_eig[None, :]) @ eigenvectors.T
                    ) / trace_exp

                projected = centered_gradients @ density_matrix  
                loss_values = np.sum(projected * centered_gradients, axis=1)
                np.maximum(loss_values, 0.0, out=loss_values)
                average_loss_value += float(inner_weights @ loss_values)

                multiplicative_factors = np.maximum(1.0 - eta_w_eff * loss_values, 0.0,)
                weights_tilde = inner_weights * multiplicative_factors
                inner_weights = self._project_onto_capped_simplex_relative_entropy(weights_tilde)

            outer_weights = averaged_weights / self.step_max_inner
            averaged_gain_matrix /= self.step_max_inner
            average_loss_value /= self.step_max_inner

            weighted_mean = self._compute_weighted_mean(self.per_sample_gradients, outer_weights)
            centered_about_weighted_mean = (self.per_sample_gradients - weighted_mean[None, :])
            weighted_covariance_about_mean = self._compute_weighted_second_moment(centered_about_weighted_mean,outer_weights,)
            spectral_norm = self._compute_spectral_norm(averaged_gain_matrix)

            previous_weights = self.sample_weights.copy()
            weight_change = float(np.linalg.norm(outer_weights - previous_weights, ord=1))
            center_shift = float(np.linalg.norm(weighted_mean - current_fixed_center))
            if self.normalizing_scale is None:
                raise RuntimeError("normalizing_scale must be available when the outer loop runs.")
            reference_scale = max(math.sqrt(self.normalizing_scale), self.geom_median_tol)
            relative_center_shift = self._relative_center_shift(current_fixed_center, weighted_mean, reference_scale,)

            self.sample_weights_history.append(outer_weights.copy())
            self.spectral_norm_history.append(spectral_norm)
            self.averaged_inner_loss_history.append(average_loss_value)
            self.weight_change_history.append(weight_change)
            self.center_shift_history.append(center_shift)
            self.relative_center_shift_history.append(relative_center_shift)

            self.sample_weights = outer_weights.copy()
            self.fixed_center = current_fixed_center.copy()
            self.weighted_mean = weighted_mean.copy()
            self.location_estimate = weighted_mean.copy()
            self.weighted_second_moment = averaged_gain_matrix.copy()
            self.weighted_covariance_about_mean = weighted_covariance_about_mean.copy()
            self.weighted_covariance = averaged_gain_matrix.copy()
            self.spectral_norm = spectral_norm

            if self.verbose:
                print(
                    f"Outer iteration {outer_step + 1:02d}: "
                    f"spectral_norm = {spectral_norm:.6e}, "
                    f"weight_change_L1 = {weight_change:.6e}, "
                    f"center_shift = {center_shift:.6e}, "
                    f"relative_center_shift = {relative_center_shift:.6e}"
                )

            if self.threshold_const is not None and spectral_norm <= self.threshold_const:
                self.location_estimate_history.append(self.location_estimate.copy())
                self.center_update_step_size_history.append(1.0)
                self.center_update_accepted_history.append(True)
                self.center_safeguard_score_history.append(np.nan)
                self.converged_ = True
                self.stopped_reason_ = "spectral_norm_threshold"
                break

            if self.threshold_const is None:
                stabilization_ready = (self.n_outer_iter_ >= self.min_outer_iterations_for_stabilization)
                is_stable_this_round = (stabilization_ready and weight_change <= self.target_accuracy and relative_center_shift <= self.target_accuracy)
                if is_stable_this_round:
                    consecutive_stable_outer_steps += 1
                else:
                    consecutive_stable_outer_steps = 0

                if consecutive_stable_outer_steps >= self.stabilization_patience:
                    self.location_estimate_history.append(self.location_estimate.copy())
                    self.center_update_step_size_history.append(1.0)
                    self.center_update_accepted_history.append(True)
                    self.center_safeguard_score_history.append(np.nan)
                    self.converged_ = True
                    self.stopped_reason_ = "outer_stabilized"
                    break

            if self.use_outer_center_safeguard:
                next_center, update_accepted, step_size, safeguard_score = (self._safeguarded_center_update(self.per_sample_gradients, current_fixed_center, weighted_mean,))
                self.center_update_step_size_history.append(step_size)
                self.center_update_accepted_history.append(update_accepted)
                self.center_safeguard_score_history.append(safeguard_score)
                self.location_estimate_history.append(self.location_estimate.copy())

                if self.verbose:
                    print(
                        f"    safeguard: accepted = {update_accepted}, "
                        f"step_size = {step_size:.6e}, score = {safeguard_score:.6e}"
                    )

                if (not update_accepted) and self.stop_on_rejected_center_update:
                    self.converged_ = False
                    self.stopped_reason_ = "center_update_rejected"
                    break

                current_fixed_center = next_center.copy()
            else:
                current_fixed_center = weighted_mean.copy()
                self.location_estimate_history.append(self.location_estimate.copy())
                self.center_update_step_size_history.append(1.0)
                self.center_update_accepted_history.append(True)
                self.center_safeguard_score_history.append(np.nan)

            previous_outer_weights = outer_weights.copy()

        if self.stopped_reason_ is None:
            self.stopped_reason_ = "max_outer_iterations_reached"
            self.converged_ = False

        return self
