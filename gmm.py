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
from scipy.optimize import minimize
from functools import reduce

from robust_gradient_reweighting import RobustGradientReweighting
from gm_moments import GaussianMixtureMoments
from utils import theta_to_params, params_to_theta, params_to_theta_grad
from NystromApprox import compute_moment_sums, compute_exact_moment_sums

from sympy.utilities.iterables import multiset_permutations
from sklearn.metrics import normalized_mutual_info_score

# -----------------------------------------------------------------------------
# The following line is added for DGMM to set seed for reproducibility: 
import random
SEED = 4142
random.seed(SEED)                  # Python built-in RNG
np.random.seed(SEED)               # NumPy global RNG (for any legacy code)
rng = np.random.default_rng(SEED)  # New NumPy Generator
# -----------------------------------------------------------------------------

class GMM(object):

    def __init__(self, 
                 theta_init, 
                 rank, 
                 n_components, 
                 L = 4, 
                 W_init = "diagonal", 
                 W_step = 'diagonal', 
                 step_max = 10, 
                 step_tol = 1e-04, 
                 iter_max = 200, 
                 softmax_reparam = True, softmax_temperature = 1.5, 
                 vec_vec_sum = None, vec_vec_diag = None, M_sample = None, M_model = None):
        """ Initialize the GMM objective wrapper.

        Parameters
        ----------
        theta_init : array
            The array collecting all initial parameters of the mixture model.
        
        rank : int
            The maximum rank of the covariance matrices of the mixture components. 
        
        n_components: int 
            The number of mixture components. 
        
        L : int
            The highest moment order, that is, to match moments of order 1, 2, ..., L.
        
        W_init : array or str 
            Weighting matrix option for the initial step:
            - given_order_weights, array of shape (L+1)
            - given_W, array of shape (n_moment_conditions, n_moment_conditions)
            - 'identity'
            - 'diagonal'
            - 'oracle' 
            - 'diagonal_truncation'
            - 'diagonal_truncation_average'
            - 'identity_direct'
            - 'full_inverse'
        
        W_step : array or str 
            Weighting matrix option for the subsequent step:
            - given_order_weights, array of shape (L+1)
            - given_W, array of shape (n_moment_conditions, n_moment_conditions)
            - 'identity'
            - 'diagonal'
            - 'oracle' 
            - 'diagonal_truncation'
            - 'diagonal_truncation_average'
            - 'identity_direct'
            - 'full_inverse'
        
        step_max : int
            The maximum number of steps for the GMoM estimation.
        
        step_tol : float
            The tolerance for the convergence of the GMoM estimation.

        softmax_reparam, softmax_temperature: parameters used in theta_to_params.
            - softmax_reparam: if True, the softmax reparameterization is used.
            - softmax_temperature: the temperature used in the softmax reparameterization.

        (optiopnal) vec_vec_sum : array, shape (2L + 1, n_points)
            Pre-computed sum_n' <y_n^⊗k, y_n'^⊗k> for k = 0, 1, 2, ..., 2L + 1 and n = 0, ..., n_points - 1. (None if not already pre-computed)

        (optional) vec_vec_diag :
            Pre-computed <y_n^⊗k, y_n^⊗k> for k = 0, 1, 2, ..., 2L + 1 and n = 0, ..., n_points - 1. (None if not already pre-computed)

        (optional) M_sample : list of sample moments. (None if not already pre-computed)
        
        (optional) M_model : list of model moments. (None if not already pre-computed)
        
        Attributes
        ----------
        theta_opt_ : array
            The optimized parameters of the model after fitting.
        n_iter_ : int
            The total number of iterations performed during the optimization.
    
        Methods
        ----------
        compute_order_weights_diagonal_GMM(theta, X = None, L = None)
            Compute for order_weights from diagonal GMM.
        compute_sample_moments(X = None)
            Compute sample moments up to order L for data X (shape n_dim x n_points).
        compute_sample_moments_one_point(x)
            Compute sample moments up to order L for one data point x.
        compute_moment_residue_vec(theta, X = None, L = None)
            Compute moment residue vector f(theta, y_n).
        compute_S_hat(theta, X = None, L = None)
            Compute estimated asymptotic variance.
        F_and_grad_F_order_weights(theta)
            Returns the objective function value and its gradient given theta, when the order weights are used.
        F_direct(theta)
            Returns the objective function value and its gradient given theta, when the weighting matrix is used.
        one_step_gmm(theta_init_current_step, W_option_current_step, W_current_step, step_i = 1)
            One step of GMM estimation.
        i_step_gmm()
            Iterative GMM estimation.
        fit(X)
            Fit the GMM to the data.
        
        ----------------------
        v1.x.x features:
        score_samples(X, theta)
            Return the per-sample likelihood of the data under the model.
        predict(X)
            Predict label for data.
        acc(y_pred_true, y_pred_est)           
            Compute clustering accuracy (ACC) between ground-truth and predicted labels.
        nmi(y_pred_true, y_pred_est)
            Compute Normalized Mutual Information (NMI) between ground-truth and predicted labels.
        bic(X)
            Compute the Bayesian Information Criterion (BIC) for the GMM model.
        aic(X)
            Compute the Akaike Information Criterion (AIC) for the GMM model.
        """
        
        
        # input initial parameters:
        self.theta_init = theta_init
        
        # to store the results:
        self.n_iter_ = 0
        self.theta_opt_ = None

        # input data to be given later:
        self.X, self.n_dim, self.n_points = None, None, None

        # input hyperparameters:
        self.rank = rank
        self.n_components = n_components
        self.L = L
        self.W_init = W_init
        self.W_step = W_step
        self.step_max = step_max
        self.step_tol = step_tol 
        self.iter_max = iter_max
        self.softmax_reparam = softmax_reparam
        self.softmax_temperature = softmax_temperature
        
        # optional parameters:
        self.vec_vec_sum = vec_vec_sum
        self.vec_vec_diag = vec_vec_diag
        self.M_sample = M_sample
        self.M_model = M_model

        # to be computed later if needed:
        self.order_weights = None
        self.W = None
        self.n_moment_conditions = None
        self.n_moment_conditions_k = None

    

    def compute_order_weights_diagonal_GMM(self, theta, X = None, L = None):
        """ Compute for order_weights from diagonal GMM.
        Returns
        -------
        order_weights : array, shape (L+1)
        """
        if self.X is None:
            self.X = X
            self.n_dim, self.n_points = self.X.shape
        if self.L is None:
            self.L = L
        self.order_weights = np.zeros(self.L+1)

        pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)
        gmm = GaussianMixtureMoments(pi, A, V, self.X, self.L, np.ones(self.L+1))
        
        mom_mom = gmm.compute_inner_product_of_moment_moment()
        # mom_mom is of shape (L+1)

        # when computing order_weights, the order_weights in GaussianMixtureMoments = None.
        mom_vec, mom_vec_n = gmm.compute_inner_product_of_moment_vector(for_each_sample=True)
        # mom_vec is of shape (L+1)
        # mom_vec_n is of shape (L+1, n_points)

        if self.vec_vec_sum is None or self.vec_vec_diag is None:
            raise RuntimeError(
                "vec_vec_sum and vec_vec_diag are required for diagonal DGMM weights. "
                "Pre-compute them with compute_exact_moment_sums or compute_moment_sums "
                "before calling compute_order_weights_diagonal_GMM."
            )
        # elif self.vec_vec_sum == 'exact' or self.vec_vec_diag == 'exact':
        #     self.vec_vec_sum, self.vec_vec_diag = compute_exact_moment_sums(self.X, self.L)
        # elif self.vec_vec_sum == 'approx' or self.vec_vec_diag == 'approx':
        #     self.vec_vec_sum, self.vec_vec_diag = compute_moment_sums(self.X, self.L, self.n_components, self.rank, m_landmarks=None)
        expected_shape = (2 * self.L + 1, self.n_points)
        if np.asarray(self.vec_vec_sum).shape != expected_shape:
            raise ValueError(
                "vec_vec_sum must have shape {}, got {}.".format(
                    expected_shape, np.asarray(self.vec_vec_sum).shape
                )
            )

        for k in range(1, self.L+1):

            numerator_k = 0
            denominator_k = 0

            if self.vec_vec_diag is not None:
                for n in range(self.n_points):
                    numerator_k += mom_mom[k] - 2 * mom_vec_n[k, n] + self.vec_vec_diag[k, n]
            
            else:
                for n in range(self.n_points):
                    numerator_k += mom_mom[k] - 2 * mom_vec_n[k, n] + np.inner(self.X[:, n], self.X[:, n]) ** k

            numerator_k = numerator_k / self.n_points

            for k_prime in range(1, self.L+1):
                denominator_k += 2 * mom_vec[k] * mom_vec[k_prime]
                
                for n in range(self.n_points):
                    denominator_k += self.n_points * mom_mom[k] * mom_mom[k_prime] \
                                    - 2 * mom_mom[k]       * mom_vec[k_prime] \
                                    - 2 * mom_mom[k_prime] * mom_vec[k] 
                    
                    denominator_k += mom_mom[k]            * self.vec_vec_sum[k_prime, n] \
                                + mom_mom[k_prime]      * self.vec_vec_sum[k, n] \
                            + 2 * self.n_points * mom_vec_n[k, n]       * mom_vec_n[k_prime, n] \
                            - 2 * mom_vec_n[k, n]       * self.vec_vec_sum[k_prime, n] \
                            - 2 * mom_vec_n[k_prime, n] * self.vec_vec_sum[k, n] \
                                + self.vec_vec_sum[k + k_prime, n]
            denominator_k = denominator_k / (self.n_points ** 2)
            self.order_weights[k] = numerator_k / denominator_k

    def compute_sample_moments(self, X = None):
        """
        Compute sample moments up to order L for data X (shape d×N).

        Returns
        -------
        M_sample : list
            M_sample[k] is the k-th moment tensor:
            - M_sample[0] = 1.0, by convention
            - M_sample[1] shape (n_dim,)
            - M_sample[2] shape (n_dim, n_dim)
            - …
            - M_sample[L] shape (n_dim,) * L
        """
        if self.X is None:
            self.X = X
            self.n_dim, self.n_points = self.X.shape

        M_sample = [1.0]  # 0th moment = 1 by convention

        for k in range(1, self.L+1):
            shape = (self.n_dim,) * k
            M_k = np.zeros(shape, dtype=float)

            for i in range(self.n_points):
                x = self.X[:, i]
                # compute x ⊗ x ⊗ … (k times)
                outer_x_k = reduce(lambda a, b: np.multiply.outer(a, b), [x]*k)
                M_k += outer_x_k

            M_k /= self.n_points
            M_sample.append(M_k)

        return M_sample
    
    def compute_sample_moments_one_point(self, x):
        """
        Compute sample moments up to order L for one data point x.

        Returns
        -------
        M_sample : list
            M_sample[k] is the k-th moment tensor:
            - M_sample[0] = 1.0, by convention
            - M_sample[1] shape (n_dim,)
            - M_sample[2] shape (n_dim, n_dim)
            - …
            - M_sample[L] shape (n_dim,) * L
        """
        if self.n_dim is None:
            self.n_dim = x.shape[0]
  
        M_sample_one_point = [1.0]  # 0th moment = 1 by convention

        for k in range(1, self.L+1):
            shape = (self.n_dim,) * k
            M_k = np.zeros(shape, dtype=float)

            # compute x \otimes x \otimes … \otimes x (k times)
            outer_x_k = reduce(lambda a, b: np.multiply.outer(a, b), [x]*k)
            M_k += outer_x_k
                
            M_sample_one_point.append(M_k)

        return M_sample_one_point
    

    def compute_moment_residue_vec(self, theta, X = None, L = None):
        """ Compute moment residue vector f(theta, y_n).

        Returns
        -------
        moment_residue_vec : array, shape (n_moment_conditions, 1)
        """
        if self.X is None:
            self.X = X
            self.n_dim, self.n_points = self.X.shape
        if self.L is None:
            self.L = L
        if self.M_sample is None:
            self.M_sample = self.compute_sample_moments()

        pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)
        self.M_model = GaussianMixtureMoments(pi, A, V, self.X, self.L, None).compute_model_moments()

        moment_residue_vec = np.ravel((self.M_model[1] - self.M_sample[1])).reshape(self.n_moment_conditions_k[1], 1)
        for k in range(2, self.L+1):
            moment_residue_vec_k = np.ravel((self.M_model[k] - self.M_sample[k])).reshape(self.n_moment_conditions_k[k], 1)
            moment_residue_vec = np.vstack((moment_residue_vec, moment_residue_vec_k))
        return moment_residue_vec
    
    def compute_S_hat(self, theta, X = None, L = None):
        """ Compute estimated asymptotic variance, S_hat.

        Returns
        -------
        S_hat : array, shape (n_moment_conditions, n_moment_conditions)
        """
        if self.X is None:
            self.X = X
            self.n_dim, self.n_points = self.X.shape
        if self.L is None:
            self.L = L
        if self.n_moment_conditions is None:
                self.n_moment_conditions_k = np.zeros(self.L+1, dtype=int)
                for k in range(1, self.L+1):
                    self.n_moment_conditions_k[k] = int(self.n_dim ** k)
                self.n_moment_conditions = int(np.sum(self.n_moment_conditions_k))
        if self.M_sample is None:
            self.M_sample = self.compute_sample_moments()


        pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)    
        self.M_model = GaussianMixtureMoments(pi, A, V, self.X, self.L, None).compute_model_moments()
    
        S_hat = np.zeros((self.n_moment_conditions, self.n_moment_conditions))
        moment_residue_all_points = np.zeros((self.n_moment_conditions, self.n_points))

        for i in range(self.n_points):
            M_sample_one_point = self.compute_sample_moments_one_point(self.X[:, i])
            moment_residue_one_point = np.ravel((self.M_model[1] - M_sample_one_point[1])).reshape(self.n_moment_conditions_k[1], 1)
            for k in range(2, self.L+1):
                moment_residue_one_point_k = np.ravel((self.M_model[k] - M_sample_one_point[k])).reshape(self.n_moment_conditions_k[k], 1)
                moment_residue_one_point = np.vstack((moment_residue_one_point, moment_residue_one_point_k))
            moment_residue_all_points[:, i] = moment_residue_one_point[:, 0]
        moment_residue_all_points_centered = moment_residue_all_points - moment_residue_all_points.mean(axis = 1).reshape(-1,1)
        
        for i in range(self.n_points):
            # when computing the cov, outer product first, then sum:
            # .reshape(-1,1) makes a column vector
            # .reshape(1,-1) makes a row vector
            S_hat += moment_residue_all_points_centered[:,i].reshape(-1,1) @ moment_residue_all_points_centered[:,i].reshape(1,-1)
        S_hat = (1/self.n_points) * S_hat
        # S_hat = (moment_residue_all_points_centered @ moment_residue_all_points_centered.T) / self.n_points

        return S_hat

    def F_and_grad_F_order_weights(self, theta):
        """
        Returns the objective function value and its gradient given theta, when the order weights are used.
        """
        pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)

        F, grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V = GaussianMixtureMoments(pi, A, V, self.X, self.L, self.order_weights).F_and_grad_F()

        grad_F_wrt_theta = params_to_theta_grad(pi, grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V, self.softmax_reparam, self.softmax_temperature)

        return F, grad_F_wrt_theta
    
    def F_direct(self, theta):
        """
        Returns the objective function value and its gradient given theta, when the weighting matrix is used.
        """
        moment_residue_vec = self.compute_moment_residue_vec(theta)

        F = moment_residue_vec.T @ self.W @ moment_residue_vec

        # print("residuals min/max:", moment_residue_vec.min(), moment_residue_vec.max())
        # print("W cond:", np.linalg.cond(self.W ))
        # assert np.all(np.isfinite(moment_residue_vec)), "moment_residue_vec contains NaN/Inf"
        # assert np.all(np.isfinite(self.W )), "W contains NaN/Inf"

        F = float(F[0, 0]) # convert to scalar

        return F
    

    def one_step_gmm(self, theta_init_current_step, W_option_current_step, W_current_step, step_i = 1):
        """ One step of GMM estimation.

        Returns
        -------
        gmm_objective, float
            GMM objective function, F = g^T W g.
        """
        
        if W_option_current_step == 'given_order_weights':
            self.order_weights = W_current_step
                
        elif W_option_current_step == 'identity':
            self.order_weights = np.ones(self.L+1)

        elif W_option_current_step == 'diagonal':
            # Implicit computation of order_weights:
            self.compute_order_weights_diagonal_GMM(theta_init_current_step)
        
        else:
            if self.n_moment_conditions is None:
                self.n_moment_conditions_k = np.zeros(self.L+1, dtype=int)
                for k in range(1, self.L+1):
                    self.n_moment_conditions_k[k] = int(self.n_dim ** k)
                self.n_moment_conditions = int(np.sum(self.n_moment_conditions_k))
            if self.M_sample is None:
                self.M_sample = self.compute_sample_moments()

            pi_current_step, A_current_step, V_current_step = theta_to_params(theta_init_current_step, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)
            self.M_model = GaussianMixtureMoments(pi_current_step, A_current_step, V_current_step, self.X, self.L, None).compute_model_moments()

            if W_option_current_step == 'given_W':
                self.W = np.copy(W_current_step)
            elif W_option_current_step == 'identity_direct':
                self.W = np.identity(self.n_moment_conditions)
            else:
                S_hat = self.compute_S_hat(theta_init_current_step)

                if W_option_current_step == 'full_inverse':
                    self.W = np.linalg.pinv(S_hat, rcond = 1e-10)
                
                elif W_option_current_step == 'diagonal_truncation':
                    self.W = np.diag(np.diagonal(np.linalg.pinv(S_hat)))
                
                elif W_option_current_step == 'diagonal_truncation_average':
                    W_diag = np.diagonal(np.linalg.pinv(S_hat))
                    index_start_k = 0 
                    index_end_k = 0 
                    self.order_weights = np.ones(self.L+1)
                    for k in range(1, self.L+1):
                        index_start_k += self.n_moment_conditions_k[k-1] # 0, n_dim, n_dim + n_moment_conditions_k[2]
                        index_end_k += self.n_moment_conditions_k[k] # n_dim, n_dim + n_moment_conditions_k[2], n_dim + n_moment_conditions_k[2] + n_moment_conditions_k[3]
                        self.order_weights[k] = W_diag[index_start_k : index_end_k].sum() / self.n_moment_conditions_k[k]

        # optimization if order weights are used:
        if W_option_current_step == 'diagonal' or W_option_current_step == 'identity' or W_option_current_step == 'given_order_weights' or W_option_current_step == 'diagonal_truncation_average':
            # print("order_weights used in the optimization:")
            # print(self.order_weights)
            if step_i == 1:
                ftol =   1e-8
                gtol =  1e-8
            else:
                ftol =  1e-8 * 1e-02 ** (step_i - 1)
                gtol =  1e-8 * 1e-02 ** (step_i - 1)
                if ftol < 1e-14:
                    ftol = 1e-14
                elif gtol < 1e-14:
                    gtol = 1e-14
            # L-BFGS-B optimization:
            result = minimize(fun=self.F_and_grad_F_order_weights, x0=theta_init_current_step, args=(), method="L-BFGS-B", 
                                             jac=True, options={'maxcor':50, 'maxiter': self.iter_max, 'maxls': 20, 'disp': True, 'ftol': ftol, 'gtol': gtol}) # 2.220446049250313e-9
            # we use the scipy default maxls = 20
            self.theta_opt_ = result.x
            self.n_iter_ += result.nit
            # print("L-BFGS successful?:" + str(result.success))
            # print("L-BFGS iterations:" + str(result.nit))
        
        # optimization if a weighting matrix is used:
        else:
            # self.theta_opt_ = manopt(theta_init, X, rank, n_components, A_true, covariances_true, L, order_weights)
            if step_i == 1:
                ftol =   1e-8
                gtol =  1e-8
            else:
                ftol =  1e-8 * 1e-02 ** (step_i - 1)
                gtol =  1e-8 * 1e-02 ** (step_i - 1)
                if ftol < 1e-14:
                    ftol = 1e-14
                elif gtol < 1e-14:
                    gtol = 1e-14

            # Note: using finite-difference gradient approximation:
            result = minimize(fun = self.F_direct, x0 = theta_init_current_step, args = (), method="L-BFGS-B", 
                                                   jac="2-point", 
                                                   options={'maxcor':50, 'maxiter': self.iter_max, 'maxls': 20, 'disp': True, 'ftol': ftol, 'gtol': gtol}) # 2.220446049250313e-9
            
            self.theta_opt_ = result.x 
            self.n_iter_ += result.nit
            # print("L-BFGS successful?:" + str(result.success))
            # print("L-BFGS iterations:" + str(result.nit))
    
    def i_step_gmm(self):

        if type(self.W_init) != str:
            if np.ndim(self.W_init) == 1:
                W_option_current_step = 'given_order_weights'
                W_current_step = np.copy(self.W_init)
            elif np.ndim(self.W_init) == 2:
                W_option_current_step = 'given_W'
                W_current_step = np.copy(self.W_init)
        else:
            W_option_current_step = self.W_init
            W_current_step = None
        theta_init_current_step = np.copy(self.theta_init)
        
        for i in range(1, self.step_max+1):

            self.one_step_gmm(theta_init_current_step, W_option_current_step, W_current_step, i)

            # check convergence:
            if np.linalg.norm(self.theta_opt_ - theta_init_current_step) < self.step_tol:
                # if convergence is reached, stop the loop:
                break
            else:
                # if convergence is not reached, update W and theta_init for the next step:
                if type(self.W_step) != str:
                    if np.ndim(self.W_step) == 1:
                        W_option_current_step = 'given_order_weights'
                        W_current_step = np.copy(self.W_step)
                    elif np.ndim(self.W_step) == 2:
                        W_option_current_step = 'given_W'
                        W_current_step = np.copy(self.W_step)
                else:
                    W_option_current_step = self.W_step
                theta_init_current_step = np.copy(self.theta_opt_)


    def score_samples(self, X, theta = None):
        """Return the per-sample likelihood of the data under the model.

        Compute the log probability of X under the model and
        return the posterior distribution (responsibilities) of each
        mixture component for each element of X.

        Returns
        -------
        logprob_GMM  : array_like, shape (n_points,)
            Log probabilities of each data point in X.

        responsibilities : array_like, shape (n_points, n_components)
            Posterior probabilities of each mixture component for each
            observation
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        if theta is None:
            theta = self.theta_opt_

        pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)

        logprob_Gaussian = GaussianMixtureMoments(pi, A, V, self.X, self.L).compute_logprob_Gaussian()

        logprob_GMM = GaussianMixtureMoments(pi, A, V, self.X, self.L).compute_logprob_GMM()

        responsibilities = np.exp(
            logprob_Gaussian
            + np.log(np.maximum(pi, np.finfo(np.float64).tiny))
            - logprob_GMM[:, np.newaxis]
        )
        return logprob_GMM, responsibilities


    def fit(self, X):
        """Fit the GMM to the data.
        Parameters
        ----------
        X : array-like, shape (n_dim, n_points)
            The input data.

        Returns
        -------
        self.theta_opt_ : array, shape depending on the model
            The optimized parameters of the model.
        """
        
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        # if isinstance(self.L, tuple):
        #     self.L_min, self.L_max = self.L
        
        if self.W_step is None or self.step_max == 1:
            self.one_step_gmm(self.theta_init, self.W_init, None, 1)
        else:
            self.i_step_gmm()

        return self.theta_opt_, self.n_iter_

    def fit_predict(self, X):
        """Fit the GMM to the data and predict labels.
        Parameters
        ----------
        X : array-like, shape (n_dim, n_points)
            The input data.

        Returns
        -------
        labels : array, shape (n_points,)
            The predicted labels for each data point.
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        self.fit(self.X)
        logprob, responsibilities = self.score_samples(self.X, self.theta_opt_)
        
        return responsibilities.argmax(axis=1)
    
    def predict(self, X, theta = None):
        """Predict label for data.
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        if theta is None:
            theta = self.theta_opt_

        logprob, responsibilities = self.score_samples(X, theta)
        return responsibilities.argmax(axis=1)


    def prediction_scores(self, X, labels_true, labels_pred = None, theta = None):
        """
        Try every permutation of the K components: permute the weights pi,
        the means A, and the covariance stack, then predict & score.
        Returns the best accuracy and the argmax permutation tuple.
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        # if we already have labels_pred, no need to provide theta, and we can skip the permutation search
        if labels_pred is not None:
            # Compute accuracy
            acc = (labels_pred == labels_true).sum() / self.n_points
            nmi = normalized_mutual_info_score(labels_true, labels_pred)

        else:
            if theta is None:
                theta = self.theta_opt_

            pi, A, V = theta_to_params(theta, self.n_dim, self.rank, self.n_components, self.softmax_reparam, self.softmax_temperature)
            
            acc = 0
            indices = None

            for indices_permuted in multiset_permutations(np.array(range(self.n_components))):
                
                pi_permuted = np.take(pi, indices_permuted, axis=-1)
                A_permuted = np.take(A, indices_permuted, axis=-1)
                V_permuted = np.take(V, indices_permuted, axis=-1)

                theta_permuted = params_to_theta(pi_permuted, A_permuted, V_permuted, self.softmax_reparam, self.softmax_temperature)

                # predict under this re-ordering of components
                labels_permuted = self.predict(X, theta_permuted)
                
                # Compute accuracy
                acc_permuted = (labels_permuted == labels_true).sum() / self.n_points

                if acc_permuted > acc:
                    labels_pred = labels_permuted.copy()
                    acc  = acc_permuted
                    indices = indices_permuted
                    nmi = normalized_mutual_info_score(labels_true, labels_pred)

        return labels_pred, acc, nmi, indices


    def bic(self, X):
        """
        Compute the Bayesian Information Criterion (BIC) for the GMM model.
        Parameters
        ----------
            X: array-like of shape (n_samples, n_features), input data
        Returns
        -------
            bic: BIC value (float)
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        self.fit(self.X)
        logprob_GMM, _ = self.score_samples(self.X, self.theta_opt_)

        # Compute BIC
        bic = -2 * np.sum(logprob_GMM) + self.theta_opt_.shape[0] * np.log(self.n_points)
        # Note: BIC = -2 * log-likelihood + number of parameters * log(n_points)

        return bic

    def aic(self, X):
        """
        Compute the Akaike Information Criterion (AIC) for the GMM model.
        Parameters
        ----------
            X: array-like of shape (n_samples, n_features), input data
        Returns
        -------
            aic: AIC value (float)
        """
        self.X = X
        self.n_dim, self.n_points = self.X.shape

        self.fit(self.X)

        logprob_GMM, _ = self.score_samples(self.X, self.theta_opt_)

        # Compute AIC
        aic = -2 * np.sum(logprob_GMM) + 2 * self.theta_opt_.shape[0]
        # Note: AIC = -2 * log-likelihood + 2 * number of parameters
        return aic
