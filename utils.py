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
from random import gauss
from sympy.utilities.iterables import multiset_permutations
import math
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
from scipy.optimize import linear_sum_assignment
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from collections import OrderedDict


def softmax(x, softmax_temperature, axis=None):
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp((x - x_max) / softmax_temperature)
    sum_e_x = np.sum(e_x, axis=axis, keepdims=True)
    return e_x / sum_e_x

def symmetrization(tensor):
    n_order = len(tensor.shape)
    tensor_symmetrized = np.zeros(tensor.shape)

    indices = np.array(range(n_order))
    for indices_perm in multiset_permutations(indices):
        tensor_symmetrized += np.transpose(tensor, indices_perm)
    tensor_symmetrized = tensor_symmetrized / math.factorial(n_order)

    return tensor_symmetrized

def theta_to_params(theta, n_dim, rank, n_components, softmax_reparam, softmax_temperature):

    if softmax_reparam is True:
        pi_reparam = np.reshape(theta[0:n_components], (1, n_components))
        pi = softmax(pi_reparam, softmax_temperature)
    
    else: 
        pi = np.reshape(theta[0:n_components], (1, n_components))
    
    A = np.reshape(theta[n_components:(n_components + n_dim * n_components)], (n_components, n_dim)).T

    V = np.zeros((n_dim, rank, n_components))
    count = 0
    for j in range(n_components):
        index_start = n_components + n_dim * n_components + count * n_dim * rank
        index_end = n_components + n_dim * n_components + (count + 1) * n_dim * rank
        V[:,:,j] = np.reshape(theta[index_start:index_end], (rank, n_dim)).T
        count += 1

    return pi, A, V

def params_to_theta(pi, A, V, softmax_reparam, softmax_temperature):

    n_dim, n_components = A.shape

    if softmax_reparam is True:
        pi_safe = np.maximum(np.asarray(pi, dtype=np.float64), np.finfo(np.float64).tiny)
        pi_safe = pi_safe / np.sum(pi_safe, axis=1, keepdims=True)
        pi_reparam = softmax_temperature * np.log(pi_safe)  # Shape: (1, n_components)
    
        theta = pi_reparam.flatten(order='F')   
    
    else: 
        theta = pi.flatten(order='F') 
    
    theta = np.hstack((theta, A.flatten(order='F')))

    for j in range(n_components):
        theta = np.hstack((theta, V[:,:,j].flatten(order='F')))

    return theta

def params_to_theta_grad(pi, grad_F_wrt_pi, grad_F_wrt_A, grad_F_wrt_V, softmax_reparam, softmax_temperature):
    
    n_dim, n_components = grad_F_wrt_A.shape

    if softmax_reparam is True: 
        s = pi.flatten()
        J = (1.0/softmax_temperature) * (np.diag(s) - np.outer(s, s))
        grad_F_wrt_pi_reparam = grad_F_wrt_pi @ J

        grad_F_wrt_theta = grad_F_wrt_pi_reparam.flatten(order='F')

    else: 
        grad_F_wrt_theta = grad_F_wrt_pi.flatten(order='F')

    grad_F_wrt_theta = np.hstack((grad_F_wrt_theta, grad_F_wrt_A.flatten(order='F'))) 

    for j in range(n_components):
        grad_F_wrt_theta = np.hstack((grad_F_wrt_theta, grad_F_wrt_V[:,:,j].flatten(order='F')))

    return grad_F_wrt_theta

def params_to_theta_grad_per_sample(pi, grad_pi_n, grad_A_n, grad_V_n, softmax_reparam, softmax_temperature):
    N, K = grad_pi_n.shape
    n_dim = grad_A_n.shape[1]
    
    if softmax_reparam:
        s = pi.flatten() 
        J = (1.0/softmax_temperature) * (np.diag(s) - np.outer(s, s))
        grad_pi_flat = grad_pi_n @ J
    else:
        grad_pi_flat = grad_pi_n
        
    parts = [grad_pi_flat]
    grad_A_flat = grad_A_n.transpose(0, 2, 1).reshape(N, -1)
    parts.append(grad_A_flat)
    

    for j in range(K):
        g_V_j = grad_V_n[:, :, :, j]
        g_V_j_flat = g_V_j.transpose(0, 2, 1).reshape(N, -1)
        parts.append(g_V_j_flat)
        
    return np.hstack(parts)

def compute_average_error_no_permutation(pi, A, covariances, pi_true, A_true, covariances_true):
    n_dim, n_components = A.shape

    avg_error_pi = 0
    avg_error_A = 0
    avg_error_covariances = 0

    for j in range(n_components):
        avg_error_pi += np.linalg.norm(pi[0,j] - pi_true[0,j]) / np.linalg.norm(pi_true[:,j])
        avg_error_A += np.linalg.norm(A[:,j] - A_true[:,j]) / np.linalg.norm(A_true[:,j])
        avg_error_covariances += np.linalg.norm(covariances[:,:,j] - covariances_true[:,:,j]) / np.linalg.norm(covariances_true[:,:,j])

    avg_error_pi /= n_components
    avg_error_A /= n_components
    avg_error_covariances /= n_components

    return avg_error_pi, avg_error_A, avg_error_covariances

def compute_average_error(pi, A, covariances, pi_true, A_true, covariances_true):
    n_dim, n_components = A.shape
    avg_error_pi = 0 
    avg_error_A = 0
    avg_error_covariances = 0

    cost_matrix = np.zeros((n_components, n_components), dtype=np.float64)
    for j1 in range(n_components):
        for j2 in range(n_components):
            denominator = np.linalg.norm(covariances_true[:,:,j2])
            if denominator <= 0.0:
                denominator = np.finfo(np.float64).eps
            cost_matrix[j1, j2] = np.linalg.norm(covariances[:,:,j1] - covariances_true[:,:,j2]) / denominator

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    index = np.zeros(n_components, dtype=int)
    index[row_ind] = col_ind

    for j1 in range(n_components):
        avg_error_covariances += cost_matrix[j1, index[j1]]

        avg_error_pi += np.linalg.norm(pi[0,j1] - pi_true[0,index[j1]]) / np.linalg.norm(pi_true[0,index[j1]])
        avg_error_A += np.linalg.norm(A[:,j1] - A_true[:,index[j1]]) / np.linalg.norm(A_true[:,index[j1]])

    avg_error_pi /= n_components
    avg_error_A /= n_components
    avg_error_covariances /= n_components

    return avg_error_pi, avg_error_A, avg_error_covariances, index.tolist()

def make_rand_vector(dims):
    vec = np.asarray([gauss(0, 1) for i in range(dims)], dtype=np.float64)
    mag = np.linalg.norm(vec)
    if mag <= 0.0:
        raise ValueError("The sampled vector has zero norm.")
    res = vec / mag
    return res / np.linalg.norm(res)

def make_rand_vector_no_normalization(dims):
    vec = np.asarray([gauss(0, 1) for i in range(dims)], dtype=np.float64)
    mag = np.linalg.norm(vec)
    if mag <= 0.0:
        raise ValueError("The sampled vector has zero norm.")
    return vec / mag

def generate_ground_truth(n_dim, n_components, n_points, rank_min, rank_max, w_true_min, w_true_max, normalization = True, center_radius_min = 1, center_radius_max = 1):

    pi_true = np.zeros((1, n_components))
    # to prevent extremely small or large mixing probabilities:
    for j in range(n_components - 1):
        pi_true[0,j] = np.random.uniform(0.1, 0.9/(n_components-1))
    pi_true[0,-1] = 1 - np.sum(pi_true[0,:])


    A_true = np.zeros((n_dim, n_components))
    V_true = np.zeros((n_dim, rank_max, n_components))
    covariances_true = np.zeros((n_dim, n_dim, n_components))

    if normalization == True:
        for j in range(n_components):
            A_true[:, j] = make_rand_vector(n_dim) * np.random.uniform(center_radius_min, center_radius_max)
    else:
        for j in range(n_components):
            A_true[:, j] = make_rand_vector_no_normalization(n_dim)
    
    ranks_true = np.random.choice(range(rank_min, rank_max+1), size=n_components, replace=True)
    
    w_true = np.zeros((n_components, rank_max))

    for j in range(n_components):
        G = np.random.normal(0, 1, (n_dim, n_dim))
        U, _, V_T = np.linalg.svd(G)

        rank_j = ranks_true[j]
        V_true[:, 0:rank_j, j] = (U @ V_T)[:,0:rank_j]

        for r in range(rank_j):
            w_true[j, r] = np.random.uniform(w_true_min[j], w_true_max[j])
            V_true[:, r, j] = V_true[:, r, j] / np.linalg.norm(V_true[:, r, j]) * w_true[j, r]

        covariances_true[:,:,j] = V_true[:,:,j] @ V_true[:,:,j].T

    labels_true = np.zeros((n_points), dtype=int)

    X = np.zeros((n_dim, n_points))
    for i in range(n_points):
        component_index = np.random.choice(range(n_components), p = pi_true[0,:])
        
        labels_true[i] = component_index

        x_i = np.random.multivariate_normal(A_true[:, component_index], covariances_true[:, :, component_index], size=1)
        X[:, i] = x_i[0,:]

    return X, pi_true, A_true, V_true, covariances_true, ranks_true, w_true, labels_true

def generate_initial_parameters(n_dim, rank, n_components, init_method='random', X = None):

    pi_init = np.ones((1, n_components)) * 1/n_components
    A_init = np.zeros((n_dim, n_components))
    V_init = np.zeros((n_dim, rank, n_components))
    covariances_init = np.zeros((n_dim, n_dim, n_components))

    if init_method == 'random':

        for j in range(n_components):
            A_init[:, j] = make_rand_vector(n_dim)
            G = np.random.normal(0, 1, (n_dim, n_dim))
            U, _, V_T = np.linalg.svd(G)
            V_init[:, :, j] = (U @ V_T)[:,0:rank]
            for r in range(rank):
                V_init[:, r, j] = V_init[:, r, j] / np.linalg.norm(V_init[:, r, j])
            covariances_init[:,:,j] = V_init[:,:,j] @ V_init[:,:,j].T

    elif init_method == 'kmeans':
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=n_components, random_state=42).fit(X.T)

        centroids = kmeans.cluster_centers_.T  # shape (n_dim, n_components)
        A_init = centroids

        for j in range(n_components):
            G = np.random.normal(0, 1, (n_dim, n_dim))
            U, _, V_T = np.linalg.svd(G)
            V_init[:, :, j] = (U @ V_T)[:,0:rank]
            for r in range(rank):
                V_init[:, r, j] = V_init[:, r, j] / np.linalg.norm(V_init[:, r, j])
            covariances_init[:,:,j] = V_init[:,:,j] @ V_init[:,:,j].T

    elif init_method == 'kmeans_plusplus':
        from sklearn.cluster import kmeans_plusplus

        centroids, _ = kmeans_plusplus(X[:, :].T, n_clusters=n_components, random_state=42)
        A_init = centroids.T

        for j in range(n_components):
            G = np.random.normal(0, 1, (n_dim, n_dim))
            U, _, V_T = np.linalg.svd(G)
            V_init[:, :, j] = (U @ V_T)[:,0:rank]
            for r in range(rank):
                V_init[:, r, j] = V_init[:, r, j] / np.linalg.norm(V_init[:, r, j])
            covariances_init[:,:,j] = V_init[:,:,j] @ V_init[:,:,j].T

    else:
        raise ValueError("Invalid init_method. Choose 'random', 'kmeans', or 'kmeans_plusplus'.")

    return pi_init, A_init, V_init, covariances_init

def additive_noise(X, noise_covariance):
    n_dim, n_points = X.shape
    noise = np.random.multivariate_normal(np.zeros(n_dim), noise_covariance, size=n_points).T
    return X + noise

def huber_epsilon_contamination(X, eps_true, rng=None, outlier_std=None, return_mask=False):

    if X.ndim != 2:
        raise ValueError("X must have shape (n_dim, n_points)")
    n_dim, n_points = X.shape
    if not (0.0 <= eps_true < 1.0):
        raise ValueError("eps_true must satisfy 0 <= eps_true < 1")

    if rng is None:
        rng = np.random.default_rng()
    if outlier_std is None:
        outlier_std = float(n_dim)  

    outlier_mask = rng.random(n_points) < eps_true
    n_bad = int(outlier_mask.sum())

    if n_bad == 0:
        return (X.copy(), outlier_mask) if return_mask else X.copy()

    X_bad = rng.normal(loc=0.0, scale=outlier_std, size=(n_dim, n_bad)).astype(X.dtype, copy=False)

    X_contaminated = X.copy()
    bad_idx = np.nonzero(outlier_mask)[0]
    X_contaminated[:, bad_idx] = X_bad

    return (X_contaminated, outlier_mask) if return_mask else X_contaminated

def _draw_single_panel(ax, X, weights, means, covariances, X_epsilon=None, labels=None, plot_data=True, plot_params=True, title=None, base_font_size=50):
    weights = np.asarray(weights).ravel()
    X2         = X[0:2, :]
    means_plot = means[:, 0:2]
    covs_plot  = covariances[:, 0:2, 0:2]

    if plot_data:
        if X_epsilon is None:
            if labels is None:
                ax.scatter(X2[0], X2[1], s=1, c='k', alpha=0.1)
            elif isinstance(labels, str):
                ax.scatter(X2[0], X2[1], s=1, c='k', alpha=0.1, label=labels)
            else:
                pal = sns.color_palette("husl", np.unique(labels).size)
                sns.scatterplot(x=X2[0], y=X2[1], hue=labels, palette=pal, s=1, alpha=0.1, ax=ax, legend=False)
        else:
            ax.scatter(X2[0], X2[1], s=3, c='k', alpha=0.1, label=r'$y_n$')
            ax.scatter(X_epsilon[0], X_epsilon[1], s=3, c='r', alpha=0.1, label=r'$\tilde y_n$')

    if plot_params:
        ws = 0.2 / weights.max()
        ci1 = itertools.cycle(["#4DBEEE","#D95319","#77AC30", "#EDB120","#7E2F8E","#A2142F"])
        for i, (w, mu, cov, col) in enumerate(zip(weights[::-1], means_plot[::-1],covs_plot[::-1], ci1)):
            eigvals, eigvecs = np.linalg.eigh(cov)
            eigvals = 2*np.sqrt(2)*np.sqrt(np.maximum(eigvals, 0))
            u = eigvecs[:, 0] / np.linalg.norm(eigvecs[:, 0])
            angle = np.degrees(np.arctan2(u[1], u[0]))
            alpha=2.5 * float(w) * ws
            if np.linalg.matrix_rank(cov) == 1:
                alpha = 1
            ell = Ellipse(mu, width=eigvals[0], height=eigvals[1], angle=180+angle, alpha=alpha, color=col, label=rf'$\Sigma_{{{i+1}}}$')
            ax.add_patch(ell)

        ci2 = itertools.cycle(["blue","red","green","orange", "purple","#A2142F"])
        for i, (w, mu, col) in enumerate(zip(weights[::-1], means_plot[::-1], ci2)):
            ax.scatter(mu[0], mu[1], marker='+', s=1000, linewidths=4, color=col, label=rf'$\mu_{{{i+1}}}$')

    ax.tick_params(axis='both', labelsize=base_font_size, pad=12)
    if title is not None:
        ax.set_title(title, fontsize=base_font_size + 6, fontweight='bold')

def plot_comparison_figure(data_dict, filename=None, axlim_data=15, axlim_param=7, legend_position=(1.05, 0.7)):
    sns.set_style("whitegrid")
    panels = [(t, args) for t, args in data_dict if t is not None]
    n = len(panels)
    cols = 3
    rows = math.ceil(n/cols)
    fig = plt.figure(figsize=(40, 27), constrained_layout=False)
    outer = GridSpec(rows, 1, figure=fig, hspace=0.30)

    axes = []
    idx = 0
    for r in range(rows):
        if r < rows - 1:
            nc = cols
        else:
            nc = n - (rows - 1) * cols
        row_wspace = 0.22 if nc == 3 else 0.08
        inner = GridSpecFromSubplotSpec(1, nc, subplot_spec=outer[r], wspace=row_wspace)
        for c in range(nc):
            title, args = panels[idx]
            ax = fig.add_subplot(inner[0, c])
            _draw_single_panel(ax, *args, plot_data=(idx == 0), plot_params=(idx != 0), title=title, base_font_size=50)
            rng = axlim_data if idx == 0 else axlim_param
            ticks = np.linspace(-rng, rng, 5)
            ax.set_xlim(-rng, rng)
            ax.set_ylim(-rng, rng)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.set_aspect('equal')
            axes.append(ax)
            idx += 1

    legend_dict = OrderedDict()
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for hh, ll in zip(h, l):
            legend_dict[ll] = hh
    fig.legend(legend_dict.values(), legend_dict.keys(), loc='lower right', bbox_to_anchor=legend_position, bbox_transform=fig.transFigure, fontsize=65, frameon=True, edgecolor='black', borderpad=0.6)
    fig.subplots_adjust(left=0.05, right=0.92, top=0.93, bottom=0.12)
    if filename is not None:
        fig.savefig(filename + '.jpg', format='JPEG', dpi=300, bbox_inches='tight')
    plt.show()

def plot_results_MoM(X, weights, means, covariances, title, filename=None, X_epsilon=None, labels = None):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import itertools
    from matplotlib.patches import Ellipse
    import matplotlib as mpl

    n_components, n_dim = means.shape
    _, n_points = X.shape

    weights_scale = 0.2 / weights.max()
    means_plot = np.zeros((n_components, 2))
    covariances_plot = np.zeros((n_components, 2, 2))
    X_plot = np.zeros((2, n_points))
    id_dimensions_plot = [0, 1]
    for i, idx in enumerate(id_dimensions_plot):
        means_plot[:, i] = means[:, idx]
        X_plot[i, :] = X[idx, :]
        for j, idy in enumerate(id_dimensions_plot):
            covariances_plot[:, i, j] = covariances[:, idx, idy]

    mpl.rcParams.update({'figure.autolayout': True})
    mpl.rcParams['text.usetex'] = True

    fig, ax = plt.subplots(constrained_layout=True)
    
    fig_width, _ = fig.get_size_inches()
    base_font_size = int(fig_width) * 2

    sns.set_style("whitegrid", {'font.size': base_font_size})
    
    if X_epsilon is None:
        if labels is not None:
            palette = sns.color_palette("husl", np.unique(labels).size)
            sns.scatterplot(x=X_plot[0], y=X_plot[1], s=5, hue=labels, palette=palette, alpha=0.5, ax=ax)
        else:
            sns.scatterplot(x=X_plot[0], y=X_plot[1], s=5, color='k', alpha=0.5, ax=ax)
    else:
        sns.scatterplot(x=X_plot[0], y=X_plot[1], s=3, color='k', alpha=0.6, label=r'$y_n$', ax=ax)
        sns.scatterplot(x=X_epsilon[0, :], y=X_epsilon[1, :], s=3, color='r', alpha=0.5, label=r'$\tilde{y}_n$', ax=ax)

    color_iter = itertools.cycle(["#4DBEEE", "#D95319", "#77AC30", "#EDB120", "#7E2F8E", "#A2142F"])
    for i, (weight, mean, covariance, color) in enumerate(zip(weights[::-1], means_plot[::-1, :], covariances_plot[::-1, :, :], color_iter)):
        eigvals, eigvecs = np.linalg.eigh(covariance)
        eigvals = np.maximum(eigvals, 0)
        eigvals = 2.0 * np.sqrt(2.0) * np.sqrt(eigvals)
        u = eigvecs[0] / np.linalg.norm(eigvecs[0])
        angle = np.degrees(np.arctan(u[1] / u[0]))
        for nsig in range(1, 2):
            ell = Ellipse(mean, nsig * eigvals[0], nsig * eigvals[1], angle=180.0 + angle, alpha=2.5 * weight * weights_scale, color=color, label=r'$\Sigma_{}$'.format(i + 1))
            ax.add_patch(ell)

    color_iter = itertools.cycle(["blue", "red", "green", "orange", "purple", "#A2142F"])
    for i, (weight, mean, covariance, color) in enumerate(zip(weights[::-1], means_plot[::-1, :], covariances_plot[::-1, :, :], color_iter)):
        ax.scatter(mean[0], mean[1], color=color, marker='+', linewidths=2.5, s=250, label=r'$\mu_{}$'.format(i + 1))

    ax.tick_params(labelsize=base_font_size)

    if title is not None:
        if X_epsilon is None:
            ax.set_title(r'\textbf{' + title + '}', fontsize=base_font_size+10)
        else:
            ax.set_title(r'\textbf{' + title + r'} $(\sigma = 1)$', fontsize=base_font_size+10)

    handles, labels = ax.get_legend_handles_labels()
    ncols = n_components
    n = len(handles)
    new_order = [None] * n
    nrows = 2
    for i in range(nrows):
        for j in range(ncols):
            original_index = i * ncols + j
            if original_index < n:
                new_index = i + j * nrows
                new_order[new_index] = original_index
    new_handles = [handles[i] for i in new_order if i is not None]
    new_labels = [labels[i] for i in new_order if i is not None]
    
    ax.legend(new_handles, new_labels, loc='upper center', bbox_to_anchor=(0.5, -0.08),
              ncol=ncols, borderaxespad=0.0, fontsize=base_font_size)
    
    ax.tick_params(axis='x', labelsize=base_font_size)
    ax.tick_params(axis='y', labelsize=base_font_size)
    
    if filename is not None:
        plt.savefig(filename + '.jpg', format='JPEG', dpi=300, bbox_inches='tight')
    plt.show()
