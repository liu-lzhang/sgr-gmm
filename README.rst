.. Copyright (c) 2026 Liu Zhang
.. SPDX-License-Identifier: GPL-3.0-only

================================================================================
Robust generalized method of moments via spectral gradient reweighting (SGR-GMM)
================================================================================

This repository contains the code and data supplementary to the paper:

*Robust Moment-Based Estimation via Spectral Gradient Reweighting* [1]_.

Installation
============

Clone the repository and install all dependencies in a virtual environment. Installing the dependencies may take some time.

.. code-block:: bash

   # Clone this repository
   git clone https://github.com/sgr-gmm.git
   cd sgr-gmm

   # Create & activate virtual environment
   python3.11 -m venv sgr-gmm-venv
   source sgr-gmm-venv/bin/activate

   # Install pinned dependencies
   pip3 install -r requirements.txt

   # Install Jupyter kernel
   pip3 install ipykernel
   python3.11 -m ipykernel install --user --name=sgr-gmm-venv

Launch Jupyter notebooks ``sgr_numerical_experiments.ipynb`` and ``robust_dgmm_numerical_experiments.ipynb`` and select the kernel named ``sgr-gmm-venv``. Then, run the respective sections in the Jupyter notebooks.

We recommend using the pinned versions listed below to ensure reproducibility and compatibility:

- ``python==3.11.11``
- ``numpy==1.26.4``
- ``scipy==1.15.3``
- ``scikit-learn==1.7.0``
- ``sympy==1.14.0``
- ``matplotlib==3.7.2``
- ``seaborn==0.12.2``

License
=======

This code is released under the ``GPL-3.0-only`` license. See ``LICENSE`` for details.


Index of files
==============

Code:
-----

- Base DGMM source code:

  - ``gm_moments.py``: implicit moment computation for low-rank Gaussian mixtures.
  - ``gmm.py``: variants of generalized method of moments compared in the paper (MM, GMM, DGMM).
  - ``NystromApprox.py``: module for Nyström approximation of the inner product kernel.
  - (optional) ``upstream/``: code adapted from the Randomly-Pivoted-Cholesky project: https://github.com/eepperly/Randomly-Pivoted-Cholesky

- Robust DGMM source code:
  
  - ``robust_dgmm.py``: specialization of SGR-GMM to DGMM for Gaussian mixtures. 
  - ``robust_gradient_reweighting.py``: spectral gradient reweighting primitive.
  - ``utils.py``: utility functions.

- Notebooks for numerical experiments: 

  - ``robust_dgmm_numerical_experiments.ipynb``: numerical experiments for robust DGMM. 
  - ``sgr_numerical_experiments.ipynb``: numerical experiments for spectral gradient reweighting.

Test data and figures:
----------------------

- ``tests/test_data/`` 

  - ``a1_data.mat``
  - ``a1_error_history.csv``
  - ``a1_outlier_history.csv``
  - ``b1_df.csv``
  - ``b1_summary.csv``
  - ``c1_df.csv``
  - ``c1_summary.csv``
  - ``contamination_results_df.csv``
  - ``contamination_summary_df.csv``
  - ``diagnostic_history.mat``
  - ``diagnostic_results_df.csv``
  - ``epsilon_assumption_results_df.csv``
  - ``epsilon_assumption_summary_df.csv``

- ``figures/`` 

  - ``fig_comparison.pdf``
  - ``fig_contamination.pdf``
  - ``fig_convergence_diagnostics.pdf``
  - ``fig_epsilon_sensitivity.pdf``
  - ``fig_outer_iterations.pdf``
  - ``fig_repeated_trials.pdf``

Other files:
------------

- ``CITATION.cff``
- ``COPYRIGHT.txt``
- ``LICENSE``
- ``README.rst`` 
- ``requirements.txt``
  

Citing
======

If you use this code in your work, please cite the article [1]_.

.. code-block:: bibtex

   @article{SGR-GMM,
     author  = {Zhang, Liu and Singer, Amit},
     title   = {Robust Moment-Based Estimation via Spectral Gradient Reweighting},
     journal = {arXiv preprint arXiv:2605.27718},
     year    = {2026},
     doi     = {https://doi.org/10.48550/arXiv.2605.27718},
   }

.. [1] L. Zhang and A. Singer,
   *Robust Moment-Based Estimation via Spectral Gradient Reweighting*,
   2026.

*Last edit: Liu Zhang - May 27, 2026*