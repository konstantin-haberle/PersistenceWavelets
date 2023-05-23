# Wavelet-Based Density Estimation for Persistent Homology

This repository provides an implementation of a wavelet-based estimator for persistence diagrams. Moreover, it contains the code for the numerical experiments in [Wavelet-Based Density Estimation for Persistent Homology](https://arxiv.org/abs/2305.08999). 

The repository is organised as follows:

- `wavelet_estimator.py`: class of wavelet estimators;
- `ot_dist.py`: contains methods for the computation of the `OT` loss;
- `comp_minimax_rates.py`: computes the `OT` losses and saves them in the folder `results`;
- `plot_minimax_rates.ipynb`: notebook which plots the minimax rates that are saved in the folder results;
- `data`: folder containing persistence diagrams of torus and double torus;
- `results`: folder containing the `OT` losses.

## Academic Use
For academic use of this project, please cite it using the following BibTeX entry:
```bibtex
@article{haberle2023PersistenceWavelets,
  title={Wavelet-Based Density Estimation for Persistent Homology},
  author={H{\"a}berle, Konstantin and Bravi, Barbara and Monod, Anthea},
  journal={arXiv preprint arXiv:2305.08999},
  year={2023}
}
```
