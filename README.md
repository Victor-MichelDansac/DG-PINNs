# Approximately well-balanced Discontinuous Galerkin methods using bases enriched with Physics-Informed Neural Networks

This work concerns the enrichment of Discontinuous Galerkin (DG) bases, so that the resulting scheme provides a much better approximation of steady solutions to hyperbolic systems of balance laws. The basis enrichment leverages a prior – an approximation of the steady solution – which we compute using a Physics-Informed Neural Network (PINN). Convergence results and error estimates prove that the basis with prior does not change the order of convergence, and that the error constant is improved. This repository provides an approximate solution to the parameterized advection equation:
1. first, a PINN is trained to learn the steady solution; 
2. then, the DG scheme can be run with and without prior, on three test cases from the paper.

If you find this repository useful, please consider citing the companion article (citation available at the end of this file).\
Emmanuel Franck, Victor Michel-Dansac and Laurent Navoret. *"[Approximately well-balanced Discontinuous Galerkin methods using bases enriched with Physics-Informed Neural Networks.](https://arxiv.org/abs/2310.14754)"* arXiv preprint arXiv:2310.14754 (2023).

## Citation

```
@misc{FraMicNav2023,
      title={Approximately well-balanced {D}iscontinuous {G}alerkin methods using bases enriched with {P}hysics-{I}nformed {N}eural {N}etworks}, 
      author={Emmanuel Franck and Victor Michel-Dansac and Laurent Navoret},
      year={2023},
      eprint={2310.14754},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

### Installation, first possibility: use Google colab (Google account required)

1. Go to https://colab.research.google.com/ and login using your Google account
2. Paste https://github.com/Victor-MichelDansac/DG-PINNs.git under the `GitHub` tab, click the magnifying glass icon and select the notebook (`enriched_DG_bases_transport.ipynb`)

### Installation, second possibility: local installation

0. Install (or create a virtual environment with) the Python packages `scipy`, `ipykernel`, `matplotlib` and `torch`
1. Close the repo: `git clone https://github.com/Victor-MichelDansac/DG-PINNs.git`
2. Navigate to the downloaded folder and open the `enriched_DG_bases_transport.ipynb` notebook
