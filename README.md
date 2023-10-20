# Approximately well-balanced Discontinuous Galerkin methods using bases enriched with Physics-Informed Neural Networks

This repository contains some code associated to the article "*Approximately well-balanced Discontinuous Galerkin methods using bases enriched with Physics-Informed Neural Networks*".
The included code provides an approximate solution to the parameterized advection equation with the technique described in the companion article.

Installation instructions follow.

### First possibility: use Google colab (Google account required)

1. Go to https://colab.research.google.com/ and login using your Google account
2. Paste https://github.com/Victor-MichelDansac/DG-PINNs.git under the `GitHub` tab, click the magnifying glass icon and select the notebook (`enriched_DG_bases_transport.ipynb`)

### Second possibility: local installation

0. Install (or create a virtual environment with) the Python packages `scipy`, `ipykernel`, `matplotlib` and `torch`
1. Close the repo: `git clone https://github.com/Victor-MichelDansac/DG-PINNs.git`
2. Navigate to the downloaded folder and open the `enriched_DG_bases_transport.ipynb` notebook
