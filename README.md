# Hyperparameter Tuning and Model Evaluation in Causal Effect Estimation
Investigating the interplay between causal estimators, ML base learners, hyperparameters and model evaluation metrics.

## Paper
This code accompanies the paper:

D. Machlanski, S. Samothrakis, and P. Clarke, ‘Hyperparameter Tuning and Model Evaluation in Causal Effect Estimation’. arXiv, Mar. 02, 2023. doi: 10.48550/arXiv.2303.01412. [link](https://arxiv.org/abs/2303.01412).

## Data
All datasets are available [here](https://doi.org/10.6084/m9.figshare.30244936.v1).

## Replicating the paper
Follow the steps below.

1. Download datasets from [here](https://doi.org/10.6084/m9.figshare.30244936.v1) and put them under 'datasets' folder.
2. Prepare Python environment.
    1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
    2. If you intend to run Neural Networks, run `conda env create -f environment_tf.yml`.
    3. Otherwise, you can use the default environment `conda env create -f environment.yml`.
3. Go to 'scripts' folder and run `bash paper.sh`. This will run ALL the experiments.
4. Go to 'analysis' folder.
5. Run `python compare_save_prob.py` and `python compare_save_hyper_prob.py` to post-process the results.
6. Use `plot_hyperparams_prob.ipynb` and `plot_metrics_prob.ipynb` to replicate Figures 2-4 from the paper.

Note that running all experiments (step 3.) may take a LONG time (weeks, likely months). Highly parallelised computing environments are recommended.

It is possible to replicate the plots without re-running the experiments as the most important CSV files obtained as part of the paper are included in this repository (analysis/tables).

## Project organisation
The following description explains only the most important files and directories necessary to replicate the paper.

    ├── environment.yml                     <- Replicate the environment to run all the scripts.
    ├── environment_tf.yml                  <- As above but with Tensorflow (required to run neural networks).
    │
    ├── analysis
    │   ├── compare_save_xxx.py             <- Post-processes 'results' into CSV files.
    │   ├── tables                          <- CSVs from above are stored here.
    │   ├── utils.py                        <- Important functions used by `compare_save.py'.
    │   ├── plot_hyperparams_prob.ipynb     <- Replicate Figure 2 and 3.
    │   └── plot_metrics_prob.ipynb         <- Replicate Figure 4.
    │
    ├── datasets                            <- All four datasets go here (IHDP, Jobs, Twins and News).
    │
    ├── helpers                             <- General helper functions.
    │
    ├── models
    │   ├── data                            <- Models for datasets.
    │   ├── estimators                      <- Implementations of CATE estimators.
    │   ├── estimators_tf                   <- Code for Neural Networks (Tensorflow).
    │   └── scorers                         <- Implementations of learning-based metrics.
    │
    ├── results
    │   ├── metrics                         <- Conventional, non-learning metrics (MSE, R^2, PEHE, etc.).
    │   ├── predictions                     <- Predicted outcomes and CATEs.
    │   ├── scorers                         <- Predictions of scorers (plugin, matching and rscore).
    │   └── scores                          <- Actual scores (combines 'predictions' and 'scorers').
    │
    └── scripts
        └── paper.sh                        <- Replicate all experiments from the paper.
    
