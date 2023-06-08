# Hyperparameter Tuning and Model Evaluation in Causal Effect Estimation

Investigating the interplay between causal estimators, ML base learners, hyperparameters and model evaluation metrics.

## Paper
This code accompanies the paper _Hyperparameter Tuning and Model Evaluation in Causal Effect Estimation_ [arxiv](https://arxiv.org/abs/2303.01412).

## Data and Results
All datasets and results (in progress) are available [here](https://essexuniversity.box.com/s/xi6ptui0162dlcaokg8vv0274napxcsi).

## Replicating the paper
Follow the steps below.

1. Download datasets from [here](https://essexuniversity.box.com/s/xi6ptui0162dlcaokg8vv0274napxcsi) and put them under 'datasets' folder.
2. Prepare Python environment.
    1. Install [miniconda](https://docs.conda.io/en/latest/miniconda.html).
    2. If you intend to run Neural Networks, run `conda env create -f environment_tf.yml`.
    3. Otherwise, you can use the default environment `conda env create -f environment.yml`.
3. Go to 'scripts' folder and run `bash paper.sh`. This will run ALL the experiments.
4. Go to 'analysis' folder.
5. If you want the results in the form of latex tables:
    1. Go to `utils.py` and set `RESULTS = 'latex'`.
    2. Run `python compare_save_latex.py`.
    3. Now you can use: `metrics_meta_latex.ipynb`, `correlations_meta_latex.ipynb` and `test_correlations.ipynb`.
6. If you want the results visualised with plots:
    1. Use: `plot_estimators.ipynb`, `plot_hyperparams.ipynb`.
    2. In order to use `plot_metrics.ipynb`, you have to perform some extra steps.
    3. Go to `utils.py` and set `RESULTS = 'mean'`.
    4. Run `python compare_save_mean.py`.
    5. Now you can use `plot_metrics.ipynb`.

Note that running all experiments (step 3.) may take a LONG time (weeks, likely months). Highly parallelised computing environments are recommended.

It is possible to skip step 3. by downloading our results from [here](https://essexuniversity.box.com/s/xi6ptui0162dlcaokg8vv0274napxcsi).

It is also possible to skip scripts `compare_save_xxx.py` as the most important CSV files obtained as part of the paper are included in this repository.

## Project organisation
The following description explains only the most important files and directories necessary to replicate the paper.

    ├── environment.yml                     <- Replicate the environment to run all the scripts.
    ├── environment_tf.yml                  <- As above but with Tensorflow (required to run neural networks).
    │
    ├── analysis
    │   ├── compare_save.py                 <- Post-processes 'results' into CSV files.
    │   ├── tables                          <- CSV from above are stored here.
    │   ├── utils.py                        <- Important functions used by `compare_save.py'.
    │   ├── plot_estimators.ipynb           <- Visualise performance of CATE estimators.
    │   ├── plot_hyperparams.ipynb          <- Visualise performance against types of hyperparameters.
    │   ├── plot_metrics.ipynb              <- Visualise performance of metrics.
    │   ├── test_correlations.ipynb         <- Compute correlations between test metrics (e.g., ATE and PEHE).
    │   ├── correlations_meta_latex.ipynb   <- Compute correlations between validation and test metrics (e.g, MSE and PEHE).
    │   └── metrics_meta_latex.ipynb        <- Compute all metrics (latex format).
    │
    ├── datasets                            <- All four datasets go here (IHDP, Jobs, Twins and News).
    │
    ├── helpers                             <- General helper functions.
    │
    ├── models
    │   ├── data                            <- Models for datasets.
    │   ├── estimators                      <- Implementations of CATE estimators.
    │   ├── estimators_tf                   <- Code for Neural Networks (Tensorflow).
    │   └── scorers                         <- The original, immutable data dump.
    │
    ├── results
    │   ├── metrics                         <- Conventional, non-learning metrics (MSE, R^2).
    │   ├── predictions                     <- Predicted outcomes and CATEs.
    │   ├── scorers                         <- Predictions of scorers (plugin, matching and rscore).
    │   └── scores                          <- Actual scores (combines 'predictions' and 'scorers').
    │
    └── scripts
        └── paper.sh                        <- Replicate all experiments from the paper.
    
