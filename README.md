# Explainable Data-Driven Optimization: From Context to Decision and Back Again
This code can be used to reproduce all figures and results in the paper titled "Explainable Data-Driven Optimization: From Context to Decision and Back Again" by Alexandre Forel, Axel Parmentier and Thibaut Vidal published in the *Proceedings of the fortieth International Conference on Machine Learning*, 2023, in press.

[Paper available here](https://arxiv.org/abs/2301.10074)   |   [Short video presentation](https://www.youtube.com/watch?v=Y29FjSClloc)


## Installation
The project requires the Gurobi solver to be installed with an authorized license. Free academic licenses can be obtained at: https://www.gurobi.com/academia/academic-program-and-licenses/ .

The packages required are listed in the file `environment.yml`. A virtual python environment can be created using an Anaconda Distribution (https://www.anaconda.com/products/distribution) by using:
`conda env create -f environment.yml`
in the project root directory. This will automatically install all required packages and their dependencies.

## Content
The main scripts that generate data or results are included in the root folder. The folders have the following contents:
* `data`: all data used for the experiments with Uber movement data,
* `ocean`: a copy of the OCEAN package taken from https://github.com/vidalt/OCEAN that has been adapted,
* `src`: all local functions needed to generate and analyze the experimental results.

## How to reproduce the paper results
All experiments are run using the scripts starting with the `run` prefix. The results are stored in the `\output\` folder.
Running the scripts will generate the data for the following results:
* `run_uber_movement_path`: Figure 1 and Table 3,
* `run_synthetic_experiment`: Tables 2, 4, 5, 7, and 8, and Figure 6,
* `run_path_features_sensitivity`: Figure 2 and 8,
* `run_problem_complex_sens`: Figures 3 and 4,
* `run_forest_depth`: Table 6,
* `run_dual_formulation`: Figure 9,
* `run_spurious_experiment`: Figures 10 and 11.

Run the script `analyze_results` to generate the figures, tables, and csv files used in the paper.
