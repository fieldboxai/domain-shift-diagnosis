# Domain Shift Diagnosis

Welcome to code repository of the paper "Analyse de shifts dans des données industrielles de capteurs
par AutoEncodeur Variationnel parcimonieux", published at [EGC2023](https://egc2023.sciencesconf.org/). In English the title would be "Domain Shift Diagnosis In Industrial Process Data With Sparse Variational AutoEncoder"

You will find all the results of experiments presented in the paper and the implementation of the models in `keras` and `tensorflow`.

It uses the `kedro` framework ([DOC](https://kedro.readthedocs.io/en/stable/) )  for structuring the code into pipelines, executable from the command line interface.
It also uses `mlflow` ([DOC](https://mlflow.org/) ) to track the various assets produced by the experiments. ML FLOW offers a web interface to visualize the results of the experiments.

## Environment set-up

### Creating a virtual environment

You need to create a virtual environment to safely work on the project. We recommend
using `pyenv`.

```console
$ pyenv virtualenv 3.8.8 domain-shift-diagnosis-venv
$ pyenv activate domain-shift-diagnosis-venv
```

### Installing the dependencies

```console
$ pip install -r src/requirements.txt
```

## Exploring the repository
The structure of the repository follows the default kedro [template](https://kedro.readthedocs.io/en/stable/faq/architecture_overview.html).
The `src/` folder contains the models implementation, the pipelines, the synthetic data generator used for the experiments on synthetic data.

### Models implementation
The implementation of the VAE models are in `src/domain_shift_diagnosis/models/`, alongside custom tensorflow layers.

### Models parameters
The parameters used to train the models are the one set under `conf/base/parameters/models/` and should not be changed in order to reproduce the experiments listed in the paper.

### Synthetic Data Generator
The code that generates the source and target dataset is in `src/domain_shift_diagnosis/datasets/`

## Starting ML Flow tracking server

```console
$ mlflow ui
```

## Running experiments

### Loading synthetic data
The first mandatory step is the generation of the synthetic dataset. To that end, run:

```console
$ kedro run --pipeline load_data
```
After running that command, you should see files generated under `data/05_model_input/`.

### Running the experiment pipeline on all models at once
There are 7 models to be tested: LassoVAE, LinearVAE, SparseVAE, SparsePCA16 and SparsePCA32, PPCA16, PPCA32

All the experiments can be run at once with the following command:
```console
$ bash run_synthetic_data_experiments.sh
```

### Running the experiment pipeline on a single model
One experiment correspond to the training and testing of one model.

A single experiment run is triggered with the following command:

```console
$ kedro run --pipeline experiment --params model_name:$MODEL_NAME
```

## Tracking results

### ML Flow web interface
Once all experiments have been run, the results (figures and data) can be explored in the ML Flow interface available at http://127.0.0.1:5000/#/experiments/1.

### Notebooks
The notebook "notebooks/Source Data Generator.ipynb" showcases our synthetic data generator.

And we used two other notebooks to generate the additional figures.
- "notebooks/Synthetic Data Results.ipynb" load and aggregates the results from experiments on the synthetic data.
- "notebooks/Experiments on TEP.ipynb" applies the same experiment to the Tennesse Eastman Process data.

