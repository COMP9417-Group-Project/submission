# COMP9417 Project - Code Running Instruction

Please unzip the submitted zip file first

## Data folder structure

The code and dataset should be placed under the same root file, so they are suppose to be structured as:

```
your_work_dir
├── all data set csv files (include X_train.cvs, X_test_1.csv, X_test_2.csv, etc.)
├── environment.yaml
├── requirement.txt
├── preds_1.npy
├── preds_2.npy
├── work_pipeline.py
├── n_binary_classification.py
├── C-SVM.py
├── predictions.py
├── EDA.ipynb
├── domain_shifting.py
```

## Environment

We recommend using conda to manage the environment. Run following command on your terminal under `your_work_dir` to create a conda environment:

```
conda env create -f environment.yaml
conda activate jupyter_env
```

Or use pip as an alternative way:

```
pip install -r requirements.txt
```

## Evaluation

### Models

To evaluate our model performance via five folds cross validation, you may run following commands on your terminal under the ` your_work_dir` directory.

To reproduce result of Semi-supervised Adaptive Generalised Ensemble (SAGE) model:

```
python work_pipeline.py
```

To reproduce result of  N Minus One Binary Classification (NMOBC) model:

```
python n_binary_classification.py
```

To reproduce result of  Cluster-wise SVM (C-SVM) model:

```
python C-SVM.py
```

To reproduce result of baseline models:

```
python reproduce.py
```

## Predictions

To generate our two prediction files `preds_1.npy` and `preds_2.npy` using our best model **SAGE**, you can run via:

```
python predictions.py
```

## EDA

Our exploratory data analysis under `EDA.ipynb`, the output is generated for visualization, you can also run it by:

1. First add the current environment as a new Jupyter kernel (we assume you are using conda), and then start the notebook server from your terminal and 

```
python -m ipykernel install --user --name=jupter_env --display-name "jupyter_env"
jupyter notebook
```

2. Then the notebook should be open in your browser, you need switch the kernel into “jupyter_env”, then open the file to run the code.

## Data Distribution Shifting

A comprehensive distribution shift analysis included in `domain_shifting.py`. You can visualize all kinds of data distribution shifting in the file, and our analysis is commented in that file as well.

To run the file via:

```
python domain_shifting.py
```