# COMP9417 Project - Code Running Instruction

Please unzip the submitted zip file first

## Data folder structure

The code and dataset should be placed under the same root file, so they are supposed to be structured as:

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

We recommend using conda to manage the environment. Run the following command on your terminal under `your_work_dir` to create a conda environment:

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

To reproduce the result of the Semi-supervised Adaptive Generalised Ensemble (SAGE) model:

```
python work_pipeline.py
```

To reproduce the result of N Minus One Binary Classification (NMOBC) model:

```
python n_binary_classification.py
```

To reproduce the result of Cluster-wise SVM (C-SVM) model:

```
python C-SVM.py
```

To reproduce results of the  baseline models:

```
python reproduce.py
```

## Predictions

To generate our two prediction files `preds_1.npy` and `preds_2.npy` using our best model **SAGE**, you can run via:

```
python predictions.py
```

## EDA

To generate output for the Exploratory Data Analysis(EDA), run:

```
python EDA.py
```

## Data Distribution Shifting

A comprehensive distribution shift analysis included in `domain_shifting.py`. You can visualize all kinds of data distribution shifting in the file, and our analysis is commented in that file as well.

To run the file via:

```
python domain_shifting.py
```
