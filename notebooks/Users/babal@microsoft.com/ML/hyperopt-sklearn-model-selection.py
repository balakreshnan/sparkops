# Databricks notebook source
# MAGIC %md
# MAGIC # Model selection using scikit-learn, Hyperopt, and MLflow
# MAGIC 
# MAGIC [Hyperopt](https://github.com/hyperopt/hyperopt) is a Python library for hyperparameter tuning. Databricks Runtime for Machine Learning includes an optimized and enhanced version of Hyperopt, including automated MLflow tracking and the `SparkTrials` class for distributed tuning.  
# MAGIC 
# MAGIC This notebook shows how to use Hyperopt to identify the best model from among several different scikit-learn algorithms and sets of hyperparameters for each model. It also shows how to use MLflow to track Hyperopt runs so you can examine them later.
# MAGIC 
# MAGIC This tutorial covers the following steps:
# MAGIC 1. Prepare the dataset.
# MAGIC * Define the function to minimize.
# MAGIC * Define the search space over hyperparameters.
# MAGIC * Select the search algorithm.
# MAGIC * Use Hyperopt's `fmin()` function to find the best combination of hyperparameters.

# COMMAND ----------

import numpy as np

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from hyperopt import fmin, tpe, hp, SparkTrials, STATUS_OK, Trials

import mlflow

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare the dataset
# MAGIC 
# MAGIC This notebook uses the [California housing dataset](https://scikit-learn.org/stable/datasets/index.html#california-housing-dataset) included with scikit-learn. The dataset is based on data from the 1990 US census. It includes the median house value in over 20,000 census blocks in California along with information about the block such as the income, number of people per household, number of rooms and bedrooms per house, and so on. 

# COMMAND ----------

X, y = fetch_california_housing(return_X_y=True)

# COMMAND ----------

# MAGIC %md ### Scale the predictor values
# MAGIC The predictor columns are median income, house age, average number of rooms in a house, average number of bedrooms, block population, average house occupancy, latitude, and longitude. The ranges of these predictors varies significantly. Block population is in the thousands, but the average number of rooms in a house is around 5. To prevent the predictors with large values from dominating the calculations, it's a good idea to [normalize the predictor values](https://scikit-learn.org/stable/modules/preprocessing.html#standardization-or-mean-removal-and-variance-scaling) so they are all on the same scale. To do this, you can use the scikit-learn function [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn-preprocessing-standardscaler).

# COMMAND ----------

# Review the mean value of each column in the dataset. You can see that they vary by several orders of magnitude, from 1425 for block population to 1.1 for average number of bedrooms. 
X.mean(axis=0)

# COMMAND ----------

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = scaler.fit_transform(X)

# COMMAND ----------

# After scaling, the mean value for each column is close to 0. 
X.mean(axis=0)

# COMMAND ----------

# MAGIC %md ### Convert the numeric target column to discrete values

# COMMAND ----------

# MAGIC %md The target value in this dataset is the value of the house, a continuous or numeric value. This notebook illustrates the use of classification functions, so the first step is to convert the target value to a categorical value. The next cell converts the original target values into two discrete levels: 0 if the value of the house is below the median, or 1 if the value of the house is above the median. 

# COMMAND ----------

y_discrete = np.where(y < np.median(y), 0, 1)

# COMMAND ----------

# MAGIC %md ## Define the function to minimize

# COMMAND ----------

# MAGIC %md
# MAGIC In this notebook, you examine three algorithms available in scikit-learn: support vector machines (SVM), random forest, and logistic regression.  
# MAGIC 
# MAGIC In the following cell, you define a parameter `params['type']` for the model name. This function also runs the training and calculates the cross-validation accuracy. 

# COMMAND ----------

def objective(params):
    classifier_type = params['type']
    del params['type']
    if classifier_type == 'svm':
        clf = SVC(**params)
    elif classifier_type == 'rf':
        clf = RandomForestClassifier(**params)
    elif classifier_type == 'logreg':
        clf = LogisticRegression(**params)
    else:
        return 0
    accuracy = cross_val_score(clf, X, y_discrete).mean()
    
    # Because fmin() tries to minimize the objective, this function must return the negative accuracy. 
    return {'loss': -accuracy, 'status': STATUS_OK}

# COMMAND ----------

# MAGIC %md ## Define the search space over hyperparameters
# MAGIC 
# MAGIC See the [Hyperopt documentation](https://github.com/hyperopt/hyperopt/wiki/FMin) for details on [defining a search space](https://github.com/hyperopt/hyperopt/wiki/FMin#2-defining-a-search-space) and [parameter expressions](https://github.com/hyperopt/hyperopt/wiki/FMin#21-parameter-expressions).

# COMMAND ----------

# MAGIC %md
# MAGIC Use `hp.choice` to select different models. 

# COMMAND ----------

search_space = hp.choice('classifier_type', [
    {
        'type': 'svm',
        'C': hp.lognormal('SVM_C', 0, 1.0),
        'kernel': hp.choice('kernel', ['linear', 'rbf'])
    },
    {
        'type': 'rf',
        'max_depth': hp.quniform('max_depth', 2, 5, 1),
        'criterion': hp.choice('criterion', ['gini', 'entropy'])
    },
    {
        'type': 'logreg',
        'C': hp.lognormal('LR_C', 0, 1.0),
        'solver': hp.choice('solver', ['liblinear', 'lbfgs'])
    },
])

# COMMAND ----------

# MAGIC %md ## Select the search algorithm
# MAGIC 
# MAGIC The two main choices are:
# MAGIC * `hyperopt.tpe.suggest`: Tree of Parzen Estimators, a Bayesian approach that iteratively and adaptively selects new hyperparameter settings to explore based on previous results
# MAGIC * `hyperopt.rand.suggest`: Random search, a non-adaptive approach that samples over the search space

# COMMAND ----------

algo=tpe.suggest

# COMMAND ----------

# MAGIC %md ## Use Hyperopt's `fmin()` function to find the best combination of hyperparameters.

# COMMAND ----------

# MAGIC %md 
# MAGIC `SparkTrials` takes 2 optional arguments:  
# MAGIC * `parallelism`: Number of models to fit and evaluate concurrently. The default is the number of available Spark task slots.
# MAGIC * `timeout`: Maximum time (in seconds) that `fmin()` can run. The default is no maximum time limit.

# COMMAND ----------

spark_trials = SparkTrials()

# COMMAND ----------

# MAGIC %md When you call `mlflow.start_run()` before calling `fmin()` as shown in the example below, the Hyperopt runs are automatically tracked with MLflow.  
# MAGIC 
# MAGIC `max_evals` is the maximum number of points in hyperparameter space to test. This is the maximum number of models Hyperopt fits and evaluates.

# COMMAND ----------

with mlflow.start_run():
  best_result = fmin(
    fn=objective, 
    space=search_space,
    algo=algo,
    max_evals=32,
    trials=spark_trials)

# COMMAND ----------

# MAGIC %md ### Print the hyperparameters that produced the best result

# COMMAND ----------

import hyperopt
print(hyperopt.space_eval(search_space, best_result))

# COMMAND ----------

# MAGIC %md To view the MLflow experiment associated with the notebook, click the **Experiment** icon in the notebook context bar on the upper right.  There, you can view all runs. To view runs in the MLflow UI, click the icon at the far right next to **Experiment Runs**. 
# MAGIC 
# MAGIC To examine the effect of tuning a specific hyperparameter:
# MAGIC 
# MAGIC 1. Select the resulting runs and click **Compare**.
# MAGIC 1. In the Scatter Plot, select the hyperparameter from the X-axis drop-down menu and select **loss** from the Y-axis drop-down menu.