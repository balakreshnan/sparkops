# Databricks notebook source
# MAGIC %md
# MAGIC # XGBoost Classification in Python

# COMMAND ----------

# MAGIC %md ## Setup
# MAGIC 
# MAGIC If you are running Databricks Runtime, uncomment the appropriate line in Cmd 3 to install the xgboost library.  
# MAGIC If you are running Databricks Runtime ML, xgboost is already installed. Skip to Cmd 4. 

# COMMAND ----------

# If you are running Databricks Runtime 7.1 or above, uncomment this line and run this cell:
#%pip install xgboost

# If you are running Databricks Runtime 6.4 to 7.0, uncomment this line and run this cell:
#dbutils.library.installPyPI("xgboost")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prepare data

# COMMAND ----------

import pandas as pd
import xgboost as xgb

# COMMAND ----------

raw_input = pd.read_csv("/dbfs/databricks-datasets/Rdatasets/data-001/csv/datasets/iris.csv",
                        header = 0,
                       names=["item","sepal length","sepal width", "petal length", "petal width","class"])
new_input = raw_input.drop(columns=["item"])
new_input["class"] = new_input["class"].astype('category')
new_input["classIndex"] = new_input["class"].cat.codes
print(new_input)

# COMMAND ----------

from sklearn.model_selection import train_test_split
# Split to train/test
training_df, test_df = train_test_split(new_input)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train XGBoost Model with Pandas DataFrames

# COMMAND ----------

dtrain = xgb.DMatrix(training_df[["sepal length","sepal width", "petal length", "petal width"]], label=training_df["classIndex"])

# COMMAND ----------

param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'multi:softmax'}
param['nthread'] = 4
param['eval_metric'] = 'auc'
param['num_class'] = 6

# COMMAND ----------

num_round = 10
bst = xgb.train(param, dtrain, num_round)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prediction

# COMMAND ----------

dtest = xgb.DMatrix(test_df[["sepal length","sepal width", "petal length", "petal width"]])
ypred = bst.predict(dtest)

# COMMAND ----------

from sklearn.metrics import precision_score

pre_score = precision_score(test_df["classIndex"],ypred, average='micro')

print("xgb_pre_score:",pre_score)