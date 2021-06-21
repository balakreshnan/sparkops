# Databricks notebook source
import os
import uuid
import shutil
import pandas as pd
 
from mlflow.tracking import MlflowClient

# COMMAND ----------

oriondskey = dbutils.secrets.get(scope = "allsecrects", key = "oriondskey")

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.orionstoreds.blob.core.windows.net",
  oriondskey)

# COMMAND ----------

FeeRatiossdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/FeeRatios.csv")

# COMMAND ----------

display(FeeRatiossdf)

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

churn = ["590","1316", "1008", "98", "902", "320"]

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
 
from distutils.version import LooseVersion
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

display(FeeRatiossdf.groupBy("AccountAge").agg(f.mean('MarketValue'), f.count('MarketValue')))

# COMMAND ----------

FeeRatiossdf = FeeRatiossdf.withColumn("churned" , f.when(f.col('DataSourceKey').isin(churn) , 1))

# COMMAND ----------

FeeRatiossdf = FeeRatiossdf.withColumn("churned" , f.when(f.col('churned') == 1 , 1).otherwise(0))

# COMMAND ----------

FeeRatiossdf = FeeRatiossdf.drop("IsDemo","AccountLastBilledDate")

# COMMAND ----------

display(FeeRatiossdf)

# COMMAND ----------

display(FeeRatiossdf.groupBy("churned").count().orderBy("churned"))

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="churned", outputCol="indexedLabel").fit(FeeRatiossdf)

# COMMAND ----------

features = ["FeeSummaryKey", "DataSourceKey", "SnapshotMonthKey", "FirmKey", "AccountKey", "RepresentativeKey", "AccountAssetRangeKey", "AccountFeeRangeKey", "HouseholdAssetRangeKey", "HouseholdFeeRangeKey", "IsNewKey", "IsQualifiedKey", "IsIRAKey", "CID", "AID", "AccountAnnualizedFeeAmount", "AccountMarketValue", "AccountEffectiveFeePercentage", "HouseholdAnnualizedFeeAmount", "HouseholdMarketValue", "HouseholdEffectiveFeePercentage", "ManagementExpenseAmount", "AccountAnnualizedFeeAmountWithManagementExpense", "AccountEffectiveFeePercentageWithManagementExpense", "HouseholdAnnualizedFeeAmountWithManagementExpense", "HouseholdEffectiveFeePercentageWithManagementExpense", "AccountFeeRangeWithManagementExpenseKey", "HouseholdFeeRangeWithManagementExpenseKey", "HouseholdManagementExpenseAmount"]

# COMMAND ----------

(trainingData, testData) = FeeRatiossdf.randomSplit([0.7, 0.3])

# COMMAND ----------

stages = []
label_stringIdx = StringIndexer(inputCol="churned", outputCol="label")
stages += [label_stringIdx]
assembler = VectorAssembler(inputCols=features, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(FeeRatiossdf)
preppedDataDF = pipelineModel.transform(FeeRatiossdf)

# COMMAND ----------

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# COMMAND ----------

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

selected = preppedDataDF.select("label", "features", "churned")
display(selected)

# COMMAND ----------

selectedcols = ["label", "features", "churned"] + features
dataset = preppedDataDF.select(selectedcols)
display(dataset)

# COMMAND ----------

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = dataset.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
 
# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)
 
# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

# Make predictions on test data using the transform() method.
# LogisticRegression.transform() will only use the 'features' column.
predictions = lrModel.transform(testData)

# COMMAND ----------

selected = predictions.select("label", "prediction", "probability", "churned")
display(selected)

# COMMAND ----------

from pyspark.ml.evaluation import BinaryClassificationEvaluator
 
# Evaluate model
evaluator = BinaryClassificationEvaluator()
evaluator.evaluate(predictions)

# COMMAND ----------

evaluator.getMetricName()

# COMMAND ----------

print(lr.explainParams())

# COMMAND ----------



# COMMAND ----------

