# Databricks notebook source


# COMMAND ----------

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

webeventsdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/webevent.csv")

# COMMAND ----------

display(webeventsdf)

# COMMAND ----------

AUMAccountsRepsdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/AUMAccountsReps.csv")

# COMMAND ----------

display(AUMAccountsRepsdf)

# COMMAND ----------

FeeRatiossdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/FeeRatios.csv")

# COMMAND ----------

display(FeeRatiossdf)

# COMMAND ----------

OptimizationMetricsdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/OptimizationMetrics.csv")

# COMMAND ----------

display(OptimizationMetricsdf)

# COMMAND ----------

QuaterlyDetailAccountingdf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/QuaterlyDetailAccounting")

# COMMAND ----------

display(QuaterlyDetailAccountingdf)

# COMMAND ----------

TWRPerformancedf = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/DSPP/TWRPerformance.csv")

# COMMAND ----------

display(TWRPerformancedf)

# COMMAND ----------

display(webeventsdf)

# COMMAND ----------

display(webeventsdf.groupBy("LoginLevel").count())

# COMMAND ----------

display(webeventsdf.groupBy("DataSourceKey").count())

# COMMAND ----------

display(webeventsdf.groupBy("RequestDateKey").count())

# COMMAND ----------

display(webeventsdf.describe())

# COMMAND ----------

display(webeventsdf.summary())

# COMMAND ----------

webeventspandas = webeventsdf.toPandas()

# COMMAND ----------

clientswebeventsdf = webeventsdf.where("LoginLevel == 'Client'")

# COMMAND ----------

display(clientswebeventsdf)

# COMMAND ----------

display(clientswebeventsdf.groupBy("BaseRoute").count())

# COMMAND ----------

display(TWRPerformancedf)

# COMMAND ----------

display(TWRPerformancedf.describe())

# COMMAND ----------

display(TWRPerformancedf.summary())

# COMMAND ----------

display(TWRPerformancedf.groupBy("DataSourceKey").count().orderBy("DataSourceKey"))

# COMMAND ----------

from pyspark.sql import functions as F
display(TWRPerformancedf.groupBy("AccountAge").agg(F.mean('MarketValue'), F.count('MarketValue')))

# COMMAND ----------

display(TWRPerformancedf.groupBy("DataSourceKey").agg(F.mean('MarketValue'), F.count('MarketValue')))

# COMMAND ----------

display(TWRPerformancedf.groupBy("HouseholdAge").agg(F.mean('MarketValue'), F.count('MarketValue')))

# COMMAND ----------

# (590, 1316, 1008, 98, 902, 320, --churned
# 195, 203, 424, 1531, 1971, 1590 --not churned)

# COMMAND ----------

import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{col, lit, typedLit, when}
import org.apache.spark.sql.types.IntegerType

# COMMAND ----------

TWRPerformancedf = TWRPerformancedf.withColumn("churned", TWRPerformancedf["FiveYear"])

# COMMAND ----------

display(TWRPerformancedf)

# COMMAND ----------

from pyspark.sql import functions as f

# COMMAND ----------

TWRPerformancedf.printSchema()

# COMMAND ----------

#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 590 , 1))


# COMMAND ----------

#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 1316 , 1))
#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 1008 , 1))
#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 98 , 1))
#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 902 , 1))
#TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey') == 329 , 1))

# COMMAND ----------

churn = ["590","1316", "1008", "98", "902", "320"]

# COMMAND ----------

TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('DataSourceKey').isin(churn) , 1))

# COMMAND ----------

TWRPerformancedf = TWRPerformancedf.withColumn("churned" , f.when(f.col('churned') == 1 , 1).otherwise(0))

# COMMAND ----------



# COMMAND ----------

display(TWRPerformancedf)

# COMMAND ----------

display(TWRPerformancedf.groupBy("churned").count().orderBy("churned"))

# COMMAND ----------

TWRPerformancedf.columns

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="churned", outputCol="indexedLabel").fit(TWRPerformancedf)

# COMMAND ----------

features = ["AccountTWRPerformanceKey", "DataSourceKey", "AsOfDateKey", "AccountKey", "MTD", "QTD", "YTD", "OneYear", "ThreeYear", "FiveYear", "AgeRangeKey", "AccountAge", "HouseholdAge", "MarketValue"]

# COMMAND ----------

import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
 
from distutils.version import LooseVersion

# COMMAND ----------

(trainingData, testData) = TWRPerformancedf.randomSplit([0.7, 0.3])

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------



# COMMAND ----------

stages = []

# COMMAND ----------

label_stringIdx = StringIndexer(inputCol="churned", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

assembler = VectorAssembler(inputCols=features, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
  
partialPipeline = Pipeline().setStages(stages)
pipelineModel = partialPipeline.fit(TWRPerformancedf)
preppedDataDF = pipelineModel.transform(TWRPerformancedf)

# COMMAND ----------

# Fit model to prepped data
lrModel = LogisticRegression().fit(preppedDataDF)

# COMMAND ----------

# ROC for training data
display(lrModel, preppedDataDF, "ROC")

# COMMAND ----------

display(lrModel, preppedDataDF)

# COMMAND ----------

(trainingData, testData) = TWRPerformancedf.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

display(preppedDataDF)

# COMMAND ----------

selectedcols = ["label", "features", "churned"] + features
dataset = preppedDataDF.select(selectedcols)
display(dataset)

# COMMAND ----------

#cols = dataset.columns

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

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
 
# Create ParamGrid for Cross Validation
paramGrid = (ParamGridBuilder()
             .addGrid(lr.regParam, [0.01, 0.5, 2.0])
             .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])
             .addGrid(lr.maxIter, [1, 5, 10])
             .build())

# COMMAND ----------

# Create 5-fold CrossValidator
cv = CrossValidator(estimator=lr, estimatorParamMaps=paramGrid, evaluator=evaluator, numFolds=5)

# COMMAND ----------

# Run cross validations
cvModel = cv.fit(trainingData)
# this will likely take a fair amount of time because of the amount of models that we're creating and testing

# COMMAND ----------

# Use the test set to measure the accuracy of the model on new data
predictions = cvModel.transform(testData)

# COMMAND ----------

print('Model Intercept: ', cvModel.bestModel.intercept)

# COMMAND ----------

weights = cvModel.bestModel.coefficients
weights = [(float(w),) for w in weights]  # convert numpy type to float, and to tuple
weightsDF = sqlContext.createDataFrame(weights, ["Feature Weight"])
display(weightsDF)

# COMMAND ----------

selected = predictions.select("label", "prediction", "probability", "churned")
display(selected)

# COMMAND ----------

display(FeeRatiossdf)

# COMMAND ----------

display(FeeRatiossdf.groupBy("DataSourceKey").count())

# COMMAND ----------

display(FeeRatiossdf.groupBy("AccountAge").agg(F.mean('MarketValue'), F.count('MarketValue')))

# COMMAND ----------

FeeRatiossdf = FeeRatiossdf.withColumn("churned" , f.when(f.col('DataSourceKey').isin(churn) , 1))

# COMMAND ----------

FeeRatiossdf = FeeRatiossdf.withColumn("churned" , f.when(f.col('churned') == 1 , 1).otherwise(0))

# COMMAND ----------

display(FeeRatiossdf.groupBy("churned").count().orderBy("churned"))

# COMMAND ----------

FeeRatiossdf.columns

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import pyspark
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorAssembler
 
from distutils.version import LooseVersion

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="churned", outputCol="indexedLabel").fit(FeeRatiossdf)

# COMMAND ----------

features = ["FeeSummaryKey", "DataSourceKey", "SnapshotMonthKey", "FirmKey", "AccountKey", "RepresentativeKey", "AccountAssetRangeKey", "AccountFeeRangeKey", "HouseholdAssetRangeKey", "HouseholdFeeRangeKey", "IsNewKey", "IsQualifiedKey", "IsIRAKey", "CID", "AID", "AccountAnnualizedFeeAmount", "AccountMarketValue", "AccountEffectiveFeePercentage", "HouseholdAnnualizedFeeAmount", "HouseholdMarketValue", "HouseholdEffectiveFeePercentage", "ManagementExpenseAmount", "AccountAnnualizedFeeAmountWithManagementExpense", "AccountEffectiveFeePercentageWithManagementExpense", "HouseholdAnnualizedFeeAmountWithManagementExpense", "HouseholdEffectiveFeePercentageWithManagementExpense", "AccountFeeRangeWithManagementExpenseKey", "HouseholdFeeRangeWithManagementExpenseKey", "HouseholdManagementExpenseAmount"]

# COMMAND ----------

(trainingData, testData) = FeeRatiossdf.randomSplit([0.7, 0.3])

# COMMAND ----------

stages = []

# COMMAND ----------

label_stringIdx = StringIndexer(inputCol="churned", outputCol="label")
stages += [label_stringIdx]

# COMMAND ----------

assembler = VectorAssembler(inputCols=features, outputCol="features")
stages += [assembler]

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression

# COMMAND ----------

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

### Randomly split data into training and test sets. set seed for reproducibility
(trainingData, testData) = FeeRatiossdf.randomSplit([0.7, 0.3], seed=100)
print(trainingData.count())
print(testData.count())

# COMMAND ----------

from pyspark.ml.classification import LogisticRegression
 
# Create initial LogisticRegression model
lr = LogisticRegression(labelCol="label", featuresCol="features", maxIter=10)

# COMMAND ----------

# Train model with Training Data
lrModel = lr.fit(trainingData)

# COMMAND ----------

