# Databricks notebook source
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

oriondskey = dbutils.secrets.get(scope = "allsecrects", key = "oriondskey")

# COMMAND ----------

spark.conf.set(
  "fs.azure.account.key.orionstoreds.blob.core.windows.net",
  oriondskey)

# COMMAND ----------

titanicds = spark.read.option("header","true").option("inferSchema", "true").csv("wasbs://dsppsample@orionstoreds.blob.core.windows.net/titanic/Titanic.csv")

# COMMAND ----------

display(titanicds)

# COMMAND ----------

titanicds.columns

# COMMAND ----------

display(titanicds.printSchema())

# COMMAND ----------

#titanicds = titanicds.na.fill(0).show()
titanicds1 = titanicds.na.fill(0)

# COMMAND ----------

display(titanicds1)

# COMMAND ----------

features = ['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoder, StringIndexer

# COMMAND ----------

stringIndexer = StringIndexer(inputCol="Name,Sex,Ticket,Cabin,Embarked", outputCol="categoryIndex")

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
assembler = VectorAssembler(inputCols=["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"],outputCol="features")

# COMMAND ----------

display(stringIndexer)

# COMMAND ----------

features = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch',  'Fare']

# COMMAND ----------

#titanicds = titanicds.na.drop()

# COMMAND ----------



# COMMAND ----------

from pyspark.sql import functions as f

#titanicds = titanicds.withColumn("Age1" , f.when(f.col('Age') == "null" , 0))

# COMMAND ----------

display(titanicds1)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

featureindexer = StringIndexer(inputCol="Name,Sex,Ticket,Cabin,Embarked", outputCol="featureIndex")

# COMMAND ----------

display(titanicds1)

# COMMAND ----------

(trainingData, testData) = titanicds1.randomSplit([0.7, 0.3])

# COMMAND ----------

featurescol = ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="Survived", outputCol="indexedLabel").fit(titanicds1)

#featureIndexer =\
#    VectorIndexer(inputCol="PassengerId, Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked", outputCol="indexedFeatures").fit(titanicds1)


(trainingData, testData) = titanicds1.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)
assembler = VectorAssembler(inputCols=features, outputCol="features")

# Chain indexers and forest in a Pipeline
pipeline = Pipeline(stages=[labelIndexer, assembler, rf, labelConverter])

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "Survived", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = ", accuracy)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only

# COMMAND ----------



# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import IndexToString, StringIndexer, VectorIndexer
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# COMMAND ----------

categoricalColumns = [item[0] for item in titanicds1.dtypes if item[1].startswith('string') ]

# COMMAND ----------

categoricalColumns

# COMMAND ----------

#featurescol = ["PassengerId", "Pclass", "Age", "SibSp", "Parch", "Fare"]
featurescol = ["PassengerId", "Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]

# COMMAND ----------

(trainingData, testData) = titanicds1.randomSplit([0.7, 0.3])

# COMMAND ----------

#assembler = VectorAssembler(inputCols=featurescol, outputCol="features")

# COMMAND ----------

stages = []

# COMMAND ----------

#iterate through all categorical values
for categoricalCol in categoricalColumns:
    #create a string indexer for those categorical values and assign a new name including the word 'Index'
    stringIndexer = StringIndexer(inputCol = categoricalCol, outputCol = categoricalCol + 'Index')

    #append the string Indexer to our list of stages
    stages += [stringIndexer]

# COMMAND ----------

stages

# COMMAND ----------

labelIndexer = StringIndexer(inputCol="Survived", outputCol="indexedLabel").fit(titanicds1)

#assembler = VectorAssembler(inputCols=featurescol, outputCol="features")
assembler = VectorAssembler(inputCols=featurescol, outputCol="features")

#featureIndexer = VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(titanicds1)


(trainingData, testData) = titanicds1.randomSplit([0.7, 0.3])

# Train a RandomForest model.
rf = RandomForestClassifier(labelCol="indexedLabel", featuresCol="features", numTrees=10)

# Convert indexed labels back to original labels.
labelConverter = IndexToString(inputCol="prediction", outputCol="predictedLabel",
                               labels=labelIndexer.labels)


# Chain indexers and forest in a Pipeline
#pipeline = Pipeline(stages=[labelIndexer, assembler, rf, labelConverter])
stages += [labelIndexer]
stages += [assembler]
stages += [rf]
stages += [labelConverter]

pipeline = Pipeline(stages=stages)

# Train model.  This also runs the indexers.
model = pipeline.fit(trainingData)

# Make predictions.
predictions = model.transform(testData)

# Select example rows to display.
predictions.select("predictedLabel", "Survived", "features").show(5)

# Select (prediction, true label) and compute test error
evaluator = MulticlassClassificationEvaluator(
    labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Accuracy = ", accuracy)
print("Test Error = %g" % (1.0 - accuracy))

rfModel = model.stages[2]
print(rfModel)  # summary only