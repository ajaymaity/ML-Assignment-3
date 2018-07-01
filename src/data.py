""" This module contains code which relate to data and preprocessing. """

from __future__ import print_function
from os.path import isfile, join
from numpy.random import permutation
from pandas import read_csv
from seaborn import boxplot, distplot, heatmap, pairplot
from six.moves.urllib.request import urlretrieve
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from zipfile import ZipFile

class DataFrameSelector(BaseEstimator, TransformerMixin):
    """ Selects a dataframe based on attributeNames. """

    def __init__(self, attributeNames):
        """ Constructor.
        Get attribute names and create an instance variable. """
        self.attributeNames = attributeNames

    def fit(self, X, y=None):
        """ Fit data. """
        return self

    def transform(self, X):
        """ Transform X based on fitted model. """
        return X[self.attributeNames].values

def fetchData(url, datasetsFolder, zipFileName):
    """ Fetch data from 'url', unzip 'zipFileName', and store in 'datasetsFolder'. """

    print("Checking if data is fetched from URL...")
    zipFolder = join(datasetsFolder, zipFileName)
    if (isfile(zipFolder)):
        print("YES! Moving on!")
    else:
        print("NO! Fetching data from URL...")
        urlretrieve(url, zipFolder)

        # unzip file
        zipFile = ZipFile(zipFolder, "r")
        zipFile.extractall(zipFolder.rsplit(".", 1)[0])
        zipFile.close()
        print("Done.")

def loadData(datasetsFolder, zipFileName, fileName):
    """ Loads the CSV file 'FILENAME' from unzipped folder. """
    return read_csv(join(datasetsFolder, zipFileName.rsplit(".", 1)[0], fileName), sep=";")

def printData(dataframe):
    """ Prints few instances of 'data', as well as its info and description. """

    print("\n\nData:")
    print(dataframe.head())

    print("\nInfo:")
    print(dataframe.info())

    print("\nDescription:")
    print(dataframe.describe())

def plotDistribution(variable, axes):
    """ Plots a distribution of a 'variable' of dataframe. """
    distplot(variable, ax=axes)

def calculateSkewness(variable):
    """ Calculates skewness of 'variable' of dataframe. """
    return variable.skew()

def calculateKurtosis(variable):
    """ Calculates kurtosis of 'variable' of dataframe. """
    return variable.kurt()

def plotHeatMap(twoDData, axes, xLabels, yLabels=False):
    """ Plots a heatmap of 'twoDData'. """
    heatmap(twoDData, annot=True, annot_kws={"size": 10}, xticklabels=xLabels, yticklabels=yLabels, ax=axes)

def plotBox(variable, axes):
    """ Plots a boxplot of 'variable'. """
    boxplot(x=variable, ax=axes)

def plotScatter(dataframe):
    """ Plots a scatter plot of 'dataframe'. """
    return pairplot(dataframe)

def removeOutliers(dataframe):
    """ Removes all the instances which has an outlier variable. """
    quantile1 = dataframe.quantile(0.25)
    quantile3 = dataframe.quantile(0.75)
    iqr = quantile3 - quantile1
    fenceLow = quantile1 - 1.5 * iqr
    fenceHigh = quantile3 + 1.5 * iqr
    fenceOut = (dataframe > fenceLow) & (dataframe < fenceHigh)
    notOutliers = fenceOut[fenceOut.columns[0]]
    for i, col in enumerate(list(fenceOut.columns)):
        if (i == 0): continue
        notOutliers = notOutliers & fenceOut[col]
    return dataframe.loc[notOutliers]

def separateVariable(dataframe, variable):
    """ Separates 'variable' from dataframe, and returns two dataframes. """
    dataframe1 = dataframe.drop(variable, axis=1)
    dataframe2 = dataframe[variable]
    return dataframe1, dataframe2

def createPipeline(features):
    """ Creates a pipeline for data cleansing and preparation. """

    pipeline = Pipeline([
        ("selector", DataFrameSelector(features)),
        ("imputer", Imputer(strategy="median")),
        ("std_scaler", StandardScaler())
    ])

    finalPipeline = FeatureUnion(transformer_list=[
        ("pipeline", pipeline)
    ])

    return finalPipeline

def splitData(featuresDF, labelDF, testRatio=0.3):
    """ Split "featuresDF" and "labelDF" into train and test, such that test is 30 percent of features and label. """

    assert len(featuresDF) == len(labelDF)
    shuffledIndices = permutation(len(labelDF))
    testSize = int(len(labelDF) * testRatio)
    testIndices = shuffledIndices[:testSize]
    trainIndices = shuffledIndices[testSize:]
    return featuresDF.iloc[trainIndices], featuresDF.iloc[testIndices], labelDF.iloc[trainIndices], labelDF.iloc[testIndices]
