""" This module should be run for Assignment 3. It contains "main" method, and references "data" and "model" modules. """

""" Wine Quality Ratings and Chemicals """
from data import calculateKurtosis, calculateSkewness, createPipeline, fetchData, loadData, plotBox, plotDistribution, plotHeatMap, plotScatter, printData, removeOutliers, separateVariable, splitData
from matplotlib.pyplot import show, subplots, xticks, yticks
from model import Model
from numpy import array, corrcoef
from os import makedirs
from os.path import isdir, join
from pandas import DataFrame
import configparser
import matplotlib.pyplot as plt

config = configparser.ConfigParser()
config.read("config.ini")

DATASETS_FOLDER = join("..", config["DEFAULT"]["datasetsFolder"]) # one level up
GRAPHS_FOLDER = join("..", config["DEFAULT"]["graphsFolder"])
ZIP_FILENAME = config["DEFAULT"]["zipFileName"]
URL = config["DEFAULT"]["url"]
FILENAME = config["DEFAULT"]["fileName"]
LABEL = config["DEFAULT"]["label"]

def calculateDataCharAndPlotDist(data, cols, iter):
    """ Calculates Skewness and Kurtosis of 'data', and return as a 2-D array.
        Also plot the distribution of each variable in 'data'. """

    outlierInfo = "(Without Outliers)" if iter == "2" else ""
    rows = 3 if iter == "2" else 4
    fig, axes = subplots(rows, 3, figsize=(25, 15))
    fig.suptitle("Distribution of each variable " + outlierInfo, fontsize=40)
    dataChar = list() # a 2D array which stores each column's skewness and kurtosis (data characteristics).
    for i in range(cols.shape[0]):
        temp = list()
        dataCol = data[cols[i]]
        plotDistribution(dataCol, axes[i / 3][i % 3])
        plt.savefig(join(GRAPHS_FOLDER, "distributions_(" + iter + ").png"), format="png")
        temp.append(calculateSkewness(dataCol))
        temp.append(calculateKurtosis(dataCol))
        dataChar.append(temp)
    return dataChar

def visualizeData(data, iter):
    """ Visualize univariate and bivariate data with the help of histograms, box plots, heat maps and scatter plots. """

    # Plot univariate distributions and calculate data characteristics
    outlierInfo = "(Without Outliers)" if iter == "2" else ""
    rows = 3 if iter == "2" else 4
    cols = data.columns
    dataChar = calculateDataCharAndPlotDist(data, cols, iter)
    fig, (ax1, ax2) = subplots(1, 2, figsize=(25, 15))
    fig.suptitle("Heatmap of Skewness and Kurtosis " + outlierInfo, fontsize=40)
    plotHeatMap(array([row[0] for row in dataChar]).reshape(cols.shape[0], 1), ax1, ["Skewness"])
    plotHeatMap(array([row[1] for row in dataChar]).reshape(cols.shape[0], 1), ax2, ["Kurtosis"], cols)
    yticks(rotation=0)
    plt.savefig(join(GRAPHS_FOLDER, "heatmap_of_skewness_and_kurtosis_(" + iter + ").png"), format="png")

    # Plot box plots of univariates.
    fig, axes = subplots(rows, 3, figsize=(25, 15))
    fig.suptitle("Box plot of each variable " + outlierInfo, fontsize=40)
    for i in range(cols.shape[0]):
        plotBox(data[cols[i]], axes[i / 3][i % 3])
    plt.savefig(join(GRAPHS_FOLDER, "box_plots_(" + iter + ").png"), format="png")

    # Bi-variate analysis
    # Find correlation between variables.
    correlationMatrix = data.corr()
    qualityCorrColSorted = correlationMatrix.nlargest(data.shape[1], LABEL)[LABEL].index
    corrCoef = corrcoef(data[qualityCorrColSorted].values.T)
    fig, axes = subplots(figsize=(25, 15))
    fig.suptitle("Correlation Heatmap of all variables w.r.t. " + LABEL + " " + outlierInfo, fontsize=40)
    plotHeatMap(corrCoef, axes, qualityCorrColSorted.values, qualityCorrColSorted.values)
    xticks(rotation=90)
    yticks(rotation=0)
    plt.savefig(join(GRAPHS_FOLDER, "correlation_heatmap_of_all_variables_wrt_" + LABEL + "_(" + iter + ").png"), format="png")

    # Plot scatter charts
    scatter = plotScatter(data[qualityCorrColSorted])
    scatter.set(xticklabels=[])
    scatter.set(yticklabels=[])
    scatter.fig.suptitle("Scatter plot of all variables w.r.t. " + LABEL + " " + outlierInfo, fontsize=40)
    scatter.fig.subplots_adjust(top=.9)
    plt.savefig(join(GRAPHS_FOLDER, "scatter_plot_of_all_variables_wrt_" + LABEL + "_(" + iter + ").png"), format="png")

if __name__ == "__main__":
    if not isdir(DATASETS_FOLDER): makedirs(DATASETS_FOLDER)
    if not isdir(GRAPHS_FOLDER): makedirs(GRAPHS_FOLDER)
    fetchData(URL, DATASETS_FOLDER, ZIP_FILENAME)
    data = loadData(DATASETS_FOLDER, ZIP_FILENAME, FILENAME)
    printData(data)

    visualizeData(data, "1")
    """ From the distributions and heatmaps, we see chlorides has the highest skewness and kurtosis.
        Also, other variables like volatile acidity, free sulfur dioxide, density, etc have a lot of outliers. """
    """ From the boxplot, we confirm that chlorides have the highest number of outliers,
        and all the remaining variables have outliers except alcohol. """
    """ From the heatmap, residual sugar and density has correlation = 0.84,
        free sulfur dioxide and total sulfur dioxide has correlation = 0.62,
        alcohol and density has correlation = -0.78. """
    """ From the scatterplot, density seems to have a linear or constant relationship with every other variable.
        Also, heatmap suggests same. Hence, we will drop density.
        Out of free sulfur dioxide and total sulfur dioxide, we drop free sulfur dioxide since it has lower correlation with quality.
        We also drop the lowest correlated variable, citric acid. """

    data.drop(["density", "free sulfur dioxide", "citric acid"], axis=1, inplace=True)

    data = removeOutliers(data)
    """ Removing the outliers reduced the data from 4898 to 4039. """

    # Let's visualize data again...
    printData(data)
    visualizeData(data, "2")

    ### Data preprocessing and cleansing...
    print("\n")
    print("Separating features and label...")
    features, label = separateVariable(data, LABEL)
    print("Creating a pipeline for imputing missing values, and standardizing all the data...")
    pipeline = createPipeline(features.columns.values)
    print("Starting the pipeline to transform features...")
    featuresFinal = pipeline.fit_transform(features)
    print("Splitting the data into train and test set...")
    featuresTrain, featuresTest, labelTrain, labelTest = splitData(DataFrame(featuresFinal), DataFrame(label))
    print("Done.")

    print("\n")
    print("Create Regression Models...")
    linearModel = Model(featuresTrain, labelTrain, Model.LINEAR_REG)
    svrModel = Model(featuresTrain, labelTrain, Model.SVR)
    print("Done.")

    # print("\n")
    # print("Computing best parameters for Support Vector Regression...")
    # tunedParameters = [{"kernel": ["linear", "rbf"],
    #                     "gamma": [1e-3, 1e-4],
    #                     "C": [1, 10, 100, 1000]}]
    # bestParams = svrModel.computeSVRParams(tunedParameters)
    # print(bestParams)

    svrModel.setSVRParams("rbf", 0.001, 1000)

    print("\n")
    print("Performing cross validation on training data for Linear Regression...")
    rmseTr = linearModel.rmseCrossVal()
    r2Tr = linearModel.r2CrossVal()
    print("Model Trained.")
    print("Training RMSE: %.4f (+/- %.4f)" % (rmseTr.mean(), rmseTr.std() * 2))
    print("Training R2: %.4f (+/- %.4f)" % (r2Tr.mean(), r2Tr.std() * 2))
    print("Done.")

    print("\n")
    print("Performing cross validation on training data for Support Vector Regression...")
    rmseTr = svrModel.rmseCrossVal()
    r2Tr = svrModel.r2CrossVal()
    print("Model Trained.")
    print("Training RMSE: %.4f (+/- %.4f)" % (rmseTr.mean(), rmseTr.std() * 2))
    print("Training R2: %.4f (+/- %.4f)" % (r2Tr.mean(), r2Tr.std() * 2))
    print("Done.")

    print("\n")
    print("Prediction on test data for Linear Regression...")
    linearModel.fit()
    prediction = linearModel.predict(featuresTest)
    rmseTe = linearModel.calculateRmse(labelTest, prediction)
    r2Te = linearModel.calculateR2(labelTest, prediction)
    print("Test RMSE: %.4f" % rmseTe)
    print("Test R2: %.4f" % r2Te)
    print("Done.")

    print("\n")
    print("Prediction on test data for Support Vector Regression...")
    svrModel.fit()
    prediction = svrModel.predict(featuresTest)
    rmseTe = svrModel.calculateRmse(labelTest, prediction)
    r2Te = svrModel.calculateR2(labelTest, prediction)
    print("Test RMSE: %.4f" % rmseTe)
    print("Test R2: %.4f" % r2Te)
    print("Done.")

    show()
