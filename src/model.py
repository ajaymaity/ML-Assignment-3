""" This module contains code which relate to model creation, training and evaluation. """

from numpy import sqrt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.svm import SVR

class Model:
    """ This class contains methods to create sklearn models, train and evaluate them. """

    LINEAR_REG = "LINEAR_REG"
    SVR = "SVR"

    def __init__(self, features, label, modelName):
        """ Constructor.
        Initializes the model specified in 'modelName', and creates 'features' and 'labels' class variables. """

        self.features = features
        self.label = label
        self.modelName = modelName

        if modelName == self.LINEAR_REG:
            self.model = LinearRegression()
        elif modelName == self.SVR:
            self.model = SVR()

    def fit(self):
        """ Fit the model on class' 'features' and 'label'. """
        self.model.fit(self.features, self.label)

    def predict(self, data):
        return self.model.predict(data)

    def calculateRmse(self, actualData, predictedData):
        """ Calculate the root mean square error between 'actualData' and 'predictedData'. """

        mse = mean_squared_error(actualData, predictedData)
        return sqrt(mse)

    def calculateR2(self, actualData, predictedData):
        """ Calculate the R2 between 'actualData' and 'predictedData'. """

        return r2_score(actualData, predictedData)

    def rmseCrossVal(self, cv=10):
        nmse = cross_val_score(self.model, self.features, self.label.values.ravel(), scoring="neg_mean_squared_error", cv=cv)
        return sqrt(-nmse)

    def r2CrossVal(self, cv=10):
        return cross_val_score(self.model, self.features, self.label.values.ravel(), scoring="r2", cv=cv)

    def computeSVRParams(self, tunedParameters, cv=10):
        assert self.modelName == self.SVR
        model = GridSearchCV(SVR(), tunedParameters, cv=cv, scoring="neg_mean_squared_error")
        model.fit(self.features, self.label.values.ravel())

        print("Grid scores:")
        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, param in zip(means, stds, model.cv_results_["params"]):
            print("%.4f (+/-%.4f) for %r" % (mean, std * 2, param))

        return model.best_params_

    def setSVRParams(self, kernel, gamma, C):
        assert self.modelName == self.SVR
        self.model = SVR(kernel=kernel, gamma=gamma, C=C)
