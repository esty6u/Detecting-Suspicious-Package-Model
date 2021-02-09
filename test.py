# Miriam sadeh 315149021
# Esty Sicsu 312184732

import csv
import base64
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sklearn
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from numpy.random import randint
from sklearn import metrics
from normalization import *
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import KFold 
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
#from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import math
from logistic import *
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDClassifier
#from kernelApproximation import *
from SupportVectorMachine import *



pd.options.mode.chained_assignment = None  # default='warn'

#### normalization & standardization #######################
print('standardization & normalization of the Data set + features selection')
n = NormalizedDF()
nDF = n.getNormalizeDF()


x, y = n.getNormalizeXY()


#### logistic regression + L2 regulation #######################
iterations = 600
logisticL2 = Logistic(n.labels)



# logisticL2.trainingByGamma(x, y, iterations)
# logisticL2.ROC_AUC(x, y, iterations)
# logisticL2.over_underFitting(x, y, iterations, True)

# logisticL2.trainingNoGamma(x, y, iterations)
# logisticL2.over_underFitting(x, y, iterations, False)


#### SVM #######################################################

svmModel = SVM(x, y, n.labels)
#svmModel.linear_kernel()
# svmModel.ROC_AUC('linear')
# svmModel.over_under_fitting('linear')

# svmModel.polynomial_kernel()
# svmModel.ROC_AUC('poly')
svmModel.over_under_fitting('poly')

#svmModel.gaussian_kernel()
# svmModel.ROC_AUC('rbf')
# svmModel.over_under_fitting('rbf')


