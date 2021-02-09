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
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.metrics import log_loss


class Logistic:
    def __init__(self, labels):

        self.labels = labels
        self.Acc_errByGamma = list()
        self.noRegAcc_err = None
        self.minGamma = None
        self.minIndex = None
        self.min_error = None

        self.Acc_errOnTrain = []
        self.Acc_errOnTest = []

        self.logregOptimal = None
        self.optimalLoss = None

        self.auc = None

        self.inf_c = 1e100
        self.c = []
        self.gamma = []
        tmp = 10.0**6

        for i in range(10):
            self.c.append(tmp)
            self.gamma.append((1/tmp))
            tmp = (tmp /100)


        self.logGamma = [] # for the graph
        for i in range(len(self.gamma)):
            self.logGamma.append(math.log10( self.gamma[i] ))

    #################################################################################################################################
    def trainingByGamma(self, x, y, iterations):
        print('Training logistic regression model with regulation L2; choosing optimal gamma...')
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
        for i in range(len(self.c)):
            logreg = LogisticRegression(penalty='l2', dual=False, C = self.c[i], max_iter = iterations, solver='liblinear')
            logreg = logreg.fit(x_train.astype('float'), y_train.astype('float'))
            self.Acc_errByGamma.append(1 - logreg.score(x_test.astype('float'), y_test.astype('float')))

        self.min_error = min(self.Acc_errByGamma)
        self.minIndex = self.Acc_errByGamma.index(self.min_error)
        self.minGamma = self.gamma[self.minIndex]

        #self.optimalLoss = log_loss(y_train.astype('float'), y_pred)
        
        l = len(self.Acc_errByGamma)
        print('The Accuracy Error with Regulation')
        for i in range(l):
            print('For gamma = ', self.gamma[i], ' The Accuracy Error is: ', self.Acc_errByGamma[i])

        print('Optimal gamma = ', self.minGamma)
        graphAcc_errLogGamma(self.Acc_errByGamma, self.logGamma, self.min_error, self.minGamma)

        
    #################################################################################################################################        

    def trainingNoGamma(self, x, y, iterations):
        print('Training logistic regression model without regulation...')  
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
        logreg = LogisticRegression(penalty='l2', dual=False, C = self.inf_c, max_iter = iterations, solver='liblinear')
        logreg = logreg.fit(x_train.astype('float'), y_train.astype('float'))
        y_pred = logreg.predict(x_test)
        self.noRegAcc_err = 1 - logreg.score(x_test.astype('float'), y_test.astype('float'))
        print('Accuracy Error without Regulation: ', self.noRegAcc_err)
        plotCM(y_test,y_pred, self.labels)

    def trainingByMinGamma(self, x, y, iterations):
        print('Trainig with rgulation L2 by optimal gamma = ', self.minGamma)
        if self.minGamma != None:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30)
            self.logregOptimal = LogisticRegression(penalty='l2', dual=False, C = (1/self.minGamma), max_iter = iterations, solver='liblinear')
            self.logregOptimal = self.logregOptimal.fit(x_train.astype('float'), y_train.astype('float'))
            y_pred = self.logregOptimal.predict(x_test)
            plotCM(y_test,y_pred, self.labels)
    #################################################################################################################################
    def over_underFitting(self, x, y, iterations, mg):
        plt.clf()
        r, c = x.shape
        # print(r,c)
        fixedTestIndex = np.arange(300000, r)
        x_test, y_test = x.iloc[fixedTestIndex], y.iloc[fixedTestIndex]
        samplesIndex = []
        samplesNum = []
        tmp = 0

        if self.minGamma == None or mg == False:
            print('Over-fitting and under-fitting graph (logistic regression); No optimal gamma -> model without regulation')
        else:
            print('Over-fitting and under-fitting graph (logistic regression); model by optimal gamma: ', self.minGamma)


        for i in range(20):
            tmp = tmp + (300000/20)
            samplesIndex = np.arange(0, tmp)
            x_train, y_train = x.iloc[samplesIndex], y.iloc[samplesIndex]
            #print(len(x_train))
            if self.minGamma == None or mg == False:
                logreg = LogisticRegression(penalty='l2', dual=False, C = self.inf_c, max_iter = iterations, solver='liblinear')
            else:
                logreg = LogisticRegression(penalty='l2', dual=False, C = (1/self.minGamma), max_iter = iterations, solver='liblinear')
            logreg = logreg.fit(x_train.astype('float'), y_train.astype('float'))

            self.Acc_errOnTrain.append(1 - logreg.score(x_train.astype('float'), y_train.astype('float')))
            self.Acc_errOnTest.append(1 - logreg.score(x_test.astype('float'), y_test.astype('float')))
            samplesIndex = []
            samplesNum.append(tmp)

        graphAcc_errOnTrainTestByOptimalGamma(samplesNum, self.Acc_errOnTrain, 
                            self.Acc_errOnTest, self.minGamma, mg)


    #################################################################################################################################
    def ROC_AUC(self, x, y, iterations):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
        if self.minGamma == None:
            print('Roc curve and AUC; No optimal gamma -> model without regulation')
            logreg = LogisticRegression(penalty='l2', dual=False, C = self.inf_c, max_iter = iterations, solver='liblinear')
            logreg = logreg.fit(x_train.astype('float'), y_train.astype('float'))
        elif self.logregOptimal != None:
            print('Roc curve and AUC; model by optimal gamma: ', self.minGamma)
            logreg = self.logregOptimal
        else: 
            self.trainingByMinGamma(x, y, iterations)
            logreg = self.logregOptimal
        
        
        probas_= logreg.predict_proba(x_test.astype('float'))
        ##Computing false and true positive rates
        fpr, tpr, threshold = metrics.roc_curve(y_test.astype('float'), probas_[:, 1],
                                        drop_intermediate=False)
        print('fpr, tpr, threshold', fpr, tpr, threshold)

        self.auc = roc_auc_score(logreg.predict(x_test.astype('float')),y_test.astype('float'))
        print("auc", self.auc)

        plt.figure()
        ##Adding the ROC
        plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % self.auc)
        ##Random FPR and TPR
        plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
        ##Title and label
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.show()

        
######################################################################################################################################################
def graphAcc_errLogGamma(Acc_errByGamma, logGamma, min_error, minGamma):

    plt.plot(logGamma, Acc_errByGamma, 'b', label = 'Min_Acc_err = %0.10f' % min_error, color = 'g')
    plt.plot(logGamma, Acc_errByGamma, 'b', label = 'minGamma = %0.10f' % minGamma, color = 'g')

    plt.title('Accuracy_error by gamma ')
    plt.legend(loc = 'lower right')
    plt.xlim([logGamma[0]-1, logGamma[-1]+1])
    plt.ylim([0, 0.05])
    plt.ylabel('accuracy-error')
    plt.xlabel('log10(gamma)')
    plt.show()



def graphAcc_errOnTrainTestByOptimalGamma(samplesNum, Acc_errOnTrain, Acc_errOnTest, minGamma, mg):
    
    plt.plot(samplesNum, Acc_errOnTrain, 'b', color = 'g')
    plt.text(samplesNum[-1], Acc_errOnTrain[-1], 'train', fontsize=10, color='g')
    if minGamma == None or mg == False:
        plt.plot(samplesNum, Acc_errOnTest, 'b', color = 'b')
    else:
        plt.plot(samplesNum, Acc_errOnTest, 'b', 
                label = 'Optimal gamma = %0.10f' % minGamma, color = 'b')          
    plt.text(samplesNum[-1], Acc_errOnTest[-1], 'test', fontsize=10, color='b')

    plt.title('Accuracy_error on train and test')
    plt.legend(loc = 'lower right')
    plt.xlim([0, samplesNum[-1] + 20])
    plt.ylim([-0.5, 1.2])
    plt.ylabel('accuracy-error')
    plt.xlabel('samples')
    plt.show()


def plotCM(y_test,y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    plot_confusion_matrix(cm_normalized, labels, title='Normalized confusion matrix')
    plt.show() 

def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')