

import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC  
from sklearn.metrics import classification_report, confusion_matrix  

from sklearn import svm, datasets
from sklearn.svm import SVC 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score 


class SVM:
    def __init__(self,x, y, labels):
        self.x = x
        self.y = y
        self.labels = labels

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size = 0.30)
        self.linear = None
        self.polynomial = None
        self.gaussian = None
        self.sigmoid = None

        self.linearModel = None
        self.polynomialModel = None
        self.gaussianModel = None
        self.sigmoidModel = None
        

        self.Acc_errOnTrain = []
        self.Acc_errOnTest = []


    def linear_kernel(self):
        print('Linear Kernel')
        svclassifier = SVC(C=1.0, cache_size=200, gamma='auto', kernel='linear', max_iter=-1, 
                    probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 

        svclassifier.fit(self.x_train, self.y_train) 
        y_pred = svclassifier.predict(self.x_test)

        self.linearModel = svclassifier

        cm = confusion_matrix(self.y_test,y_pred)
        cr = classification_report(self.y_test,y_pred) 
        print(cr)
        self.linear = cr

        plotCM(self.y_test,y_pred, self.labels)
          
    def polynomial_kernel(self):
        print('Polynomial Kernel')
        svclassifier = SVC(C=1.0, cache_size=200, degree=3, gamma='auto', kernel='poly', max_iter=-1, 
                        probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)  
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test) 

        self.polynomialModel = svclassifier

        cm = confusion_matrix(self.y_test,y_pred)
        cr = classification_report(self.y_test,y_pred) 
        print(cr)
        self.polynomial = cr

        plotCM(self.y_test,y_pred, self.labels)

    
    def gaussian_kernel(self):
        print('Gaussian Kernel')
        svclassifier = SVC(C=1.0, cache_size=200, gamma='auto', kernel='rbf', max_iter=-1, 
                        probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)  
        svclassifier.fit(self.x_train, self.y_train)
        y_pred = svclassifier.predict(self.x_test) 

        self.gaussianModel = svclassifier

        cm = confusion_matrix(self.y_test,y_pred)
        cr = classification_report(self.y_test,y_pred)  
        print(cr)
        self.gaussian = cr

        plotCM(self.y_test,y_pred, self.labels)

    def sigmoid_kernel(self):
        print('Sigmoid Kernel')
        svclassifier = SVC(C=1.0, cache_size=200, gamma='auto', kernel='sigmoid', max_iter=-1, 
                        probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)  
        svclassifier.fit(self.x_train, self.y_train) 
        y_pred = svclassifier.predict(self.x_test)

        self.sigmoidModel = svclassifier

        cm = confusion_matrix(self.y_test,y_pred)
        cr = classification_report(self.y_test,y_pred)  
        print(cr)
        self.sigmoid = cr 

        plotCM(self.y_test,y_pred, self.labels)


            
    ###############################################################################################################3
    def over_under_fitting(self, type):

        if type == 'linear' or type == 'poly' or type == 'rbf' or type == 'sigmoid':
            print('Over-fitting and under-fitting graph (SVM), type: ', type)
        else:
            print('Kernel type error')
            return

        r, c = self.x.shape
        fixedTestIndex = np.arange(150000, r)
        x_test, y_test = self.x.iloc[fixedTestIndex], self.y.iloc[fixedTestIndex]
        samplesIndex = []
        samplesNum = []
        tmp = 0


        for i in range(5):
            tmp = tmp + (int)(150000/5)
            samplesIndex = np.arange(0, tmp)
            x_train, y_train = self.x.iloc[samplesIndex], self.y.iloc[samplesIndex]
            print('samples number: ',tmp)
            if type == 'poly':
                svclassifier = SVC( C=1.0, cache_size=200, gamma='auto', kernel = type, degree=5, max_iter=-1, 
                        probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
            else: 
                svclassifier = SVC(C=1.0, cache_size=200, gamma='auto', kernel=type, max_iter=-1, 
                        probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False) 
                
            svclassifier.fit(x_train, y_train)

            self.Acc_errOnTrain.append(1 - svclassifier.score(x_train.astype('float'), y_train.astype('float')))
            self.Acc_errOnTest.append(1 - svclassifier.score(x_test.astype('float'), y_test.astype('float')))
            samplesIndex = []
            samplesNum.append(tmp)

        graphAcc_errOnTrainTest(samplesNum, self.Acc_errOnTrain, self.Acc_errOnTest)

    def ROC_AUC(self, type):
        if type == 'linear' or type == 'poly' or type == 'rbf' or type == 'sigmoid':
            print('ROC curve & AUC (SVM), type: ', type)
            if type == 'linear':
                calc(self.linearModel, self.x_test, self.y_test)
            if type == 'poly':
                calc(self.polynomialModel, self.x_test, self.y_test)
            if type == 'rbf':
                calc(self.gaussianModel, self.x_test, self.y_test)
            if type == 'sigmoid':
                calc(self.sigmoidModel, self.x_test, self.y_test)
        else:
            print('Kernel type error')
            return
        

#####################################################################################
def calc(model, x_test, y_test):
    ##Computing false and true positive rates
    fpr, tpr, threshold = roc_curve(y_test.astype('float'),  model.predict(x_test.astype('float')) ,drop_intermediate=False)
    print('fpr, tpr, threshold', fpr, tpr, threshold)

    auc = roc_auc_score(y_test.astype('float'), model.predict(x_test.astype('float')))
    #auc = roc_auc_score(model.predict(x_test.astype('float')),y_test.astype('float'))
    print("auc", auc)

    plt.figure()
    ##Adding the ROC
    plt.plot(fpr, tpr, color='red', lw=2, label='ROC curve (area = %0.2f)' % auc)
    ##Random FPR and TPR
    plt.plot([0, 1], [0, 1], color='blue', lw=2, linestyle='--')
    ##Title and label
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
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

def graphAcc_errOnTrainTest(samplesNum, Acc_errOnTrain, Acc_errOnTest):
    plt.plot(samplesNum, Acc_errOnTrain, 'b', color = 'g')
    plt.text(samplesNum[-1], Acc_errOnTrain[-1], 'train', fontsize=10, color='g')
        
    plt.text(samplesNum[-1], Acc_errOnTest[-1], 'test', fontsize=10, color='b')

    plt.title('Accuracy_error on train and test')
    plt.legend(loc = 'lower right')
    plt.xlim([0, samplesNum[-1] + 20])
    plt.ylim([-0.5, 0.5])
    plt.ylabel('accuracy-error')
    plt.xlabel('samples')
    plt.show()
