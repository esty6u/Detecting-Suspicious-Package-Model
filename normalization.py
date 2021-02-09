


import csv
import base64
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from numpy.random import randint
from sklearn import metrics
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

from scipy.stats.stats import pearsonr

from featureSelection import *



# Encode text values to dummy variables(i.e. [1,0,0],[0,1,0],[0,0,1] for red,green,blue)
def encode_text_dummy(df, name):
    dummies = pd.get_dummies(df[name])
    for x in dummies.columns:
        dummy_name = f"{name}-{x}"
        df[dummy_name] = dummies[x]
    df.drop(name, axis=1, inplace=True)


# Encode text values to a single dummy variable.  The new columns (which do not replace the old) will have a 1
# at every location where the original column (name) matches each of the target_values.  One column is added for
# each target value.
def encode_text_single_dummy(df, name, target_values):
    for tv in target_values:
        l = list(df[name].astype(str))
        l = [1 if str(x) == str(tv) else 0 for x in l]
        name2 = f"{name}-{tv}"
        df[name2] = l


# Encode text values to indexes(i.e. [1],[2],[3] for red,green,blue).
def encode_text_index(df, name ,main=None):
    newOutcome = []
    if main is not None:
        for other in df[name]:
            if other != main:
                newOutcome.append("other")
            else:
                newOutcome.append(main)
        df[name] = newOutcome
    #print(df[name][490950:491050])           
    le = preprocessing.LabelEncoder()
    df[name] = le.fit_transform(df[name])
    return le.classes_

###########################################################################################3
# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd
###########################################################################################
# Convert all missing values in the specified column to the median
def missing_median(df, name):
    med = df[name].median()
    df[name] = df[name].fillna(med)
###########################################################################################


# Convert all missing values in the specified column to the default
def missing_default(df, name, default_value):
    df[name] = df[name].fillna(default_value)


# Convert a Pandas dataframe to the x,y inputs that TensorFlow needs
def to_xy(df, target):
    result = []
    for x in df.columns:
        if x != target:
            result.append(x)

    return df[result], df[[target]]

# Nicely formatted time string
def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m:>02}:{s:>05.2f}"


# Regression chart.
def chart_regression(pred, y, sort=True):
    t = pd.DataFrame({'pred': pred, 'y': y.flatten()})
    if sort:
        t.sort_values(by=['y'], inplace=True)
    plt.plot(t['y'].tolist(), label='expected')
    plt.plot(t['pred'].tolist(), label='prediction')
    plt.ylabel('output')
    plt.legend()
    plt.show()
##################################################################################
# Remove all rows where the specified column is +/- sd standard deviations
def remove_outliers(df, name, sd):
    drop_rows = df.index[(np.abs(df[name] - df[name].mean())
                          >= (sd * df[name].std()))]
    df.drop(drop_rows, axis=0, inplace=True)


# Encode a column to a range between normalized_low and normalized_high.
def encode_numeric_range(df, name, normalized_low=-1, normalized_high=1,
                         data_low=None, data_high=None):
    if data_low is None:
        data_low = min(df[name])
        data_high = max(df[name])

    df[name] = ((df[name] - data_low) / (data_high - data_low)) \
        * (normalized_high - normalized_low) + normalized_low


# This function submits an assignment.  You can submit an assignment as much as you like, only the final
# submission counts.  The paramaters are as follows:
# data - Pandas dataframe output.
# key - Your student key that was emailed to you.
# no - The assignment class number, should be 1 through 1.
# source_file - The full path to your Python or IPYNB file.  This must have "_class1" as part of its name.  
# .             The number must match your assignment number.  For example "_class2" for class assignment #2.
def submit(data,key,no,source_file=None):
    if source_file is None and '__file__' not in globals(): raise Exception('Must specify a filename when a Jupyter notebook.')
    if source_file is None: source_file = __file__
    suffix = '_class{}'.format(no)
    if suffix not in source_file: raise Exception('{} must be part of the filename.'.format(suffix))
    with open(source_file, "rb") as image_file:
        encoded_python = base64.b64encode(image_file.read()).decode('ascii')
    ext = os.path.splitext(source_file)[-1].lower()
    if ext not in ['.ipynb','.py']: raise Exception("Source file is {} must be .py or .ipynb".format(ext))
    r = requests.post("https://api.heatonresearch.com/assignment-submit",
        headers={'x-api-key':key}, json={'csv':base64.b64encode(data.to_csv(index=False).encode('ascii')).decode("ascii"),
        'assignment': no, 'ext':ext, 'py':encoded_python})
    if r.status_code == 200:
        print("Success: {}".format(r.text))
    else: print("Failure: {}".format(r.text))



##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
##################################################################################################################################################
class NormalizedDF:


    def __init__(self):

        pd.options.mode.chained_assignment = None  # default='warn'


        path = './data/'
        filenam_read = os.path.join(path, 'kddcup.data_10_percent_corrected')
        #file_name = os.path.join(path, 'kddcup.data_10_percent_corrected')
        df = pd.read_csv(filenam_read)
        df = pd.read_csv(filenam_read, na_values=['NA', '?'])
        #df = pd.read_csv("C:/ML/FinalProjectML/data/kddcup.data_10_percent_corrected", header=None)

        print("Read {} rows.".format(len(df)))
        # df = df.sample(frac=0.1, replace=False) # Uncomment this line to sample only 10% of the dataset
        df.dropna(inplace=True,axis=1) # For now, just drop NA's (rows with missing values)

        # The CSV file has no column heads, so add them
        df.columns = [
            'duration',
            'protocol_type',
            'service',
            'flag',
            'src_bytes',
            'dst_bytes',
            'land',
            'wrong_fragment',
            'urgent',
            'hot',
            'num_failed_logins',
            'logged_in',
            'num_compromised',
            'root_shell',
            'su_attempted',
            'num_root',
            'num_file_creations',
            'num_shells',
            'num_access_files',
            'num_outbound_cmds',
            'is_host_login',
            'is_guest_login',
            'count',
            'srv_count',
            'serror_rate',
            'srv_serror_rate',
            'rerror_rate',
            'srv_rerror_rate',
            'same_srv_rate',
            'diff_srv_rate',
            'srv_diff_host_rate',
            'dst_host_count',
            'dst_host_srv_count',
            'dst_host_same_srv_rate',
            'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate',
            'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate',
            'dst_host_srv_serror_rate',
            'dst_host_rerror_rate',
            'dst_host_srv_rerror_rate',
            'outcome'
        ]
        df = df.drop(columns=['is_host_login' , 'is_guest_login' , 'logged_in'])

        encode_numeric_zscore(df, 'duration')
        encode_text_dummy(df, 'protocol_type')
        encode_text_dummy(df, 'service')
        encode_text_dummy(df, 'flag')
        encode_numeric_zscore(df, 'src_bytes')
        encode_numeric_zscore(df, 'dst_bytes')
        encode_text_dummy(df, 'land')
        encode_numeric_zscore(df, 'wrong_fragment')
        encode_numeric_zscore(df, 'urgent')
        encode_numeric_zscore(df, 'hot')
        encode_numeric_zscore(df, 'num_failed_logins')
        #encode_text_dummy(df, 'logged_in')
        encode_numeric_zscore(df, 'num_compromised')
        encode_numeric_zscore(df, 'root_shell')
        encode_numeric_zscore(df, 'su_attempted')
        encode_numeric_zscore(df, 'num_root')
        encode_numeric_zscore(df, 'num_file_creations')
        encode_numeric_zscore(df, 'num_shells')
        encode_numeric_zscore(df, 'num_access_files')
        encode_numeric_zscore(df, 'num_outbound_cmds')
        #encode_text_dummy(df, 'is_host_login')
        #encode_text_dummy(df, 'is_guest_login')
        encode_numeric_zscore(df, 'count')
        encode_numeric_zscore(df, 'srv_count')
        encode_numeric_zscore(df, 'serror_rate')
        encode_numeric_zscore(df, 'srv_serror_rate')
        encode_numeric_zscore(df, 'rerror_rate')
        encode_numeric_zscore(df, 'srv_rerror_rate')
        encode_numeric_zscore(df, 'same_srv_rate')
        encode_numeric_zscore(df, 'diff_srv_rate')
        encode_numeric_zscore(df, 'srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_count')
        encode_numeric_zscore(df, 'dst_host_srv_count')
        encode_numeric_zscore(df, 'dst_host_same_srv_rate')
        encode_numeric_zscore(df, 'dst_host_diff_srv_rate')
        encode_numeric_zscore(df, 'dst_host_same_src_port_rate')
        encode_numeric_zscore(df, 'dst_host_srv_diff_host_rate')
        encode_numeric_zscore(df, 'dst_host_serror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_serror_rate')
        encode_numeric_zscore(df, 'dst_host_rerror_rate')
        encode_numeric_zscore(df, 'dst_host_srv_rerror_rate')

        ##########################################################################



        # drop = df.drop(df[(df['outcome'] == 'neptune.')].index).copy()

        # complete = df[(df['outcome'] == 'neptune.')].copy()
        # print(len(drop),len(complete),len(df))
        # outcomes = encode_text_index(complete, 'outcome',"normal.")
        # print("---------------")
        # print(outcomes)

        # complete.dropna(inplace=True,axis=1)

        # outcomes2 = encode_text_index(drop, 'outcome',"normal.")
        # drop.dropna(inplace=True,axis=1)
        # print("---------------")
        # print(outcomes2)


        # noNormal = df.drop(df[(df['outcome'] == 'normal.')].index).copy()
        # noOther = df.drop(df[(df['outcome'] != 'normal.')].index).copy()
        # smurf = df.drop(df[(df['outcome'] != 'smurf.')].index).copy()
        # neptune = df.drop(df[(df['outcome'] != 'neptune.')].index).copy()
        # print('len(neptune), len(smurf), len(noNormal), len(noOther), len(df)', 
        #         len(neptune), len(smurf), len(noNormal), len(noOther), len(df))

        self.labels = encode_text_index(df, 'outcome',"normal.")
        df.dropna(inplace=True,axis=1)
        print('------------------------------------------------------------------------------------------')
        print(self.labels)

        ##########################################################################
        # Break into X (predictors) & y (prediction)
        self.df = df
        self.y = df['outcome']
        #self.y = np.asarray(df['AHD']) 
        tmp = df
        tmp.drop('outcome', 1, inplace=True)

        self.x = tmp

        

        #print(df.corr(method = 'pearson'))

        ##########################################################################

        train = self.x
        train_labels = self.y

        fs = FeatureSelector(data = train, labels = train_labels)
        fs.identify_all(selection_params = {'missing_threshold': 0.6,    
                                    'correlation_threshold': 0.98, 
                                    'task': 'classification',    
                                    'eval_metric': 'auc', 
                                    'cumulative_importance': 0.99})

        # print('fs.feature_importances, record_low_importance', fs.feature_importances, fs.record_low_importance)
        # print('The features required for comulative importance of 0.99 are: ' )
        #len(self.feature_importances) - len(self.record_low_importance)


        fs.feature_importances.head(20)
        fs.plot_collinear(plot_all = True)
        fs.plot_collinear()
        fs.plot_feature_importances(threshold = 0.99, plot_n = 35)
        fs.plot_unique()
        plt.clf()

        # print('record_collinear: the pairs of collinear variables with a correlation coefficient above the threshold')
        # print(fs.record_collinear)
        # print('record_zero_importance: the zero importance features in the data according to the gbm')
        # print(fs.record_zero_importance)
        # print('record_low_importance: the lowest importance features not needed to reach the threshold of cumulative importance according to the gbm')
        # print(fs.record_low_importance)

        train_removed_all = fs.remove(methods = 'all', keep_one_hot = True)
        #train_removed_all_once = fs.remove(methods = 'all', keep_one_hot = True)


        print('Original Number of Features', train.shape[1])
        print('Final Number of Features: ', train_removed_all.shape[1])



        ##########################################################################

        self.x = train_removed_all
        #print(self.x.corr(method = 'pearson'))
        print('')
        print('')
        print("Features final set: ", self.x.columns.values.tolist())
        print('')
        print('------------------------------------------------------------------------------------------')

        self.x.insert(loc=0, column='ForBias', value = 1)
        # self.x = np.c_[np.ones((self.x.shape[0], 1)), self.x]
        # self.y = np.c_[np.ones((self.y.shape[0], 1)), self.y]
        #self.x = self.x.values



    def getNormalizeDF(self):
        return self.df


    def getNormalizeXY(self):
        return self.x, self.y



############################################################################################################



