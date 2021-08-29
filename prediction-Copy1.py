#!/usr/bin/env python
# coding: utf-8

# In[1]:

from __future__ import print_function
import argparse
import os
import warnings
from sklearn import tree
#import boto3
#import sagemaker
#from sagemaker import get_execution_role
#from sagemaker.sklearn.processing import SKLearnProcessor

#region = boto3.session.Session().region_name

#role = get_execution_role()
#sklearn_processor = SKLearnProcessor(
#    framework_version="0.20.0", role=role, instance_type="ml.m5.xlarge", instance_count=1
#)


# # How to use this script
# 
# This script is used in AWS Sagemaker to build and deploy machine learning models and has 2 main functions:
# 1. A main function to prepcrocess, train and save the model 
# 2. Load and return the saved model

# In[2]:


from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
import pandas as pd


# In[3]:


#%%writefile preprocessing.py


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler
from sklearn.compose import make_column_transformer

from sklearn.exceptions import DataConversionWarning

warnings.filterwarnings(action="ignore", category=DataConversionWarning)


# In[24]:


# bucket='sagemaker-us-east-2-681954894216'

# path = 'data/dataset.csv'

# s3uri = 's3://{}/{}'.format(bucket, path)

# df = pd.read_csv(s3uri, sep=';')


# In[25]:


#df.sample(5)


# In[26]:



import argparse
import os
import pandas as pd

from sklearn import tree
from sklearn.externals import joblib
#import joblib

#label = ['default']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Sagemaker specific arguments. Defaults are set in the environment variables.

    #Saves Checkpoints and graphs
    #parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))


    #Save model artifacts
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    #parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])

    #Train data
    #parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAIN'))
    #parser.add_argument('--train', type=str, default=os.environ)

    args, _ = parser.parse_known_args()

    print("parse the arguments")
    
    file = os.path.join(args.train, 'dataset.csv')
    df = pd.read_csv(file, sep=';')
    
    print("read the files")
    cat_cols = ['account_status', 
            'account_worst_status_0_3m', 
            'account_worst_status_12_24m', 
               'account_worst_status_3_6m', 
            'account_worst_status_6_12m', 
            'worst_status_active_inv',
            'status_last_archived_0_24m', 
            'status_2nd_last_archived_0_24m', 
            'status_3rd_last_archived_0_24m',
            'status_max_archived_0_6_months', 
            'status_max_archived_0_12_months', 
            'status_max_archived_0_24_months',
            'has_paid',
            'name_in_email', 
            'merchant_group',
            'merchant_category',
            'uuid']
    # for c in cat_cols:
    #     df[c]= df[c].values.astype(str)
    cat_features = cat_cols[:-1]
    num_features= [col for col in df.columns.tolist() if col not in (cat_cols)]
    label = ['default']
    num_features = num_features[1:]
    
        # Removing highly correlated features
    highly_corelated_features = ['max_paid_inv_0_12m',
                                 'avg_payment_span_0_3m',
                                 'sum_capital_paid_account_12_24m', 
                                 'num_arch_ok_0_12m']

    df = df.drop(highly_corelated_features, axis=1)

    num_features=[col for col in num_features if col not in highly_corelated_features]

    ## Missing Data Imputation 1 
    ### for some columns, the median of default = 1 is higher than default = 0, hence, I impute the missing data with median grouping by the category
    feature_higher_1 = ['avg_payment_span_0_12m',
                        'account_days_in_rem_12_24m',
                        'num_active_div_by_paid_inv_0_12m' ]

    na_cols = ['account_days_in_dc_12_24m',
     'account_days_in_term_12_24m',
     'account_incoming_debt_vs_paid_0_24m',
     'num_arch_written_off_0_12m',
     'num_arch_written_off_12_24m']

    for c in feature_higher_1:
        df[c] = df[df['default'] != 'nan'][c].transform(lambda x: x.fillna(x.median()))

    for c in na_cols:
        df[c] = df[df['default'] != 'nan'][c].transform(lambda x: x.fillna(x.median()))

    # Label encoding for ordinal categorical features
    #label encoding better for ordinal data with high cardinality
    #one hot encoding better for low cardinality and not ordinal data

    from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler,MinMaxScaler

    labelencoder = LabelEncoder()
    for feat in cat_features[0:11]:
        df[feat] = labelencoder.fit_transform(df[feat].astype(str))


    #Binary encoding for Boolean columns
    dict1 = {True: 0, False:1}
    var = 'has_paid'
    df[var+'_ordinal'] = df['has_paid'].map(dict1)
    df = df.drop(var, axis=1)


    scaler = MinMaxScaler()
    df[num_features] = scaler.fit_transform(df[num_features])

    for c in cat_features[13:]:
        dummies = pd.get_dummies(df[c])
        df[dummies.columns] = dummies

    df = df.drop(cat_features[13:], axis=1)

    #find missing values

    df_train = df[df['default'].notnull()]
    df_test = df[df['default'].isnull()]

    y = df_train['default']
    X = df_train.drop(['uuid', 'default'],axis = 1)

    cat_features[-4]= 'has_paid_ordinal'

    print("finishing data preprocessing")
    
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 43)
    
    print("finishing data preprocessing")
    from sklearn.linear_model import LinearRegression
    regressor = LogisticRegression(max_iter=1000)
    regressor.fit(X_train, y_train)

    # Print the coefficients of the trained classifier, and save the coefficients
    joblib.dump(regressor, os.path.join(args.model_dir, "model.joblib"))


def model_fn(model_dir):
    """Deserialized and return fitted model
    
    Note that this should have the same name as the serialized model in the main method
    """
    regressor = joblib.load(os.path.join(model_dir, "model.joblib"))
    return regressor


# In[6]:



