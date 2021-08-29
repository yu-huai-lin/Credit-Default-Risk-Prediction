# Credit-Default-Risk-Prediction

### General Info
This repository trains an algorithem to predict the probability of default inside the environment of AWS sagemaker's notebook instances and invoke a model endpoint using AWS API Gateway and AWS Lambda.

### Description
This goal of this project is to predict the probability that a customer will default, which result in a binary classification problem. The dataset is heavily class imbalanced, with most records being labeled as `default = 0`. After training a logistic regression model, I evaluate the model against a hold-out test dataset, with cross validation stratified K fold approach, and save the evaluation metrics, including F1 score, accuracy, ROC AUC, confusion matrix for the model. To fine tune the model, I did Hyperparameter tuning with grid search method.
I used python and AWS for deployment. In short, I trained the model and did data preprocessing using AWS Sagemaker's notebook, and invoke a model endpoint deployed by SageMaker using AWS API Gateway and AWS Lambda.
In the attached csv file final_prediction_result.csv contained the prediction results

### Requirements

python version 3.6

#### Libraries
Package             Version
## ------------------- ---------
boto3               1.18.27
botocore            1.21.27
matplotlib          3.4.3
matplotlib-inline   0.1.2
numpy               1.21.2
packaging           21.0
pandas              1.3.2
plotly              5.2.1
s3transfer          0.5.0
scikit-learn        0.24.2
scipy               1.7.1
seaborn             0.11.2
sklearn             0.0
statsmodels         0.12.2

### Instructions

Instrusctions on how to query the endpoint is in the pdf summary file attached.

You can upload the script and the data in to AWS sagemaker notebook instance, and run the AWS_train.ipynb script, which generates a SageMaker SkLearn Estimator and run the prediction-Copy1.py script and train and model. To deploy the model, you need to create a function on AWS lambda which runs the function described in the lambda_function.py file, in this file, we update our lambda_handler to download our model and run a prediction. Finally, you can create a simple API using Amazon API Gateway, when calling the API through an HTTPS endpoint, Amazon API Gateway invokes the Lambda function.

`EDA.ipynb`
this file contains the exploratory data analysis

`prediction.ipynb`
this file is the notebook version of the prediction-Copy1.py file, it also contains my process of data transformation, model evaludation, and hyperparameter tuning

`prediction-Copy1.py`
this file is the actual file to run train the model and run the prediction

`lambda_function.py`
This script contains the lambda function which download the model and run prediction

`final_prediction_result.csv`
This file is the output of the prediction results for missing variables (default= NaN)
