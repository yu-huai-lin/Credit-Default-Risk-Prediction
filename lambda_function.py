import os
import io
import boto3
import json

# grab environment variables
ENDPOINT_NAME = "sagemaker-scikit-learn-2021-08-25-21-32-49-542"
runtime= boto3.client('runtime.sagemaker')

def lambda_handler(event, context):
    print("Received event: " + json.dumps(event, indent=2))
    
    data = json.loads(json.dumps(event))
    payload = data['data']
    print(payload)
    
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                      Body=json.dumps(payload))
    print(response)
    result = json.loads(response['Body'].read().decode())
    print(result)
    predicted_label = '0' if result[0] == 0 else '1'
    
    return result[0]



