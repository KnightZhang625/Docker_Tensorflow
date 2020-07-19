# coding:utf-8
import json
import base64
import requests
import numpy as np

args = [{'input': [1.0, 2.0, 5.0, 3.2, 5.2, 6.1, 8.0, 1.1, 1.2, 3.2]}]
data = json.dumps({
    "instances": args})
headers = {"content-type": "application/json"}
json_response = requests.post(
    'http://localhost:8500/v1/models/toyModel:predict',
    data=data, headers=headers)
  
print(json_response.text)
print(json.loads(json_response.text)['predictions'])

# ab -n 10000 -c 10 -T "Content-Type:application/json" -p ./post.txt http://localhost:8501/v1/models/toyModel:predict