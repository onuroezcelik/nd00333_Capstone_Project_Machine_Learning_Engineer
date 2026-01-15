import requests
import json

# URL for the web service, should be similar to:
# 'http://8530a665-66f3-49c8-a953-b82a2d312917.eastus.azurecontainer.io/score'
scoring_uri = 'http://91ede4c6-ad25-480d-b9ef-35d3f85b5b04.southcentralus.azurecontainer.io/score'
# If the service is authenticated, set the key or token
key = 'vFEYUzKT8W6hEveaDzT7YtPQeM3h3Fyp'

# Two sets of data to score, so we get two results back
data = {
    "data":
    [
      {
        "age": 51, 
        "anaemia": 0, 
        "creatinine_phosphokinase": 600, 
        "diabetes": 1, 
        "ejection_fraction": 20, 
        "high_blood_pressure": 1, 
        "platelets": 265000, 
        "serum_creatinine": 1.9, 
        "serum_sodium": 130, 
        "sex": 0, 
        "smoking": 1,
        "time": 4
      },
      {
        "age": 25, 
        "anaemia": 0, 
        "creatinine_phosphokinase": 1380, 
        "diabetes": 1, 
        "ejection_fraction": 25, 
        "high_blood_pressure": 1, 
        "platelets": 271000, 
        "serum_creatinine": 0.9, 
        "serum_sodium": 100, 
        "sex": 1, 
        "smoking": 0,
        "time": 5
      },
    ]
}
# Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
    _f.write(input_data)

# Set the content type
headers = {'Content-Type': 'application/json'}
# If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

# Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
