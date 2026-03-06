
import requests
import json
import time

url = "http://localhost:8000/chat"
headers = {"Content-Type": "application/json"}
data = {"query": "Hello?"}

print(f"Sending request to {url} with query: {data['query']}")
start_time = time.time()
try:
    response = requests.post(url, headers=headers, json=data, timeout=120)
    end_time = time.time()
    print(f"Response received in {end_time - start_time:.2f} seconds")
    print(f"Status Code: {response.status_code}")
    try:
        json_resp = response.json()
        print("Response JSON:")
        print(json.dumps(json_resp, indent=2))
        
        if not json_resp.get("answer"):
             print("\nWARNING: 'answer' field is empty or missing!")
        else:
             print("\nAnswer received successfully.")

    except Exception as e:
        print(f"Error parsing JSON: {e}")
        print(f"Raw Response: {response.text}")
except Exception as e:
    print(f"Request failed: {e}")
