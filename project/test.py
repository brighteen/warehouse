import requests

url = "http://202.31.230.150:5000/api/query"
data = {"query": "테스트 문장"}
res = requests.post(url, json=data)
print(res.json())
