import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'Transacciones_ATM':210000, 'Ticket_promedio_ATM':850})

print(r.json())