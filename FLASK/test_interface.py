import requests
import base64
import os

root = r'C:\Users\yuhe\Desktop\model_weights'
res = open(os.path.join(root, '8.jpg'), 'rb').read()
s = base64.b64encode(res)
img1 = s.decode('ascii')

res = open(os.path.join(root, '9.jpg'), 'rb').read()
s = base64.b64encode(res)
img2 = s.decode('ascii')

tenant_id = "29b804d2ed9e45b88099e5bbd2076aa6"
headers = {'tenantId': tenant_id}
info = {"startimg": img1, 'endimg': img2}
# response = requests.post('http://10.10.13.11:8080/sdyd-box-ai/qualityDetector', headers=headers, json=info, timeout=5)
info = {"image": img2}
response = requests.post('http://10.10.13.11:8080/sdyd-box-ai/wasteDetector', headers=headers, json=info, timeout=5)
data_output = response.json()  # 数据获取成功
print(data_output)
