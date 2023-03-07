import requests
import hashlib
import base64
import time
import os

root = r'C:\Users\yuhe\Desktop\model_weights'
res = open(os.path.join(root, '8.jpg'), 'rb').read()
s = base64.b64encode(res)
img1 = s.decode('ascii')

# print(type(img1))
# m = hashlib.sha3_512()
#
# t = time.time()
# m.update(img1.encode())
# resultHex = m.hexdigest()
# print(resultHex, len(resultHex), time.time() - t)

# tenant_id = "29b804d2ed9e45b88099e5bbd2076aa6"
# headers = {'tenantId': tenant_id}
# info = {"startimg": img1, 'endimg': img2}
info = {"image": 11}

# inputs = {"download_url": 'http://222.92.212.123:8003/cloudFile/common/downloadFile',
#           "file_name": '8e67d03eb132493db502d595ee390d0e',
#           "save_name": '1'}
t = time.time()
response = requests.post('http://10.10.13.11:6666/test', json=info, timeout=5)
print(response.headers)
print(response.apparent_encoding)
print(response.status_code)
print(time.time() - t)
# data_output = response.json()  # 数据获取成功
# print(data_output)
