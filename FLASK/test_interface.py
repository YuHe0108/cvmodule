import requests
import base64
import os

# root = r'C:\Users\yuhe\Desktop\model_weights'
# res = open(os.path.join(root, '8.jpg'), 'rb').read()
# s = base64.b64encode(res)
# img1 = s.decode('ascii')


# tenant_id = "29b804d2ed9e45b88099e5bbd2076aa6"
# headers = {'tenantId': tenant_id}
# info = {"startimg": img1, 'endimg': img2}
# info = {"image": img2}

inputs = {"download_url": 'http://222.92.212.123:8003/cloudFile/common/downloadFile',
          "file_name": '8e67d03eb132493db502d595ee390d0e',
          "save_name": '1'}
response = requests.post('http://10.10.10.190:6666/AI_Interface/download_file', json=inputs, timeout=5)
data_output = response.json()  # 数据获取成功
print(data_output)
