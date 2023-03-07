"""数据的加密 + 解密"""

import rsa
import json
import base64

"""
# 生成密钥
(pubkey, privkey) = rsa.newkeys(1024)
# 保存密钥
with open('public.pem', 'w+') as f:
    f.write(pubkey.save_pkcs1().decode())
with open('private.pem', 'w+') as f:
    f.write(privkey.save_pkcs1().decode())
"""


def encrypt_data(message):
    data = {'message': message}
    print("明文: %s" % data)
    # 用A的私钥加签
    str_data = json.dumps(data)
    with open('private.pem', 'r') as f:
        privkey = rsa.PrivateKey.load_pkcs1(f.read().encode())
        signature = rsa.sign(str_data.encode('utf-8'), privkey, 'SHA-1')
        sign = base64.b64encode(signature)
        data['sign'] = sign.decode()
    str_data = json.dumps(data)

    # 用B的公钥加密
    finally_data = {}
    PAGE_SIZE = 110  # 每页大小 因为加密最大长度为117 需要进行切片加密
    data_lengh = len(str_data) - 1
    pages = data_lengh // PAGE_SIZE if (data_lengh % PAGE_SIZE) == 0 else data_lengh // PAGE_SIZE + 1
    with open('public.pem', 'r') as f:
        pubkey = rsa.PublicKey.load_pkcs1(f.read().encode())
    for page in range(pages):
        if page < pages - 1:
            value = str_data[PAGE_SIZE * page:PAGE_SIZE * (page + 1)]
        else:
            value = str_data[PAGE_SIZE * page:]
        encrypt_data = rsa.encrypt(value.encode('utf-8'), pubkey)
        finally_data[page] = base64.b64encode(encrypt_data).decode()

    print("加签后的密文:", finally_data)
    return finally_data


def decry_data(encrypt_message):
    # B的私钥解密
    with open('private.pem', 'r') as f:
        privkey = rsa.PrivateKey.load_pkcs1(f.read().encode())
    final_data = ''
    for index in encrypt_message:
        temp_data = base64.b64decode(encrypt_message[index])
        data = rsa.decrypt(temp_data, privkey).decode()
        final_data = final_data + data
    print("解密后的明文:%s" % final_data)
    # A的公钥验签
    with open('public.pem', 'r') as f:
        pubkey = rsa.PublicKey.load_pkcs1(f.read().encode())
    message = {}
    final_data = json.loads(final_data)
    message['message'] = final_data['message']
    sign = base64.b64decode(final_data['sign'])
    message_str = json.dumps(message)
    result = rsa.verify(message_str.encode("utf-8"), sign, pubkey)
    print("明文:%s,验签结果:%s" % (message_str, result))
    return


if __name__ == '__main__':
    encrpt_data_ = encrypt_data('helloworld')
    decry_data(encrpt_data_)
