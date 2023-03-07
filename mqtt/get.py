import sys
import os
import time
import paho.mqtt.client as mqtt

sys.path.append(os.path.abspath(os.path.dirname(__file__) + '/' + '..'))
sys.path.append("..")

REPORT_TOPIC = ['RoadWaste', "heart"]  # 主题
IP_ADDRESS = '10.10.13.11'
PORT = 1883


def on_connect(client, userdata, flags, rc):
    print('connected to mqtt with result code', rc)
    for report_topic in REPORT_TOPIC:
        client.subscribe(report_topic)  # 订阅主题


def on_message(client, userdata, msg):
    message = msg.payload.decode()
    print(message)


def server_connect(client):
    client.username_pw_set("yuhe", '123')
    client.on_connect = on_connect  # 启用订阅模式
    client.on_message = on_message  # 接收消息
    client.connect(IP_ADDRESS, PORT)  # 链接10
    client.loop_forever()  # 以forever方式阻塞运行。


def server_stop(client):
    client.loop_stop()  # 停止服务端
    sys.exit(0)


def server_main():
    client_id = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
    client = mqtt.Client(client_id, transport='tcp')
    server_connect(client)


if __name__ == '__main__':
    server_main()
