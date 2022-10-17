import paho.mqtt.client as mqtt
import time
import json

IP_ADDRESS = "10.10.13.11"
PORT = 1883


def post_message(topic, info):
    client = mqtt.Client()
    client.connect(IP_ADDRESS, PORT)
    client.publish(topic, info)
    return


if __name__ == '__main__':
    post_message("RoadWaste", 'ok')
