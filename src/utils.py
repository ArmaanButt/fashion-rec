import base64
import requests


def get_and_encode_image(image_url):
    response = requests.get(image_url)
    return base64.b64encode(response.content).decode("utf-8")
