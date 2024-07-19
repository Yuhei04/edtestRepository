import os
import cv2
import torch
import numpy as np
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage

app = Flask(__name__)

# LINE Botのアクセストークンとチャネルシークレットを環境変数から取得
LINE_CHANNEL_ACCESS_TOKEN = os.getenv('LINE_CHANNEL_ACCESS_TOKEN')
LINE_CHANNEL_SECRET = os.getenv('LINE_CHANNEL_SECRET')

line_bot_api = LineBotApi(LINE_CHANNEL_ACCESS_TOKEN)
handler = WebhookHandler(LINE_CHANNEL_SECRET)

# モデルのロード
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
CLASSES = model.names

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    if event.message.text.lower() == '人数を教えて':
        count = detect_people()
        reply = f'現在の人数は {count} 人です。'
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text=reply))

def detect_people():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if not ret:
        return 0

    results = model(frame)
    detections = results.pred[0]
    people_count = sum(1 for *box, conf, cls in detections if CLASSES[int(cls)] == 'person')

    cap.release()
    return people_count

if __name__ == "__main__":
    app.run()
