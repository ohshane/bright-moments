import base64
import os
import random
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import face_recognition
import numpy as np
import requests
import torch
import torch.nn.functional as F
import zmq
from dotenv import load_dotenv
from PIL import Image

from misc import *
from model.FER2013_VGG19.VGG import VGG

load_dotenv(verbose=True)
TARGET_URL = 'https://notify-api.line.me/api/notify'
TOKEN = os.getenv('LINE_TOKEN')

root_dir = Path(__file__).parent

def tprint(msg):
    print(msg)
    sys.stdout.flush()

class ServerTask(threading.Thread):
    def __init__(self, num_workers=5):
        threading.Thread.__init__(self)
        self.num_workers = num_workers

    def run(self):
        context = zmq.Context()
        frontend = context.socket(zmq.ROUTER)
        frontend.bind('tcp://*:5570')

        backend = context.socket(zmq.DEALER)
        backend.bind('inproc://backend')

        workers = []
        for _ in range(self.num_workers):
            worker = ServerWorker(context)
            worker.start()
            workers.append(worker)


        zmq.proxy(frontend, backend)

        frontend.close()
        backend.close()
        context.term()

class ServerWorker(threading.Thread):
    def __init__(self, context):
        threading.Thread.__init__ (self)
        self.context = context
        self.id = uuid.uuid4()
        self.model = VGG('VGG19')
        checkpoint = torch.load(root_dir / 'model' / 'FER2013_VGG19' / 'PrivateTest_model.t7', map_location="cpu")
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

    def run(self):
        worker = self.context.socket(zmq.DEALER)
        worker.connect('inproc://backend')
        identity = f"Worker-{str(self.id).split('-')[-1]}"
        tprint(f"[{identity}] started")

        while True:
            ident, msg = worker.recv_multipart()
            msg = str(msg, encoding='ascii')
            ts, shape_str, msg = str(msg).split('__', 2)
            shape = tuple(map(int, shape_str.split('_')))

            msg = base64.decodebytes(bytes(msg, encoding='ascii'))
            msg = np.frombuffer(msg, dtype=np.uint8)
            frame = np.reshape(msg, shape)
            image = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            boxes = face_recognition.face_locations(image)

            for box in boxes:
                top, right, bottom, left = box
                frame_roi = frame[top:bottom, left:right]
                frame_roi = cv2.equalizeHist(frame_roi)
                frame_roi = cv2.resize(frame_roi, dsize=(48,48), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                
                cv2.imwrite('frame_roi.jpg', frame_roi)
                
                inputs = [[frame_roi, frame_roi, frame_roi]]
                ncrops, channel, height, width = np.shape(inputs)

                outputs = self.model(torch.Tensor(inputs)/255)
                outputs_avg = outputs.view(ncrops, -1).mean(0)
                score = F.softmax(outputs_avg)
                score = dict(zip(['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'], score))
                score = dict(sorted(score.items(), key=lambda x: x[1], reverse=True))

                worker.send_multipart([ident, bytes(ts + '__', encoding='ascii') + bytes(list(score.keys())[0], encoding='ascii')])
                tprint(f"[{identity}] ...pong")
                break
        worker.close()

class ClientTask(threading.Thread):
    def __init__(self, url=0):
        threading.Thread.__init__(self)
        self.id = uuid.uuid4()
        self.url = url

    def run(self):
        context = zmq.Context()
        socket = context.socket(zmq.DEALER)
        identity = f"Client-{str(self.id).split('-')[-1]}"
        socket.identity = identity.encode('ascii')
        socket.connect('tcp://localhost:5570')
        tprint(f"{socket.identity} started")
        poll = zmq.Poller()
        poll.register(socket, zmq.POLLIN)

        cv2.namedWindow(identity, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(identity, 800, 800)
        font = cv2.FONT_HERSHEY_SIMPLEX

        using = False
        cap = cv2.VideoCapture(self.url)
        history = History(size=60)
        fps = FPS()
        __start = datetime.now()
        frames = []
        prev_detected = False
        while True:
            ret, frame = cap.read()
            if not ret:
                raise ValueError

            key = cv2.waitKey(1)
            if key == 27:
                tprint(f"[{identity}] exit")
                break

            cv2.imshow(identity, frame)

            frame_reduced = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_reduced = cv2.resize(frame_reduced, dsize=(320, 240))

            shape_bytes = ''
            for i in frame_reduced.shape:
                shape_bytes += str(i) + '_'
            shape_bytes += '_'
            shape_bytes = bytes(shape_bytes, encoding='ascii')

            ts = str(time.perf_counter())
            history.enqueue([ts, frame, None])
            socket.send_string(ts + '__' + str(shape_bytes + base64.b64encode(frame_reduced), encoding='ascii'))
            tprint(f"[{identity}] ping...")

            sockets = dict(poll.poll(200))
            if socket in sockets:
                msg = socket.recv()
                msg = str(msg, encoding='ascii')
                ts, emotion = msg.split('__', 1)
                history[history.find(ts, place=lambda x: x[0])][2] = emotion

            current_ts = time.perf_counter()
            emotions = [emotion for ts, _, emotion in history if (current_ts > float(ts) > current_ts-1) and emotion is not None]
            tprint(emotions)
            if all(list(map(lambda x: x == 'Happy', emotions))):
                tprint("happy!")
                if prev_detected:
                    frames.append(frame)
                else:
                    frames 
                    prev_detected = True
                    frames = [frame for ts, frame, _ in history if (current_ts > float(ts) > current_ts-1)]

            else:
                if prev_detected:
                    # images = list(map(lambda x: Image.fromarray(cv2.cvtColor(x, cv2.COLOR_BGR2RGB)), frames))
                    # images[0].save('happy.gif', save_all=True, append_images=images[1:], optimizer=False, duration=100, loop=0)
                    # gif = open(root_dir/'happy.gif', 'rb')
                    cv2.imwrite('happy.jpg', frames[len(frames)//2])

                    res = requests.post(
                        TARGET_URL,
                        headers = {'Authorization': 'Bearer ' + TOKEN},
                        files = {
                            'imageFile': open(root_dir/'happy.jpg', 'rb')
                        },
                        data = {
                            'message': f"\n\nYour {random.choice(['cute', 'adorable', 'beautiful', 'sweet', 'lovely'])} {random.choice(['baby', 'angel'])} is smiling {random.choice('😀😃😄😊🙂😉🥰🥳')}\n\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n© Bright Moments",
                            'stickerPackageId': 11539,
                            'stickerId': random.choice([
                                52114112,
                                52114118,
                                52114119,
                                52114124,
                                52114131,
                                52114147]),
                        },
                    )
                prev_detected = False
                frames = []
            
            tprint(f"FPS\t{fps}\n")

        cap.release()
        cv2.destroyAllWindows()

        __end = datetime.now()
        tprint(f"start_ts  {__start}")
        tprint(f"end_ts    {__end}")
        tprint(f"uptime    {__end - __start}")

        socket.close()
        context.term()

def main():
    server = ServerTask(num_workers=5)
    server.start()

    time.sleep(1)

    client = ClientTask(url=0)
    client.start()

    server.join()


if __name__ == "__main__":
    main()
