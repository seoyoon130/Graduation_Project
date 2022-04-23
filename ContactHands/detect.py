# -*- coding: utf-8 -*- 
import random
import cv2
import os
import argparse
import numpy as np 
import torch
import tqdm
from detectron2.config import get_cfg
from contact_hands_two_stream import CustomVisualizer
from detectron2.data import MetadataCatalog
from contact_hands_two_stream import add_contacthands_config
from datasets import load_voc_hand_instances, register_pascal_voc
from contact_hands_two_stream.engine import CustomPredictor
from detectron2.modeling import build_model
from detectron2.data import MetadataCatalog
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
import socket
from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
class CustomPredictorTwoStream:

    def __init__(self, cfg):
        self.cfg = cfg.clone()  
        self.model = build_model(self.cfg)
        self.model.eval()
        self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image, model2):

        with torch.no_grad():  
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = torch.as_tensor(original_image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            
            second_stream_outputs = inference_second_stream(model2, original_image)
            predictions = self.model([inputs], second_stream_outputs)[0]
            return predictions

def inference_second_stream(model, image):
    outputs = model(image)   
    return outputs 
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)
def prepare_second_stream():
    cfg2 = get_cfg()
    cfg2.merge_from_file('./configs/second_stream.yaml')
    cfg2.MODEL.WEIGHTS = "detectron2://COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x/138205316/model_final_a3ec72.pkl"
    model2 = CustomPredictor(cfg2)
    return model2
def list_chunk(lst, n):
    return [lst[i:i+n] for i in range(0, len(lst), n)]
def prepare_first_stream(cfg_file, weights, roi_score_thresh):
    cfg1 = get_cfg()
    add_contacthands_config(cfg1)
    cfg1.merge_from_file(cfg_file)
    cfg1.MODEL.ROI_HEADS.SCORE_THRESH_TEST = roi_score_thresh
    cfg1.MODEL.WEIGHTS = weights
    model1 = CustomPredictorTwoStream(cfg1)
    
    return model1
def start():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    width = 640
    height = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    score = []
    score2 = []
    cluster = []
    cluster2 = []
    boxes = []
    boxes2 = []
    usedsize = 0

    while True:

        ret, img_color = cap.read(0)

        if ret == False:
            break;
        img_input = img_color.copy()
        ratio = height / width
        im = cv2.resize(img_input, (720, int(720 * ratio)))
        outputs = model1(im, model2)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        fps = 30
        out = cv2.VideoWriter('video.avi', fourcc, fps, (int(width), int(height)))
        time.sleep(0.2)
        c = 'pred_cats'
        d = 'pred_boxes'
        e = 'scores'
        for key, value in outputs.items():
            score.clear()
            boxes.clear()
            cluster.clear()
            howmuch = str(value)[str(value).find('num_instances=') + 14:str(value).find('num_instances=') + 15]
            tmp = str(value)[str(value).find(c):len(str(value))]
            if len(str(value)[str(value).find(c):len(str(value))]) > 60:
                score.append(tmp[tmp.find('[') + 2:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                score.append(tmp[1:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                score.append(tmp[1:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                score.append(tmp[1:tmp.find(']')])
                for i in range(int(howmuch) - 1):
                    tmp = tmp[tmp.find('['):len(tmp)]
                    score.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(']')])
            tmp = str(value)[str(value).find(d):str(value).find(e)]
            if len(str(value)[str(value).find(d):str(value).find(e)]) > 70:
                boxes.append(tmp[tmp.find('[') + 2:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                boxes.append(tmp[1:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                boxes.append(tmp[1:tmp.find(',')])
                tmp = tmp[tmp.find(',') + 1:len(tmp)]
                boxes.append(tmp[1:tmp.find(']')])
                for i in range(int(howmuch) - 1):
                    tmp = tmp[tmp.find('['):len(tmp)]
                    boxes.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(']')])

        size = int(len(score) / 4)
        for j in range(size):
            q = 0
            for i in score[j * 4:j * 4 + 4]:
                if (i == max(score[j * 4:j * 4 + 4])):
                    cluster.append(q)
                    break
                q += 1
        boxes = list_chunk(boxes, 4)

        clustertmp = cluster
        check = 0

        position = 0
        if len(boxes2) > 0:
            for i in range(len(boxes)):
                for j in range(len(boxes2)):
                    if abs(float(boxes[i][0]) - float(boxes2[j][0])) > 50:
                        print(abs(float(boxes[i][0]) - float(boxes2[j][0])))
                        continue
                    if abs(float(boxes[i][1]) - float(boxes2[j][1])) > 50:
                        print(abs(float(boxes[i][1]) - float(boxes2[j][1])))
                        continue
                    if abs(float(boxes[i][2]) - float(boxes2[j][2])) > 50:
                        print(abs(float(boxes[i][2]) - float(boxes2[j][2])))
                        continue
                    if abs(float(boxes[i][3]) - float(boxes2[j][3])) > 50:
                        print(abs(float(boxes[i][3]) - float(boxes2[j][3])))
                        continue
                    check += 1

        if len(boxes2) > 0:
            if check != len(boxes2):
                print("THEIF")
                client_sock.send(data)
                data2 = 10
                client_sock.send(data2.to_bytes(4, byteorder='little'))

        boxes2.clear()
        cluster2.clear()
        for i in clustertmp:
            if i == 3:
                cluster2.append(position)
            position += 1

        if (len(cluster2) > 0):
            for i in cluster2:
                boxes2.append(boxes[i])

        v = CustomVisualizer(im[:, :, ::-1], MetadataCatalog.get("ContactHands_test"), scale=1,
                             scores_thresh=contact_thresh)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        out_im = add_legend(v.get_image()[:, :, ::-1])
        cv2.imshow("r", out_im)
        out.write(out_im)
        ret, buffer = cv2.imencode('.jpg', out_im)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        if cv2.waitKey(1) & 0xFF == ord('q'):
            client_sock.close()
            server_sock.close()
            break

    cap.release()
    cv2.destroyAllWindows()


def setup_cfg(args):
    # load config from file and command-line arguments
    # Set score_threshold for builtin models

    return args
def add_legend(im):
    cyan, magenta, red, yellow = (255, 255, 0), (255, 0, 255), (0, 0, 255),  (0, 255, 255)
    labels = ["No", "Self", "Person", "Object"]
    map_idx_to_color = {}
    map_idx_to_color[0], map_idx_to_color[1], map_idx_to_color[2],  map_idx_to_color[3] = \
    cyan, magenta, red, yellow

    font = cv2.FONT_HERSHEY_SIMPLEX
    h, w = im.shape[:2]
    image = 255*np.ones((h+50, w, 3), dtype=np.uint8)
    image[:h, :w, :] = im 
    h, w = image.shape[:2]
    offset = 0

    for itr, word in enumerate(labels):
        offset += int(w / len(labels)) - 50
        cv2.putText(image, word, (offset, h-15), font, 1, map_idx_to_color[itr], 3)

    return image

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Arguments for evaluation')
    parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
    parser.add_argument('--image_dir', required=False, metavar='path to images', help='path to images')
    parser.add_argument('--ROI_SCORE_THRESH', required=False, metavar='threshold for hand detections', \
    	help='hand detection score threshold', default=0.7)
    parser.add_argument('--sc', required=False, metavar='threshold for self-contact', 
        help='threshold for self-contact', default=0.5)
    parser.add_argument('--pc', required=False, metavar='threshold for person-contact', 
        help='threshold for self-contact', default=0.3)
    parser.add_argument('--oc', required=False, metavar='threshold for object-contact', 
        help='threshold for self-contact', default=0.6)

    args = parser.parse_args()
    roi_score_thresh = float(args.ROI_SCORE_THRESH)
    sc_thresh = float(args.sc)
    pc_thresh = float(args.pc)
    oc_thresh = float(args.oc)
    contact_thresh = [0.5, sc_thresh, pc_thresh, oc_thresh]


    # if the scores for all contact states is less than corresponding thresholds, No-Contact is predicted; 0.5 is dummy here, it is not used.

    model2 = prepare_second_stream()
    model1 = prepare_first_stream('./configs/ContactHands.yaml', './models/combined_data_model.pth', roi_score_thresh)

    count = 0
    if args.webcam:

        app = Flask(__name__)



        host=socket.gethostbyname(socket.gethostname())
        print(host)

          # 호스트 ip를 적어주세요
        port = 8080  # 포트번호를 임의로 설정해주세요

        server_sock = socket.socket(socket.AF_INET)
        server_sock.bind((host, port))
        server_sock.listen(1)

        print("기다리는 중")
        client_sock, addr = server_sock.accept()

        print('Connected by', addr)
        data = client_sock.recv(1024)
        print(data.decode("utf-8"), len(data))




        @app.route('/')
        @app.route('/index')
        def index():
            return render_template('index.html')


        print("hhhhh")





        @app.route('/video_feed')
        def video_feed():
            print("inini")
            return Response(start(), mimetype='multipart/x-mixed-replace; boundary=frame')



        @app.route("/favicon.ico")
        def favicon():
            return "", 200


        app.run('192.168.0.9')
        """
        while(True):
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            width = 640
            height = 480
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            score = []
            score2 = []
            cluster = []
            cluster2 = []
            boxes = []
            boxes2 = []
            usedsize = 0

            ret, img_color=cap.read(0)

            if ret==False:
                break;
            img_input=img_color.copy()
            ratio = height / width
            im=cv2.resize(img_input,(720,int(720*ratio)))
            outputs = model1(im, model2)
            c = 'pred_cats'
            d = 'pred_boxes'
            e = 'scores'
            for key, value in outputs.items():
                score.clear()
                boxes.clear()
                cluster.clear()
                howmuch = str(value)[str(value).find('num_instances=') + 14:str(value).find('num_instances=') + 15]
                tmp = str(value)[str(value).find(c):len(str(value))]
                if len(str(value)[str(value).find(c):len(str(value))]) > 60:
                    score.append(tmp[tmp.find('[') + 2:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    score.append(tmp[1:tmp.find(']')])
                    for i in range(int(howmuch) - 1):
                        tmp = tmp[tmp.find('['):len(tmp)]
                        score.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        score.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        score.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        score.append(tmp[1:tmp.find(']')])
                tmp = str(value)[str(value).find(d):str(value).find(e)]
                if len(str(value)[str(value).find(d):str(value).find(e)]) > 70:
                    boxes.append(tmp[tmp.find('[') + 2:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(']')])
                    for i in range(int(howmuch) - 1):
                        tmp = tmp[tmp.find('['):len(tmp)]
                        boxes.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(']')])

            size = int(len(score) / 4)
            for j in range(size):
                q = 0
                for i in score[j * 4:j * 4 + 4]:
                    if (i == max(score[j * 4:j * 4 + 4])):
                        cluster.append(q)
                        break
                    q += 1
            boxes = list_chunk(boxes, 4)

            clustertmp = cluster
            check = 0

            position = 0
            if len(boxes2) > 0:
                for i in range(len(boxes)):
                    for j in range(len(boxes2)):
                        if abs(float(boxes[i][0]) - float(boxes2[j][0])) > 50:
                            print(abs(float(boxes[i][0]) - float(boxes2[j][0])))
                            continue
                        if abs(float(boxes[i][1]) - float(boxes2[j][1])) > 50:
                            print(abs(float(boxes[i][1]) - float(boxes2[j][1])))
                            continue
                        if abs(float(boxes[i][2]) - float(boxes2[j][2])) > 50:
                            print(abs(float(boxes[i][2]) - float(boxes2[j][2])))
                            continue
                        if abs(float(boxes[i][3]) - float(boxes2[j][3])) > 50:
                            print(abs(float(boxes[i][3]) - float(boxes2[j][3])))
                            continue
                        check += 1

            if len(boxes2)>0:
                if check != len(boxes2):
                    print("THEIF")
                    client_sock.send(data)
                    data2=10
                    client_sock.send(data2.to_bytes(4, byteorder='little'))






            boxes2.clear()
            cluster2.clear()
            for i in clustertmp:
                if i == 3:
                    cluster2.append(position)
                position += 1

            if (len(cluster2) > 0):
                for i in cluster2:
                    boxes2.append(boxes[i])



            v = CustomVisualizer(im[:, :, ::-1], MetadataCatalog.get("ContactHands_test"), scale=1,
                                 scores_thresh=contact_thresh)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_im = add_legend(v.get_image()[:, :, ::-1])
            cv2.imshow("r",out_im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                client_sock.close()
                server_sock.close()
                break









        cap.release()
        cv2.destroyAllWindows()
"""

    if args.image_dir:
        images_path = args.image_dir
        images = sorted(os.listdir(images_path))

        score = []
        score2=[]
        cluster=[]
        cluster2=[]
        boxes=[]
        boxes2=[]
        usedsize=0
        for img in images:
            count += 1
            print(count)
            im = cv2.imread(os.path.join(images_path, img))
            height, width = im.shape[0], im.shape[1]
            ratio = height / width
            im = cv2.resize(im, (720, int(720*ratio)))
            outputs = model1(im, model2)
            coun = 0;
            c='pred_cats'
            d='pred_boxes'
            e='scores'
            for key, value in outputs.items():
                score.clear()
                boxes.clear()
                cluster.clear()
                howmuch=str(value)[str(value).find('num_instances=')+14:str(value).find('num_instances=')+15]
                tmp = str(value)[str(value).find(c):len(str(value))]
                if len(str(value)[str(value).find(c):len(str(value))]) > 60:
                    score.append(tmp[tmp.find('[')+2:tmp.find(',')])
                    tmp=tmp[tmp.find(',')+1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp=tmp[tmp.find(',')+1:len(tmp)]
                    score.append(tmp[1:tmp.find(',')])
                    tmp=tmp[tmp.find(',')+1:len(tmp)]
                    score.append(tmp[1:tmp.find(']')])
                    for i in range(int(howmuch)-1):
                        tmp=tmp[tmp.find('['):len(tmp)]
                        score.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                        tmp=tmp[tmp.find(',')+1:len(tmp)]
                        score.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        score.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        score.append(tmp[1:tmp.find(']')])
                tmp=str(value)[str(value).find(d):str(value).find(e)]
                if len(str(value)[str(value).find(d):str(value).find(e)])>70:
                    boxes.append(tmp[tmp.find('[') + 2:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(',')])
                    tmp = tmp[tmp.find(',') + 1:len(tmp)]
                    boxes.append(tmp[1:tmp.find(']')])
                    for i in range(int(howmuch) - 1):
                        tmp = tmp[tmp.find('['):len(tmp)]
                        boxes.append(tmp[tmp.find('[') + 1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(',')])
                        tmp = tmp[tmp.find(',') + 1:len(tmp)]
                        boxes.append(tmp[1:tmp.find(']')])

            size=int(len(score)/4)
            for j in range(size):
                q=0
                for i in score[j*4:j*4+4]:
                    if(i==max(score[j*4:j*4+4])):
                        cluster.append(q)
                        print((score[j*4:j*4+4]))
                        print(i)
                        break
                    q+=1
            print(cluster)
            boxes=list_chunk(boxes,4)
            print(boxes)
            clustertmp=cluster
            check=0

            position=0
            if len(boxes2)>1:
                for i in range(len(boxes)):
                    for j in range (len(boxes2)):
                        if abs(boxes[i][0]-boxes2[j][0])>20:
                            break
                        if abs(boxes[i][1] - boxes2[j][1]) > 20:
                            break
                        if abs(boxes[i][2] - boxes2[j][2]) > 20:
                            break
                        if abs(boxes[i][3] - boxes2[j][3]) > 20:
                            break
                        check+=1
            if check!=len(boxes2):
                print("THEIF")
            for i in clustertmp:
                if i==3:
                    cluster2.append(position)
                position += 1
            print(cluster2)
            if(len(cluster2)>0):
                for i in cluster2:
                    boxes2.append(boxes[i])

            print(cluster2)
            print(boxes2)
            boxes.clear()
            cluster2.clear()
            v = CustomVisualizer(im[:, :, ::-1], MetadataCatalog.get("ContactHands_test"), scale=1, scores_thresh=contact_thresh)
            v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            out_im = add_legend(v.get_image()[:, :, ::-1])
            cv2.imwrite('./results/res_' + img, out_im)

