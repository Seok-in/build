import time
import argparse
import dlib

import numpy
import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import light_remover as lr
import datetime
import torch
from scipy.spatial import distance as dist

from imutils import face_utils

from ErrorCode import CODE
from config import db
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, apply_classifier, set_logging
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


# # database를 사용하기 위한 cursor를 세팅합니다.
cursor = db.cursor()

app = Flask(__name__)
cors = CORS(app, resources={r"/flask/*": {"origins": "*"}})

# dlib이용 얼굴을 감지하기
detector = dlib.get_frontal_face_detector()
# dlib이용 얼굴을 각각의 좌표로 바꿔주기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

totalRoom = {}

net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
OPEN_EAR = 0
EAR_THRESH = 0
classes = []

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
opt = parser.parse_args()


weights, view_img, save_txt, imgsz, trace = opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace

# Initialize
device = select_device(opt.device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
model = attempt_load(weights, map_location=device)  # load FP32 model
stride = int(model.stride.max())  # model stride

imgsz = check_img_size(imgsz, s=stride)  # check img_size

if trace:
    model = TracedModel(model, device, opt.img_size)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def init_open_ear():
    ear_list = []
    for i in range(7):
        ear_list.append(both_ear)
        time.sleep(1)
    global OPEN_EAR
    OPEN_EAR = sum(ear_list) / len(ear_list)

def init_close_ear():
    ear_list = []
    for i in range(7):
        ear_list.append(both_ear)
    CLOSE_EAR = sum(ear_list) / len(ear_list)
    global EAR_THRESH
    EAR_THRESH = (((OPEN_EAR - CLOSE_EAR) / 2) + CLOSE_EAR)

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 업로드 HTML 렌더링
@app.route('/upload')
def render_file():
    return render_template('upload.html')

def penaltRecord(memberId, roomId) :
    # 패널티 기록
    sql = """INSERT INTO penalty_log(`created_at`,`member_id`, `room_id`, `status`) VALUES ('%s','%s', '%s', '%s');""" % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), memberId, roomId, "MODE2");
    cursor.execute(sql)
    db.commit();


    sql = """SELECT sprint_id FROM sprint where room_id = '%s' """ % (roomId)
    # sql = """select email from member where member_id = 1;"""
    cursor.execute(sql)
    row = cursor.fetchone()
    if(row == None) :
        return 200
    else :
        sprintId = row[0]
    print(sprintId)

    sql = """SELECT fee, penalty_money FROM sprint_info WHERE sprint_id = '%s' """ % (sprintId)
    cursor.execute(sql)
    row = cursor.fetchone()
    fee = row[0]
    penalty = row[1]

    sql = """SELECT SUM(point_value) FROM point_log WHERE member_id = '%s' AND sprint_id = '%s' group by member_id""" % (memberId, sprintId)
    cursor.execute(sql)
    row = cursor.fetchone()
    if(row == None):
        leftoverPoint = fee
    else :
        leftoverPoint = fee + row[0]

    print("leftover: ")
    print(leftoverPoint)

    if leftoverPoint < penalty:
        sql = """UPDATE member_sprint_log SET `status` = 'DELETE' WHERE(`member_id` = '%s') AND (`sprint_id` = '%s')""" % (memberId, sprintId)
        cursor.execute(sql)
        db.commit();
        # 강퇴 처리 보내기
        return 202

    sql = """INSERT INTO point_log(`created_at`,`point_value`, `member_id`, `sprint_id`) VALUES ('%s', '%s', '%s', '%s')""" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), -1 * penalty, memberId, sprintId)
    cursor.execute(sql)
    db.commit();

    # detect
    return 205

# 파일 업로드 처리
@app.route('/openCV', methods=['POST'])
def upload_file():
    result = {}

    flaskDTO = request.form.get("flaskDTO")

    dict = json.loads(flaskDTO)

    nickName = dict.get("nickName")
    roomId = dict.get("roomId")
    mode = dict.get("mode")

    f = request.files['img']

    img2 = cv2.imdecode(numpy.frombuffer(f.read(), numpy.uint8), cv2.IMREAD_COLOR)

    sql = """SELECT member_id FROM member where nick_name = '%s' """ % (nickName)
    cursor.execute(sql)
    row = cursor.fetchone()

    # E : NO MEMBER
    if(row == None):
        code = 504
        result.update({"code": code, "MESSAGE": CODE[code]})
        return result
    memberId = row[0]


    sql = """SELECT * FROM room where room_id = '%s'""" %(roomId)
    cursor.execute(sql)
    row = cursor.fetchone()

    if(row == None):
        code = 522
        result.update({"code": code, "MESSAGE": CODE[code]})
        return result


    code = 200

    if (totalRoom.get(roomId) == None):
        totalRoom[roomId] = {}
        totalRoom[roomId][nickName] = [0, 0, 0, 0]

    elif(totalRoom.get(roomId).get(nickName) == None):
        totalRoom[roomId][nickName] = [0, 0, 0, 0]

    if (mode == "MODE1"):
        return ({"responseCode": 200})

    # 졸음 인식
    if (mode == "MODE2"):

        # 조명제거
        L, gray = lr.light_removing(img2)

        # 그레이스케일링
        rects = detector(gray, 0)

        # 화면 감지
        for rect in rects:

            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            # (leftEAR + rightEAR) / 2 => both_ear.
            both_ear = (leftEAR + rightEAR) * 500  # I multiplied by 1000 to enlarge the scope.

            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(img2, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(img2, [rightEyeHull], -1, (0, 255, 0), 1)

            print(both_ear)
            EAR_THRESH = 150

            if both_ear < EAR_THRESH:
                totalRoom[roomId][nickName][0] += 1

            else:
                totalRoom[roomId][nickName][0] = 0

            if totalRoom[roomId][nickName][0] >= 1:
                totalRoom[roomId][nickName][1] += 1
                totalRoom[roomId][nickName][0] = 0

                code = penaltRecord(memberId, roomId)

                result.update({"code":code, "MESSAGE": CODE[code]})
                return result
        result.update({"code":code, "MESSAGE": CODE[code]})

    if (mode == "MODE3"):
        img = letterbox(img2, 640, stride=32)[0]
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = numpy.ascontiguousarray(img)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names

        t0 = time.time()

        img = torch.from_numpy(img).to(device)
        # print(type(img))
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s, im0 = '', img2

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            if len(det):
                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if(names[int(cls)] == 'cell phone'):
                        totalRoom[roomId][nickName][2] += 1

                    else :
                        totalRoom[roomId][nickName][2] = 0

                    if(totalRoom[roomId][nickName][2] >= 10):
                        totalRoom[roomId][nickName][3] += 1
                        totalRoom[roomId][nickName][2] = 0
                        code = penaltRecord(memberId, roomId)
                        result.update({"code": code, "MESSAGE": CODE[code]})
                        return result

        result.update({"code": code, "MESSAGE": CODE[code]})
    return result

if __name__ == '__main__':
    # 서버 실행
    app.run(host="0.0.0.0", port=5000)
