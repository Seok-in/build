import gc
import time
import dlib
# # databse 접근
import pymysql



import numpy
import json
import tracemalloc

from flask import Flask, render_template, request, jsonify, make_response
from flask_cors import CORS
import cv2
import light_remover as lr
import datetime
from pytz import timezone
import torch
from scipy.spatial import distance as dist

from imutils import face_utils

from ErrorCode import CODE
# from config import db
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device, time_synchronized, TracedModel


my_snapshot = None

# # database를 사용하기 위한 cursor를 세팅합니다.
# cursor = db.cursor()
tracemalloc.start()
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# dlib이용 얼굴을 감지하기
detector = dlib.get_frontal_face_detector()
# dlib이용 얼굴을 각각의 좌표로 바꿔주기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

totalRoom = {}

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
OPEN_EAR = 0
EAR_THRESH = 0
classes = []

classes_to_filter = ['train']



opt = {
    "weights": "yolov7-tiny.pt",  # Path to weights file default weights are for nano model
    "yaml": "data/coco.yaml",
    "img-size": 320,  # default image size
    "conf-thres": 0.25,  # confidence threshold for inference.
    "iou-thres": 0.45,  # NMS IoU threshold for inference.
    "device": 'cpu',  # device to run our model i.e. 0 or 0,1,2,3 or cpu
    "classes": classes_to_filter  # list of classes to filter or None
}

weights, imgsz = opt['weights'], opt['img-size']
device = select_device(opt['device'])
half = device.type != 'cpu'
model = attempt_load(weights, map_location=device)  # load FP32 model

# print(type(model))
# print(model)
# print(device)

stride = int(model.stride.max())  # model stride
imgsz = check_img_size(imgsz, s=stride)  # check img_size
if half:
    model.half()
names = model.module.names if hasattr(model, 'module') else model.names
if device.type != 'cpu':
    model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

sleepCnt = 5
maxPenaltyCnt = 3

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


def penaltRecord(nickName, memberId, roomId):
    db = pymysql.connect(host='54.180.134.240',
                         port=3306,
                         user='seokin',
                         password='ghdtjrdls777!',
                         db='WatchMe',
                         charset='utf8')
    cursor = db.cursor()
    # 패널티 기록
    sql = """INSERT INTO penalty_log(`created_at`,`member_id`, `room_id`, `status`) VALUES ('%s','%s', '%s', '%s');""" % (
        datetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S'), memberId, roomId, "MODE2");
    cursor.execute(sql)
    db.commit();


    sql = """SELECT sprint_id FROM sprint where room_id = '%s' """ % (roomId)
    cursor.execute(sql)
    row = cursor.fetchone()
    if(row == None) :

        if totalRoom[roomId][nickName][1] + totalRoom[roomId][nickName][2] >=maxPenaltyCnt:
            print("EXILE ROOM")
            sql = """UPDATE member_room_log SET `status` = 'DELETE' WHERE(`member_id` = '%s') AND (`room_id` = '%s')""" % (memberId, roomId)
            cursor.execute(sql)
            db.commit();
            db.close();
            return 202
        db.close();
        return 205

    sprintId = row[0]

    sql = """SELECT fee, penalty_money FROM sprint_info WHERE sprint_id = '%s' """ % (sprintId)
    cursor.execute(sql)
    row = cursor.fetchone()
    fee = row[0]
    penalty = row[1]

    sql = """SELECT COUNT(*) FROM penalty_log WHERE member_id = '%s' AND sprint_id = '%s'""" % (memberId, sprintId)
    cursor.execute(sql)
    row = cursor.fetchone()
    if(row == None):
        db.close();
        return 553
    else :
        count = row[0]


    if fee < penalty * count:
        sql = """UPDATE member_sprint_log SET `status` = 'DELETE' WHERE(`member_id` = '%s') AND (`sprint_id` = '%s')""" % (memberId, sprintId)
        cursor.execute(sql)
        db.commit();
        # 강퇴 처리 보내기
        db.close();
        return 202

    sql = """INSERT INTO point_log(`created_at`,`point_value`, `member_id`, `sprint_id`) VALUES ('%s', '%s', '%s', '%s')""" %  (datetime.atetime.now(timezone('Asia/Seoul')).strftime('%Y-%m-%d %H:%M:%S'), -1 * penalty, memberId, sprintId)
    cursor.execute(sql)
    db.commit();
    db.close();
    # detect
    return 205



# 파일 업로드 처리
@app.route('/openCV', methods=['POST'])
def upload_file():
    db = pymysql.connect(host='54.180.134.240',
                         port=3306,
                         user='seokin',
                         password='ghdtjrdls777!',
                         db='WatchMe',
                         charset='utf8')
    cursor = db.cursor()
    # # 값을 계속 사용해야 하므로 전역 변수에 저장한다
    # global my_snapshot
    # if not my_snapshot:
    #     # 최초 메모리 상태를 저장한다
    #     my_snapshot = tracemalloc.take_snapshot()
    # else:
    #     lines = []
    #     # 현재 메모리 상태를 최초와 비교하여 얼마나 차이가 나는지에 대한 통계를 구한다
    #     top_stats = tracemalloc.take_snapshot().compare_to(my_snapshot, 'lineno')
    #     # 메모리 사용량이 많은 순서대로 10개를 구하여 출력한다
    #     for stat in top_stats[:10]:
    #         lines.append(str(stat))
    #     print('\n'.join(lines), flush=True)
    #
    # snapshot = tracemalloc.take_snapshot()
    # for idx, stat in enumerate(snapshot.statistics('lineno')[:5], 1):
    #     print(str(stat), flush=True)
    # # 메모리 사용량이 가장 많은 부분에 대한 정보를 상세히 출력한다
    # traces = tracemalloc.take_snapshot().statistics('traceback')
    # for stat in traces[:1]:
    #     print("memory_blocks=", stat.count, "size_kB=", stat.size / 1024, flush=True)
    #     for line in stat.traceback.format():
    #         print(line, flush=True)

    result = {}

    # print(request)
    # print(request.files)
    files = request.files
    flaskDTO = files.get('flaskDTO')
    # print(flaskDTO)

    dict = json.load(flaskDTO)

    nickName = dict.get("nickName")
    roomId = dict.get("roomId")
    mode = dict.get("mode")

    print(nickName)
    print(roomId)
    print(mode)

    f = request.files['img']

    img2 = cv2.imdecode(numpy.frombuffer(f.read(), numpy.uint8), cv2.IMREAD_COLOR)


    del dict
    del f
    del request.files
    gc.collect(0)
    print(nickName)
    sql = """SELECT member_id FROM member where nick_name = '%s' """ % (nickName)
    cursor.execute(sql)
    row = cursor.fetchone()

    # E : NO MEMBER
    if(row == None):
        code = 504
        result.update({"code": code, "MESSAGE": CODE[code]})
        db.close()
        return result
    memberId = row[0]

    print(memberId)

    # memberId = 1

    print(roomId)
    sql = """SELECT * FROM room where room_id = '%s'""" % roomId
    cursor.execute(sql)
    row = cursor.fetchone()

    if (row == None):
        code = 522
        result.update({"code": code, "MESSAGE": CODE[code]})
        db.close()
        return result

    code = 200

    if (totalRoom.get(roomId) == None):
        print("init roomData")
        totalRoom[roomId] = {}
        totalRoom[roomId][nickName] = [0, 0, 0, 0]

    if totalRoom.get(roomId).get(nickName) == None:
        print("init room nickname")
        totalRoom[roomId][nickName] = [0, 0, 0, 0]

    if (mode == "MODE1"):
        return ({"responseCode": 200})

    # 졸음 인식
    if (mode == "MODE2"):
        # print(f'count{sys.getrefcount(img2)}')
        # 조명제거
        L, gray = lr.light_removing(img2)
        # print(f'count{sys.getrefcount(img2)}')
        print()

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

            # leftEyeHull = cv2.convexHull(leftEye)
            # rightEyeHull = cv2.convexHull(rightEye)
            # cv2.drawContours(img2, [leftEyeHull], -1, (0, 255, 0), 1)
            # cv2.drawContours(img2, [rightEyeHull], -1, (0, 255, 0), 1)

            print(both_ear)
            EAR_THRESH = 200


            if both_ear < EAR_THRESH:
                totalRoom[roomId][nickName][0] += 1

            else:
                totalRoom[roomId][nickName][0] = 0

            if totalRoom[roomId][nickName][0] >= sleepCnt:
                totalRoom[roomId][nickName][1] += 1
                totalRoom[roomId][nickName][0] = 0
                code = penaltRecord(nickName, memberId, roomId)
                result.update({"code": code, "MESSAGE": CODE[code]})
                return result
            code = 205
        print(totalRoom[roomId][nickName][0])
        print(totalRoom[roomId][nickName][1])
        del rects
        del L
        del gray
        gc.collect()
        result.update({"code": code, "MESSAGE": CODE[code]})
        db.close()
        return result

    if mode == "MODE3":
        with torch.no_grad():
            print("mode3")

            img0 = img2
            img = letterbox(img0, imgsz, stride=stride)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
            img = numpy.ascontiguousarray(img)
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            # t1 = time_synchronized()
            pred = model(img, augment=True)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt['conf-thres'], opt['iou-thres'], classes=[67], agnostic=False)
            t2 = time_synchronized()
            for i, det in enumerate(pred):
                gn = torch.tensor(img0.shape)[[1, 0, 1, 0]]
                if len(det):
                    print('cell phone')

                    code = penaltRecord(nickName, memberId, roomId)
                    # code = 200

                    gc.collect()
                    result.update({"code": code, "MESSAGE": CODE[code]})
                    db.close()
                    return result
            del pred
            gc.collect()

    del img2
    gc.collect()

    result.update({"code": code, "MESSAGE": CODE[code]})
    db.close()
    return result


if __name__ == '__main__':

    # 서버 실행
    app.run(host="0.0.0.0", port=5000)

