import time
import dlib
import numpy
import pymysql
import json

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import cv2
import light_remover as lr
import datetime
from scipy.spatial import distance as dist

from imutils import face_utils

from threading import Thread

time.strftime('%Y-%m-%d %H:%M:%S')

# # databse 접근
db = pymysql.connect(host='54.180.134.240',
                     port=3306,
                     user='seokin',
                     password='ghdtjrdls777!',
                     db='WatchMe',
                     charset='utf8')

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

@app.route('/test', methods=['POST'])
def test():
    # sql = """select email from member where member_id = 1;"""
    # cursor.execute(sql)
    # rows = cursor.fetchone()
    # print(rows)
    # return ({"email": rows})

    # sql = """insert into penalty_log(member_id,room_id,status) values (1, 1, 'MODE1')"""
    # cursor.execute(sql)
    # print(sql)
    # print(cursor.execute(sql, (1, 1, 'MODE1')))

    sql = """INSERT INTO WatchMe.penalty_log(`created_at`,`member_id`, `room_id`, `status`) VALUES ('%s','%d', '%d', '%s');""" % (
        datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 1, 1, "MODE3");
    print(sql)

    cursor.execute(sql)
    db.commit();

    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # cursor.execute(sql)
    # db.commit();
    return ({"test": "test"})

# 파일 업로드 처리
@app.route('/openCV', methods=['POST'])
def upload_file():

    result = {}
    nickName = request.form.get("nickName")
    roomId = request.form.get("roomId")
    mode = request.form.get("MODE")

    if (totalRoom.get(roomId) == None):
        totalRoom[roomId] = {}
        totalRoom[roomId][memberId] = [0, 0, 0]

    elif(totalRoom.get(roomId).get(memberId) == None):
        totalRoom[roomId][memberId] = [0, 0, 0]

    print(totalRoom)
    print(totalRoom[roomId][memberId][0])

    f = request.files['file']
    img2 = cv2.imdecode(numpy.fromstring(f.read(), numpy.uint8), cv2.IMREAD_COLOR)

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

            if totalRoom[roomId][nickName][0] > 10:
                totalRoom[roomId][nickName][1] += 1

                totalRoom[roomId][nickName][0] = 0

                # DB INSERT
                sql = """INSERT INTO penalty_log(`created_at`,`member_id`, `room_id`, `status`) VALUES ('%s','%s', '%s', '%s');""" % (
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), nickName, roomId, "MODE2");

                cursor.execute(sql)
                db.commit();

                sql = """SELECT * FROM  """

                result.update({"code":205 , "responseMessage": "CLOSE PENALTY OCCURRED",
                               "PenaltyCount": totalRoom[roomId][nickName][1]})
                return result
            else:
                result.update({"code": 200})

    # 사물 인식 yolo v3 - tiny
    if (mode == "MODE3"):
        # img = cv2.resize(img2, None, fx=0.4, fy=0.4)
        img = img2
        height, width, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(output_layers)
        class_ids = []
        confidences = []
        is_checked = 0

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = numpy.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    print(str(classes[class_id]))

                    if str(classes[class_id]) == 'cellphone':
                        # 핸드폰 감지
                        totalRoom[roomId][nickName][2] += 1

                        # DB INSERT
                        sql = """INSERT INTO WatchMe.penalty_log(`created_at`,`member_id`, `room_id`, `status`) VALUES ('%s','%s', '%s', '%s');""" % (
                            datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), nickName, roomId, "MODE3");
                        cursor.execute(sql)
                        db.commit();
                        result.update({"Code": 205, "responseMessage": "CELL PHONE PENALTY OCCURRED",
                                       "PenaltyCellPhoneCount": totalRoom[roomId][nickName][2]})
                        return result


            result.update({"ObjectDetect": 200})


    return result
if __name__ == '__main__':
    # 서버 실행
    app.run(debug=True)