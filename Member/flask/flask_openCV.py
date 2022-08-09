import time
import dlib
import numpy
import pymysql
import json

from flask import Flask, render_template, request, jsonify
import cv2
import light_remover as lr
from scipy.spatial import distance as dist


from imutils import face_utils
from threading import Thread

# # databse 접근
# db = pymysql.connect(host='43.200.6.20',
#                      port=3306,
#                      user='seokin',
#                      password='12345',
#                      db='watchme',
#                      charset='utf8')
#
# # database를 사용하기 위한 cursor를 세팅합니다.
# cursor = db.cursor()

app = Flask(__name__)

# dlib이용 얼굴을 감지하기
detector = dlib.get_frontal_face_detector()
# dlib이용 얼굴을 각각의 좌표로 바꿔주기
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

cellphoneCount = {}
closeEyeCount = {}
totalCount = {}
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]


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




(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
OPEN_EAR = 0
EAR_THRESH = 0
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# 업로드 HTML 렌더링
@app.route('/upload')
def render_file():
    return render_template('upload.html')


# 파일 업로드 처리
@app.route('/openCV', methods=['POST'])
def upload_file():

    result = {}
    memberId = request.form.get("member_id")
    roomId = request.form.get("room_id")
    mode1 = request.form.get("mode1")
    mode2 = request.form.get("mode2")
    mode3 = request.form.get("mode3")
    mode4 = request.form.get("mode4")

    if (closeEyeCount.get(memberId) == None):
        closeEyeCount[memberId] = 0;

        f = request.files['file']
        img2 = cv2.imdecode(numpy.fromstring(f.read(), numpy.uint8), cv2.IMREAD_UNCHANGED)

        if(mode1 != None):
            return ({"responseCode": 200})

        # 졸음 인식
        if(mode2 != None or mode4 != None):

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
                    closeEyeCount[memberId] = closeEyeCount.get(memberId) + 1;

                else:
                    closeEyeCount[memberId] = 0;

            if closeEyeCount[memberId] > 10:
                if (totalCount.get(memberId) == None):
                    totalCount[memberId] = 0
                totalCount[memberId] = totalCount.get(memberId) + 1

                closeEyeCount[memberId] = 0

                # DB UPDATE
                # sql="""UPDATE watchme.penaltys set MODE1 = MODE1 + 1 WHERE member_id = %d AND room_id = %d;""" % (memberId, roomId)
                # cursor.execute(sql)

                result.update({"CloseDetect": 205, "responseMessage": "CLOSE PENALTY OCCURRED", "MODE": "MODE2","PenaltyCount": totalCount[memberId]})
            else :
                result.update({"CloseDetect": 200})

        # 사물 인식 yolo v3 - tiny
        if(mode3 != None or mode4 != None):
            img = cv2.resize(img2, None, fx=0.4, fy=0.4)
            height, width, channels = img.shape

            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)

            class_ids = []
            confidences = []

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
            result.update({"ObjectDetect": 200})

    return result

if __name__ == '__main__':
    # 서버 실행
    app.run(debug=True)