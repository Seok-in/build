from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
cors = CORS(app, resources={r"/flask/*": {"origins": "*"}})

@app.route('/filetest', methods=['POST'])
def file_test():
    f = request.files['img']
    print(f)
    print(type(f))

    return {"code":200, "message":"FILE RESPONSE"}

@app.route('/jsontest', methods=['POST'])
def json_test():
    flaskDTO = request.json
    print(flaskDTO)
    print(type(flaskDTO))
    print(flaskDTO['nickName'])

    return {"code":200, "message":"JSON RESPONSE"}

@app.route('/json_test', methods=['POST'])
def json_test():
    flaskDTO = request.get_json
    print(flaskDTO)
    print(type(flaskDTO))
    print(flaskDTO['nickName'])

    return {"code":200, "message":"JSON RESPONSE"}

@app.route('/multytest', methods=['POST'])
def file_test():
    f = request.files['img']
    flaskDTO = request.get_json("flaskDTO")
    print(f)
    print(type(f))

    print(flaskDTO)
    print(type(flaskDTO))
    print(flaskDTO['nickName'])

    return {"code":200, "message":"MULTY RESPONSE"}

if __name__ == '__main__':
    # 서버 실행
    app.run(host="0.0.0.0", port=5000)



