from flask import Flask, request, make_response, jsonify
from utils import YOLOModel

app = Flask(__name__)

yolo_model = YOLOModel(model_path="api/models/best.pt")
yolo_model.load_model()

@app.route('/', methods=["GET"])
def get_status():
    return str("Healthy")

@app.route('/score', methods=["POST"])
def score():

    data = request.json
    imageb64 = data["imageb64"]
    result = yolo_model.detect_objects(imageb64=imageb64)

    return make_response(
        jsonify(
            {
                "Resultado": result 
            }
        )
    )

if __name__ == '__main__':
    app.run()