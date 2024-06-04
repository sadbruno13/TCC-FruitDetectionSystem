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

    if request.is_json:
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
    elif request.content_type.startswith('multipart/form-data'):
        image_file = request.files["image_file"]
        result = yolo_model.detect_objects_from_form(image_file=image_file)
        return make_response(
            jsonify(
                {
                    "Resultado": result 
                }
            )
        )
    else:
        return jsonify({"message": "Conteúdo desconhecido"}), 400


if __name__ == '__main__':
    app.run(host="0.0.0.0")