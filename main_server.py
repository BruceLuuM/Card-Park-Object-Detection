import os
import cv2
from flask_cors import CORS
import numpy as np
import easyocr
from flask import Flask, request, jsonify
import util

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define constants
model_cfg_path = os.path.join('.', 'model', 'cfg', 'darknet-yolov3.cfg')
model_weights_path = os.path.join('.', 'model', 'weights', 'model.weights')
class_names_path = os.path.join('.', 'model', 'class.names')

# Load class names
with open(class_names_path, 'r') as f:
    class_names = [j[:-1] for j in f.readlines() if len(j) > 2]

# Load model
net = cv2.dnn.readNetFromDarknet(model_cfg_path, model_weights_path)

# Initialize EasyOCR Reader
reader = easyocr.Reader(['en'])

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    # Check if an image file is included in the POST request
    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400
 
    file = request.files['image']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_UNCHANGED)

    # Ensure the image has 3 channels
    if img.shape[2] == 4:  # If the image has an alpha channel
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    elif img.shape[2] == 1:  # If the image is grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    H, W, _ = img.shape

    # Convert image
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), True)

    # Get detections
    net.setInput(blob)
    detections = util.get_outputs(net)

    # bboxes, class_ids, confidences
    bboxes = []
    class_ids = []
    scores = []

    for detection in detections:
        # [x1, x2, x3, x4, x5, x6, ..., x85]
        bbox = detection[:4]

        xc, yc, w, h = bbox
        bbox = [int(xc * W), int(yc * H), int(w * W), int(h * H)]

        bbox_confidence = detection[4]

        class_id = np.argmax(detection[5:])
        score = np.amax(detection[5:])

        bboxes.append(bbox)
        class_ids.append(class_id)
        scores.append(score)

    # Apply NMS
    bboxes, class_ids, scores = util.NMS(bboxes, class_ids, scores)

    # Initialize results list
    results = []

    for bbox_ in bboxes:
        xc, yc, w, h = bbox_

        license_plate = img[int(yc - (h / 2)): int(yc + (h / 2)), int(xc - (w / 2)): int(xc + (w / 2)), :].copy()
        license_plate_gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)
        _, license_plate_thresh = cv2.threshold(license_plate_gray, 64, 255, cv2.THRESH_BINARY_INV)

        output = reader.readtext(license_plate_gray)
        for out in output:
            text_bbox, text, text_score = out
            results.append({'text': text, 'score': text_score})

    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
