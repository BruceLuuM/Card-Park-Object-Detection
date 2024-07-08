from flask import Flask, send_file

app = Flask(__name__)

@app.route('/')
def get_image():
    # Replace 'image.jpg' with your actual image file path
    image_path = 'D:/Coding/DDat/yolov3-from-opencv-object-detection/data/Screenshot (2).png'
    return send_file(image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True,port=5001)
