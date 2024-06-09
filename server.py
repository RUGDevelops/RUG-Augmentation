import threading

from PIL import Image
from pathlib import Path
from flask import Flask, request, Response, jsonify

from ModelCreator import ModelCreator

app = Flask(__name__)


is_busy = False



class ThreadClass:
    def __init__(self, user):
        global is_busy
        is_busy = True
        self.user = user
        thread = threading.Thread(target=self.run, args=())
        thread.start()                             # Start the execution
        print("Thread started")

    def run(self):
        test = ModelCreator(None, self.user)
        test.create_model()
        global is_busy
        is_busy = False



@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"




@app.route("/<user>/validate", methods=['POST'])
def validate_user(user):
    file = request.files['image']
    # Read the image via file.stream
    img = Image.open(file.stream)
    # save image to disk
    # create directory if not exist
    Path(f'temporary').mkdir(parents=True, exist_ok=True)

    rgb_im = img.convert('RGB')
    rgb_im.save(f'temporary/{user}.jpg')
    # call validation function and validate the image in synchronous mode (not in thread)
    test = ModelCreator(None, user)

    if not test.does_model_exist():
        return jsonify({'msg': 'error', 'error': 'Model for this uer doesn\'t exist'})

    is_valid = test.verify_image()

    if is_valid:
        return jsonify({'msg': 'success', 'user': user, 'size': [img.width, img.height]})
    else:
        return jsonify({'msg': 'error', 'error': 'Invalid face'})



@app.route("/<user>/setup", methods=['POST'])
def setup_user(user):
    if 'video' not in request.files:
        return jsonify({'msg': 'error', 'error': 'No video file found'})
    video = request.files['video']

    if(video.filename == ''):
        return jsonify({'msg': 'error', 'error': 'No selected file'})

    if is_busy:
        return jsonify({'msg': 'error', 'error': 'Server is busy processing another request'})

    Path(f'temporary').mkdir(parents=True, exist_ok=True)
    video.save(f'temporary/{user}.mp4')


    try:
        ThreadClass(user)
    except Exception as e:
        print(str(e))

    return jsonify({'msg': 'progress', 'user': user})




if __name__ == "__main__":
    app.run(debug=True)