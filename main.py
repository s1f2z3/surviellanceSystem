# from https://github.com/pornpasok/opencv-stream-video-to-web
from flask import Flask, url_for, redirect, render_template, Response
#from flask import Response
#from flask import Flask
#from flask import render_template
import threading
import datetime
import time
import cv2
import psutil
# from FaceDetection.preEnd import lib
import test as lib

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
cpu = 40
ram = 10
lib.link_hoh = ""
outputFrame = None
do_detection = False
do_tracking = False
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
resizeValue = 4
# initialize the video stream and allow the camera sensor to
# warmup
# vs = VideoStream(usePiCamera=1).start()
# vs = VideoStream(src=0).start()
cameraId = 0
maxCamera = 2
#for maxCamera in range(10):
#    print(maxCamera)
#    vs = cv2.VideoCapture(maxCamera)
#    if not vs.isOpened():
#        break
#    else:
#        vs.    ()
# kkk = input(maxCamera)
vs = cv2.VideoCapture(cameraId)
time.sleep(2.0)


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/PEOPLE")
def PEOPLE():
    # return the rendered template
    return render_template("PEOPLE.html")


def detect_motion():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, do_detection

    # loop over frames from the video stream
    while True:
        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        timer = cv2.getTickCount()
        _, frame = vs.read()
        frame = lib.process(frame, do_detection=do_detection, do_tracking=do_tracking)
        # frame = cv2.resize(frame, None)
        # frame = imutils.resize(frame)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        # acquire the lock, set the output frame, and release the
        # lock
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        
        cv2.putText(frame, f'FPS: {int(fps)}', (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/people_saved")
def people_saved():
    with open('static/DataBases/people.csv', 'r') as f:
        data_list = f.readlines()
    return data_list

@app.route("/default")
def default():
    return str(do_tracking)+"-"+ str(do_detection)

@app.route("/cpu_ram")
def cpu_ram():
    global cpu, ram
    cpu = psutil.cpu_percent()
    ram = psutil.virtual_memory().percent
    return str(cpu) + " " + str(ram)


@app.route("/change_camera")
def change_camera():
    global vs, cameraId
    cameraId += 1
    if cameraId == maxCamera:
        cameraId = 0
    print("yesssss")
    vs.release();
    vs = cv2.VideoCapture(cameraId)
    print("yesssss")


@app.route("/change_type")
def change_type():
    global do_detection
    if do_detection:
        do_detection = False
    else:
        do_detection = True
    return "nothing"

@app.route("/change_track")
def change_track():
    global do_tracking
    if do_tracking:
        do_tracking = False
    else:
        do_tracking = True
    return "nothing"


# check to see if this is the main thread of execution
if __name__ == '__main__':
    t = threading.Thread(target=detect_motion)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(debug=True, threaded=True, use_reloader=False, host='0.0.0.0')
# release the video stream pointer
vs.release()

#
# if __name__ == '__main__':
#     app.run(debug=True)
#     # app.run(debug=True, host='0.0.0.0')
