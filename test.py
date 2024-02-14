from datetime import datetime
import cv2
import face_recognition
from os import listdir
import threading
import time
from FaceDetection.preEnd.Tracker import *

link_hoh = "../../"
link_hoh = ""
tracker = EuclideanDistTracker()
faces_found = 0
encodeEls = []
names = []
people = 1
with open(link_hoh + 'static/DataBases/people.csv', 'r') as f:
    data_list = f.readlines()
    for line in data_list:
        if "person" in line:
            print(line)
            people += 1
    faces_found = len(data_list) - 1
    print(faces_found)
# for file in listdir('DataBases/peopleFaces'):
#     faces_found += 1
print(people)
print(faces_found)
t_lock = threading.Lock()


def mark_people(name, pic_path):

    with open(link_hoh + 'static/DataBases/people.csv', 'r+') as f:
        lines = f.readlines()
        people_name = []
        for line in lines:
            #for i in range(5):
            #    print("++++++++++++++++++++++++++++++++++")
            #print(line)
            #for i in range(5):
            #    print("++++++++++++++++++++++++++++++++++")
            entry = line.split(',')
            people_name.append(entry[0])
        now = datetime.now()
        dt_string = now.strftime('%H:%M:%S')
        if name not in people_name:
            print(5555555555555)
            f.writelines(f'\n{name},{dt_string},{pic_path}')
            return 1
        else:
            print("+++++++++++")
            print(name)
            if name == 'Unknown':
                f.writelines(f'\n{name},{dt_string},{pic_path}')
                return 1
            else:
                return 0




def dataBaseCreate():
    directorySource = 'static/DataBases/gg/'

    encodeEl = []
    nameEn = []

    for file in listdir(directorySource):
        # print(file)
        resizeValue1 = 1
        path = directorySource + file
        img1 = face_recognition.load_image_file(path)
        xx = img1.shape[1]
        while xx > 400:
            resizeValue1 += 1
            yy = img1.shape[0] // resizeValue1
            xx = img1.shape[1] // resizeValue1
        img1 = cv2.resize(img1, (xx, yy))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        name = file.replace('.jpg', '')
        face_location1 = face_recognition.face_locations(img1)[0]
        encodeEl += [face_recognition.face_encodings(img1)[0]]
        nameEn += [name]
        cv2.rectangle(img1, (face_location1[3], face_location1[0]), (face_location1[1], face_location1[2]), (255, 0, 0),
                      3)
        # cv2.imshow(f'Image {name}', img1)
    return encodeEl, nameEn


def face_name(img_work, faces_location, face_names, id):
    global faces_found, t_lock, encodeEls, names, people
    time.sleep(2)
    stime = time.time()
    encode = face_recognition.face_encodings(img_work, [faces_location])
    file = open("resulta/timeResultRecog.txt","a")
    file.write(str(round(time.time()-stime, 3))+"\n")
    # print(faces_location)
    # cv2.imshow('yes', img_work)
    for face_code in encode:
        time.sleep(1)
        j = 0
        # print(len(encode))
        # print(encode)
        with t_lock:
            for ele in encodeEls:
                res = face_recognition.compare_faces([ele], face_code)
                if True in res:
                    if mark_people(face_names[j], f'/peopleFaces/{faces_found}.jpg') == 1:
                        face_pic = cv2.cvtColor(img_work, cv2.COLOR_RGB2BGR)
                        face_pic = face_pic[faces_location[0]:faces_location[2], faces_location[3]:faces_location[1]]
                        cv2.imwrite(link_hoh + f'static/DataBases/peopleFaces/{faces_found}.jpg', face_pic)
                        faces_found += 1
                    # print(face_names[j])
                    if "person" in face_names[j]:
                        tracker.name_update(id, "Unknown")
                        return 'Unknown'
                    else:
                        tracker.name_update(id, face_names[j])
                        return face_names[j]
                        # print(face_names[j])
                        # cv2.imshow('yes', img_work)
                        # input('Hehe')
                        # for ii in range(10):
                        #     print('=========================')
                        # cv2.rectangle(face_pic, (faces_location[3], faces_location[0]), (faces_location[1],
                        # faces_location[2]), (255, 0, 0), 3)
                j += 1
            tracker.name_update(id, 'Unknown')
            if mark_people(f'person {people}', f'/peopleFaces/{faces_found}.jpg') == 1:
                face_pic = cv2.cvtColor(img_work, cv2.COLOR_RGB2BGR)
                face_pic = face_pic[faces_location[0]:faces_location[2], faces_location[3]:faces_location[1]]
                cv2.imwrite(link_hoh + f'static/DataBases/peopleFaces/{faces_found}.jpg', face_pic)
                faces_found += 1
                encodeEls += encode
                names += [f'person {people}']
                people += 1
    # print('Unknown')
    # for ii in range(10):
    #     print('=========================')
    return 'Unknown'


def process(frame, resize_value=4, do_detection=False, do_tracking = False):

    pp = frame
    y = frame.shape[0] // resize_value
    x = frame.shape[1] // resize_value
    frame = cv2.resize(frame, (x, y))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    start_time = time.time()
    face_location = face_recognition.face_locations(img)
    #ttt = cv2.getTickCount() - timer
    #fppppss = cv2.getTickFrequency() / (ttt)
    file = open("resulta/time.txt","a")
    file.write(str(round(time.time()-start_time, 3))+"\n")
    
    detection = []
    for face in face_location:
        detection.append(face)
        if do_detection:
            cv2.rectangle(pp, (face[3] * resize_value, face[0] * resize_value), (face[1]*resize_value, face[2] *
                                                                                 resize_value), (0, 255, 0), 3)
    
    start_time = time.time()
    boxes_ids = tracker.update(detection,resize_value)
    if do_tracking:
        pp = tracker.draw_line(pp)
    pp = tracker.draw_track(pp,resize_value)
    file = open("resulta/timeResultTrac.txt","a")
    file.write(str(str(round(time.time()-start_time, 7)))+"\n")
    for bid in boxes_ids:
        if bid[5] == '':
            # should make it with fork and give a value until calc end like "Recognition"
            face = [bid[0], bid[1], bid[2], bid[3]]
            recognition_thread = threading.Thread(target=face_name, args=(img, face, names, bid[4],))
            recognition_thread.start()
            bid[5] = 'Recognition'
            # print('Name:    ', bid[5])
            k = tracker.name_update(bid[4], bid[5])
            # print(k)
            # print(bid)
        if do_detection:
            if bid[5] == 'Unknown':
                cv2.putText(pp, 'Unknown', (bid[3] * resize_value, bid[0]*resize_value-10), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 0, 255), 1)
            else:
                cv2.putText(pp, f'{bid[5]}', (bid[3] * resize_value, bid[0] * resize_value - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)

    return pp


resourceNumber = 0
path = 'static/DataBases/img/'
for file in listdir(path):
    resourceNumber += 1

encodeEls, names = dataBaseCreate()
# print(encodeEls[0])


def start():
    global link_hoh
    # link_hoh = "../../"
    koko = 0
    for file in listdir('C:/Users/accer/Pictures/test/test'):
        print(file)
        img = cv2.imread('C:/Users/accer/Pictures/test/test/'+file)
        # cv2.imshow(f"{koko}", img)
        koko += 1
        process(img, do_detection=True)

    for file in listdir('C:/Users/accer/Pictures/test/test'):
        print(file)
        img = cv2.imread('C:/Users/accer/Pictures/test/test/' + file)
        # cv2.imshow(f"{koko}", img)
        koko += 1
        process(img, do_detection=True)


#start()


def work():

    resizeValue = 4

    cap = cv2.VideoCapture(1)

    # names = []
    # boxes_ids = []
    # print("Press Enter to Start")
    move = 1
    while 1:

        c = cv2.waitKey(move)
        timer = cv2.getTickCount()
        _, frame = cap.read()
        pp = frame
        y = frame.shape[0] // resizeValue
        x = frame.shape[1] // resizeValue
        frame = cv2.resize(frame, (x, y))
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_location = face_recognition.face_locations(img)
        

        detection = []
        for face in face_location:
            # print(face, '//////////////////////////////')
            detection.append(face)
            cv2.rectangle(pp, (face[3] * resizeValue, face[0] * resizeValue), (face[1] * resizeValue, face[2] * resizeValue)
                          , (0, 255, 0), 3)
        # print(detection)

        boxes_ids = tracker.update(detection)

        # print('+++++', boxes_ids)
        for bid in boxes_ids:
            # print('55555', bid)
            if bid[5] == '':
                # for i in range(4):
                #     print("+++++++++++++++++++++++++++++++++++++++++++++++++")
                # should make it with fork and give a value until calc end like "Recognition"
                face = [bid[0], bid[1], bid[2], bid[3]]
                # print('////////////////////////', face, '////////////////')
                recognition_thread = threading.Thread(target=face_name, args=(img, face, names, bid[4],))
                recognition_thread.start()
                # recognition_thread = threading.Thread(target=face_name, args=(img, face_location, names, bid[4],))
                # recognition_thread.start()
                bid[5] = 'Recognition'
                # print('Name:    ', bid[5])
                k = tracker.name_update(bid[4], bid[5])
                # print(k)
                # print(bid)
                # for i in range(4):
                #     print("+++++++++++++++++++++++++++++++++++++++++++++++++")
            if bid[5] == 'Unknown':
                cv2.putText(pp, 'Unknown', (bid[3] * resizeValue, bid[0] * resizeValue - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6,
                            (0, 0, 255), 1)
            else:
                cv2.putText(pp, f'{bid[5]}', (bid[3] * resizeValue, bid[0] * resizeValue - 10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    # for face, code in zip(face_location, encodeEl):
    #     j = 0
    #     # for ele in encodeEls:
    #     #     result = face_recognition.compare_faces([ele], code)
    #     #     dist = face_recognition.face_distance([ele], code)
    #     #     if True in result:
    #     #         break
    #     #     j += 1
    #     dist = 0
    #     result = [False]
    #     if result == [False]:
    #         cv2.putText(pp, 'Unknown', (face[3]*resizeValue, face[0]*resizeValue - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6,
    #                     (0, 0, 255), 1)
    #         # cv2.putText(frame, 'Unknown', (face[3], face[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
    #     else:
    #         cv2.putText(pp, f'{names[j]} {round(dist[0], 2)}', (face[3]*resizeValue, face[0]*resizeValue - 10),
    #                     cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
    #         # cv2.putText(frame, f'{names[j]} {round(dist[0], 2)}', (face[3], face[0] - 10), cv2.FONT_HERSHEY_COMPLEX,
    #         #            0.6, (255, 255, 255), 1)
    #     # cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), 3)
    #     FPS = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
    #     cv2.putText(pp, f'FPS: {int(FPS)}', (10, 18), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 0, 0), 1)
    #     cv2.imshow("Live", pp)
    #     # cv2.imshow("Frame", frame)
    #     if c == ord('q'):
    #         break
    #     elif c == ord('+'):
    #         resizeValue += 1
    #         print(resizeValue)
    #     elif c == ord('-'):
    #         if resizeValue > 1:
    #             resizeValue -= 1
    #         print(resizeValue)
    #     elif c == ord(' '):
    #         if move != 0:
    #             move = 0
    #         else:
    #             move = 10
    #     elif c == ord('p'):
    #         cv2.imwrite(path+f'pic{resourceNumber}.png', pp)
    #         resourceNumber += 1
    #
    # # recognition_thread.join()
    # cv2.destroyAllWindows()


'''
import cv2
import numpy as np
import face_recognition
from os import listdir


def dataBaseCreate():
    directorySource = 'C:/Users/accer/Pictures/test/source/'

    encodeEl = []
    nameEn = []

    for file in listdir(directorySource):
        # print(file)
        resizeValue1 = 1
        path = directorySource + file
        img1 = face_recognition.load_image_file(path)
        xx = img1.shape[1]
        while xx > 400:
            resizeValue1 += 1
            yy = img1.shape[0] // resizeValue1
            xx = img1.shape[1] // resizeValue1
        img1 = cv2.resize(img1, (xx, yy))
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        name = file.replace('.jpg', '')
        face_location1 = face_recognition.face_locations(img1)[0]
        encodeEl += [face_recognition.face_encodings(img1, face_location1)[0]]
        nameEn += [name]
        cv2.rectangle(img1, (face_location1[3], face_location1[0]), (face_location1[1], face_location1[2]), (255, 0, 0), 3)
        cv2.imshow(f'Image {name}', img1)
    return encodeEl, nameEn


encodeEls, names = dataBaseCreate()

resizeValue = 1

cap = cv2.VideoCapture(0)

while 1:
    c = cv2.waitKey(10)
    _, frame = cap.read()
    pp = frame
    y = frame.shape[0] // resizeValue
    x = frame.shape[1] // resizeValue
    frame = cv2.resize(frame, (x, y))
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_location = face_recognition.face_locations(img)
    encodeEl = face_recognition.face_encodings(img, face_location)

    for face, code in zip(face_location, encodeEl):
        j = 0
        for ele in encodeEls:
            result = face_recognition.compare_faces([ele], code)
            dist = face_recognition.face_distance([ele], code)
            if True in result:
                break
            j += 1
        if result == [False]:
            cv2.putText(frame, 'Unknown', (face[3], face[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
        else:
            cv2.putText(frame, f'{names[j]} {round(dist[0], 2)}', (face[3], face[0] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
        cv2.rectangle(frame, (face[3], face[0]), (face[1], face[2]), (0, 255, 0), 3)

    cv2.imshow("Live", frame)
    if c == ord('q'):
        break
    elif c == ord('+'):
        resizeValue += 1
        print(resizeValue)
    elif c == ord('-'):
        if resizeValue > 1:
            resizeValue -= 1
        print(resizeValue)


cv2.destroyAllWindows()

'''


