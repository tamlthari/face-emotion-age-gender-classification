import cv2
import threading
import time
import face_recognition
import numpy as np
import tensorflow as tf

thread = None
emo_model = tf.keras.models.load_model("./models/emotion_classifier.h5")
gen_model = tf.keras.models.load_model("./models/gender_classifier.h5")
age_model = tf.keras.models.load_model("./models/age_detection.h5")
genders = ('female', 'male')
ages = ('0-15', '16-25', '26-35', '36-45', '>46')
emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


def predict_gender_age(detected_face):
    #preprocess image
    detected_face_gen = tf.compat.v1.image.resize(detected_face, [192,192])#resize to 192x192
    face_gen = tf.keras.preprocessing.image.img_to_array(detected_face_gen)
    face_gen = np.expand_dims(face_gen, axis = 0)
    face_gen = tf.reshape(face_gen, [1,192,192,3])

    #predict gender
    gen_predictions = gen_model.predict(face_gen)
    age_predictions = age_model.predict(face_gen)
    
    # max_index_gen = np.argmax(gen_predictions[0])
    max_index_age = np.argmax(age_predictions[0])

    # gender = genders[max_index_gen]
    age = ages[max_index_age]
    return gender, age
    return age


def predict_emotion(detected_face):
    #preprocess image
    detected_face_emo = tf.image.rgb_to_grayscale(detected_face) #convert image to grayscale
    detected_face_emo = tf.compat.v1.image.resize(detected_face_emo, [48,48]) #resize image
    face_emo = tf.keras.preprocessing.image.img_to_array(detected_face_emo)
    face_emo = np.expand_dims(face_emo, axis = 0)
    face_emo = tf.reshape(face_emo, [1,48,48,1])


    face_emo = face_emo/255.0 #pixels are in scale of [0, 255]. normalize all pixels in scale of [0, 1]
    emo_predictions = emo_model.predict(face_emo) #store probabilities of 7 expressions

    #find max indexed array 0: angry, 1:disgust, 2:fear, 3:happy, 4:sad, 5:surprise, 6:neutral
    max_index = np.argmax(emo_predictions[0])
    emotion = emotions[max_index]
    return emotion


class Camera:
    def __init__(self,fps=20,video_source=0):
        self.fps = fps
        self.video_source = video_source
        self.camera = cv2.VideoCapture(self.video_source)
        # We want a max of 5s history to be stored, thats 5s*fps
        self.max_frames = 5 * self.fps
        self.frames = []
        self.gender = 'No result'
        self.age = 'No result'
        self.emotion = 'No result'
        self.isrunning = False
    def run(self):
        global thread
        if thread is None:
            thread = threading.Thread(target=self._capture_loop)
            print("Starting thread...")
            thread.start()
            self.isrunning = True
    def _capture_loop(self):
        dt = 1/self.fps
        print("Observing...")
        while self.isrunning:
            v,im = self.camera.read()
            faces = face_recognition.face_locations(im)

            for face_location in faces:
                # Print the location of each face in this image
                top, right, bottom, left = face_location
                detected_face = im[int(top-35):int(bottom+35), int(left-35):int(right+35)]
                self.emotion = predict_emotion(detected_face)
                cv2.rectangle(im, (left,top), (right, bottom), (0,255,0), 2)
                if self.gender == 'No result' and self.age == 'No result':
                    self.gender, self.age = predict_gender_age(detected_face)
                    self.age = predict_gender_age(detected_face)


            if v:
                if len(self.frames)==self.max_frames:
                    self.frames = self.frames[1:]
                    
                self.frames.append(im)
            time.sleep(dt)
    def stop(self):
        self.isrunning = False
    def get_frame(self, bytes=True):
        if len(self.frames)>0:
            if bytes:
                img = cv2.imencode('.png',self.frames[-1])[1].tobytes()
            else:
                img = self.frames[-1]
        else:
            with open("images/not_found.jpeg","rb") as f:
                img = f.read()
        return img
        