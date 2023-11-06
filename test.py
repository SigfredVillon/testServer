
import tensorflow as tf
from typing import List
import cv2
import os 
import dlib
import ffmpeg
from matplotlib import pyplot as plt 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten


def load_model() -> Sequential: 
    model = Sequential()

    model.add(Conv3D(128, 3, input_shape=(75,46,140,1), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(256, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(Conv3D(75, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPool3D((1,2,2)))

    model.add(TimeDistributed(Flatten()))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Bidirectional(LSTM(128, kernel_initializer='Orthogonal', return_sequences=True)))
    model.add(Dropout(.5))

    model.add(Dense(41, kernel_initializer='he_normal', activation='softmax'))

    model.load_weights('9models101/checkpoint')

    return model

def load_video(path:str) -> List[float]: 
    #print(path)
    pwd = os.getcwd()
    hog_face_detector = dlib.get_frontal_face_detector()

    dlib_facelandmark = dlib.shape_predictor(pwd + "/data/shape_predictor_68_face_landmarks.dat")
    cap = cv2.VideoCapture(path)
    frames = []

    
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))): 
        ret, frame = cap.read()
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = hog_face_detector(frame)

        #plt.imshow(gray2)
        face=faces[0]
        face_landmarks = dlib_facelandmark(gray, face)
                  
        # cv2.imshow('frames',frame)
        # start_point = ((face_landmarks.part(52).x) - 70, face_landmarks.part(52).y - 10)
        # end_point = ((face_landmarks.part(52).x) + 69, face_landmarks.part(52).y + 35)

        ytop=(face_landmarks.part(52).y)-10
        ybottom=(face_landmarks.part(52).y)+36
        xtop=(face_landmarks.part(52).x)-70
        xbottom=(face_landmarks.part(52).x)+70

        print(f'frame[{ytop}:{ybottom},{xtop}:{xbottom},:]')
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[ytop:ybottom,xtop:xbottom,:])
    cap.release()
    
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    print(f'mean:{mean}, std:{std}')
    print(f'frame-shape: {frames[0].shape}')
    cast = tf.cast((frames - mean), tf.float32) / std
    print(f'cast: {cast.shape}')
    return cast
    
    

def load_data(path: str): 
    path = bytes.decode(path.numpy())
    file_name = path.split('/')[-1].split('.')[0]
    # File name splitting for windows
    #file_name = path.split('\\')[-1].split('.')[0]
    video_path = os.path.join(f'{file_name}.mpg')
    frames = load_video(video_path) 

    
    return frames

vocab = [x for x in "abcdefghijklmnopqrstuvwxyz'?!123456789 "]
char_to_num = tf.keras.layers.StringLookup(vocabulary=vocab, oov_token="")
 # Mapping integers back to original characters
num_to_char = tf.keras.layers.StringLookup(
vocabulary=char_to_num.get_vocabulary(), oov_token="", invert=True)



video= load_data(tf.convert_to_tensor(os.path.join('bbaf2n.mpg')))

#print(tf.expand_dims(video, axis=0).shape)

model = load_model()
yhat = model.predict(tf.expand_dims(video, axis=0))
decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()


# Convert prediction to text
converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')

print(converted_prediction)