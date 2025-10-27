import os
import pandas as pd
import cv2
import numpy as np

def process_data(emotion_raw):
    size = len(emotion_raw)
    emotion_unknown = [0.0]*size
    emotion_unknown[-2] = 1.0

    # remove emotions with a single vote (outlier removal) 
    for i in range(size):
        if emotion_raw[i] < 1.0 + np.finfo(float).eps:
            emotion_raw[i] = 0.0
    
    sum_list = sum(emotion_raw)
    emotion  = [0.0]*size
    sum_part = 0
    count    = 0
    valid_emotion = True
    while sum_part < 0.75*sum_list and count < 3 and valid_emotion:
        maxval = max(emotion_raw) 
        for i in range(size): 
            if emotion_raw[i] == maxval: 
                emotion[i] = maxval
                emotion_raw[i] = 0
                sum_part += emotion[i]
                count += 1
                if i >= 8:  # unknown or non-face share same number of max votes 
                    valid_emotion = False
                    if sum(emotion) > maxval:   # there have been other emotions ahead of unknown or non-face
                        emotion[i] = 0
                        count -= 1
                    break
    if sum(emotion) <= 0.5*sum_list or count > 3: # less than 50% of the votes are integrated, or there are too many emotions, we'd better discard this example
        emotion = emotion_unknown   # force setting as unknown
    return np.array(emotion, dtype=float)

def loadData(split="FER2013Train"):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "dataset")
    label_path = os.path.join(data_path, "Labels", split, "label.csv")
    df = pd.read_csv(label_path, header=None)

    images_path = os.path.join(data_path, "Images", split)
    images_name = df.iloc[:, 0].values
    df_values = df.iloc[:, 2:].astype(float).values

    X, labels = [], []
    for name, row in zip(images_name, df_values):
        emotion = process_data(list(row))
        idx = np.argmax(emotion)
        if idx < 8:
            emotion = emotion[:-2] 
            emotion = emotion / np.sum(emotion)  
            labels.append(emotion)

            path = os.path.join(images_path, name)
            image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if image is not None:
                image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
                image = image.astype("float32") / 255.0
                X.append(image)

    X = np.expand_dims(np.array(X), axis=-1)
    labels = np.array(labels)
    return X, labels









































