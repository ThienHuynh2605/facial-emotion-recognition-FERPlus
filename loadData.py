import os
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt

def process_data(emotion_raw, emotion_unknown):
    emotion_raw = emotion_raw.copy()
    sum_list = sum(emotion_raw)
    
    emotion = np.zeros_like(emotion_raw)
    sum_part = 0
    count = 0
    valid_emotion = True
    size = len(emotion_raw)
    
    while sum_part < 0.75 * sum_list and count < 3 and valid_emotion:
        maxval = max(emotion_raw)
        for i in range(size):
            if emotion_raw[i] == maxval:
                emotion[i] = maxval
                emotion_raw[i] = 0
                sum_part += emotion[i]
                count += 1
                if i >= 8:  # unknown hoặc non-face
                    valid_emotion = False
                    if sum(emotion) > maxval:
                        emotion[i] = 0
                        count -= 1
                    break
        if sum(emotion) <= 0.5 * sum_list or count > 3:
            emotion = emotion_unknown.copy()  
    return [float(i)/sum(emotion) for i in emotion]

def loadData(split = "FER2013Train"):
    current_path = os.getcwd()
    data_path = os.path.join(current_path, "dataset")
    label_path = os.path.join(data_path, "Labels", split, "label.csv")
    df = pd.read_csv(label_path, header=None)

    images_path = os.path.join(data_path, "Images", split)
    images_name = df.iloc[:, 0].values
    df_values = df.iloc[:, 2:].astype(float).values
    df_values[df_values == 1] = 0
    emotion_unknown = np.zeros(size)
    emotion_unknown[-2] = 1.0
    filtered = np.array([process_data(row, emotion_unknown) for row in df_values])
    
    labels = []
    for i in filtered:
        idx = np.random.choice(len(i), p=i)
        new_targets = np.zeros_like(i)
        new_targets[idx] = 1.0
        labels.append(new_targets)
    labels = np.array(labels)

    X=[]
    for name in images_name:
        path = os.path.join(images_path, name)
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image_resize = cv2.resize(image, (64, 64), interpolation=cv2.INTER_LINEAR)
        image_resize = image_resize.astype("float32")/255.0
        X.append(image_resize)
    X = np.array(X)
    X = np.expand_dims(X, axis=-1)

    return X, labels









































