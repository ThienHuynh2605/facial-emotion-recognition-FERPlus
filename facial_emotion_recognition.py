import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Concatenate, Activation
from tensorflow.keras.models import Model
from loadData import loadData
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

emotion_classes = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
#-----------------------------------------------------------------------------------
X_train, y_train = loadData("FER2013Train")
X_val, y_val     = loadData("FER2013Valid")
X_test, y_test   = loadData("FER2013Test")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)

num_classes = y_train.shape[1]
print("Number of classes:", num_classes)

#-----------------------------------------------------------------------------------
model = build_model(input_shape=(64,64,1), num_classes=num_classes)
model.compile(
	optimizer=tf.keras.optimizers.Adam(1e-3),
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.summary()

#-----------------------------------------------------------------------------------
early_stop = EarlyStopping(
        monitor='val_loss',
        patience=4,
        restore_best_weights=True
)

#-----------------------------------------------------------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=2,     
    min_lr=1e-6,
    verbose=1
)

#-----------------------------------------------------------------------------------
history = model.fit(
	X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=5,
	batch_size=32,
    callbacks=[early_stop, lr_scheduler]
)

#------------------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

#------------------------------------------------------------------------------------
sample_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_sample = X_test[sample_idx]

pred = model.predict(X_sample)             # shape: (5, 8)
pred_class = np.argmax(pred, axis=1)       # index class
pred_emotions = [emotion_classes[i] for i in pred_class]

print("\nPrediction results:")
for i, emo in zip(sample_idx, pred_emotions):
    print(f"Sample {i}: {emo}")











