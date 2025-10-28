import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

from loadData import loadData
from model import build_model
from load_plotter import LossPlotter


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
epochs = 200
model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
)

model.summary()

#-----------------------------------------------------------------------------------
early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True,
        verbose=1
)

#-----------------------------------------------------------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=7,     
    min_lr=1e-6,
    verbose=1
)

#---------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",     
    save_best_only=True,
    verbose=1
)

#-----------------------------------------------------------------------------------
history = model.fit(
	X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=epochs,
	batch_size=32,
    callbacks=[lr_scheduler, early_stop, checkpoint],
    verbose=1
)

#------------------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

#------------------------------------------------------------------------------------
plotter = LossPlotter(history)
plotter.plot() 

#------------------------------------------------------------------------------------
sample_idx = list(range(0, 21))
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

pred = model.predict(X_sample) 
pred_class = np.argmax(pred, axis=1) 
true_class = np.argmax(y_sample, axis=1)

pred_emotions = [emotion_classes[i] for i in pred_class]
true_emotions = [emotion_classes[i] for i in true_class]

print("\nPrediction results vs Ground Truth:")
for i, pred_emo, true_emo in zip(sample_idx, pred_emotions, true_emotions):
    print(f"Sample {i}: Predicted = {pred_emo}, {pred_emo==true_emo} = {true_emo}")











