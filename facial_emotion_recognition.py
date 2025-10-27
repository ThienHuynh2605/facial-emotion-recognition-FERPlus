import os
import pandas as pd
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Concatenate, Activation
from tensorflow.keras.models import Model
from loadData import loadData
from model import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
        patience=50,
        restore_best_weights=True
)

#-----------------------------------------------------------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=10,     
    min_lr=1e-6
)

#---------------------------------------------------------------------------------
checkpoint = ModelCheckpoint(
    "best_model.keras",
    monitor="val_loss",     
    save_best_only=True,    
    verbose=2
)

#-----------------------------------------------------------------------------------
history = model.fit(
	X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=200,
	batch_size=64,
    callbacks=[early_stop, lr_scheduler, checkpoint]
)

#------------------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print("Test Accuracy:", test_acc)

# ---------------------- Save original model ----------------------
saved_model_dir = "saved_model"
model.save(saved_model_dir, save_format="tf")
print(f"Saved original model to: {saved_model_dir}")

# ---------------------- Convert model to TensorRT (FP16) ----------------------
from tensorflow.python.compiler.tensorrt import trt_convert as trt

params = trt.DEFAULT_TRT_CONVERSION_PARAMS._replace(precision_mode='FP16')
converter = trt.TrtGraphConverterV2(input_saved_model_dir=saved_model_dir, conversion_params=params)
converter.convert()
trt_saved_model_dir = "saved_model_trt"
converter.save(trt_saved_model_dir)
print(f"Converted model to TensorRT at: {trt_saved_model_dir}")

# ---------------------- Inference using TensorRT ----------------------
trt_model = tf.saved_model.load(trt_saved_model_dir)
infer = trt_model.signatures["serving_default"]

#------------------------------------------------------------------------------------
sample_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
X_sample = X_test[sample_idx]

# TensorRT model output
pred = infer(tf.convert_to_tensor(X_sample))["dense_1"]  # "dense_1" là tên layer output
pred_class = np.argmax(pred.numpy(), axis=1)
pred_emotions = [emotion_classes[i] for i in pred_class]

print("\nPrediction results:")
for i, emo in zip(sample_idx, pred_emotions):
    print(f"Sample {i}: {emo}")











