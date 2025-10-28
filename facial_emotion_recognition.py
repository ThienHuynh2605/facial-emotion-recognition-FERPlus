import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
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
initial_lr=5e-4
steps_per_epoch = len(X_train)
total_steps = epochs * steps_per_epoch
warmup_steps = 5 * steps_per_epoch 
    
def lr_schedule(step):
    min_lr = 1e-6  
    if step < warmup_steps:
        return min_lr + (initial_lr - min_lr) * (step / warmup_steps)
    else:
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return initial_lr * 0.5 * (1 + tf.cos(3.14159 * progress))
        
model.compile(
    optimizer=tf.keras.optimizers.AdamW(
        learning_rate=initial_lr,
        weight_decay=1e-4
    ),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_acc')]
)

model.summary()

#-----------------------------------------------------------------------------------
# Callbacks
callbacks = [
    tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=0),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=20,  # Increased patience
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=8,
        min_lr=1e-7,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        verbose=1
    )
]

#-----------------------------------------------------------------------------------
history = model.fit(
	X_train, y_train,
	validation_data=(X_val, y_val),
	epochs=epochs,
	batch_size=32,
    callbacks=callbacks,
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











