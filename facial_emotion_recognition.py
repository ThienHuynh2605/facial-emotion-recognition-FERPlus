import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from loadData import loadData
from model import build_model


emotion_classes = ["neutral", "happiness", "surprise", "sadness", "anger", "disgust", "fear", "contempt"]
#-----------------------------------------------------------------------------------

X_train, y_train = loadData("FER2013Train")
X_val, y_val     = loadData("FER2013Valid")
X_test, y_test   = loadData("FER2013Test")
print(X_train.shape, y_train.shape)
print(X_val.shape, y_val.shape)
print(X_test.shape, y_test.shape)
#-----------------------------------------------------------------------------------

train_datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    brightness_range=[0.95, 1.05],
    horizontal_flip=True
)

# val_datagen = ImageDataGenerator()
#-----------------------------------------------------------------------------------

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
        patience=10,
        restore_best_weights=True,
        verbose=1
)

#-----------------------------------------------------------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=4,     
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
batch_size = 128
history = model.fit(
    train_datagen.flow(X_train, y_train, batch_size=batch_size),
    validation_data=(X_val, y_val),  # Hoặc dùng val_datagen.flow(X_val, y_val, batch_size=128)
    epochs=20,
    steps_per_epoch=len(X_train) // batch_size, 
    callbacks=[early_stop, lr_scheduler, checkpoint],
    verbose=1
)

#------------------------------------------------------------------------------------
test_loss, test_acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print("Test Accuracy:", test_acc)

#------------------------------------------------------------------------------------
sample_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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











