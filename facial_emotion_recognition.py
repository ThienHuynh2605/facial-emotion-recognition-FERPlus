import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import (RandomFlip, RandomRotation, RandomZoom, RandomTranslation, RandomContrast, RandomBrightness)

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

data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal"),
    # RandomRotation(0.02),           # 0.03 * 2π ≈ 10 độ
    # RandomZoom(0.05),
    # RandomTranslation(0.01, 0.01),
    # RandomContrast(0.03),
    RandomBrightness(0.01)
], name='data_augmentation')

# val_datagen = ImageDataGenerator()
#-----------------------------------------------------------------------------------

num_classes = y_train.shape[1]
print("Number of classes:", num_classes)

#-----------------------------------------------------------------------------------
model = build_model(input_shape=(64,64,1), num_classes=num_classes)
model.compile(
	optimizer=tf.keras.optimizers.Adam(1e-2),
	loss='categorical_crossentropy',
	metrics=['accuracy']
)

model.summary()

#-----------------------------------------------------------------------------------
early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True,
        verbose=1
)

#-----------------------------------------------------------------------------------
lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,     
    min_lr=1e-5,
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
BATCH_SIZE = 128

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.shuffle(buffer_size=1000)
train_ds = train_ds.batch(BATCH_SIZE)
train_ds = train_ds.map(
    lambda x, y: (data_augmentation(x, training=True), y),
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_ds = val_ds.batch(BATCH_SIZE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop, lr_scheduler, checkpoint],
    verbose=1
)

#------------------------------------------------------------------------------------
# Evaluate
test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.batch(BATCH_SIZE)
test_loss, test_accuracy = model.evaluate(test_ds)
print(f"Test Accuracy: {test_accuracy:.4f}")

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











