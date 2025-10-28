import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D,
    Concatenate, Activation, Dropout, MaxPooling2D, Dense
)
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation,
    RandomContrast, RandomBrightness
)

from tensorflow.keras.regularizers import l2


#------------------------------------------------------------------------------------------
def FireA(x, s, e, l2_reg):
    # Squeeze 1x1
    squeeze = Conv2D(s, (3,3), padding="same", kernel_regularizer=l2(l2_reg)) (x)
    squeeze = BatchNormalization()(squeeze)
    squeeze = ReLU()(squeeze)

    # Expand 1x1
    expand1 = Conv2D(e, (1,1), padding="same", kernel_regularizer=l2(l2_reg))(squeeze)
    expand1 = BatchNormalization()(expand1)
    expand1 = ReLU()(expand1)

    # Expand 3x3
    expand3 = Conv2D(e, (3,3), padding="same", kernel_regularizer=l2(l2_reg))(squeeze)
    expand3 = BatchNormalization()(expand3)
    expand3 = ReLU()(expand3)

    return Concatenate(axis=-1)([expand1, expand3])

#-----------------------------------------------------------------------------------------
def FireB(x, s, e, l2_reg):
    # Squeeze 1x1
    squeeze = Conv2D(s, (1,1), padding="same", kernel_regularizer=l2(l2_reg))(x)
    squeeze = BatchNormalization()(squeeze)
    squeeze = ReLU()(squeeze)

    # Expand 1x1
    expand1 = Conv2D(e, (1,1), padding="same", kernel_regularizer=l2(l2_reg))(squeeze)
    expand1 = BatchNormalization()(expand1)
    expand1 = ReLU()(expand1)

    # Expand 3x3
    expand3 = Conv2D(e, (3,3), padding="same", kernel_regularizer=l2(l2_reg))(squeeze)
    expand3 = BatchNormalization()(expand3)
    expand3 = ReLU()(expand3)

    return Concatenate(axis=-1)([expand1, expand3])

#------------------------------------------------------------------------------------------
def build_model(input_shape=(64,64,1), num_classes=8):
    inputs = Input(shape=input_shape)
    l2_reg = 1e-4
    # First Conv
    x = Conv2D(32, (3,3), padding="same", kernel_regularizer=l2(l2_reg))(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    # Fire blocks
    x = FireA(x, s=16, e=32, l2_reg=l2_reg)
    x = FireB(x, s=16, e=32, l2_reg=l2_reg)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(2)(x)

    x = FireA(x, s=32, e=64, l2_reg=l2_reg)
    x = FireB(x, s=32, e=64, l2_reg=l2_reg)
    x = Dropout(0.3)(x)
    x = MaxPooling2D(2)(x)

    x = FireA(x, s=64, e=128, l2_reg=l2_reg)
    x = FireB(x, s=64, e=128, l2_reg=l2_reg)
    x = Dropout(0.3)(x)
    x = MaxPooling2D(2)(x)

    # Final conv layers
    x = Conv2D(64, (3,3), padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(num_classes, (1,1), padding="same", kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)

    # GAP + Softmax
    x = GlobalAveragePooling2D()(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_reg))(x)
    x = Dropout(0.5)(x)

    outputs = tf.keras.layers.Activation("softmax")(x)

    model = Model(inputs, outputs)
    
    return model
















