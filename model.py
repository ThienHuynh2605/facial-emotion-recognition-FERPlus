import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D,
    Concatenate, Activation, Dropout, MaxPooling2D
)
from tensorflow.keras.layers import (
    RandomFlip, RandomRotation, RandomZoom, RandomTranslation,
    RandomContrast, RandomBrightness
)


#------------------------------------------------------------------------------------------
def FireA(x, s, e):
    # Squeeze 1x1
    squeeze = Conv2D(s, (3,3), padding="same") (x)
    squeeze = BatchNormalization()(squeeze)
    squeeze = ReLU()(squeeze)

    # Expand 1x1
    expand1 = Conv2D(e, (1,1), padding="same")(squeeze)
    expand1 = BatchNormalization()(expand1)
    expand1 = ReLU()(expand1)

    # Expand 3x3
    expand3 = Conv2D(e, (3,3), padding="same")(squeeze)
    expand3 = BatchNormalization()(expand3)
    expand3 = ReLU()(expand3)

    return Concatenate(axis=-1)([expand1, expand3])

#-----------------------------------------------------------------------------------------
def FireB(x, s, e):
    # Squeeze 1x1
    squeeze = Conv2D(s, (1,1), padding="same")(x)
    squeeze = BatchNormalization()(squeeze)
    squeeze = ReLU()(squeeze)

    # Expand 1x1
    expand1 = Conv2D(e, (1,1), padding="same")(squeeze)
    expand1 = BatchNormalization()(expand1)
    expand1 = ReLU()(expand1)

    # Expand 3x3
    expand3 = Conv2D(e, (3,3), padding="same")(squeeze)
    expand3 = BatchNormalization()(expand3)
    expand3 = ReLU()(expand3)

    return Concatenate(axis=-1)([expand1, expand3])

#------------------------------------------------------------------------------------------
def build_model(input_shape=(64,64,1), num_classes=8):
    inputs = Input(shape=input_shape)

    # First Conv
    x = Conv2D(64, (3,3), padding="same")(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    # Fire blocks
    x = FireA(x, s=32, e=64)
    x = FireB(x, s=32, e=64)
    x = Dropout(0.1)(x)
    x = MaxPooling2D(2)(x)

    x = FireA(x, s=64, e=128)
    x = FireB(x, s=64, e=128)
    x = Dropout(0.3)(x)
    x = MaxPooling2D(2)(x)

    x = FireA(x, s=128, e=256)
    x = FireB(x, s=128, e=256)
    x = Dropout(0.3)(x)
    x = MaxPooling2D(2)(x)

    # Final conv layers
    x = Conv2D(64, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(2)(x)

    x = Conv2D(num_classes, (1,1), padding="same")(x)
    x = Dropout(0.5)(x)

    # GAP + Softmax
    x = GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    model = Model(inputs, outputs)
    
    return model






























