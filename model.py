from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, GlobalAveragePooling2D, Concatenate, Activation, Dropout, RandomFlip, RandomRotation, RandomZoom, RandomTranslation,RandomContrast, RandomBrightness
from tensorflow.keras.models import Model
import tensorflow as tf

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

    x = RandomFlip("horizontal")(inputs)
    x = RandomRotation(0.1)(x)      
    x = RandomZoom(0.1)(x)              
    x = RandomTranslation(0.1,0.1)(x)   
    x = RandomContrast(0.1)(x)      
    x = RandomBrightness(0.1)(x)  

    # First Conv
    x = Conv2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Fire blocks
    x = FireA(x, s=16, e=32)
    x = FireB(x, s=16, e=32)
    x = FireA(x, s=32, e=64)
    x = FireB(x, s=32, e=64)
    x = Dropout(0.2)(x)
    x = FireA(x, s=64, e=128)
    x = FireB(x, s=64, e=128)
    x = Dropout(0.5)(x)

    # Final conv layers
    x = Conv2D(32, (3,3), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(num_classes, (1,1), padding="same")(x)
    x = Dropout(0.5)(x)

    # GAP + Softmax
    x = GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Activation("softmax")(x)

    model = Model(inputs, outputs)
    
    return model






























