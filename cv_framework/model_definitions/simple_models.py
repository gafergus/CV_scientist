import keras
import gin

@gin.configurable
def simple_CNN(input_shape=(None, None, 1), classes=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='scnn_input')
    conv1     = keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2     = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool2)
    flatten   = keras.layers.Flatten()(conv3)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True,
                                   kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(classes, activation='softmax', use_bias=True,
                                   kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model

@gin.configurable
def simple_sep_CNN(input_shape=(None, None, 1), classes=3):
    ''' A simple CNN for testing'''
    input     = keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='scnn_input')
    conv1     = keras.layers.SeparableConv2D(filters=128, kernel_size=(5, 5), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(input)
    max_pool1 = keras.layers.MaxPooling2D(pool_size=(3,3))(conv1)
    conv2     = keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool2)
    flatten   = keras.layers.Flatten()(conv3)
    dense1    = keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = keras.layers.Dense(classes, activation='softmax', use_bias=True,
                                   kernel_initializer='he_uniform')(dense1)
    model = keras.Model(inputs=input, outputs=pred)
    return model

