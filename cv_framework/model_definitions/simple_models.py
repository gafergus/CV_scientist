import tensorflow as tf
import gin

@gin.configurable
def simple_CNN(input_shape=(None, None, 1), classes=3):
    ''' A simple CNN for testing'''
    input     = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='scnn_input')
    conv1     = tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(input)
    max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv1)
    conv2     = tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                    kernel_initializer='he_uniform')(max_pool2)
    flatten   = tf.keras.layers.Flatten()(conv3)
    dense1    = tf.keras.layers.Dense(64, activation='relu', use_bias=True,
                                   kernel_initializer='he_uniform')(flatten)
    pred      = tf.keras.layers.Dense(classes, activation='softmax', use_bias=True,
                                   kernel_initializer='he_uniform')(dense1)
    return tf.keras.Model(inputs=input, outputs=pred)

@gin.configurable
def simple_sep_CNN(input_shape=(None, None, 1), classes=3):
    ''' A simple CNN for testing'''
    input     = tf.keras.layers.Input(shape=(input_shape[0], input_shape[1], input_shape[2]), name='scnn_input')
    conv1     = tf.keras.layers.SeparableConv2D(filters=128, kernel_size=(5, 5), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(input)
    max_pool1 = tf.keras.layers.MaxPooling2D(pool_size=(3,3))(conv1)
    conv2     = tf.keras.layers.SeparableConv2D(filters=64, kernel_size=(3, 3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool1)
    max_pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2,2))(conv2)
    conv3     = tf.keras.layers.SeparableConv2D(filters=32, kernel_size=(3,3), activation='relu',use_bias=True,
                                             kernel_initializer='he_uniform')(max_pool2)
    flatten   = tf.keras.layers.Flatten()(conv3)
    dense1    = tf.keras.layers.Dense(64, activation='relu', use_bias=True, kernel_initializer='he_uniform')(flatten)
    pred      = tf.keras.layers.Dense(classes, activation='softmax', use_bias=True,
                                   kernel_initializer='he_uniform')(dense1)
    return tf.keras.Model(inputs=input, outputs=pred)

