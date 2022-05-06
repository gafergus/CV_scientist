import tensorflow as tf


def Xception(input_shape=None, pooling=None,classes=3):
    '''Instantiates the Xception architecture.'''
    img_input = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, kernel_initializer='he_normal',
                            name='block1_conv1')(img_input)
    x = tf.keras.layers.BatchNormalization(name='block1_conv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv1_act')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), use_bias=False, kernel_initializer='he_normal', name='block1_conv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block1_conv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block1_conv2_act')(x)

    residual = tf.keras.layers.Conv2D(128, (1, 1), strides=(2, 2), padding='same', kernel_initializer='he_normal',
                                   use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block2_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block2_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block2_sepconv2_act')(x)

    x = tf.keras.layers.SeparableConv2D(128, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block2_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block2_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = tf.keras.layers.add([x, residual])

    residual = tf.keras.layers.Conv2D(256, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block3_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block3_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block3_sepconv1_bn')(x)

    x = tf.keras.layers.Activation('relu', name='block3_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(256, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block3_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block3_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same',
                            name='block3_pool')(x)
    x = tf.keras.layers.add([x, residual])

    residual = tf.keras.layers.Conv2D(728, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block4_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block4_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block4_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block4_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block4_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block4_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = tf.keras.layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = f'block{str(i + 5)}'
        x = tf.keras.layers.Activation('relu', name=f'{prefix}_sepconv1_act')(x)

        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                         pointwise_initializer='he_normal', name=prefix + '_sepconv1')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_sepconv1_bn')(x)
        x = tf.keras.layers.Activation('relu', name=f'{prefix}_sepconv2_act')(x)

        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                         pointwise_initializer='he_normal', name=prefix + '_sepconv2')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_sepconv2_bn')(x)
        x = tf.keras.layers.Activation('relu', name=f'{prefix}_sepconv3_act')(x)

        x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                         pointwise_initializer='he_normal', name=prefix + '_sepconv3')(x)
        x = tf.keras.layers.BatchNormalization(name=f'{prefix}_sepconv3_bn')(x)
        x = tf.keras.layers.add([x, residual])

    residual = tf.keras.layers.Conv2D(1024, (1, 1), strides=(2, 2), padding='same', use_bias=False)(x)
    residual = tf.keras.layers.BatchNormalization()(residual)

    x = tf.keras.layers.Activation('relu', name='block13_sepconv1_act')(x)
    x = tf.keras.layers.SeparableConv2D(728, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block13_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block13_sepconv1_bn')(x)

    x = tf.keras.layers.Activation('relu', name='block13_sepconv2_act')(x)
    x = tf.keras.layers.SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block13_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block13_sepconv2_bn')(x)

    x = tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = tf.keras.layers.add([x, residual])

    x = tf.keras.layers.SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block14_sepconv1')(x)
    x = tf.keras.layers.BatchNormalization(name='block14_sepconv1_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block14_sepconv1_act')(x)

    x = tf.keras.layers.SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, depthwise_initializer='he_normal',
                                     pointwise_initializer='he_normal', name='block14_sepconv2')(x)
    x = tf.keras.layers.BatchNormalization(name='block14_sepconv2_bn')(x)
    x = tf.keras.layers.Activation('relu', name='block14_sepconv2_act')(x)

    if not pooling:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = tf.keras.layers.Dense(classes, activation='softmax', name='predictions')(x)
    elif pooling == 'avg':
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        x = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        raise ValueError('Pooling must be None, avg, or max. Current value is {}'.format(pooling))

    return tf.keras.models.Model(img_input, x, name='xception')

