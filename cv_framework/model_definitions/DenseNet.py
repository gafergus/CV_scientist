import tensorflow as tf


def DenseNet(blocks, input_shape=None, pooling=None, classes=3):
    '''Instantiates the DenseNet architecture.'''
    img_input =  tf.keras.layers.Input(shape=input_shape)
    x =  tf.keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x =  tf.keras.layers.Conv2D(64, 7, strides=2, use_bias=False, kernel_initializer='he_normal', name='conv1/conv')(x)
    x =  tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1/bn')(x)
    x =  tf.keras.layers.Activation('relu', name='conv1/relu')(x)
    x =  tf.keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x =  tf.keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x =  tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
    x =  tf.keras.layers.Activation('relu', name='relu')(x)

    if not pooling:
        x = tf.keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        mod_output = tf.keras.layers.Dense(classes, activation='softmax', name='fc')(x)
    elif pooling == 'avg':
        mod_output = tf.keras.layers.GlobalAveragePooling2D()(x)
    elif pooling == 'max':
        mod_output = tf.keras.layers.GlobalMaxPooling2D()(x)
    else:
        raise ValueError('Pooling must be set to None, avg, or max. Current value: {}'.format(pooling))

    # Create model.
    if blocks == [6, 12, 24, 16]:
        return tf.keras.models.Model(img_input, mod_output, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        return tf.keras.models.Model(img_input, mod_output, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        return tf.keras.models.Model(img_input, mod_output, name='densenet201')
    else:
        return tf.keras.models.Model(img_input, mod_output, name='densenet')

def DenseNet121(input_shape=None, classes=3):
    return DenseNet([6, 12, 24, 16], input_shape=input_shape, classes=classes)

def DenseNet169(input_shape=None, classes=3):
    return DenseNet([6, 12, 32, 32], input_shape=input_shape, classes=classes)

def DenseNet201(input_shape=None,classes=3):
    return DenseNet([6, 12, 48, 32], input_shape=input_shape, classes=classes)

def DenseNetCustom(blocks, input_shape=None, classes=3):
    if not isinstance(blocks, list):
        raise ValueError('Blocks must be a list')
    elif len(blocks) != 4 or all(type(x) is int for x in blocks):
        raise ValueError('Blocks must be a list of four integers!')
    else:
        return DenseNet(blocks, input_shape=input_shape, classes=classes)

def dense_block(x, blocks, name):
    '''A dense block.'''
    for i in range(blocks):
        x = conv_block(x, 32, name=f'{name}_block{str(i + 1)}')
    return x

def transition_block(x, reduction, name):
    '''A transition blocki'''
    x = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_bn')(x)
    x = tf.keras.layers.Activation('relu', name=f'{name}_relu')(x)
    x = tf.keras.layers.Conv2D(int(tf.keras.backend.int_shape(x)[3] * reduction), 1, use_bias=False,
                            kernel_initializer='he_normal', name=name + '_conv')(x)
    x = tf.keras.layers.AveragePooling2D(2, strides=2, name=f'{name}_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    '''A building block for a dense block.'''
    x1 = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = tf.keras.layers.Activation('relu', name=f'{name}_0_relu')(x1)
    x1 = tf.keras.layers.Conv2D(
        4 * growth_rate,
        1,
        use_bias=False,
        kernel_initializer='he_normal',
        name=f'{name}_1_conv',
    )(x1)

    x1 = tf.keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = tf.keras.layers.Activation('relu', name=f'{name}_1_relu')(x1)
    x1 = tf.keras.layers.Conv2D(
        growth_rate,
        3,
        padding='same',
        use_bias=False,
        kernel_initializer='he_normal',
        name=f'{name}_2_conv',
    )(x1)

    x = tf.keras.layers.Concatenate(axis=3, name=f'{name}_concat')([x, x1])
    return x
