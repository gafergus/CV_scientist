import keras

def DenseNet(blocks, input_shape=None, pooling=None, classes=3):
    '''Instantiates the DenseNet architecture.'''
    img_input =  keras.layers.Input(shape=input_shape)
    x =  keras.layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(img_input)
    x =  keras.layers.Conv2D(64, 7, strides=2, use_bias=False, kernel_initializer='he_normal', name='conv1/conv')(x)
    x =  keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='conv1/bn')(x)
    x =  keras.layers.Activation('relu', name='conv1/relu')(x)
    x =  keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x =  keras.layers.MaxPooling2D(3, strides=2, name='pool1')(x)

    x = dense_block(x, blocks[0], name='conv2')
    x = transition_block(x, 0.5, name='pool2')
    x = dense_block(x, blocks[1], name='conv3')
    x = transition_block(x, 0.5, name='pool3')
    x = dense_block(x, blocks[2], name='conv4')
    x = transition_block(x, 0.5, name='pool4')
    x = dense_block(x, blocks[3], name='conv5')

    x =  keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name='bn')(x)
    x =  keras.layers.Activation('relu', name='relu')(x)

    if not pooling:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        mod_output = keras.layers.Dense(classes, activation='softmax', name='fc')(x)
    else:
        if pooling == 'avg':
            mod_output = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            mod_output = keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise ValueError('Pooling must be set to None, avg, or max. Current value: {}'.format(pooling))

    # Create model.
    if blocks == [6, 12, 24, 16]:
        model = keras.models.Model(img_input, mod_output, name='densenet121')
    elif blocks == [6, 12, 32, 32]:
        model = keras.models.Model(img_input, mod_output, name='densenet169')
    elif blocks == [6, 12, 48, 32]:
        model = keras.models.Model(img_input, mod_output, name='densenet201')
    else:
        model = keras.models.Model(img_input, mod_output, name='densenet')
    return model

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
        x = conv_block(x, 32, name=name + '_block' + str(i + 1))
    return x

def transition_block(x, reduction, name):
    '''A transition blocki'''
    x = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_bn')(x)
    x = keras.layers.Activation('relu', name=name + '_relu')(x)
    x = keras.layers.Conv2D(int(keras.backend.int_shape(x)[3] * reduction), 1, use_bias=False,
                            kernel_initializer='he_normal', name=name + '_conv')(x)
    x = keras.layers.AveragePooling2D(2, strides=2, name=name + '_pool')(x)
    return x

def conv_block(x, growth_rate, name):
    '''A building block for a dense block.'''
    x1 = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_0_bn')(x)
    x1 = keras.layers.Activation('relu', name=name + '_0_relu')(x1)
    x1 = keras.layers.Conv2D(4 * growth_rate, 1, use_bias=False, kernel_initializer='he_normal',
                             name=name + '_1_conv')(x1)
    x1 = keras.layers.BatchNormalization(axis=3, epsilon=1.001e-5, name=name + '_1_bn')(x1)
    x1 = keras.layers.Activation('relu', name=name + '_1_relu')(x1)
    x1 = keras.layers.Conv2D(growth_rate, 3, padding='same', use_bias=False,  kernel_initializer='he_normal',
                             name=name + '_2_conv')(x1)
    x  = keras.layers.Concatenate(axis=3, name=name + '_concat')([x, x1])
    return x
