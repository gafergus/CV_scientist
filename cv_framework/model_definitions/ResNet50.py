import keras
import gin

@gin.configurable
def ResNet50(input_shape=None, pooling=None, classes=3):
    """Instantiates the ResNet50 architecture."""
    img_input = keras.layers.Input(shape=input_shape)
    x = keras.layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = keras.layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal',
                            name='conv1')(x)
    x = keras.layers.BatchNormalization(axis=3, name='bn_conv1')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = keras.layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    if not pooling:
        x = keras.layers.GlobalAveragePooling2D(name='avg_pool')(x)
        x = keras.layers.Dense(classes, activation='softmax', name='fc')(x)
    else:
        if pooling == 'avg':
            x = keras.layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = keras.layers.GlobalMaxPooling2D()(x)
        else:
            raise ValueError('Pooling must be set to None, avg, or max. Current value is {}'.format(pooling))

    # Create model.
    model = keras.models.Model(img_input, x, name='resnet50')
    return model

def identity_block(input_tensor, kernel_size, filters, stage, block):
    '''The identity block is the block that has no conv layer at shortcut.'''
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = keras.layers.Conv2D(filters1, (1, 1), kernel_initializer='he_normal',
                            name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters2, kernel_size, padding='same',kernel_initializer='he_normal',
                            name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)

    x = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal',
                            name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    x = keras.layers.add([x, input_tensor])
    x = keras.layers.Activation('relu')(x)
    return x

def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    '''A block that has a conv layer at shortcut.'''
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    x = keras.layers.Conv2D(filters1, (1, 1), strides=strides, kernel_initializer='he_normal',
                            name=conv_name_base + '2a')(input_tensor)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2a')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2b')(x)
    x = keras.layers.Activation('relu')(x)
    x = keras.layers.Conv2D(filters3, (1, 1), kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '2c')(x)
    shortcut = keras.layers.Conv2D(filters3, (1, 1), strides=strides, kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = keras.layers.BatchNormalization(axis=3, name=bn_name_base + '1')(shortcut)
    x = keras.layers.add([x, shortcut])
    x = keras.layers.Activation('relu')(x)
    return x

