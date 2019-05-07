import keras
from keras.backend import tf as ktf

def Multiscale_CNN(input_shape=None, classes=3):
    '''Instantiates the Multiscale CNN architecture.'''
    img_input = keras.layers.Input(shape=input_shape)
    # Scale 1  network
    scale_01_conv=conv2d(img_input, filters=[16,16,16], kernel_size=(5,5), strides=1, pad=2, activation='relu',
                         pool_kernel=64,pool_stride=64,name='scl_01',scale_shape=(input_shape[0]//1,input_shape[1]//1))
    # Scale 2  network
    scale_02_conv=conv2d(img_input, filters=[16,16,16], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=32,pool_stride=32,name='scl_02',scale_shape=(input_shape[0]//2,input_shape[1]//2))
    # Scale 4  network
    scale_04_conv=conv2d(img_input, filters=[16,16,16], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=16,pool_stride=16,name='scl_04',scale_shape=(input_shape[0]//4,input_shape[1]//4))
    # Scale 8  network
    scale_08_conv=conv2d(img_input, filters=[32,32,32], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=8,pool_stride=8,name='scl_08',scale_shape=(input_shape[0]//8,input_shape[1]//8))
    # Scale 16 network
    scale_16_conv=conv2d(img_input, filters=[32,32,32], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=4,pool_stride=4,name='scl_16',scale_shape=(input_shape[0]//16,input_shape[1]//16))
    # Scale 32 network
    scale_32_conv=conv2d(img_input, filters=[32,32,32], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=2,pool_stride=2,name='scl_32',scale_shape=(input_shape[0]//32,input_shape[1]//32))
    # Scale 64 network
    scale_64_conv=conv2d(img_input, filters=[64,64,64], kernel_size=(5 , 5), strides=1, pad=2, activation='relu',
                         pool_kernel=1,pool_stride=1,name='scl_64',scale_shape=(input_shape[0]//64,input_shape[1]//64))
    # Concatenate subnetworks
    x = keras.layers.concatenate([scale_01_conv, scale_02_conv, scale_04_conv, scale_08_conv, scale_16_conv,
                                  scale_32_conv, scale_64_conv])
    # Final convolution
    x = keras.layers.Conv2D(1024, (1, 1), strides=1, kernel_initializer='he_normal',
                            name='conv2d_final')(x)
    x = keras.layers.Activation('relu', name= 'act_conv_final')(x)
    x = keras.layers.MaxPool2D(pool_size=2, strides=2)(x)
    # Add dense layer
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(512, use_bias=True, kernel_initializer='he_uniform')(x)
    x = keras.layers.Activation('relu', name= 'act_dense')(x)
    # prediction layer
    pred = keras.layers.Dense(classes, use_bias=True, kernel_initializer='he_uniform')(x)
    pred = keras.layers.Activation('softmax')(pred)
    model = keras.Model(inputs=img_input, outputs=pred)
    return model

def conv2d(x, filters=None, kernel_size=None,  scale_shape=None, strides=1, pad=2, activation='relu', pool_kernel=None,
           pool_stride=None, name=None):
    '''Apply the convolutional to a scale.'''
    act_base = 'act_'
    bn_base  = 'bn_'
    conv_base = 'conv2d_'
    x = keras.layers.Lambda(lambda image: ktf.image.resize_images(image, scale_shape))(x)
    x = keras.layers.ZeroPadding2D(padding=pad)(x)
    x = keras.layers.Conv2D(filters[0], kernel_size, strides=strides, kernel_initializer='he_normal',
                            name= conv_base + name + '1')(x)
    x = keras.layers.BatchNormalization(name=bn_base + name + '1')(x)
    x = keras.layers.Activation(activation, name= act_base + name + '1')(x)

    x = keras.layers.ZeroPadding2D(padding=pad)(x)
    x = keras.layers.Conv2D(filters[1], kernel_size, strides=strides, kernel_initializer='he_normal',
                            name= conv_base + name + '2')(x)
    x = keras.layers.BatchNormalization(name=bn_base + name + '2')(x)
    x = keras.layers.Activation(activation, name= act_base + name + '2')(x)

    x = keras.layers.ZeroPadding2D(padding=pad)(x)
    x = keras.layers.Conv2D(filters[1], kernel_size, strides=strides, kernel_initializer='he_normal',
                            name= conv_base + name + '3')(x)
    x = keras.layers.BatchNormalization(name=bn_base + name + '3')(x)
    x = keras.layers.Activation(activation, name = act_base + name + '3')(x)
    x = keras.layers.MaxPool2D(pool_size=pool_kernel, strides=pool_stride, name= 'pool_' + name)(x)
    return x

