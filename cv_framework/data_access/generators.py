import tensorflow as tf
import gin

@gin.configurable
def image_generator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-06,
        rotation_range=0,
        width_shift_range=0.0,
        height_shift_range=0.0,
        brightness_range=None,
        shear_range=0.0,
        zoom_range=0.0,
        channel_shift_range=0.0,
        fill_mode='nearest',
        cval=0.0,
        horizontal_flip=False,
        vertical_flip=False,
        rescale=1./255,
        preprocessing_function=None,
        data_format=None,
        validation_split=0.0):
    return tf.keras.preprocessing.image.ImageDataGenerator(
        featurewise_center=featurewise_center,
        samplewise_center=samplewise_center,
        featurewise_std_normalization=featurewise_std_normalization,
        samplewise_std_normalization=samplewise_std_normalization,
        zca_whitening=zca_whitening,
        zca_epsilon=zca_epsilon,
        rotation_range=rotation_range,
        width_shift_range=width_shift_range,
        height_shift_range=height_shift_range,
        brightness_range=brightness_range,
        shear_range=shear_range,
        zoom_range=zoom_range,
        channel_shift_range=channel_shift_range,
        fill_mode=fill_mode,
        cval=cval,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
        rescale=rescale,
        preprocessing_function=preprocessing_function,
        data_format=data_format,
        validation_split=validation_split)

@gin.configurable
def directory_flow(dir='.', image_size=None, batch_size=32, color_mode='greyscale',
                   class_mode='categorical', shuffle=False, seed=42, interpolation='nearest'):
    gen = image_generator()
    return gen.flow_from_directory(
        directory=dir,
        target_size=image_size,
        batch_size=batch_size,
        color_mode=color_mode,
        class_mode=class_mode,
        shuffle=shuffle,
        seed=seed,
        interpolation=interpolation)