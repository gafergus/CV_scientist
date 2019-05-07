############## Module is currently not used!!! #############################
import keras
from keras.callbacks import Callback
from keras import backend as K
import tensorflow as tf
import numpy as np
import sklearn
import gin
import math
from PIL import Image

class ConfusionMatrix(Callback):
    def __init__(self, val_gen):
        # !!!! A bug in keras keeps self.validation (in the Callback class) from ever being set with a generator,
        # used classes from the validation generator instead
        super().__init__()
        val_gen.reset()
        self.validation_data = val_gen
        self.validation_labels = val_gen.classes
        self.val_batch = val_gen.batch_size
        self.class_indices = val_gen.class_indices

    def on_epoch_end(self, epoch, logs=None):
        print("Calculating confusion matrix")
        #print(vars(vars(self.validation_data)['image_data_generator']))
        num_val_gen_steps = math.ceil(len(self.validation_labels)/self.val_batch)
        #black_field_001.png  black_field_023.png  black_field_034.png  black_field_044.png  black_field_050.png
        #black_field_007.png  black_field_025.png  black_field_042.png  black_field_047.png  black_field_071.png
        #
        #checkerboard_060.png  checkerboard_073.png  checkerboard_078.png  checkerboard_081.png  checkerboard_089.png
        #checkerboard_063.png  checkerboard_075.png  checkerboard_079.png  checkerboard_085.png  checkerboard_092.png
        #
        #white_field_014.png  white_field_032.png  white_field_055.png  white_field_067.png  white_field_074.png
        #white_field_023.png  white_field_037.png  white_field_066.png  white_field_070.png  white_field_500.png
        #
        #im = Image.open('/data/glferguso/cv_unit_test/images/test/class_002_2/white_field_037.png')
        #im = np.array(im.resize((128,128))) 
        #im = np.reshape(im, (1, 128, 128, 1))
        predicted_1 = self.model.predict_generator(self.validation_data, verbose=1, steps=num_val_gen_steps, )
        datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255) 
        generator = datagen.flow_from_directory(
                '/data/glferguso/cv_unit_test/images/test',
                target_size=(128, 128),
                batch_size=10,
                color_mode='grayscale',
                class_mode='categorical',  # only data, no labels
                seed=42,
                interpolation='nearest',
                shuffle=False)  # keep data in same order as labels
                                      
        probabilities = self.model.predict_generator(generator, 3) 

        #print(vars(self.validation_data))
        #print('\n')
        #print(vars(generator))

        predicted = np.argmax(probabilities, axis=1)
        predicted_1 = np.argmax(predicted_1, axis=1)
        ground = self.validation_labels
#        #print('predicted = {}'.format(predicted))
        print(predicted)
        print(predicted_1)
        print(ground)
#        cm = sklearn.metrics.confusion_matrix(ground, predicted, labels=None, sample_weight=None)
        #print(cm)
#        template = "{0:10}|{1:30}|{2:10}|{3:30}|{4:15}|{5:15}"
#        print(template.format("", "", "", "Predicted", "", ""))
#        print(template.format("", "", "Normal", "No Lung Opacity / Not Normal", "Lung Opacity", "Total true"))
#        print(template.format("", "="*28, "="*9, "="*28, "="*12, "="*12))
#        print(template.format("", "Normal",
#                              cm[0, 0], cm[0, 1], cm[0, 2], np.sum(cm[0, :])))
#        print(template.format("True", "No Lung Opacity / Not Normal",
#                              cm[1, 0], cm[1, 1], cm[1, 2], np.sum(cm[1, :])))
#        print(template.format("", "Lung Opacity",
#                              cm[2, 0], cm[2, 1], cm[2, 2], np.sum(cm[2, :])))
#        print(template.format("", "Total predicted", np.sum(
#            cm[:, 0]), np.sum(cm[:, 1]), np.sum(cm[:, 2]), ""))

def F1_score(y_true, y_pred, smooth=1.0):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true + y_pred)
    numerator = tf.constant(2.) * intersection + smooth
    denominator = union + smooth
    F1 = numerator / denominator
    return tf.reduce_mean(F1)

def sensitivity(y_true, y_pred, smooth=1.):
    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_true) + smooth)
    return coef

def specificity(y_true, y_pred, smooth=1.):
    intersection = tf.reduce_sum(y_true * y_pred)
    coef = (intersection + smooth) / (tf.reduce_sum(y_pred) + smooth)
    return coef

def muilticlass_logloss(y_true, y_pred):
    return tf.losses.log_loss(y_true, y_pred)

def dice_coef_loss(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    loss = -K.log(2. * intersection + smooth) + K.log((K.sum(y_true_f) + K.sum(y_pred_f) + smooth))
    return loss
