import keras
from keras.callbacks import Callback
from keras import backend as K
import numpy as np
import sklearn
import math

class Summary_metrics(Callback):
    def __init__(self, val_gen):
        super().__init__()
        self.validation_gen = val_gen
        self.validation_labels = val_gen.classes
        self.val_batch = val_gen.batch_size
        self.class_indices = val_gen.class_indices

    def on_epoch_end(self, epoch, logs=None):
        pass
        #print("Calculating confusion matrix")
        #self.validation_gen.reset()
        #print(vars(self.validation_gen))
        #num_val_gen_steps = math.ceil(len(self.validation_labels)/self.val_batch)
        #predicted = self.model.predict_generator(self.validation_gen, verbose=1, steps=num_val_gen_steps)
        #predicted = np.argmax(predicted, axis=1)
        #ground = self.validation_labels
        #cm = sklearn.metrics.confusion_matrix(ground, predicted, labels=None, sample_weight=None)
        #print(cm)
        #report = sklearn.metrics.classification_report(ground, predicted)
        #print('Classification Report: \n{}'.format(report))

def muilticlass_logloss(y_true, y_pred):
   return K.tf.losses.log_loss(y_true, y_pred)
