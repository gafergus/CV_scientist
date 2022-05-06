import keras
import os
import gin
import math
from cv_framework.data_access.generators import directory_flow
from cv_framework.model_definitions.model_utils import data_shape

# Images functions
@gin.configurable
def load_images(image_dir_path=None, model_name=None):
    if not model_name:
        raise Exception('No model name given, inference cannot happen in an unknown scope!')
    with gin.config_scope(model_name):
        image_size, _, _ = data_shape()
        infer_gen = directory_flow(dir=image_dir_path, shuffle=False, image_size=image_size)
    return infer_gen

# @gin.configurable
# def preprocessing():
#     pass

# Model functions
@gin.configurable
def load_model(model_name=None, model_path=None, custom_objects=None):
    load_path = os.path.join(model_path, model_name)
    return keras.models.load_model(load_path, custom_objects=custom_objects)

# @gin.configurable
# def evaluate_model(image_path=None, image_labels=None, model_name=None, model_path=None, batch_size=32):
#     images = load_images(image_dir_path=image_path)
#     model = load_model(model_name=model_name, model_path=model_path)
#     evaluation = model.evaluate(x=images, y=image_labels, batch_size=batch_size, steps=1, verbose=0)
#     return evaluation

@gin.configurable
def infer_classes(image_path=None, model_name=None, model_path=None, custom_objects=None):
    infer_gen = load_images(image_dir_path=image_path, model_name=model_name)
    model = load_model(model_name=model_name, model_path=model_path, custom_objects=custom_objects)
    num_steps = math.ceil(len(infer_gen.classes) / infer_gen.batch_size)
    return model.predict_generator(infer_gen, steps=num_steps, verbose=0)

