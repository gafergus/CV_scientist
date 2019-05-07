import keras
import os
import gin

# Images functions
@gin.configurable
def load_images(image_dir_path=None):
    images = os.listdir(image_dir_path)
    return images

# @gin.configurable
# def preprocessing():
#     pass

# Model functions
@gin.configurable
def load_model(model_name=None, model_path=None, custom_objects=None):
    load_path = os.path.join(model_path, model_name)
    model = keras.models.load_model(load_path, custom_objects=custom_objects)
    return model

# @gin.configurable
# def evaluate_model(image_path=None, image_labels=None, model_name=None, model_path=None, batch_size=32):
#     images = load_images(image_dir_path=image_path)
#     model = load_model(model_name=model_name, model_path=model_path)
#     evaluation = model.evaluate(x=images, y=image_labels, batch_size=batch_size, steps=1, verbose=0)
#     return evaluation

@gin.configurable
def infer_classes(image_path=None, model_name=None, model_path=None, proba=False, batch_size=32):
    images = load_images(image_dir_path=image_path)
    model = load_model(model_name=model_name, model_path=model_path)
    if proba:
        preds = model.predict(images, batch_size=batch_size, steps=1, verbose=0)
    else:
        preds = model.predict_proba(images, batch_size=batch_size, steps=1, verbose=0)
    return preds

