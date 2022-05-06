import tensorflow as tf
from vis.utils import utils
import vis.input_modifiers
import vis.visualization
from cv_framework.diagnostics.shap_save import shap_image_plot
import gin
import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
import shap

@gin.configurable
def ActMax(model=None, layer_name=None, filter_index=None, backprop_modifier=None, grad_modifier=None,
           lp_norm_weight=0., tv_weight=0., verbose=False, custom_objects=None, model_name=None, **opt_kwargs):

    layer_idx = vis.utils.utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = vis.utils.utils.apply_modifications(
        model,
        custom_objects=custom_objects)
    img = vis.visualization.visualize_activation(
        model, layer_idx,
        filter_indices=filter_index,
        backprop_modifier=backprop_modifier,
        grad_modifier=grad_modifier,
        lp_norm_weight=lp_norm_weight,
        tv_weight=tv_weight,
        verbose=verbose,
        **opt_kwargs)

    plt.imshow(img[..., 0], cmap='jet')

    fig_name = (
        f'/data/gferguso/cord_comp/visual/ActMax_{str(filter_index)}_'
        + str(layer_name)
        + '_'
        + model_name
        + '.png'
    )


    plt.savefig(fig_name, format='png')

def ActMaxList(model=None, layer_name=None, filter_index=None, backprop_modifier=None, grad_modifier=None,
               lp_norm_weight=0., tv_weight=0., verbose=False, custom_objects=None, model_name=None,
               categories=None, columns=3, **opt_kwargs):

    # load and transform model
    layer_idx = vis.utils.utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = vis.utils.utils.apply_modifications(model, custom_objects=custom_objects)

    vis_images = []
    image_modifiers = [vis.input_modifiers.Jitter(16)]

    for idx in categories:
        img = vis.visualization.visualize_activation(model, layer_idx, filter_indices=idx,
                                                     backprop_modifier=backprop_modifier, grad_modifier=grad_modifier,
                                                     lp_norm_weight=lp_norm_weight, tv_weight=tv_weight, verbose=verbose,
                                                     max_iter=500, input_modifiers=image_modifiers, **opt_kwargs)
        img = vis.utils.utils.draw_text(img[...,0].astype('uint8'), 'Class {}'.format(idx), color=None)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        vis_images.append(img)

    # Generate stitched images with 5 cols (so it will have 3 rows).
    #plt.rcParams['figure.figsize'] = (50, 50)
    stitched = vis.utils.utils.stitch_images(vis_images, cols=columns)
    plt.axis('off')
    plt.imshow(stitched[...,0], cmap='jet')
    fig_name = '/data/gferguso/cord_comp/visual/ActMax_list' + str(filter_index) + '_' + str(layer_name) + '_' + model_name + '.png'
    plt.savefig(fig_name, format='png')

def feature_maps_vis(model, backprop_modifier=None, grad_modifier=None, lp_norm_weight=0., tv_weight=0., verbose=False,
                     layer_name=None, columns=8, model_name=None, custom_objects=None, **opt_kwargs):
    # The name of the layer we want to visualize
    layer_idx = vis.utils.utils.find_layer_idx(model, layer_name)

    # Visualize all filters in this layer.
    filters = np.arange(vis.visualization.get_num_filters(model.layers[layer_idx]))

    # Generate input image for each filter.
    vis_images = []
    for idx in filters:
        img = vis.visualization.visualize_activation(model, layer_idx, filter_indices=idx,
                                                     backprop_modifier=backprop_modifier, grad_modifier=grad_modifier,
                                                     lp_norm_weight=lp_norm_weight, tv_weight=tv_weight,
                                                     verbose=verbose, **opt_kwargs)
        # Utility to overlay text on image.
        img = vis.utils.utils.draw_text(img[...,0].astype('uint8'), 'Filter {}'.format(idx), color=None)
        img = img.reshape((img.shape[0], img.shape[1], 1))
        vis_images.append(img)

    # Generate stitched image palette with 8 cols.
    stitched = vis.utils.utils.stitch_images(vis_images, cols=columns)
    plt.axis('off')
    plt.imshow(stitched[...,0], cmap='jet')
    fig_name = '/data/gferguso/cord_comp/visual/FM' + '_' + str(layer_idx) + '_' + model_name + '.png'
    plt.savefig(fig_name, format='png', cmap='jet')

def saliency(model, images, class_index=None, layer_name=None, model_name=None, grad_modifier=None,
             backprop_modifier=None, custom_objects=None, image_size=None):
    layer_idx = vis.utils.utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = vis.utils.utils.apply_modifications(model, custom_objects=custom_objects)
    vis_images = []
    for img in images:
        image = np.array(tf.keras.preprocessing.image.load_img(img, target_size=image_size, color_mode='grayscale'))
        image = image.reshape(image_size[0],image_size[1])
        grads = vis.visualization.visualize_saliency(model, layer_idx, filter_indices=class_index, seed_input=image,
                                                     backprop_modifier=backprop_modifier, grad_modifier=grad_modifier)
        grad_img = vis.utils.utils.draw_text(grads.astype('uint8'), 'Class {}'.format(class_index), color=None)
        grad_img = grad_img.reshape((grad_img.shape[0], grad_img.shape[1], 1))
        vis_images.append(grad_img)

    # Generate stitched image palette with 8 cols.
    stitched = vis.utils.utils.stitch_images(vis_images, cols=1)
    plt.axis('off')
    plt.imshow(stitched[...,0], cmap='jet')
    fig_name = '/data/gferguso/cord_comp/visual/Sal' + '_' + str(layer_idx) + '_' + model_name + '.png'
    plt.savefig(fig_name, format='png', cmap='jet')

def CAM(model, images, class_index=None, layer_name=None, model_name=None, grad_modifier=None,
        backprop_modifier=None, custom_objects=None, image_size=None):
    layer_idx = vis.utils.utils.find_layer_idx(model, layer_name)
    model.layers[layer_idx].activation = tf.keras.activations.linear
    model = vis.utils.utils.apply_modifications(model, custom_objects=custom_objects)
    vis_images = []
    for img in images:
        image = np.array(tf.keras.preprocessing.image.load_img(img, target_size=image_size, color_mode='grayscale'))
        image = image.reshape(image_size[0],image_size[1],1)
        grads = vis.visualization.visualize_cam(model, layer_idx, filter_indices=class_index, seed_input=image,
                                                     backprop_modifier=backprop_modifier, grad_modifier=grad_modifier)
        grad_img = vis.utils.utils.draw_text(grads.astype('uint8'), 'Class {}'.format(class_index), color=None)
        grad_img = grad_img.reshape((grad_img.shape[0], grad_img.shape[1], 1))
        heatmap = np.uint8(cm.jet(grad_img[...,0])[..., :3] * 255)
        image = np.dstack((image, np.zeros((image.shape[0], grad_img.shape[1], 2))))
        overlay = vis.visualization.overlay(heatmap, image)
        vis_images.append(overlay)

    # Generate stitched image palette with 8 cols.
    stitched = vis.utils.utils.stitch_images(vis_images, cols=1)
    plt.imshow(stitched[...,0], cmap='jet')
    fig_name = '/data/gferguso/cord_comp/visual/CAM' + '_' + str(layer_idx) + '_' + model_name + '.png'
    plt.savefig(fig_name, format='png')

def shap_maps(model, shap_images, background_images, image_size=None, model_name=None):
    # select a set of background examples to take an expectation over
    background = []
    shap_list = []
    for b_img in background_images:
        image = np.array(tf.keras.preprocessing.image.load_img(b_img, target_size=image_size, color_mode='grayscale'))
        image = image.reshape(image_size[0], image_size[1], 1)
        background.append(image)

    background = np.asarray(background)
    # explain predictions of the model on four images
    e = shap.DeepExplainer(model, background)

    for s_img in shap_images:
        image = np.array(tf.keras.preprocessing.image.load_img(s_img, target_size=image_size, color_mode='grayscale'))
        image = image.reshape(image_size[0], image_size[1], 1)
        shap_list.append(image)

    shap_list = np.asarray(shap_list).astype(np.float32)
    shap_values = e.shap_values(shap_list)
    shap_list = shap_list.astype(np.float32)

    # plot the feature attributions
    shap_image_plot(shap_values, -shap_list, model_name=model_name)

def shap_grad_maps(model, shap_images, background_images, image_size=None, model_name=None):
    pass
    # select a set of background examples to take an expectation over
    # background = []
    # shap_list = []
    # for b_img in background_images:
    #     image = np.array(keras.preprocessing.image.load_img(b_img, target_size=image_size, color_mode='grayscale'))
    #     image = image.reshape(image_size[0], image_size[1], 1)
    #     background.append(image)
    #
    # background = np.asarray(background)
    # # explain predictions of the model on four images
    # e = shap.DeepExplainer(model, background)
    #
    # for s_img in shap_images:
    #     image = np.array(keras.preprocessing.image.load_img(s_img, target_size=image_size, color_mode='grayscale'))
    #     image = image.reshape(image_size[0], image_size[1], 1)
    #     shap_list.append(image)
    #
    # shap_list = np.asarray(shap_list).astype(np.float32)
    # shap_values = e.shap_values(shap_list)
    # shap_list = shap_list.astype(np.float32)
    #
    # # plot the feature attributions
    # shap_image_plot(shap_values, -shap_list, model_name=model_name)
