import gin
from cv_framework.inference.infer import load_model
from cv_framework.diagnostics.advanced_diagnostics import shap_maps
from cv_framework.model_definitions.model_utils import set_input_output
from cv_framework.metrics.metrics import muilticlass_logloss

def run():
    gin.parse_config_file('config_vis.gin')
    # Load the model
    model_name = 'ResNet50.h5'
    #images = ['/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/72a10e64-2d93-4b61-8b90-8efef7284ced.jpg']
    image_list = [
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/3e2db6da-3b64-4388-8098-1c4c037ec03a.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/b7b1abfd-c7e8-4404-aeed-b3a3b489443f.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/820ace18-945b-49e0-99b7-386c94a92182.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/35ab0be2-1772-4ef7-b8f9-45e5e371ac11.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/9d1ceb1a-66ef-4135-a905-3d7b51b0fb4d.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/dc3f50d6-3e8c-4c5b-814b-3f2043165556.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/b18a2d86-3e3a-43ec-b9f6-0fd249bc1a0d.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/62158463-7464-4781-86b2-f43cf3f1833d.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/be0d1645-263c-4dce-99e9-77f31d6ded61.png',
        '/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/ea1cf22d-0ec1-4820-86ed-a2bce81fa924.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/45149eff-20a5-48d2-b544-b159bd153043.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/89937639-13b7-4831-9d97-0473334260f1.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/ed0ed8ae-2897-42fe-af08-ec88c69ce1b3.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/4ec56f3a-f4b0-4a4d-a3b9-fa78827b269e.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/dd0fbdcd-c045-4979-85ad-9c615d2e9038.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/483eb36d-93dc-4414-9af1-f9e10ae2754b.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/cdf38d9b-d871-4c34-a589-319491db508c.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/c3414479-5221-4c46-bd60-6dda7e4f19cf.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/6a5b9fe0-bcea-4f37-b2a5-df9b27793028.png',
        '/data/gferguso/cord_comp/images/train/class_001_No_Lung_Opacity_and_Not_Normal/325d5974-1e30-4d27-8802-ac3cf1b02e81.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/6366d2b6-c1a7-40da-998d-981aa684a7f6.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/ed4ece2f-96ca-4153-b165-79816f5a16f2.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/5278b08c-a475-469e-8910-56bff6d85210.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/8f4f0c03-bccd-420b-965f-c58d21690b12.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/d703f118-7133-48d1-91ff-56fcd50d6d9a.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/47d56dc3-c4b0-4bd0-bab7-d39549cc46a0.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/83bef258-f056-428b-adc2-2f2d72ed4586.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/37b16c54-600a-408b-b358-adbcc3ee24e5.png',
        '/data/gferguso/cord_comp/images/train/class_002_Normal/7257e3de-ff3d-4b37-9450-937abfae06dd.png'
    ]
    images = ['/data/gferguso/cord_comp/images/train/class_000_Lung_Opacity/72a10e64-2d93-4b61-8b90-8efef7284ced.png']
    custom_losses = {'muilticlass_logloss':muilticlass_logloss}
    model = load_model(model_name=model_name, model_path='/data/gferguso/cord_comp/', custom_objects=custom_losses)
    model.summary()
    image_size, _, _ = set_input_output()

    # Plot the activation maximization
    #ActMax(model=model, layer_name='fc', custom_objects=custom_losses, model_name=model_name)
    #ActMaxList(model=model, layer_name='fc', custom_objects=custom_losses, model_name=model_name, categories=[0,1,2],
    #           columns=3)
    #feature_maps_vis(model, layer_name='conv1', model_name=model_name, custom_objects=custom_losses,)
    # Broken
    #saliency(model, images, class_index=0, layer_name='fc', model_name=model_name, custom_objects=custom_losses,
    #         image_size=image_size)
    #CAM(model, images, class_index=0, layer_name='fc', model_name=model_name, custom_objects=custom_losses,
    #    image_size=image_size)
    shap_maps(model, images, image_list, image_size=image_size, model_name=model_name)

if __name__ == '__main__':
    run()
