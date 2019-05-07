from .simple_models import simple_CNN
from .ResNet50 import ResNet50
from .InceptionResNetV2 import InceptionResNetV2
from .InceptionV3 import InceptionV3
from .Multiscale_CNN import Multiscale_CNN
from .Xception import Xception
from .DenseNet import DenseNet121,DenseNet169, DenseNet201

standard_dict = {
    'simple_CNN':simple_CNN,
    'ResNet50':ResNet50,
    'InceptionResNetV2':InceptionResNetV2,
    'InceptionV3':InceptionV3,
    'Multiscale_CNN':Multiscale_CNN,
    'Xception':Xception,
    'DenseNet121':DenseNet121,
    'DenseNet169':DenseNet169,
    'DenseNet201':DenseNet201,
}

