import tensorflow as tf 
import utils
from models.FCN import get_model as get_FCN
from models.SegNet import get_model as get_SegNet
from models.SegNetVGG16 import get_model as get_SegNetVGG16 
from models.UNet import get_model as get_UNet
from models.UNetVGG16 import get_model as get_UNetVGG16
from models.UNetMobileNetV2 import get_model as get_UNetMobileNetV2

MODELS = {'fcn':get_FCN,
          'segnet':get_SegNet,
          'unet':get_UNet,
          'unetmobilenetv2': get_UNetMobileNetV2,
          'unetvgg16':get_UNetVGG16,
          'segnetvgg16':get_SegNetVGG16}

COMPILE_PARAMETERS = {"loss":tf.keras.losses.CategoricalCrossentropy(),
          "metrics":[utils.dice_coef, utils.iou_coef]}

def print_available_models():
    print('Availaible models: ')
    for model in MODELS.keys():
        print(f'\t{model}')


def get_model(model='unet', **kwargs):
    model = model.lower()
    try: 
        model_keras = MODELS[model](**kwargs)
        model_keras.compile(**COMPILE_PARAMETERS)
        return model_keras
    except KeyError: 
        print(f'Model {model} is not avalaible')
        print(f"Posible models: {', '.join(MODELS.keys())}")
        exit()