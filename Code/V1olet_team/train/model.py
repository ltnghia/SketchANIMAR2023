import tensorflow as tf

from layers import *
from utils import *

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q
from keras_cv_attention_models import efficientnet, convnext, caformer

def get_base_model(name, input_shape):
    if name == 'EfficientNetV2B1':
        return efficientnet.EfficientNetV2B1(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV2S':
        return efficientnet.EfficientNetV2S(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV2M':
        return efficientnet.EfficientNetV2M(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV2L':
        return efficientnet.EfficientNetV2L(num_classes=0, input_shape=input_shape, pretrained="imagenet21k")

    if name == 'EfficientNetV1B1':
        return efficientnet.EfficientNetV1B1(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B2':
        return efficientnet.EfficientNetV1B2(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B3':
        return efficientnet.EfficientNetV1B3(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B4':
        return efficientnet.EfficientNetV1B4(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B5':
        return efficientnet.EfficientNetV1B5(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B6':
        return efficientnet.EfficientNetV1B6(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'EfficientNetV1B7':
        return efficientnet.EfficientNetV1B7(num_classes=0, input_shape=input_shape, pretrained="imagenet")

    if name == 'ConvNeXtTiny':
        return convnext.ConvNeXtTiny(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ConvNeXtSmall':
        return convnext.ConvNeXtSmall(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ConvNeXtBase':
        return convnext.ConvNeXtBase(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    if name == 'ResNet50':
        return tf.keras.applications.resnet50.ResNet50(include_top=False, input_shape=input_shape, weights='imagenet')

    if name == 'CAFormerS18':
        return caformer.CAFormerS18(num_classes=0, input_shape=input_shape, pretrained="imagenet21k-ft1k")

    raise Exception("Cannot find this base model:", name)

def get_out_layers(name):
    if name == 'EfficientNetV2S':
        return [
                'stack_0_block1_output',
                'stack_1_block3_output',
                'stack_2_block3_output',
                'stack_4_block8_output',
                'post_swish'
            ]
    if name == 'ConvNeXtTiny':
        return [
            'stack2_downsample_ln',
            'stack3_downsample_ln',
            'stack4_downsample_ln',
            'stack4_block3_output'
        ]
    return None

def create_model(n_labels=100):
    setting_cfg = get_settings('setting.yaml')
    input_shape = (setting_cfg['im_size'],setting_cfg['im_size'],3)
    base_name = setting_cfg['base_name']
    final_dropout = setting_cfg['final_dropout']

    emb = get_base_model(base_name, input_shape)

    x = emb.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(final_dropout)(x)

    cate_output = Dense(n_labels, name='cate_output', activation='softmax')(x)

    model = Model([emb.input], [cate_output])

    return model

if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    model = create_model()
    model.summary()
