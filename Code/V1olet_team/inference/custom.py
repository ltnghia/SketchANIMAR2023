import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, regularizers, backend as K
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU, Reshape, \
DepthwiseConv2D, Multiply, Add, LayerNormalization, Conv2DTranspose

import numpy as np
import cv2
import os
import sys
import random
import math
import re
import time
import json
import matplotlib.pyplot as plt

# !pip install -U git+https://github.com/leondgarse/keras_cv_attention_models -q

class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config

class ZeroInitGain(layers.Layer):
    def __init__(self, use_bias=False, weight_init_value=0, bias_init_value=0, **kwargs):
        super().__init__(**kwargs)
        self.use_bias = use_bias
        self.ww_init = initializers.Constant(weight_init_value) if weight_init_value != 0 else "zeros"
        self.bb_init = initializers.Constant(bias_init_value) if bias_init_value != 0 else "zeros"

    def build(self, input_shape):
        self.gain = self.add_weight(name="gain", shape=(), initializer=self.ww_init, dtype="float32", trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name="bias", shape=(), initializer=self.bb_init, dtype="float32", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return (inputs * self.gain + self.bias) if self.use_bias else (inputs * self.gain)

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"use_bias": self.use_bias})
        return base_config

def auto_select_accelerator():
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
        print('Using TPU')
    except:
        strategy = tf.distribute.MirroredStrategy()
        if strategy.num_replicas_in_sync == 1:
            strategy = tf.distribute.get_strategy()
            print('Using 1 GPU')
        else:
            print('Using GPUs')
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy

def seedEverything(seed):
    def seedTF(seed):
        tf.random.set_seed(seed)

    def seedBasic(seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    seedBasic(seed)
    seedTF(seed)

all_classes = ['0016afdafa241a8b', '00b174d953f27354', '00beb632ff36011e', '0109bd7501410481', '014cbea869fcf211', '01612cb400453597', '019470e0b56fe70a', '01fea2892d3dcb14', '0266a5a1e24613ba', '02bf2b708a98eb01', '02f910f4d565a330', '0313b6fb78613a9e', '032fb59a082e3095', '034074f6277978dc', '03b5427c9cbad7df', '0426cd835b1f0bf0', '0449f665c5c7d649', '046ba23fb2dba9b6', '0489595b95c547dd', '05c7a6c16b69e4b0', '06420791d90920aa', '067354f955919479', '068a77d6a7c8aa90', '079c96c521c73ca9', '07a182b4c5ff670d', '07b9f5a19be6449a', '0823860d2037647f', '087d185ae42e6026', '08d73e04780a95a5', '0a77cd2a91dfff53', '0a87f2d6c01fb230', '0ac1112120f956e6', '0ae627367cedcd26', '0ae964801539411d', '0b3e8d17d98f5193', '0b98619a32548d8e', '0c273d5ab5435bff', '0c27b52a88c33aae', '0c9b5c0c7881583f', '0cc1b52750c22499', '0cedfe3226dabe49', '0efe9a4279ef9c48', '0ffc2d1991f4d1e4', '0ffe68ec732dddff', '103a969e2bb1dcc9', '10436dc75a8fa9d4', '1065699438e44f04', '107cecfd59be0583', '10b05be54ad24bbf', '10bcae5f92b18612', '10c2e95fab328678', '11395e357f045559', '1180fbeb855ae874', '11d69250767f925c', '11e16848f8de0a7b', '124bea9bc82bb0c4', '12c2d1ff31beb7e9', '130ee9cf38dcc2a1', '1387e002edbe7757', '13a49ab0aecbc76f', '13e0f1c32a3c9bcc', '145639767668eaf7', '1469459f7ebe496c', '148342bfa8f41a31', '152e68ac50b67fd3', '1564b04086c2c084', '158dcfba185978e8', '15b3a52de1d155f8', '15d746d0e8530e55', '15f5184dd981d634', '169b90e868c69185', '17e243080e46a77a', '1815808c9db01bc7', '18476d26139f9f44', '188f8079c7e8e65a', '18916433442bd760', '1925dd31d7b4064c', '19be36a45ce6f9e4', '1a0240c017cd5737', '1a418f85ca7808da', '1b0305051ec67406', '1b1aee8806c0393b', '1b3d973e8c712776', '1b538c586ad2c0db', '1b7bc4b33cddb120', '1b86d214f8445508', '1ba95d8d320f5e33', '1bb180982752f261', '1c94e26b92ab97dd', '1cb8783b0b728483', '1d502beefbc7697e', '1dacb6cdad209643', '1df3b0ad86562fde', '1e76e5c9aa18cad5', '1e87934cd4d7e8ee', '1eb2e5d50f0ae982', '1f356a5d5fbd0263', '1f65711c9ef62568', '1f7b19dd8f7d532e', '2030c08c33298d28', '208ff71bed4a1459', '20b5e565c2f67b81', '20fad05da21b0518', '22148c7852d51497', '2232fed9e00d559a', '223bc7dec8a03c59', '225def4a419b8f55', '22648f17641ea2ee', '2268076b3b795a99', '22c09296590fb951', '24215bf7ce370aaf', '24e4919ee2fad075', '255d31f96c2f3723', '25a5dff1c2e3fd6c', '25d8bb2c5103d784', '25eb316bfa00fc98', '26c8931736dc43fb', '26f8ae8f8db6e224', '2721b9a1421f6599', '2736c1212e9624d8', '2846d0b05ed649cf', '28ae488982984ce3', '29550f9c0b8e7d7a', '29c20613ff8b0ea4', '29e1d80da09586cc', '29e9d7cf62ed365b', '2accc11b14dde296', '2acea48879fd7fec', '2ad8b1381890d17a', '2ae530419cd685af', '2b72aa75a44e8847', '2c5bd307d25beb0f', '2c709428b2ca75be', '2ce177087a5b7860', '2d0e5e5ef940e212', '2d4347cacab9627c', '2db0707188554ab9', '2df52539bd6c0756', '2dfa7571f2321b9e', '2e86a3625d556854', '2f97710a9ad1b2c5', '303ae000f3346db3', '3080f1b08986b040', '30aeb0f6039ef714', '30cf391e77c2cf88', '30d44706cdb396ba', '30db82e1d7e02e0f', '31cf8317f82a5815', '32e445ce45908f8f', '33fdb47f963804ae', '342d50ae23716421', '344e0698585e098a', '3575419b4b9d1e89', '36089dc42596904a', '3707cda48b44084f', '3730d6a7fdb0e4a8', '3762672a3fb6a081', '390eb1a5e57f1bd5', '39ac47af68e86109', '39fa25fa685a2a71', '3af12c73b71c656f', '3b3252444d8b457a', '3bbdb934e5cf5e7c', '3bd22b60871f3c97', '3bdb27fe2e018a7c', '3bf1ce0d8b192e34', '3c203e2dbdf1bea7', '3c2c913e87a88e57', '3c5f903557218c37', '3cf19cfa1149d500', '3d677a4457574cd7', '3d7695f8399c68c5', '3d9546c481cc1c22', '3dd66bd447d95ae1', '3ddd6a55a6d8e37c', '3df2c5b8e30103ff', '3e5de1100e3b5b82', '3e93217cf9baa08f', '40989eb3293c273e', '40e86f26ce8cc8eb', '40ffabf8d9949b3a', '410211806e61d8bb', '413505e6f4ddf95e', '426a1b89dcfa2b6d', '42e032de7218c164', '43cc39cf656a41ae', '43cfff6a23acafe7', '43fd7e94a67f9376', '4581d98eaf09d27b', '45e33d9f984acc7a', '467501864d76cbbd', '46f3bcef53438440', '47a5282b8ba9e6d2', '47d04be344220ba4', '48ea82841daadc18', '49296a65591751f1', '492ab711a91f0d60', '49b1dcc1038c7d6a', '4a295ba0a7ded399', '4a2ef78c1829d866', '4aeadac4ffc25dab', '4b433ec01cdaafff', '4b7e6d7c00dc221c', '4bc2868c75c32abd', '4c142b488845264b', '4c681e7d9db52f8b', '4ca46d226ab601bf', '4cce9c8ae0103550', '4ce0245ee81d54c8', '4d5af5360a844b3f', '4d7a48f9df08a11e', '4e3036c53e50beed', '4ea79a529433b5d6', '4ed71d8d8944949a', '4ef649c218c6ffd0', '4f144c6cac46dac5', '4f721ca24c1ea9ee', '4f7811c96a04ba51', '5061a045b4f68bc6', '50d1d60a07d09d83', '511779683b40b395', '5177b051d25ea0f7', '5192c45bf1ce718c', '52037651db34d9c6', '52568b7512b8c478', '5268c856bdbcd7ec', '52f8ea88fade532a', '52fde79d27bfeafd', '530c82165054bf61', '53139295c1a7f131', '535c19d963f3d1ba', '53e190f2111eaddb', '53e9c7c3130f2d41', '5479388fdcb8f4b3', '547f2f46f3a479eb', '553744f87e36aabb', '5544db19d7450c08', '5587d54f4108c95e', '55a985eef253a7a0', '55f44c3f728145bb', '56458bc6918ca35d', '56c67f15ec8bdb2c', '56d86e7b9f59e2b7', '56e73e1ef7d75a43', '5716d05c503ca41e', '5748b8108aea43c6', '575052b4c7230c8b', '57b24bbdb0c2985f', '581129f2e0a8962a', '587eeec34cf4d2c8', '592aba62e8462b16', '59400f104fce28c2', '59bd33136e830f43', '5a0331cd4017b611', '5a47b8a4ad093e6c', '5a96f86475d74e88', '5ae6baa4fd312d38', '5aff7c6a33aa8071', '5cf7adb1082aa77d', '5d078794e493954c', '5d33d36a11ebf237', '5d47fcc026aece1e', '5dce28d98011a696', '5e626e0bc33e7019', '5e6f1816ab0eebeb', '5e873501f1e26ca0', '5e9c1bf43507f405', '5f2f1b58e50c61bf', '5f38dbc7e09d2cdd', '5f76ff02b2bd8554', '5f779cd192cae3a3', '602f8d20449d6e58', '60a64c226b25f697', '60af2d4f86d5307c', '60d4505f1a7290eb', '62d3595fe53c81be', '62e532ede5fce4a5', '64263b59af4be01a', '64773bd12fa9b48c', '6481b377a1f2992a', '654e7e41a4575e77', '6585855871402772', '6586d4fbdf3bbddb', '658d4e36212e9c9e', '6683e5816cd62649', '672c0162a3077e07', '677063925aa95336', '67fbcae05765c5f0', '688c7d507ae9f253', '6892964be383453f', '6920236f40bf3768', '6929c827e1ce8419', '69b0631cac45fce6', '6a34b7da751a7f3f', '6a6002c745899757', '6afaaa7ebeee14b7', '6b3b93b4b0711861', '6bc2402ce628bbef', '6beb33f61a648b30', '6c280b75a4128f8b', '6c4d2f2a280f3a90', '6c74ffe6fa3a2227', '6cb7c854dc08ec70', '6d63b247b02d535d', '6d73a35aafa82787', '6dc816c6745c0b4d', '6e92439ce227a932', '6efc9529ed9e7fc5', '6f08aeec26ba66cf', '6f3e9d792ec6082d', '6f41ba76e9ffcf67', '6f4322bbfcbf58b6', '6f50fb3efc179160', '6f58684956f448da', '6fc0839548a389b2', '6fc741f92c03654d', '706e6d63c0a59998', '70c98f4c562affee', '70de0fdcbaf1302f', '7115af5fd3e62ca0', '7160a4036bd716d3', '7161147dfc135e09', '7173c9264a210587', '72996e8a76011909', '72f7a445bff185f8', '73796850790b70ca', '74703abba6aed00b', '74e121781a01280d', '75a51a0bc964ed21', '75bd8a12238f44b5', '76386d460969cd1c', '7693fede195ea017', '771f7f4cbdb88fa6', '776a9134c6bb1d03', '77e419b386f8bb46', '787871da8b795d39', '78b045d9082fa8a1', '78c02d222c27bc25', '78c5de6dd0d3af11', '7907e0dc70c44f92', '7939a5fa2ffed845', '79a002142ee6b312', '79c95869e0d37be3', '7aa5b3344260a6c8', '7aa6307caf88079d', '7af0ae61cfa42a48', '7b3c2502ab3aacc5', '7bc562527a894ac8', '7bf29d74ee81f62a', '7c1bca56ec73c461', '7c53223fa27fbb2b', '7e425f2c7bd1de87', '7f36942b8785093d', '7f9c5a47a8b18f00', '7febb03fa7121507', '800132a29257a00e', '8061c9167e0cb731', '80b20ad87a71a44f', '80f19bdb036e4d95', '811a1f305c2884ce', '819367dceddb838b', '81dc2a724ba57790', '81e25016525a9151', '8240c7ae34b9fcd0', '825b05e3f1482c22', '82c76d00ebd193b6', '82de86e0cd7f9736', '834953da8335695b', '836406771c68c193', '837f7b9ff5dcfa78', '83a6898b3cf63399', '845335c530f24a80', '846ee3d09ab0fcc8', '84cda9bd7042c100', '8504af9b6f6829dc', '85a5e7457d06795b', '85e5acde0498e53b', '868c9905d94ddb08', '86e6f532243c9e95', '872da42e17302cf3', '876489950e67fe4b', '87941f22ae04cfa1', '88028a0e4ac75667', '88652c4cf16deeda', '88bf8bc923ad639b', '88c74acaea736788', '88fe8b2185d4feb6', '8905f44a81dddb52', '890f0812b9947e09', '89defea2383d8c4c', '8b0062ec0267c5f1', '8b4f277b03b271b5', '8b632eddc3627a37', '8c85dc57e5e16770', '8d32b020126e6e8c', '8d3be739b380144a', '8e05fa0fab7d7e0f', '8e2cb9cc846f26a7', '8e2dcf928d4348fd', '8e5b0a1b440e41b1', '8f1e3d3f3eb18a62', '8f2b6f200d420b7e', '8f42f036627160ae', '9047b88a1daca816', '904cb4ada4827f50', '90595337e6108d17', '90ba6644ad322c47', '9115ebf2eb5e3dd1', '919317f18eaae8fd', '919b80269990ac22', '92daae9f6dbb128e', '93068e2de4fc820a', '9351cfef5ce54e69', '93c911b19ed62df8', '94ee98fd5cf5e505', '9556be5bb399f203', '95865abd260e2c15', '95fbfda2a354bacc', '97db07b2d5e4b197', '97eb2ab380ceac87', '98032f4c2f991db9', '983acf35534872bb', '9841cafe9b5e1f5e', '98b4aaaa2165fcd1', '98b9ecda30efcb34', '98c510f0e41accf8', '98d12855b2889504', '99467c906f40a69f', '9946e34f45632be6', '9aa017336678933a', '9b93eafce6f5a7ac', '9c0ccaca4f7af75d', '9c8ce73f3c2c12d2', '9d5d62c52379d9f0', '9d6d5dd3175f52fa', '9d9e8b57fde29a41', '9dd06e3c5098a211', '9e10ab1de79421ec', '9e2c640d53381d60', '9e89b018e25bd4ed', '9ea7c44d1e5ac264', '9edcaa9f5ddecfc2', '9f890fc4347f91b0', '9fe19c2f9343a942', 'a02aaca56afb4f17', 'a08b0c2fe44e748b', 'a1993557a877621a', 'a2a17167753efcb1', 'a2e67ac727768dcc', 'a3dc68476353e506', 'a3f337038d154765', 'a4da84ed81451ad0', 'a52636bfacafa2b6', 'a5391c778367ed6b', 'a5842e12a9200d43', 'a594a0e2b9fe9ca5', 'a609886e475e0942', 'a635e529613c6c00', 'a64e0155977494f6', 'a7c3e08cc4309ec9', 'a86071b0d9d7d0da', 'a91a958f77aaaf7a', 'a92a71ac204d0223', 'a957b00d43ef7c8f', 'aa0598ae402d2833', 'aa182a1ca33fba10', 'aa5750ffc43bcc18', 'aae580fed8c9604b', 'aaf18dc46614568f', 'ab047afa481c3134', 'ab130247173f42ba', 'ac3d3b1b6d75d3ae', 'ad3a9fd00fc69eaf', 'ad6886154dae4a8c', 'ad790b0a529e2267', 'ae09892d3ad04b4b', 'ae7e4778dfc7147b', 'ae93f2d3e0ab350d', 'afb009152b8ff33c', 'afb491e33546448e', 'afb4aa409ce3ba60', 'b0334182d0c3faeb', 'b095d9ef45ca6c13', 'b0ed6ea888551408', 'b141dda23578766d', 'b1fc2b6ab1cbb8e7', 'b2155ffa27d015f8', 'b234fb92bcacf4db', 'b245bce8888afeb4', 'b345896ab5eb2f5e', 'b3639e91e48f90f1', 'b40605d93511756d', 'b44e755a2fb42ad0', 'b4679940dd7710ef', 'b46c5d1510f7e1a4', 'b4820fa8a2c2444a', 'b4891278cf84bda4', 'b54319c9bb0ede4d', 'b55c36293cdb47bf', 'b5f24bcd658e3bd5', 'b609dc59b3039257', 'b660f7e5959a2ea0', 'b922cbe2403df658', 'ba6c17861f1fdabc', 'ba7b9da1f1c45296', 'bb306902252799be', 'bbdedc10be93fb6e', 'bbe5a083ba234197', 'bc362ef90482b082', 'bcfec2a2e7d4336a', 'bd8a7ac67339e2d5', 'bd9f8bd45593a0cc', 'bda1f6d4a4b5fe79', 'bdf5128b241959b8', 'bea831bbdeb49dbe', 'bfc8c9ed57d319d2', 'bfdb497b62c53c6d', 'bffcd34d68d28303', 'c06f7a01098b018b', 'c0711e59c0006177', 'c08708af112a0bc7', 'c08b5058ec12705d', 'c0d8aefb3b2ae9fb', 'c0dcb96c756dbfc5', 'c180a6944a796dd1', 'c18dc05027727a8d', 'c1db6aceadbe546f', 'c281a16ddb955261', 'c329a4bfc8c0a995', 'c36b74d05bd80d81', 'c423400504d96338', 'c43879676fbe4462', 'c44866f41c8768f3', 'c484f391c112c70f', 'c4cb6c13af0fba27', 'c4ee709c2845df96', 'c53efb8e05ccafb5', 'c5bc9b0cfdb5d23e', 'c6457fbbce37ec86', 'c6902f10d0efd8be', 'c6a64b1fd1e8cd43', 'c6aff24baaab3ca9', 'c6cbe6d259d1c457', 'c6f7761f5fe290fa', 'c75a9c7c8b59efc9', 'c78aacf25332539a', 'c7c9add79b919256', 'c8a68150df3727e6', 'c8c580af957a5d2c', 'c9115aa02a41a4e6', 'c9318b8622922822', 'c94760fab3f8ec44', 'c97abb4e781eaec2', 'c9d2d69ecab50fcc', 'ca028b330a788164', 'cb3b53c236312a01', 'cb4fbea952d068b4', 'cb59eadac2b90db5', 'cb6ef21b1c71cbbb', 'cbd5aaf74e9de14a', 'cbef81f72fefce7a', 'cc9d45414fb693d5', 'ccedf54cba34d96d', 'cdb03dd72471b157', 'cdd35fc56b3e2d0a', 'ce9073046498fc6f', 'd086e2ba9427d224', 'd0bc4cd0c3bc8ced', 'd0c30ad9d0522ca9', 'd0ea1adb69ca11d1', 'd135189b46e20da8', 'd19ff493ba370b6e', 'd1aef3261220290a', 'd1fa77c870715aa6', 'd2262a2bf815bca0', 'd265178183f1045a', 'd2e4667d08d1ff86', 'd30fb376f0eaea2b', 'd3377b07ed88c02c', 'd3af05d7f2f37761', 'd3ba454970b6abd8', 'd3ce00984b6ffe3a', 'd3f5de25c5cfe10b', 'd42e636946ade14e', 'd46d18fabb157728', 'd4de076de078e6ba', 'd52126f39607d7e5', 'd5bc2c74d13ee217', 'd5ce39428399616c', 'd63af3f5e7edd913', 'd6654dc72b2fc814', 'd6e0c1df4a6727ef', 'd6f67448fad3e49d', 'd7c1b172e327ab80', 'd7ec8c861271c6b9', 'd7f91c8dd06833e0', 'd8507df2c18f7908', 'd890aa3630d5e8b2', 'd91ae5ac12f20fb9', 'd93679457ff96f4f', 'da0da90b4c6c9621', 'dbfad9e0971c6734', 'dc4ff0bf65018d35', 'dcdd9defb7654a69', 'dcea5825436a03ca', 'dd8a26c4de9a4d6c', 'dd986bc1a049ef4b', 'ddfa2ef9e3c43e13', 'deed286d937453e9', 'dfbb8fdd9be16fc6', 'e072cf875c16542a', 'e0c5393e4f88864f', 'e0f5cafb98d9acf7', 'e16916d53f03e59f', 'e17edb72cc31de14', 'e18ea7c595a3d53d', 'e242947fb36276d8', 'e31ced3776eca0a7', 'e406c931589df07b', 'e448bfc8561b7f5b', 'e478035ea7a63a88', 'e505606f158f2955', 'e5ee823dd8ebfe70', 'e6e23a42e21d5d81', 'e6ec065ce21bb1e9', 'e74b025c2133655a', 'e74deefe0a3ad809', 'e76f366074e000b0', 'e7d35ff357dd270d', 'e7f4d68eb4c3f512', 'e81bcfcf9bf2c1ca', 'e84c540449aea18a', 'e84d5bcc733cb2c0', 'e8626631698592a7', 'e911271cca4a9d36', 'e9b85e6177ccf14d', 'ea0a0d3cb57de45c', 'ea200cbaf4ceb144', 'ea4680eb71358a11', 'ea774de39e97e259', 'eac21a4b02e6e24a', 'eb2ca1b686cec5b6', 'eb669ec7a2cc888a', 'ebb4e5b69643d4ab', 'ec1d8367e32b35b5', 'ed2b9dbd5a8d4f91', 'edf43557a84dc38e', 'ee08e4a284a536e8', 'ee0a634b331b5a2a', 'ee154ab1b203f278', 'ee348416503b399b', 'ee5fdac79ee65b48', 'eea729867e7273aa', 'ef26ac0097ec03ab', 'ef916ad12d81d484', 'efa061e775cc76e0', 'efc57137562e8eeb', 'efdf5b5eed9a4890', 'f0931310b6964a5e', 'f0b5c52cb89607ad', 'f1089dcdf18fa0a2', 'f17c28d4b31b1517', 'f1d410c5d6888311', 'f24db050b4065a9c', 'f2e49e6309928b53', 'f2fcaced04bd4f2e', 'f30a9b11844c24b0', 'f34aa0a2108b06c2', 'f40a50ef56006ab9', 'f45311b54d5d3fe4', 'f46502a5c792f3d2', 'f4673fcd52f52b95', 'f488c177fdbbab01', 'f4ae012475c8b50a', 'f5749752ed4a8e10', 'f5c33961fc80448c', 'f676e480e5f79f05', 'f68c68b42185c717', 'f6dddfd190b2291f', 'f7237b602f17a279', 'f72449d035d0ac79', 'f764a508c643d067', 'f770b7a17bed6938', 'f807a75c0e3429fa', 'f8287f9edce2640a', 'f858f2ac7979a7b1', 'f88ddb5d1e798ebc', 'f8b337766428bd4e', 'f8c35c11d6367ad1', 'f8cbd24719bd2d25', 'fa19e2ec35f90a50', 'fa79d53c4036fa3d', 'fb5c86ffdae2de70', 'fb8e44e358fa39c8', 'fb9353dc68b766dd', 'fbebaed970d04c62', 'fc20b4369ee117fd', 'fc2db06c4c42b0d2', 'fc7b6af833b14287', 'fc8524975220e49c', 'fca89dbdefa84fb9', 'fdfd2538c9269782', 'fe0dc4f6d17a9c13', 'fe8ba017416507eb', 'fec0ffde891590ec', 'fed99c4e5443d447', 'ff2eff3e8be69c85', 'ff35780e9a84c00c', 'ff4fb6cb32c1c467', 'ff51de34794e07fd', 'ffb83d7326fd48d9']