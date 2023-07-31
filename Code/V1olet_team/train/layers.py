import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras import layers, Model, regularizers, backend as K
from tensorflow.keras.activations import swish
from tensorflow.keras.layers import Activation, Conv2D, Input, GlobalAveragePooling2D, Concatenate, InputLayer, \
ReLU, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalMaxPooling2D, Softmax, Lambda, LeakyReLU, Reshape, \
DepthwiseConv2D, Multiply, Add, LayerNormalization, Conv2DTranspose

class NormDense(keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, append_norm=False, partial_fc_split=0, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        # self.init = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
        self.init = keras.initializers.glorot_normal()
        # self.init = keras.initializers.TruncatedNormal(mean=0, stddev=0.01)
        self.units, self.loss_top_k, self.append_norm, self.partial_fc_split = units, loss_top_k, append_norm, partial_fc_split
        self.kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self.supports_masking = False

    def build(self, input_shape):
        if self.partial_fc_split > 1:
            self.cur_id = self.add_weight(name="cur_id", shape=(), initializer="zeros", dtype="int64", trainable=False)
            self.sub_weights = self.add_weight(
                name="norm_dense_w_subs",
                shape=(self.partial_fc_split, input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        else:
            self.w = self.add_weight(
                name="norm_dense_w",
                shape=(input_shape[-1], self.units * self.loss_top_k),
                initializer=self.init,
                trainable=True,
                regularizer=self.kernel_regularizer,
            )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # tf.print("tf.reduce_mean(self.w):", tf.reduce_mean(self.w))
        if self.partial_fc_split > 1:
            # self.sub_weights.scatter_nd_update([[(self.cur_id - 1) % self.partial_fc_split]], [self.w])
            # self.w.assign(tf.gather(self.sub_weights, self.cur_id))
            self.w = tf.gather(self.sub_weights, self.cur_id)
            self.cur_id.assign((self.cur_id + 1) % self.partial_fc_split)

        norm_w = tf.nn.l2_normalize(self.w, axis=0, epsilon=1e-5)
        norm_inputs = tf.nn.l2_normalize(inputs, axis=1, epsilon=1e-5)
        output = K.dot(norm_inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        if self.append_norm:
            # Keep norm value low by * -1, so will not affect accuracy metrics.
            output = tf.concat([output, tf.norm(inputs, axis=1, keepdims=True) * -1], axis=-1)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "append_norm": self.append_norm,
                "partial_fc_split": self.partial_fc_split,
                "kernel_regularizer": keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

def conv(inputs, filters, kernel_size=3, strides=(1, 1), padding='same', activation=None):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, activation=activation)(inputs)
    return x

def bn_act(inputs, activation='swish'):
    x = BatchNormalization()(inputs)
    if activation is not None:
        x = Activation(activation)(x)
    return x

def conv_bn_act(inputs, filters, kernel_size, strides=(1, 1), padding='same', activation='swish'):
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(inputs)
    x = bn_act(x, activation=activation)
    return x

def residual_block(inputs, filters):
    x = conv_bn_act(inputs, filters, 3)
    x = conv(x, filters, 3)
    x = BatchNormalization()(x)
    skips = conv(inputs, filters, 1) if inputs.shape[-1] != filters else inputs
    x = x + skips
    x = Activation('swish')(x)
    return x

def block_multi_kernel_size(inputs, filters, kernel_sizes):
    xlist = []
    for kernel_size in kernel_sizes:
        x = conv(inputs, filters, kernel_size)
        xlist.append(x)
    x = Concatenate(axis=3)(xlist)
    x = conv_bn_act(x, filters, 1)
    return x

def self_attention(inputs, filters):
    x = conv_bn_act(inputs, filters, 1, activation=None)
    norm_x = tf.math.l2_normalize(x)
    x = tf.keras.activations.relu(x)
    x = conv_bn_act(x, filters, 1, activation=None)
    x = tf.keras.activations.softplus(x)
    x = x * norm_x
    return x

def local_branch(inputs, filters, kernel_sizes):
    x = block_multi_kernel_size(inputs, filters, kernel_sizes)
    x = self_attention(x, filters)
    return x

def orthogonal_fusion(fl, fg):
    """
    fl: local (h1, w1, c1)
    fg: global (c2)
    c1 == c2
    Return:
    (h1, w1, c1 + c2)
    """
    fl = tf.transpose(fl, [0,3,1,2])
    bs, c, w, h = fl.shape
    fl_b = tf.reshape(fl, [tf.shape(fl)[0],c,w*h])
    fl_dot_fg = tf.matmul(fg[:,tf.newaxis,:] ,fl_b)
    fl_dot_fg = tf.reshape(fl_dot_fg, [tf.shape(fl_dot_fg)[0],1,w,h])
    fg_norm = tf.norm(fg, ord=2, axis=1)
    fl_proj = (fl_dot_fg / fg_norm[:,tf.newaxis,tf.newaxis,tf.newaxis]) * fg[:,:,tf.newaxis,tf.newaxis]
    fl_orth = fl - fl_proj
    fg_rep = tf.tile(fg[:,:,tf.newaxis,tf.newaxis], multiples=(1,1,w,h))
    f_fused = tf.keras.layers.Concatenate(axis=1)([fl_orth, fg_rep])
    f_fused = tf.transpose(f_fused, [0,2,3,1])
    return f_fused

def se_module(inputs, se_ratio=4):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    reduction = filters // se_ratio
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = Conv2D(reduction, kernel_size=1, use_bias=True)(se)
    se = Activation("swish")(se)
    se = Conv2D(filters, kernel_size=1, use_bias=True)(se)
    se = Activation("sigmoid")(se)
    
    return Multiply()([inputs, se])

def mb_conv(inputs, filters, stride, expand_ratio, kernel_size=3, drop_rate=0, use_se=0):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    
    is_fused = True if use_se == 0 else False
    shortcut = True if filters == input_channel and stride == 1 else False
        
    if is_fused and expand_ratio != 1:
        x = conv(inputs, input_channel * expand_ratio, 3, stride)
        x = bn_act(x)
    elif expand_ratio != 1: # not is fused
        x = conv(inputs, input_channel * expand_ratio, 1, 1)
        x = bn_act(x)
    else:
        x = inputs
        
    if not is_fused:
        x = DepthwiseConv2D(kernel_size, padding="same", strides=stride, use_bias=False)(x)
        x = bn_act(x)
        
    if use_se:
        x = se_module(x, se_ratio=expand_ratio)
    
    # pw-linear
    if is_fused and expand_ratio == 1:
        x = conv(x, filters, 3, strides=stride)
        x = bn_act(x)
    else:
        x = conv(x, filters, 1, strides=(1, 1))
        x = bn_act(x)
        
    if shortcut:
        if drop_rate > 0:
            x = Dropout(drop_rate, noise_shape=(None, 1, 1, 1))(x)
        return Add()([inputs, x])
    else:
        return x

class wBiFPNAdd(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-4, **kwargs):
        super(wBiFPNAdd, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=(num_in,),
                                initializer=tf.keras.initializers.constant(1 / num_in),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.relu(self.w)
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        x = x / (tf.reduce_sum(w) + self.epsilon)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(wBiFPNAdd, self).get_config()
        config.update({
          'epsilon': self.epsilon
        })
        return config

class PositionEmbedding(layers.Layer):
    """Turn integers (positions) into dense vectors of fixed size.
    eg. [[-4], [10]] -> [[0.25, 0.1], [0.6, -0.2]]
    Expand mode: negative integers (relative position) could be used in this mode.
        # Input shape
            2D tensor with shape: `(batch_size, sequence_length)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, output_dim)`.
    Add mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
    Concat mode:
        # Input shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim)`.
        # Output shape
            3D tensor with shape: `(batch_size, sequence_length, feature_dim + output_dim)`.
    """
    MODE_EXPAND = 'expand'
    MODE_ADD = 'add'
    MODE_CONCAT = 'concat'

    def __init__(self,
                 input_dim,
                 output_dim,
                 mode=MODE_EXPAND,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 **kwargs):
        """
        :param input_dim: The maximum absolute value of positions.
        :param output_dim: The embedding dimension.
        :param embeddings_initializer:
        :param embeddings_regularizer:
        :param activity_regularizer:
        :param embeddings_constraint:
        :param mask_zero: The index that represents padding. Only works in `append` mode.
        :param kwargs:
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mode = mode
        self.embeddings_initializer = keras.initializers.get(embeddings_initializer)
        self.embeddings_regularizer = keras.regularizers.get(embeddings_regularizer)
        self.activity_regularizer = keras.regularizers.get(activity_regularizer)
        self.embeddings_constraint = keras.constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero is not False

        self.embeddings = None
        super(PositionEmbedding, self).__init__(**kwargs)

    def get_config(self):
        config = {'input_dim': self.input_dim,
                  'output_dim': self.output_dim,
                  'mode': self.mode,
                  'embeddings_initializer': keras.initializers.serialize(self.embeddings_initializer),
                  'embeddings_regularizer': keras.regularizers.serialize(self.embeddings_regularizer),
                  'activity_regularizer': keras.regularizers.serialize(self.activity_regularizer),
                  'embeddings_constraint': keras.constraints.serialize(self.embeddings_constraint),
                  'mask_zero': self.mask_zero}
        base_config = super(PositionEmbedding, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def build(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            self.embeddings = self.add_weight(
                shape=(self.input_dim * 2 + 1, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint,
            )
        super(PositionEmbedding, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        if self.mode == self.MODE_EXPAND:
            if self.mask_zero:
                output_mask = K.not_equal(inputs, self.mask_zero)
            else:
                output_mask = None
        else:
            output_mask = mask
        return output_mask

    def compute_output_shape(self, input_shape):
        if self.mode == self.MODE_EXPAND:
            return input_shape + (self.output_dim,)
        if self.mode == self.MODE_CONCAT:
            return input_shape[:-1] + (input_shape[-1] + self.output_dim,)
        return input_shape

    def call(self, inputs, **kwargs):
        if self.mode == self.MODE_EXPAND:
            if K.dtype(inputs) != 'int32':
                inputs = K.cast(inputs, 'int32')
            return K.gather(
                self.embeddings,
                K.minimum(K.maximum(inputs, -self.input_dim), self.input_dim) + self.input_dim,
            )
        input_shape = K.shape(inputs)
        if self.mode == self.MODE_ADD:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], input_shape[2]
        else:
            batch_size, seq_len, output_dim = input_shape[0], input_shape[1], self.output_dim
        pos_embeddings = K.tile(
            K.expand_dims(self.embeddings[:seq_len, :self.output_dim], axis=0),
            [batch_size, 1, 1],
        )
        if self.mode == self.MODE_ADD:
            return inputs + pos_embeddings
        return K.concatenate([inputs, pos_embeddings], axis=-1)

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.3
        )
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation=tf.nn.gelu), layers.Dense(embed_dim),]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]

        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'dense_dim': self.dense_dim,
            'num_heads': self.num_heads,
        })
        return config

def mkn_conv(inputs, filters, kernel_sizes=[1,3,5], padding="same", activation='swish'):
    # multi kernel size conv
    list_f = []
    for kernel_size in kernel_sizes:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   )(inputs)
        f = bn_act(f, activation)
        list_f.append(f)
    return list_f

def atrous_conv(inputs, filters, dilation_rates=[6,12,18], kernel_size=3, padding="same", activation='swish'):
    list_f = []
    for rate in dilation_rates:
        f = Conv2D(filters=filters, 
                   kernel_size=kernel_size,  
                   padding=padding, 
                   dilation_rate=rate
                   )(inputs)
        f = bn_act(f, activation)
        list_f.append(f)
    return list_f

class softmax_merge(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(softmax_merge, self).__init__(**kwargs)

    def build(self, input_shape):
        num_in = len(input_shape)
        self.w = self.add_weight(name=self.name,
                                shape=(num_in,),
                                initializer=tf.keras.initializers.constant(1 / num_in),
                                trainable=True,
                                dtype=tf.float32)

    def call(self, inputs, **kwargs):
        w = tf.keras.activations.softmax(tf.expand_dims(self.w, 0))[0]
        x = tf.reduce_sum([w[i] * inputs[i] for i in range(len(inputs))], axis=0)
        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    def get_config(self):
        config = super(softmax_merge, self).get_config()
        return config
    
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

def layer_norm(inputs, zero_gamma=False, epsilon=1e-6):
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    return LayerNormalization(axis=-1, epsilon=epsilon, gamma_initializer=gamma_initializer)(inputs)
    
def convnext_block(inputs, filters, layer_scale_init_value=1e-6, drop_rate=0, activation="gelu"):
    x = DepthwiseConv2D(kernel_size=7, padding="SAME", use_bias=True)(inputs)
    x = layer_norm(x)
    x = Dense(4 * filters)(x)
    x = Activation(activation=activation)(x)
    x = Dense(filters)(x)
    if layer_scale_init_value > 0:
        x = ChannelAffine(use_bias=False, weight_init_value=layer_scale_init_value)(x)
    x = Dropout(drop_rate)(x)
    return Add()([inputs, x])

def sample_conv(inputs, scale=2, filters=None):
    if scale == 1:
        return inputs
    if filters is None:
        filters = inputs.shape[-1]
    if scale > 1:
        s = Conv2DTranspose(filters, 
                            kernel_size=(2, 2), 
                            strides=(scale, scale), 
                            padding="same")(inputs)
    else:
        scale = int(1.0 / scale)
        s = Conv2D(filters, 
                   kernel_size=(2, 2), 
                   strides=(scale, scale), 
                   padding="same")(inputs)
    s = bn_act(s)
    return s

def sample_resize(inputs, scale=2):
    if scale == 1:
        return inputs

    ups = tf.image.resize(inputs, (int(inputs.shape[1]*scale), 
                                   int(inputs.shape[2]*scale)))
    return ups