import tensorflow as tf
import numpy as np
from tensorflow.keras.optimizers import Adam, SGD, Optimizer
import tensorflow_addons as tfa

class ArcfaceLossSimple(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossSimple, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        # self.low_pred_punish = tf.sin(np.pi - margin) * margin
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(tf.math.not_equal(y_true, 0))
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLossSimple, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ArcfaceLossSimple(tf.keras.losses.Loss):
    def __init__(self, margin=0.5, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLossSimple, self).__init__(**kwargs)
        self.margin, self.scale, self.from_logits, self.label_smoothing = margin, scale, from_logits, label_smoothing
        self.margin_cos, self.margin_sin = tf.cos(margin), tf.sin(margin)
        self.threshold = tf.cos(np.pi - margin)
        # self.low_pred_punish = tf.sin(np.pi - margin) * margin
        self.theta_min = -2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(tf.math.not_equal(y_true, 0))
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
        theta = y_pred_vals * self.margin_cos - tf.sqrt(1 - tf.pow(y_pred_vals, 2)) * self.margin_sin
        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)
        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLossSimple, self).get_config()
        config.update(
            {
                "margin": self.margin,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class ArcfaceLoss(tf.keras.losses.Loss):
    def __init__(self, margin1=1.0, margin2=0.5, margin3=0.0, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super(ArcfaceLoss, self).__init__(**kwargs)
        self.margin1, self.margin2, self.margin3, self.scale = margin1, margin2, margin3, scale
        self.from_logits, self.label_smoothing = from_logits, label_smoothing
        self.threshold = np.cos((np.pi - margin2) / margin1)  # grad(theta) == 0
        self.theta_min = (-1 - margin3) * 2
        self.batch_labels_back_up = None

    def build(self, batch_size):
        self.batch_labels_back_up = tf.Variable(tf.zeros([batch_size], dtype="int64"), dtype="int64", trainable=False)

    def call(self, y_true, norm_logits):
        if self.batch_labels_back_up is not None:
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        pick_cond = tf.where(tf.math.not_equal(y_true, 0))
        y_pred_vals = tf.gather_nd(norm_logits, pick_cond)
 
        if self.margin1 == 1.0 and self.margin2 == 0.0 and self.margin3 == 0.0:
            theta = y_pred_vals
        elif self.margin1 == 1.0 and self.margin3 == 0.0:
            theta = tf.cos(tf.acos(y_pred_vals) + self.margin2)
        else:
            theta = tf.cos(tf.acos(y_pred_vals) * self.margin1 + self.margin2) - self.margin3

        theta_valid = tf.where(y_pred_vals > self.threshold, theta, self.theta_min - theta)

        arcface_logits = tf.tensor_scatter_nd_update(norm_logits, pick_cond, theta_valid) * self.scale

        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super(ArcfaceLoss, self).get_config()
        config.update(
            {
                "margin1": self.margin1,
                "margin2": self.margin2,
                "margin3": self.margin3,
                "scale": self.scale,
                "from_logits": self.from_logits,
                "label_smoothing": self.label_smoothing,
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
        
class SupervisedContrastiveLoss(tf.keras.losses.Loss):
    def __init__(self, temperature=1, name=None):
        super(SupervisedContrastiveLoss, self).__init__(name=name)
        self.temperature = temperature

    def __call__(self, labels, feature_vectors, sample_weight=None):
        # Normalize feature vectors
        feature_vectors_normalized = tf.math.l2_normalize(feature_vectors, axis=1)
        # Compute logits
        logits = tf.divide(
            tf.matmul(
                feature_vectors_normalized, tf.transpose(feature_vectors_normalized)
            ),
            self.temperature,
        )
        return tfa.losses.npairs_loss(tf.squeeze(labels), logits)

class AdaFaceLoss(ArcfaceLossSimple):
    """
    margin_alpha:
      - When margin_alpha=0.33, the model performs the best. For 0.22 or 0.66, the performance is still higher.
      - As long as h is set such that ∥dzi∥ has some variation, margin_alpha is not very sensitive.
    margin:
      - The performance is best for HQ datasets when margin=0.4, for LQ datasets when margin=0.75.
      - Large margin results in large angular margin variation based on the image quality, resulting in more adaptivity.
    mean_std_alpha: Update pace for batch_mean and batch_std.
    """

    def __init__(self, batch_size=16, margin=0.4, margin_alpha=0.333, mean_std_alpha=0.01, scale=64.0, from_logits=True, label_smoothing=0, **kwargs):
        super().__init__(scale=scale, from_logits=from_logits, label_smoothing=label_smoothing, **kwargs)
        self.min_feature_norm, self.max_feature_norm, self.epislon = 0.001, 100, 1e-3
        self.min_feature_norm = np.array(self.min_feature_norm)
        self.max_feature_norm = np.array(self.max_feature_norm)
        self.epislon = np.array(self.epislon)
        self.margin, self.margin_alpha, self.mean_std_alpha = margin, margin_alpha, mean_std_alpha
        self.mean_std_alpha = np.array(self.mean_std_alpha, np.float32)
        self.batch_mean = tf.Variable(20, dtype="float32", trainable=False)
        self.batch_std = tf.Variable(100, dtype="float32", trainable=False)
        self.cos_max_epislon = tf.acos(-1.0) - self.epislon  # pi - epislon
        self.clip_min = np.array(-1.0 + self.epislon)
        self.clip_max = np.array(1.0 - self.epislon)
        self.batch_size = batch_size

    def __to_scaled_margin__(self, feature_norm):
        norm_mean = tf.math.reduce_mean(feature_norm)
        samples = tf.cast(tf.maximum(1, self.batch_size - 1), feature_norm.dtype)
        norm_std = tf.sqrt(tf.math.reduce_sum((feature_norm - norm_mean) ** 2) / samples)  # Torch std
        self.batch_mean.assign(tf.add(tf.math.multiply(self.mean_std_alpha, norm_mean), tf.math.multiply(tf.math.subtract(1.0, self.mean_std_alpha), self.batch_mean)))
        self.batch_std.assign(tf.add(self.mean_std_alpha * norm_std, (1.0 - self.mean_std_alpha) * self.batch_std))
        margin_scaler = (feature_norm - self.batch_mean) / (self.batch_std + self.epislon)  # 66% between -1, 1
        margin_scaler = tf.clip_by_value(margin_scaler * self.margin_alpha, -1, 1)  # 68% between -0.333 ,0.333 when h:0.333
        return tf.expand_dims(self.margin * margin_scaler, 1)

    def call(self, y_true, norm_logits_with_norm):
        if self.batch_labels_back_up is not None:  # For VPL mode
            self.batch_labels_back_up.assign(tf.argmax(y_true, axis=-1))
        # feature_norm is multiplied with -1 in NormDense layer, keeping low for not affecting accuracy metrics.
        norm_logits, feature_norm = norm_logits_with_norm[:, :-1], norm_logits_with_norm[:, -1] * -1
        norm_logits = tf.clip_by_value(tf.cast(norm_logits, tf.float32), clip_value_min=self.clip_min, clip_value_max=self.clip_max)
        feature_norm = tf.clip_by_value(feature_norm, clip_value_min=self.min_feature_norm, clip_value_max=self.max_feature_norm)
        scaled_margin = tf.stop_gradient(self.__to_scaled_margin__(feature_norm))
        # tf.print(", margin: ", tf.reduce_mean(scaled_margin), sep="", end="\r")
        # tf.print(", margin hist: ", tf.histogram_fixed_width(scaled_margin, [-self.margin, self.margin], nbins=3), sep="", end="\r")
        # ex: m=0.5, h:0.333
        # range
        #       (66% range)
        #   -1 -0.333  0.333   1  (margin_scaler)
        # -0.5 -0.166  0.166 0.5  (m * margin_scaler)

        # XLA after TF > 2.7.0 not supporting gather_nd -> tensor_scatter_nd_update method...
        arcface_logits = tf.where(
            tf.cast(y_true, dtype=tf.bool),
            tf.cos(tf.clip_by_value(tf.acos(norm_logits) - scaled_margin, self.epislon, self.cos_max_epislon)) - (self.margin + scaled_margin),
            norm_logits,
        )
        # arcface_logits = tf.minimum(arcface_logits, norm_logits) * self.scale
        arcface_logits *= self.scale
        # return arcface_logits
        return tf.keras.losses.categorical_crossentropy(y_true, arcface_logits, from_logits=self.from_logits, label_smoothing=self.label_smoothing)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "margin": self.margin,
                "margin_alpha": self.margin_alpha,
                "mean_std_alpha": self.mean_std_alpha,
                "_batch_mean_": float(self.batch_mean.numpy()),
                "_batch_std_": float(self.batch_std.numpy()),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        _batch_mean_ = config.pop("_batch_mean_", 20.0)
        _batch_std_ = config.pop("_batch_std_", 100.0)
        aa = cls(**config)
        aa.batch_mean.assign(_batch_mean_)
        aa.batch_std.assign(_batch_std_)
        return 
