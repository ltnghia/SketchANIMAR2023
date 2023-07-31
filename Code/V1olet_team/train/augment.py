import tensorflow as tf
from tensorflow.keras import backend as K 
import tensorflow_addons as tfa
import math

from utils import *

def clip_image(image):
    image -= tf.reduce_min(image)
    image_max = tf.reduce_max(image)
    if image_max != 0:
        image = image / image_max
    return image

def rotate_image(image, rot):
    """
    rot: angle (0* - 180*)
    """
    if tf.math.equal(rot, 0):
        return image

    rot = rot * tf.random.normal([1], dtype='float32')
    rotation = math.pi * rot / 180.0
    image = tfa.image.rotate(image, rotation)
    return image

def cutout(image, pad_factor, replace=None):
    if tf.math.equal(pad_factor, 0):
        return image

    replace = 1.0
    # if replace is None:
    #     replace = tf.random.uniform(shape=[], minval=0, maxval=1., dtype=tf.float32)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    z_o = tf.random.uniform(shape=[], minval=0.5, maxval=1., dtype=tf.float32)
    pad_size = tf.cast(z_o * tf.cast(pad_factor, tf.float32) * tf.cast(image_height, tf.float32), tf.int32)

    # Sample the center location in the image where the zero mask will be applied.
    cutout_center_height = tf.random.uniform(shape=[], minval=0, maxval=image_height, dtype=tf.int32)

    cutout_center_width = tf.random.uniform(shape=[], minval=0, maxval=image_width, dtype=tf.int32)

    lower_pad = tf.maximum(0, cutout_center_height - pad_size)
    upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
    left_pad = tf.maximum(0, cutout_center_width - pad_size)
    right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
    return image

def solarize(image, threshold):
    return tf.where(image < threshold, image, 1 - image)

def equalize_image(image):
    """
    Implements Equalize function from PIL using TF ops.
    image: [0, 1]
    """
    def scale_channel(im, c):
        """Scale the data in the channel to implement equalize."""
        im = clip_image(im)
        im = im * 255 

        im = tf.cast(im[:, :, c], tf.int32)
        # Compute the histogram of the image channel.
        histo = tf.histogram_fixed_width(im, [0, 255], nbins=256)

        # For the purposes of computing the step, filter out the nonzeros.
        nonzero = tf.where(tf.not_equal(histo, 0))
        nonzero_histo = tf.reshape(tf.gather(histo, nonzero), [-1])
        step = (tf.reduce_sum(nonzero_histo) - nonzero_histo[-1]) // 255

        def build_lut(histo, step):
            # Compute the cumulative sum, shifting by step // 2
            # and then normalization by step.
            lut = (tf.cumsum(histo) + (step // 2)) // step
            # Shift lut, prepending with 0.
            lut = tf.concat([[0], lut[:-1]], 0)
            # Clip the counts to be in range.  This is done
            # in the C code for image.point.
            return tf.clip_by_value(lut, 0, 255)

        # If step is zero, return the original image.  Otherwise, build
        # lut from the full histogram and step and then index from it.
        result = tf.cond(tf.equal(step, 0), lambda: im, lambda: tf.gather(build_lut(histo, step), im))

        return tf.cast(result, tf.float32) / 255

    # Assumes RGB for now.  Scales each channel independently
    # and then stacks the result.
    s1 = scale_channel(image, 0)
    s2 = scale_channel(image, 1)
    s3 = scale_channel(image, 2)
    image = tf.stack([s1, s2, s3], 2)
    return image

def color_image(img, hue, sat, cont, bri):
    img = tf.image.random_hue(img, hue) if hue is not None else img
    img = tf.image.random_saturation(img, sat[0], sat[1]) if sat is not None else img
    img = tf.image.random_contrast(img, cont[0], cont[1]) if cont is not None else img
    img = tf.image.random_brightness(img, bri) if bri is not None else img
    return img

def cutout_mask(image, pad_factor, replace=None):
    if tf.math.equal(pad_factor, 0):
        return image

    if replace is None:
        replace = tf.random.uniform(shape=[], minval=0, maxval=1., dtype=tf.float32)

    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    z_o = tf.random.uniform(shape=[], minval=0.8, maxval=1.2, dtype=tf.float32)
    pad_size = tf.cast(z_o * (1.0-tf.cast(pad_factor, tf.float32)) * tf.cast(image_height, tf.float32), tf.int32)

    # Sample the center location in the image where the zero mask will be applied.

    lower_pad = pad_size
    upper_pad = 0
    left_pad = 0
    right_pad = 0

    # edge_rate_left = tf.random.uniform(shape=[], minval=0.0, maxval=0.1, dtype=tf.float32)
    # left_pad = tf.maximum(0, tf.cast(edge_rate*tf.cast(image_width,tf.float32), tf.int32))
    # right_pad = tf.maximum(0, tf.cast(edge_rate*tf.cast(image_width,tf.float32), tf.int32))

    cutout_shape = [image_height - (lower_pad + upper_pad), image_width - (left_pad + right_pad)]
    padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
    mask = tf.pad(tf.zeros(cutout_shape, dtype=image.dtype), padding_dims, constant_values=1)
    mask = tf.expand_dims(mask, -1)
    mask = tf.tile(mask, [1, 1, 3])
    image = tf.where(tf.equal(mask, 0), tf.ones_like(image, dtype=image.dtype) * replace, image)
    return image

def build_augment():
    setting_cfg = get_settings('setting.yaml')
    aug_cfg = get_settings('augment.yaml')

    def augment_img(img):
        cnt = 0

        P0_left_right = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['flip_left_right_prob'], tf.int32)
        if P0_left_right == 1:
            img = tf.image.flip_left_right(img)

        P0_up_down = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['flip_up_down_prob'], tf.int32)
        if P0_up_down == 1:
            img = tf.image.flip_up_down(img)

        P1 = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['color_prob'], tf.int32)
        if P1 == 1:
            img = color_image(img, aug_cfg['hue'], aug_cfg['sature'], aug_cfg['contrast'], aug_cfg['brightness'])
            cnt += 1

        P2 = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['rotate_prob'], tf.int32)
        if P2 == 1:
            img = rotate_image(img, aug_cfg['rotate'])
            cnt += 1

        P3 = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['cutout_prob'], tf.int32)
        if cnt < 2 and P3 == 1:
            img = cutout(img, aug_cfg['cutout_pad_factor'])
            cnt += 1

        P4 = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['solarize_prob'], tf.int32)
        if cnt < 3 and P4 == 1:
            img = solarize(img, aug_cfg['solarize_threshold'])
            cnt += 1

        P5 = tf.cast(tf.random.uniform([], 0, 1) < aug_cfg['equalize_prob'], tf.int32)
        if cnt < 3 and P5 == 1:
            img = equalize_image(img)
            cnt += 1

        img = clip_image(img)

        if setting_cfg['im_size_before_crop'] is not None:
            P6 = tf.cast(tf.random.uniform([], 0, 1) < setting_cfg['crop_prob'], tf.int32)
            if P6 == 1:
                img = tf.image.random_crop(img, size=(setting_cfg['im_size'], setting_cfg['im_size'], 3))
            else:
                img = img = tf.image.resize(img, (setting_cfg['im_size'], setting_cfg['im_size']))

        return img

    return augment_img

if __name__ == '__main__':
    import cv2
    from utils import *

    os.environ["CUDA_VISIBLE_DEVICES"]=""

    sample_inp = '/home/minint-t14g3hk-local/hieunmt/tf_reid_SHREC23/unzip/shrec23_reid_dataset_merge/0ae964801539411d/3_Image0005_sketch.png'
    img = cv2.imread(sample_inp)
    cv2.imwrite("sample_inp.png", img)

    img = img[...,::-1]
    img = img / 255.0
    img = np.float32(img)

    augment_img = build_augment()
    img = augment_img(img)

    # img = rotate_image(img, rotate)

    # img = cutout(img, cutout_pad_factor)

    # img = color_image(img, hue, sature, contrast, brightness)

    # img = solarize(img, solarize_threshold)

    # img = equalize_image(img)

    # img = cutout_mask(img, cutout_mask_height_factor)

    img = clip_image(img)

    img = img.numpy()[...,::-1] * 255
    cv2.imwrite("sample_aug.png", img)
    
