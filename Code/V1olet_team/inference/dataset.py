import tensorflow as tf

def get_decode_function(target_size=(384, 384)):
    def decode_img(path):
        """
        path to image
        """
        file_bytes = tf.io.read_file(path)
        img = tf.io.decode_image(file_bytes, channels=3, expand_animations=False)
        img = tf.image.resize(img, target_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img
    return decode_img

def build_test_dataset(paths, target_size=(256,256), bsize=32, augment=None):
    AUTO = tf.data.experimental.AUTOTUNE
    dset = tf.data.Dataset.from_tensor_slices((paths))
    decode_fn = get_decode_function(target_size)
    dset = dset.map(decode_fn, num_parallel_calls=AUTO)
    dset = dset.map(augment, num_parallel_calls=AUTO) if augment is not None else dset
    # dset = dset.map(lambda x,y:(augment(x),y), num_parallel_calls=AUTO) if augment is not None else dset
    dset = dset.batch(bsize)
    dset = dset.prefetch(AUTO)
    return dset