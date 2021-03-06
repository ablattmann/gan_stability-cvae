import tensorflow as tf
import os
import functools
import math
import sympy
import operator


def rescale_image(images):
    '''
    Rescale images, so that the pixel intensities are in between 0 and 1
    :param images: images with intensities in between -1 and 1
    :return:
    '''
    return tf.cast((images + 1.0) / 2.0, tf.float32)

def prod(iterable):
    return functools.reduce(operator.mul,iterable,1)

def make_z_normal(batch_size, z_dim, name=None):
    '''
    Make random noises tensors with normal distribution  feeding into the generator
    :param batch_size: the batch_size for z
    :param z_dim: The dimension of the z (noise) vector.
    :return:
    '''
    shape = [batch_size, z_dim]
    z = tf.random_normal(shape, name=name, dtype=tf.float32)
    return z


def get_wikiart_batches(data_dir,
                        batch_size,
                        image_size,
                        shuffle_buffer_size=70000):
    '''
    Fetches data from a tf_records-file
    :param data_dir:  the path of the directory, the data file is placed in
    :param batch_size: batch size for the given model
    :param image_size: integer depicting the height and width of the images in the
    dataset (net is trained on images of equal height and width)
    :param shuffle_buffer_size: Number of elements to prefetch; IMPORTANT: Consider available RAM on training machine. Too
    large buffer sizes will cause the RAM to overflow; Nevertheless, choose this parameter the biggest possible.
    :return: A tuple, containing a 4D tensor, a batch of color images (batch_size, height, width, 3)
    (normalized in between -1 and 1) and their labels as a 1D tensor of integers
    '''
    filenames = tf.data.Dataset.list_files(os.path.join(data_dir, 'wiki-art.tfrecords'))
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(shuffle_buffer_size))

    def _extract_image_and_label(record):
        """Extracts and preprocesses the image and label from the record."""
        features = tf.parse_single_example(
            record,
            features={
                'image_raw': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        image.set_shape(image_size * image_size * 3)
        image = tf.reshape(image, [image_size, image_size, 3])

        image = tf.cast(image, tf.float32) * (2. / 255) - 1.

        label = tf.cast(features['label'], tf.int32)
        return image, label

    dataset = dataset.apply(tf.contrib.data.map_and_batch(lambda x: _extract_image_and_label(x), batch_size))
    iterator = dataset.make_one_shot_iterator()
    batches = iterator.get_next()

    # iterator makes the data lose shape information --> restore it
    batches[0].set_shape([batch_size, image_size, image_size, 3])
    batches[1].set_shape((batch_size,))

    return batches


def squarest_grid_size(num_images):
    """Calculates the size of the most square grid for num_images.
    Calculates the largest integer divisor of num_images less than or equal to
    sqrt(num_images) and returns that as the width. The height is
    num_images / width.
    Args:
      num_images: The total number of images.
    Returns:
      A tuple of (height, width) for the image grid.
    """
    divisors = sympy.divisors(num_images)
    square_root = math.sqrt(num_images)
    width = 1
    for d in divisors:
        if d > square_root:
            break
        width = d

    return (num_images // width, width)


def checkID(id, length):
    if id > (length - 1):
        return -1
    else:
        return id


def choose_gpu():
    '''
    Searches for available GPU devices and automatically chooses the one with the most memory available;
    asks the user via command line which GPU to choose, if none available
    :return: The GPU index of the GPU to be used
    '''
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    os.system('nvidia-smi -q | grep Processes > tmp')
    processes = [(x.find("None") != -1) for x in open('tmp', 'r').readlines()]
    max_memory = 0
    id = -1
    n = 0
    for i, processes in enumerate(processes):
        print("GPU: ", i, "Memory Available: ", memory_available[i], "MB, ",
              "%s" % ("FREE" if processes else "NOT FREE"))
        n = n + 1
        if (processes and memory_available[i] > max_memory):
            max_memory = memory_available[i]
            id = i

    if (max_memory != 0):
        print("Using GPU: ", id)
        return id

    else:
        text = input("Every GPU is currently in use. Which GPU do you want to use?")
        return checkID(int(text), n)


