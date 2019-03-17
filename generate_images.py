import os
import utils
import datetime

import numpy as np
from matplotlib import pyplot as plt

gpu = str(utils.choose_gpu())
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu

import tensorflow as tf

from utils import squarest_grid_size
from absl import flags

cur_dir = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_string('model_dir', os.path.join(cur_dir, '../../data/wiki-art/vdbgan/checkpoints/finished'),
                    'Directory, where the model to be used is located.')
flags.DEFINE_string('model','vdbgan_wiki-art_32_0.1_128_2019-01-11-08-41-49','The model to use to generate the images.')
flags.DEFINE_integer('number_images',20,'The number of generated images.')
flags.DEFINE_integer('label',1,'The label that should be generated [-1].')
flags.DEFINE_integer('n_labels',8,'The label that should be generated [-1].')
flags.DEFINE_bool('show_images',True,'Whether to show images after generation or not. [True]')
flags.DEFINE_string('results_dir',os.path.normpath(os.path.join(cur_dir, '../../results/vdbgan')),'Directory to store the results at.')
flags.DEFINE_string('generator_name','generator_cin','The name of the generator model to use for the image generation.')

FLAGS = flags.FLAGS
gfile = tf.gfile

def main(_):

    model_dir = os.path.normpath(os.path.join(FLAGS.model_dir,FLAGS.model))
    save_dir = os.path.join(FLAGS.results_dir,FLAGS.model)

    gfile.MakeDirs(save_dir)

    device = '/gpu:0'

    is_path = os.path.exists(model_dir)

    # latest_ckpt = tf.train.latest_checkpoint(model_dir,latest_filename='checkpoint')
    # filename = ".".join([latest_ckpt,'meta'])
    filename = os.path.join(model_dir,'model.ckpt-732551.meta')
    model_name = os.path.join(model_dir,'model.ckpt-732551')

    sess = tf.Session()
    graph = tf.get_default_graph()

    with graph.as_default():
        with sess.as_default():

            loader = tf.train.import_meta_graph(filename,clear_devices=True)
            loader.restore(sess,model_name)

            # sess.run(tf.global_variables_initializer())

            noise_input = graph.get_tensor_by_name('z_train:0')
            label_input = graph.get_tensor_by_name('Squeeze:0')

            batch_size = noise_input.get_shape().as_list()[0]

            output = graph.get_tensor_by_name('{}/g_final_act:0'.format(FLAGS.generator_name))
            i_images = tf.cast((output + 1) * 127.5, tf.int32)

            images = []
            n_it = np.ceil(FLAGS.number_images / batch_size).astype(np.int)

            for i in range(n_it):
                # define generator input
                # random vector
                z_gen = np.random.normal(size=noise_input.get_shape())

                # labels
                if FLAGS.label>= 0 and FLAGS.label < FLAGS.n_labels:
                    labels = FLAGS.label * np.ones(label_input.get_shape(),dtype=np.int)
                else:
                    labels = np.random.randint(0,FLAGS.n_labels-1,size=batch_size,dtype=np.int)

                # create feed dict
                feed_dict = {noise_input: z_gen, label_input: labels}

                # generate images
                images +=[sess.run(i_images,feed_dict)]

            # rescale images to integer values and split it so that the desired output size is obtained
            # images = [((image+1.0)*127.5).astype(np.int) for image in images]
            # fimages = ((images+1.0)*127.5).astype(np.int)
            # select first number_images images

            # saved_images = np.squeeze(split[0])
            saved_images = images[0]
            saved_images = saved_images[:FLAGS.number_images]

            # define grid for figure
            grid = squarest_grid_size(FLAGS.number_images)
            height_ratios = []
            for i in range(grid[0]):
                height_ratios += [saved_images.shape[1]]

            fig, axarr = plt.subplots(grid[0], grid[1], figsize=(grid[0] * 0.375, grid[1] * 0.375),
                                      gridspec_kw={'height_ratios': height_ratios})

            for nr, image in enumerate(saved_images):

                axarr[nr // grid[1]][nr % grid[1]].imshow(image)
                axarr[nr // grid[1]][nr % grid[1]].axis('off')
                axarr[nr // grid[1]][nr % grid[1]].plot()


            plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

            # show images, if required
            if FLAGS.show_images:
                plt.show()

            imagename = save_dir + '/{}_label_{}_n_{}.png'.format(FLAGS.model,FLAGS.label,FLAGS.number_images)
            plt.savefig(imagename)
            plt.close('all')

if __name__=='__main__':
    tf.app.run()