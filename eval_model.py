import os

import tensorflow as tf
from absl import flags
import numpy as np

from utils import make_z_normal, choose_gpu, get_wikiart_batches, rescale_image, squarest_grid_size

# uncomment, if starting main script and not using the evaluation function from train script
# gpu = str(choose_gpu())
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu


tfgan = tf.contrib.gan
gfile = tf.gfile

eval_dir = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_integer('n_stat_data_points', 12800, 'The number of data points that are used as input for the statistics')
flags.DEFINE_integer('statistics_batch_size', 32,
                     'The number of data points that are used as input for the statistics')
flags.DEFINE_string('image_stats_directory', os.path.join(eval_dir, '../../data/wiki-art/'),
                    'The directory, where the pre-computed activations for the current data set are saved as npy-files.')
flags.DEFINE_integer('eval_steps', 5000
                     , 'The train steps between two consecutive evaluations.')

FLAGS = flags.FLAGS


def _run_inception(images, output_tensors):
    '''
    Returns activations for certain layers of inception-v3-model, for a mini-batch of images
    :param images: the images, for which the activations should be computed
    :param output_tensors: list of layers, for which the inception output should be computed
    :return:
    '''
    images = tfgan.eval.preprocess_image(images)
    return tfgan.eval.run_inception(images, graph_def=None, output_tensor=output_tensors)


class Evaluator:

    def __init__(self, model, config, device, logdir, latent_dim = None):
        '''
        Evaluator class is for computing scores and saving checkpoints, if new score falls below old score
        :param generator: the generator function
        :param config:  parameters file
        :param device: the device to place the computations on
        :param logdir: The directory, where the event file for tensorboard should be placed
        :param latent_dim: the dimension of the latent space when evaluating a cvae_gan
        '''
        # add attributes
        self.model = model
        self.config = config
        self.device = device
        self.logdir = logdir

        self.save_dir = os.path.join(self.logdir, 'best_model')

        # create directory to store the best model, if non-existent
        gfile.MakeDirs(self.save_dir)

        # add saver in order to save only the checkpoint with the best fid
        self.best_model_saver = tf.train.Saver(max_to_keep=1, filename='best_model.ckpt', name='save_best')

        self.fid = tf.get_variable('best_fid', shape=(), dtype=tf.float32,
                                   initializer=tf.initializers.constant(100000))

        #define the latent dimension, if the generator is from cvaegan
        self.latent_dim = latent_dim

    def _save(self, fid):
        # actualize fid
        self.fid = fid

        # save best model
        with tf.Session().as_default() as sess:
            self.best_model_saver.save(sess, self.save_dir, global_step=tf.train.get_global_step)

        return tf.no_op

    def compute_stats(self):
        '''
        Adds a new tf.SummarySaverHook for logging of IS- and FID-scores to the graph and returns it
        :return: A new tf.SummarySaverHook
        '''
        # place it on the recent device
        with tf.device(self.device):
            # load stored activations for the data set
            dataset_stats_file = FLAGS.image_stats_directory + 'dataset_pools.npy'
            n_data_points = FLAGS.n_stat_data_points
            local_batch_size = self.config.batch_size

            # load the output of the inception-v3 model after final pooling layer,
            # has to be pre-computed
            dataset_pools = tf.stack(np.load(dataset_stats_file))

            # init for generated images
            if self.latent_dim is None:
                z_dim = self.config.z_g_dim
            else:
                z_dim = self.latent_dim

            n_classes = self.config.n_classes
            nf_init = self.config.gf_dim
            n_stages = self.config.n_stages

            # create generator input
            dist = make_z_normal(local_batch_size, z_dim, 'z_eval')
            label_logits = tf.zeros((self.config.batch_size, FLAGS.n_classes))
            gen_class_ints = tf.multinomial(label_logits, 1, output_dtype=tf.int32)
            gen_sparse_class = tf.squeeze(gen_class_ints)

            if self.config.use_cin:
                # if using conditional instance normalization, set is_training to False
                init_gen = self.model(dist, gen_sparse_class, n_classes, nf_init, n_stages, name='generator_cin',
                                      is_training=False)
            else:
                init_gen = self.model(dist, gen_sparse_class, z_dim, n_classes, nf_init, n_stages)

            init_gen = tf.Print(init_gen, [init_gen], 'Starting stats-inferences, first batch: ', first_n=1)

            pools, logits = _run_inception(init_gen, ['pool_3:0', 'logits:0'])

            # construct while loop, in order to avoid running out of memory
            def _while_cond(g_pools, g_logits, it):
                return tf.less(it, n_data_points // local_batch_size)

            def _while_body(g_pools, g_logits, it):
                '''
                Body of the tf.while_loop, that is excuted every iteration
                :param g_pools: The outputs of last inception-v3-pooling-layers after all mini-batch-inferences done so far
                :param g_logits: The output-logits of inception-v3 after all mini-batch-inferences done so far
                :param it: loop-variable
                :return: g_pools and g_logits with new appendices and a loop-variable that is incemented by 1
                '''
                with tf.control_dependencies([g_pools, g_logits]):
                    # create images and run inception
                    if self.config.use_cin:
                        # if using conditional instance normalization, set is_training to False
                        current_gen = self.model(dist, gen_sparse_class, n_classes, nf_init, n_stages,
                                                 name='generator_cin', is_training=False)
                    else:
                        current_gen = self.model(dist, gen_sparse_class, z_dim, n_classes, nf_init, n_stages)

                    current_pools, current_logits = _run_inception(current_gen, ['pool_3:0', 'logits:0'])

                    g_pools = tf.concat([g_pools, current_pools], axis=0)
                    g_logits = tf.concat([g_logits, current_logits], axis=0)

                    # increment it
                    increment = tf.add(it, 1)

                    return (g_pools, g_logits, increment)

            i = tf.constant(1)

            g_pools_complete, g_logits_complete, _ = tf.while_loop(_while_cond, _while_body, [pools, logits, i],
                                                                   shape_invariants=[
                                                                       tf.TensorShape([None, 2048]),
                                                                       tf.TensorShape([None, 1008]), i.get_shape()],
                                                                   parallel_iterations=1,
                                                                   back_prop=False, swap_memory=True,
                                                                   name='eval_pools_logits')

            assert dataset_pools.get_shape() == tf.TensorShape([n_data_points, 2048])

            g_pools_complete.set_shape([n_data_points, 2048])
            g_logits_complete.set_shape([n_data_points, 1008])

            # compute scores
            inception_score = tfgan.eval.classifier_score_from_logits(g_logits_complete)
            fid = tfgan.eval.frechet_classifier_distance_from_activations(dataset_pools, g_pools_complete)

            print_fid = tf.Print(fid, [fid], 'Finished fid-computation; FID = ')
            print_is = tf.Print(inception_score, [inception_score], 'Finished inception-score-computation; IS = ')

            # fixme test this implementation
            # save best_model, if actual fid is below best up to now
            # tf.cond(tf.less(fid, self.fid), lambda: self._save(fid), lambda: tf.no_op,
            #         name='conditional_checkpoint_saving')

            # add summaries
            inception_proto = tf.summary.scalar('inception_score', print_is,
                                                collections='EVALUATION_SUMMARIES')
            fid_proto = tf.summary.scalar('fid', print_fid, collections='EVALUATION_SUMMARIES')

            eval_steps = FLAGS.eval_steps
            return tf.train.SummarySaverHook(save_steps=eval_steps, output_dir=self.logdir,
                                             summary_op=[inception_proto, fid_proto])


def eval_cvae(decoder, device, config, logdir,latent_dim):
    '''
    This function evaluates a generator of a cvae with gaussian noise input instead of the latent code which is
    ectracted by the encoder network during trainig
    :param decoder:
    :param device:
    :param config:
    :param logdir:
    :param latent_dim:
    :return:
    '''
    with tf.device(device):

        batch_size = config.batch_size
        n_classes = config.n_classes
        nf_init = config.gf_dim
        n_stages = config.n_stages
        z_dim = latent_dim
        image_size = config.image_size
        save_steps = config.save_summaries_steps

        dist = tf.random_normal(shape=(batch_size, z_dim), dtype=tf.float32)

        gen_class_logits = tf.zeros((config.batch_size, FLAGS.n_classes))
        gen_class_ints = tf.multinomial(gen_class_logits, 1, output_dtype=tf.int32)
        gen_sparse_class = tf.squeeze(gen_class_ints)

        if config.use_cin:
            # if using conditional instance normalization, set is_training to False
            fake_images = decoder(dist, gen_sparse_class, n_classes, nf_init, n_stages, name='generator_cin',
                                  is_training=False)
        else:

            fake_images = decoder(dist, gen_sparse_class, z_dim, n_classes, nf_init, n_stages)

        fake_images = rescale_image(fake_images)
        assert fake_images.get_shape()[1] == image_size

        # add visualization
        vis_restored = tf.cast(fake_images * 255.0, tf.uint8)
        evaluation_grid = tf.summary.image('decoder_evaluation',
                                           tfgan.eval.image_grid(vis_restored[:config.batch_size],
                                                                 grid_shape=squarest_grid_size(config.batch_size),
                                                                 image_shape=(image_size, image_size)))

        return tf.train.SummarySaverHook(save_steps=save_steps, output_dir=logdir, summary_op=evaluation_grid)


def main(_):
    # parameters
    data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/wiki-art')
    image_size = 128
    batch_size = FLAGS.statistics_batch_size
    shuffle_buffer_size = 50000
    n_data_points = FLAGS.n_stat_data_points

    filename = FLAGS.image_stats_directory + 'dataset_pools_small.npy'

    # Set up device to use
    device = '/cpu:0'

    with tf.Session() as sess:
        with tf.device(device):
            data_init, _ = get_wikiart_batches(data_dir, batch_size, image_size, shuffle_buffer_size)
            data_pools = _run_inception(data_init, ['pool_3:0'])[0]

            it = tf.constant(1)

            def _while_cond(data_pools, it):
                return tf.less(it, n_data_points // batch_size)

            def _while_body(data_pools, it):
                '''
                Body of the tf.while_loop, that is excuted every iteration
                :param data_pools: The outputs of last inception-v3-pooling-layers after all mini-batch-inferences done so far
                :param it: loop-variable
                :return: data_pools with new appendices and a loop-variable that is incemented by 1
                '''
                with tf.control_dependencies([data_pools]):
                    # create images and run inception
                    batch, _ = get_wikiart_batches(data_dir, batch_size, image_size, shuffle_buffer_size)
                    current_pools = _run_inception(batch, ['pool_3:0'])[0]

                    data_pools = tf.concat([data_pools, current_pools], axis=0)

                    return (data_pools, tf.add(it, 1))

            data_pools, _ = tf.while_loop(_while_cond, _while_body, [data_pools, it],
                                          shape_invariants=[tf.TensorShape([None, 2048]), it.get_shape()],
                                          parallel_iterations=1, back_prop=False, swap_memory=True,
                                          name='compute_data_stats')

            print('Computing dataset stats, this may take a while...')
            output = sess.run(data_pools)
            print('Computation finished, output has shape: {}'.format(output.shape))

            assert output.shape == (n_data_points, 2048)

            # save as numpy array
            np.save(filename, output)


if __name__ == '__main__':
    tf.app.run()
