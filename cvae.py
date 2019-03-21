import os
import tensorflow as tf

from absl import flags
import utils

from discriminator import conditioned_encoder as enc
import generator as gen_modules

tfgan = tf.contrib.gan

model_dir = os.path.dirname(os.path.realpath(__file__))

flags.DEFINE_string('data_dir', os.path.join(model_dir, '../../data/wiki-art'),
                    'Directory, where the data is expected to be found.')
flags.DEFINE_float('learning_rate', 0.0001, 'The learning rate for the neural network.')
flags.DEFINE_float('optimizer_add', 0.09,
                   'Additional optimizer parameter, meaning depending on which optimizer is used.')
flags.DEFINE_float('gd_gamma', 0.99, 'The discouting factor for the history/coming gradient.')
flags.DEFINE_integer('image_size', 128, 'The size of image to use '
                                        '(will be center cropped) [128]')
flags.DEFINE_integer('z_g_dim', 512, 'Dimensionality of latent code z_g for the generator.')
flags.DEFINE_integer('gf_dim', 64, 'Dimensionality of gf. [64]')
flags.DEFINE_integer('ef_dim', 64, 'Dimensionality of ef. [64]')
flags.DEFINE_integer('n_stages', 5, 'Number of stacked Resnet-Blocks, both in Generator and Discriminator.')
flags.DEFINE_integer('n_classes', 14, 'Number of classes that are present in the data set.')

flags.DEFINE_integer('dataset_size', 80000, 'Number of training exmaples in the dataset.')
flags.DEFINE_bool('use_cin', False,
                  'Whether to use conditional_instane_normalization for the generator or not.')
flags.DEFINE_float('kl_weight', 10.0, 'Weighting factor for Kullback-Leibler-Loss.')

FLAGS = flags.FLAGS


def _kullback_leibler_loss(mus, sigmas):
    '''
    Computes the Kullback-Leibler-Loss as defined in the original paper for the vae/cvae
    :param mus: The expectation values of the latent distribution
    :param sigmas: The standard-deviations of the latent distribution
    :return: The mutual information between the 'real' latent distribution, which is assumed to be a standard normal
    distribution and the latent distribution that is mapped from the data by the encoder network
    '''
    return tf.reduce_mean(tf.reduce_sum(mus ** 2 + sigmas - tf.log(sigmas) - 1, axis=1), name='kb_loss')


def _reconstruction_loss(original, reconstructed):
    '''
    Function computes the reconstruction loss as defined in the original formulation. Note that this results in the
    mean squared norm between original and reconstructed image when trying to estimate the model
    parameters using maximum-likelihood-estimation and assuming that the model is gaussian
    :param original: The original image from the dataset
    :param reconstructed: the reconstructed decoder output
    :return: The reconstruction loss that is used for optimization of the model parameters
    '''
    return tf.reduce_mean(tf.reduce_sum(tf.square(original - reconstructed), axis=[1, 2, 3]), name='rec_loss')


class CVAE:

    def __init__(self, config, global_step, device=None):
        # take parameters from train script
        self.config = config
        self.global_step = global_step
        self.device = device

        # get params defined here
        self.learning_rate = FLAGS.learning_rate
        self.optimizer_add = FLAGS.optimizer_add
        self.image_size = FLAGS.image_size
        self.z_g_dim = FLAGS.z_g_dim
        self.gf_dim = FLAGS.gf_dim
        self.ef_dim = FLAGS.ef_dim
        self.n_stages = FLAGS.n_stages
        self.n_classes = FLAGS.n_classes
        self.image_shape = [self.image_size, self.image_size, 3]

        self._build()

    def _build(self):
        '''
        Builds up the conditional variational autoencoder model
        '''

        with tf.device(self.device):
            # define optimizer that's to be applied
            self.optimizer = tf.train.AdamOptimizer(self.learning_rate, self.optimizer_add, name='Adam_optimizer')

            self.increment_global_step = tf.assign_add(self.global_step, 1, name='incr_global_stepb')

            # add image feed ops
            self.batches = utils.get_wikiart_batches(FLAGS.data_dir, self.config.batch_size, FLAGS.image_size,
                                                     shuffle_buffer_size=self.config.shuffle_buffer_size)

            sample_images, _ = self.batches
            vis_images = tf.cast((sample_images + 1.) * 127.5, tf.uint8)
            tf.summary.image('input_image_grid',
                             tfgan.eval.image_grid(
                                 vis_images[:self.config.batch_size],
                                 grid_shape=utils.squarest_grid_size(self.config.batch_size),
                                 image_shape=(self.image_size, self.image_size)))

            images, sparse_labels = self.batches

            print('han sparse_labels.shape', sparse_labels.shape)

            class_logits = tf.zeros((self.config.batch_size, FLAGS.n_classes))
            class_ints = tf.multinomial(class_logits, 1, output_dtype=tf.int32)
            sparse_class = tf.squeeze(class_ints)
            print('han sparse_class.shape', sparse_class.shape)
            assert len(class_ints.get_shape()) == 2
            class_ints = tf.squeeze(class_ints)
            assert len(class_ints.get_shape()) == 1
            class_vector = tf.one_hot(class_ints, FLAGS.n_classes)
            assert len(class_vector.get_shape()) == 2
            assert class_vector.dtype == tf.float32

            # images for the encoder should be in between zero and one
            images = utils.rescale_image(images)

            assert images.dtype == tf.float32

            self.enc_fcn = enc
            # apply encoder function on the images
            latent_code, mus, sigmas = self.enc_fcn(images, sparse_class, self.n_classes, self.ef_dim, self.n_stages)

            if FLAGS.use_cin:
                self.dec_fcn = gen_modules.conditional_instance_normalized_generator
                restored = self.dec_fcn(latent_code, sparse_class, self.n_classes, self.gf_dim, self.n_stages)
            else:
                self.dec_fcn = gen_modules.condition_embedded_generator
                restored = self.dec_fcn(latent_code, sparse_class, self.z_g_dim, self.n_classes, self.gf_dim,
                                        self.n_stages)

            # decode and try to reconstruct conditional image; Note that, due to tanh-activation, the
            # restored images are in between -1 and 1, so it is required to transform the pixel-intensities

            restored = utils.rescale_image(restored)
            assert restored.get_shape()[1] == self.image_size

            # add visualization
            vis_restored = tf.cast(restored * 255.0, tf.uint8)
            tf.summary.image('restored_image_grid',
                             tfgan.eval.image_grid(vis_restored[:self.config.batch_size],
                                                   grid_shape=utils.squarest_grid_size(self.config.batch_size),
                                                   image_shape=(self.image_size, self.image_size)))

            # compute partial losses
            kullback_leibler_loss = _kullback_leibler_loss(mus, sigmas)
            reconstruction_loss = _reconstruction_loss(images, restored)
            # overall loss
            kl_weight = FLAGS.kl_weight
            loss = kl_weight * kullback_leibler_loss + reconstruction_loss

            # summaries
            tf.summary.scalar('overall_loss', loss)
            tf.summary.scalar('reconstruction_loss', reconstruction_loss)
            tf.summary.scalar('kullback_leibler_loss', kullback_leibler_loss)

            n_params = 0
            for var in tf.trainable_variables():
                n_params += utils.prod(var.get_shape().as_list())

            print('Current model has {} parameters that are optimized during training'.format(n_params))

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            print('Number of update ops: ', len(update_ops))

            with tf.control_dependencies(update_ops):
                self.train_op = tf.contrib.training.create_train_op(loss, self.optimizer)
